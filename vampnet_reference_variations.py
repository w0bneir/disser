"""Общие функции prompt-free прототипа вариаций на акустических токенах.

Модуль намеренно не импортирует VampNet/LAC при загрузке. Благодаря этому
валидацию, маски и обработку WAV можно тестировать в основном окружении без
загрузки весов и без обращения к GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


SAMPLE_RATE = 44_100
MINIMUM_TOTAL_VRAM_MIB = 12_000
MINIMUM_FREE_VRAM_MIB = 10_000

MODEL_ASSET_SIZES = {
    "codec.pth": 600_996_465,
    "coarse.pth": 1_332_182_321,
    "c2f.pth": 1_101_898_865,
}
MODEL_ASSET_SHA256 = {
    "codec.pth": "3db3fa43ab5d160439ddb81fc540b5573ad5ae962230de3fc5b47d218845b855",
    "coarse.pth": "78e4ad4f8398e8ec3651bc5e5c6ea2995e1080b6226be186723ccf4320c9756c",
    "c2f.pth": "b10ea2d45459d34edb773cbacd71f40f7baa1f4e75ac8bcd93b022ac69f8fa63",
}


@dataclass(frozen=True)
class AudioDescription:
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    peak: float
    rms: float


def describe_audio(audio: np.ndarray, sample_rate: int) -> AudioDescription:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        channels = 1
        frames = values.shape[0]
    elif values.ndim == 2:
        frames, channels = values.shape
    else:
        raise ValueError(f"Ожидался mono/stereo массив, получена форма {values.shape}")
    return AudioDescription(
        sample_rate=int(sample_rate),
        channels=int(channels),
        frames=int(frames),
        duration_seconds=float(frames / sample_rate),
        peak=float(np.max(np.abs(values))) if values.size else 0.0,
        rms=float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
    )


def load_reference_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Загрузить WAV, свести в mono и точно привести к частоте кодека."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Пустой аудиофайл: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"В аудиофайле есть NaN/Inf: {path}")
    mono = audio.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        divisor = int(np.gcd(source_rate, sample_rate))
        mono = signal.resample_poly(
            mono,
            sample_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)
        expected_frames = int(round(audio.shape[0] * sample_rate / source_rate))
        mono = fix_length(mono, expected_frames)
    return mono


def fix_length(audio: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if frames < 1:
        raise ValueError("Целевая длина должна быть положительной")
    if values.shape[0] >= frames:
        return values[:frames].copy()
    return np.pad(values, (0, frames - values.shape[0]))


def match_rms_and_limit(
    audio: np.ndarray,
    reference: np.ndarray,
    *,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """Вернуть громкость референса без peak-нормализации каждого дубля."""
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    source_rms = float(np.sqrt(np.mean(np.square(values))))
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    if source_rms <= 1e-9 or target_rms <= 1e-9:
        raise ValueError("Нельзя согласовать громкость тишины")
    values *= target_rms / source_rms
    peak = float(np.max(np.abs(values)))
    if peak > peak_limit:
        values *= peak_limit / peak
    return values.astype(np.float32)


def prepare_codec_input(
    audio: np.ndarray,
    *,
    target_dbfs: float = -24.0,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """Быстрая детерминированная RMS-нормализация перед токенизацией.

    Официальный UI считает integrated LUFS. На Windows реализация audiotools
    оказалась неприемлемо медленной даже для 1.7 секунды, поэтому прототип явно
    использует RMS dBFS и фиксирует это отличие в metadata.
    """
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(values))))
    if rms <= 1e-9:
        raise ValueError("Нельзя подготовить тишину")
    target_rms = 10.0 ** (target_dbfs / 20.0)
    values *= target_rms / rms
    peak = float(np.max(np.abs(values)))
    if peak > peak_limit:
        values *= peak_limit / peak
    return values.astype(np.float32)


def envelope(audio: np.ndarray, *, frame_length: int = 2048, hop: int = 512) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if values.shape[0] < frame_length:
        values = np.pad(values, (0, frame_length - values.shape[0]))
    frame_count = 1 + int(np.ceil((values.shape[0] - frame_length) / hop))
    padded = np.pad(values, (0, max(0, (frame_count - 1) * hop + frame_length - values.shape[0])))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop]
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    count = min(left.size, right.size)
    if count < 2:
        return 0.0
    left = np.asarray(left[:count], dtype=np.float64)
    right = np.asarray(right[:count], dtype=np.float64)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Диагностические, но не заменяющие прослушивание метрики."""
    count = min(reference.size, candidate.size)
    target = np.asarray(reference[:count], dtype=np.float64)
    output = np.asarray(candidate[:count], dtype=np.float64)
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    output_rms = float(np.sqrt(np.mean(np.square(output))))

    _, _, target_stft = signal.stft(
        target,
        fs=SAMPLE_RATE,
        nperseg=2048,
        noverlap=1536,
        boundary=None,
    )
    _, _, output_stft = signal.stft(
        output,
        fs=SAMPLE_RATE,
        nperseg=2048,
        noverlap=1536,
        boundary=None,
    )
    target_mag = np.abs(target_stft)
    output_mag = np.abs(output_stft)
    spectral_convergence = float(
        np.linalg.norm(output_mag - target_mag) / max(np.linalg.norm(target_mag), 1e-12)
    )
    log_spectral_distance_db = float(
        np.sqrt(
            np.mean(
                np.square(
                    20.0 * np.log10(output_mag + 1e-7)
                    - 20.0 * np.log10(target_mag + 1e-7)
                )
            )
        )
    )
    return {
        "waveform_correlation": _correlation(target, output),
        "envelope_correlation": _correlation(envelope(target), envelope(output)),
        "rms_ratio": float(output_rms / max(target_rms, 1e-12)),
        "spectral_convergence": spectral_convergence,
        "log_spectral_distance_db": log_spectral_distance_db,
    }


def technical_audio_gate(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    peak_limit: float = 1.0,
) -> tuple[bool, list[str]]:
    """Отсечь только технический брак; перцептивное качество оценивает человек."""
    failures: list[str] = []
    if candidate.size != reference.size:
        failures.append(f"длина {candidate.size} вместо {reference.size} отсчётов")
    if not np.isfinite(candidate).all():
        failures.append("обнаружены NaN/Inf")
    peak = float(np.max(np.abs(candidate))) if candidate.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(candidate)))) if candidate.size else 0.0
    if peak > peak_limit + 1e-6:
        failures.append(f"clipping: peak={peak:.4f}")
    if rms <= 1e-6:
        failures.append("результат является тишиной")
    return not failures, failures


def validate_model_assets(
    model_dir: Path,
    *,
    required: tuple[str, ...],
) -> dict[str, dict[str, int | str]]:
    report: dict[str, dict[str, int | str]] = {}
    for filename in required:
        if filename not in MODEL_ASSET_SIZES:
            raise ValueError(f"Неизвестный файл модели: {filename}")
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Не найден файл модели: {path}")
        actual_size = path.stat().st_size
        expected_size = MODEL_ASSET_SIZES[filename]
        if actual_size != expected_size:
            raise ValueError(
                f"Неверный размер {path}: {actual_size}, ожидалось {expected_size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        expected_hash = MODEL_ASSET_SHA256[filename]
        if actual_hash != expected_hash:
            raise ValueError(
                f"Неверный SHA256 {path}: {actual_hash}, ожидалось {expected_hash}"
            )
        report[filename] = {
            "path": str(path.resolve()),
            "bytes": actual_size,
            "sha256": actual_hash,
        }
    return report


def build_reference_mask(
    code_shape: tuple[int, int, int],
    *,
    upper_codebook_mask: int = 3,
    periodic_prompt: int = 7,
    periodic_offset: int = 0,
    attack_tokens: int = 0,
) -> np.ndarray:
    """Детерминированная версия conservative VampNet mask.

    1 означает пересэмплирование токена, 0 — жёсткое сохранение референса.
    Нижние codebook-и несут основной каркас; редкие временные якоря удерживают
    событие от дрейфа, но не копируют всю волну.
    """
    batch, codebooks, steps = code_shape
    if batch < 1 or codebooks < 1 or steps < 1:
        raise ValueError(f"Некорректная форма кодов: {code_shape}")
    if not 0 <= upper_codebook_mask < codebooks:
        raise ValueError("upper_codebook_mask вне диапазона codebook-ов")
    if periodic_prompt < 0:
        raise ValueError("periodic_prompt не может быть отрицательным")
    if attack_tokens < 0:
        raise ValueError("attack_tokens не может быть отрицательным")

    mask = np.zeros(code_shape, dtype=np.int64)
    mask[:, upper_codebook_mask:, :] = 1
    if periodic_prompt:
        offset = periodic_offset % periodic_prompt
        mask[:, :, offset::periodic_prompt] = 0
    if attack_tokens:
        mask[:, :, : min(attack_tokens, steps)] = 0
    return mask


def build_tiered_reference_mask(
    code_shape: tuple[int, int, int],
    *,
    coarse_start: int = 2,
    coarse_stop: int = 4,
    coarse_anchor_period: int = 7,
    coarse_anchor_offset: int = 0,
    fine_start: int = 4,
    fine_resample_period: int = 4,
    fine_resample_offset: int = 0,
    attack_tokens: int = 0,
) -> np.ndarray:
    """Сместить изменения в mid-level tokens, сохранив большую часть fine detail.

    В codebook-ах ``coarse_start:coarse_stop`` меняется большинство token-frame,
    кроме временных якорей. В верхних codebook-ах меняется только один frame из
    ``fine_resample_period``. Такая маска является причинной проверкой вывода из
    listening gate: v1 менял fine detail, давал металлическую окраску, но почти
    не давал слышимой вариативности.
    """
    batch, codebooks, steps = code_shape
    if batch < 1 or codebooks < 1 or steps < 1:
        raise ValueError(f"Некорректная форма кодов: {code_shape}")
    if not 0 <= coarse_start < coarse_stop <= codebooks:
        raise ValueError("Некорректный диапазон coarse codebook-ов")
    if not coarse_stop <= fine_start <= codebooks:
        raise ValueError("fine_start должен быть не ниже coarse_stop")
    if coarse_anchor_period < 1 or fine_resample_period < 1:
        raise ValueError("Периоды маски должны быть положительными")
    if attack_tokens < 0:
        raise ValueError("attack_tokens не может быть отрицательным")

    mask = np.zeros(code_shape, dtype=np.int64)
    mask[:, coarse_start:coarse_stop, :] = 1
    coarse_offset = coarse_anchor_offset % coarse_anchor_period
    mask[:, coarse_start:coarse_stop, coarse_offset::coarse_anchor_period] = 0

    if fine_start < codebooks:
        fine_offset = fine_resample_offset % fine_resample_period
        mask[:, fine_start:, fine_offset::fine_resample_period] = 1
    if attack_tokens:
        mask[:, :, : min(attack_tokens, steps)] = 0
    return mask


def serializable_description(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    return asdict(describe_audio(audio, sample_rate))
