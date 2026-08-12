"""Безопасная подготовка сгенерированного аудио к записи в WAV."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Сохранить форму волны, масштабировав её до безопасного пика.

    Нормализация одной константой не меняет нормированную RMS-огибающую,
    но предотвращает клиппинг при записи PCM WAV.
    """
    if not 0 < target_peak <= 1:
        raise ValueError("target_peak должен лежать в диапазоне (0, 1]")
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim not in (1, 2) or samples.size == 0:
        raise ValueError("Ожидается непустой mono- или stereo-массив аудиосэмплов")
    if not np.isfinite(samples).all():
        raise ValueError("Аудио содержит NaN или бесконечность")

    peak = float(np.max(np.abs(samples)))
    if peak == 0:
        return samples.copy()
    return samples * (target_peak / peak)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int, *, target_peak: float = 0.95) -> None:
    """Записать mono или stereo WAV в PCM_24 без клиппинга."""
    import soundfile as sf

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, peak_normalize(audio, target_peak), sample_rate, subtype="PCM_24")
