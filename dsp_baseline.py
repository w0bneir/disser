"""Воспроизводимый pitch/time/EQ baseline для SFX-вариаций."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2, pi

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


@dataclass(frozen=True)
class DspRanges:
    """Зафиксированные пределы умеренной DSP-вариации."""

    pitch_cents: float = 100.0
    time_stretch_fraction: float = 0.05
    eq_gain_db: float = 2.0
    eq_center_min_hz: float = 250.0
    eq_center_max_hz: float = 5_000.0
    eq_width_octaves: float = 1.5

    def validate(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate должен быть положительным")
        if not 0 <= self.pitch_cents <= 300:
            raise ValueError("pitch_cents должен лежать в диапазоне [0, 300]")
        if not 0 <= self.time_stretch_fraction <= 0.15:
            raise ValueError("time_stretch_fraction должен лежать в диапазоне [0, 0.15]")
        if not 0 <= self.eq_gain_db <= 6:
            raise ValueError("eq_gain_db должен лежать в диапазоне [0, 6]")
        if not 20 <= self.eq_center_min_hz < self.eq_center_max_hz:
            raise ValueError("Некорректный диапазон центральной частоты EQ")
        if self.eq_center_max_hz >= sample_rate / 2:
            raise ValueError("eq_center_max_hz должен быть ниже Nyquist")
        if not 0.25 <= self.eq_width_octaves <= 4:
            raise ValueError("eq_width_octaves должен лежать в диапазоне [0.25, 4]")


@dataclass(frozen=True)
class DspParameters:
    seed: int
    pitch_cents: float
    time_stretch_factor: float
    eq_gain_db: float
    eq_center_hz: float
    eq_width_octaves: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def parameters_from_seed(seed: int, ranges: DspRanges) -> DspParameters:
    """Получить одинаковые параметры на любой машине с NumPy PCG64."""
    rng = np.random.default_rng(int(seed))
    center_log2 = rng.uniform(
        log2(ranges.eq_center_min_hz),
        log2(ranges.eq_center_max_hz),
    )
    return DspParameters(
        seed=int(seed),
        pitch_cents=float(rng.uniform(-ranges.pitch_cents, ranges.pitch_cents)),
        time_stretch_factor=float(
            1.0
            + rng.uniform(
                -ranges.time_stretch_fraction,
                ranges.time_stretch_fraction,
            )
        ),
        eq_gain_db=float(rng.uniform(-ranges.eq_gain_db, ranges.eq_gain_db)),
        eq_center_hz=float(2**center_log2),
        eq_width_octaves=float(ranges.eq_width_octaves),
    )


def _analysis_fft_size(num_samples: int) -> int:
    if num_samples <= 0:
        raise ValueError("Аудиосигнал пуст")
    upper = min(2_048, max(256, num_samples))
    return 2 ** int(np.floor(np.log2(upper)))


def _time_stretch(
    audio: torch.Tensor,
    factor: float,
    *,
    target_length: int,
) -> torch.Tensor:
    """Изменить длительность без изменения pitch и вернуть точную длину."""
    if factor <= 0:
        raise ValueError("time-stretch factor должен быть положительным")
    n_fft = _analysis_fft_size(audio.shape[-1])
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    padded = audio
    if padded.shape[-1] < n_fft:
        padded = F.pad(padded, (0, n_fft - padded.shape[-1]))
    spectrum = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0,
        pi * hop_length,
        spectrum.shape[-2],
        device=audio.device,
        dtype=audio.dtype,
    )[..., None]
    stretched_spectrum = AF.phase_vocoder(
        spectrum,
        rate=1.0 / factor,
        phase_advance=phase_advance,
    )
    stretched_length = max(1, int(round(audio.shape[-1] * factor)))
    stretched = torch.istft(
        stretched_spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=stretched_length,
    )
    if stretched.shape[-1] < target_length:
        stretched = F.pad(stretched, (0, target_length - stretched.shape[-1]))
    return stretched[..., :target_length]


def _broad_eq(
    audio: torch.Tensor,
    sample_rate: int,
    *,
    center_hz: float,
    gain_db: float,
    width_octaves: float,
) -> torch.Tensor:
    """Применить гладкий zero-phase bell EQ в частотной области."""
    spectrum = torch.fft.rfft(audio, dim=-1)
    frequencies = torch.fft.rfftfreq(
        audio.shape[-1],
        d=1.0 / sample_rate,
        device=audio.device,
        dtype=audio.dtype,
    )
    safe_frequencies = frequencies.clamp_min(1.0)
    octave_distance = torch.log2(safe_frequencies / float(center_hz))
    bell = torch.exp(-0.5 * (octave_distance / float(width_octaves)) ** 2)
    bell[0] = 0
    linear_gain = torch.pow(
        torch.tensor(10.0, device=audio.device, dtype=audio.dtype),
        (float(gain_db) * bell) / 20.0,
    )
    return torch.fft.irfft(
        spectrum * linear_gain,
        n=audio.shape[-1],
        dim=-1,
    )


def _match_rms(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    reference_rms = reference.square().mean().sqrt()
    candidate_rms = candidate.square().mean().sqrt()
    if float(reference_rms) == 0 or float(candidate_rms) == 0:
        return candidate
    return candidate * (reference_rms / candidate_rms)


def generate_dsp_variation(
    audio: np.ndarray,
    sample_rate: int,
    *,
    parameters: DspParameters,
) -> np.ndarray:
    """Создать stereo/mono variation формы ``[frames, channels]``."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("Ожидается непустой массив [frames, channels]")
    if not np.isfinite(samples).all():
        raise ValueError("Аудио содержит NaN или бесконечность")
    if sample_rate <= 0:
        raise ValueError("sample_rate должен быть положительным")

    reference = torch.from_numpy(samples.T.copy())
    shifted = AF.pitch_shift(
        reference,
        sample_rate=sample_rate,
        n_steps=parameters.pitch_cents / 100.0,
    )
    stretched = _time_stretch(
        shifted,
        parameters.time_stretch_factor,
        target_length=reference.shape[-1],
    )
    equalized = _broad_eq(
        stretched,
        sample_rate,
        center_hz=parameters.eq_center_hz,
        gain_db=parameters.eq_gain_db,
        width_octaves=parameters.eq_width_octaves,
    )
    matched = _match_rms(equalized, reference)
    return matched.T.contiguous().numpy().astype(np.float32)

