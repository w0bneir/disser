"""CPU-метрики идентичности, структуры, спектра и не-копирования SFX."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from analyzer import extract_rms_envelope


EPSILON = 1e-10
METRIC_PROTOCOL_ID = "sfx_metrics_v1"


def load_mono_audio(
    path: str | Path,
    *,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    if samples.shape[0] == 0:
        raise ValueError(f"Аудиофайл пуст: {path}")
    mono = torch.from_numpy(samples).mean(dim=1)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        mono = AF.resample(mono, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return mono.contiguous(), int(sample_rate)


def align_length(samples: torch.Tensor, target_length: int) -> torch.Tensor:
    if samples.ndim != 1 or target_length <= 0:
        raise ValueError("Ожидается mono-сигнал и положительная target_length")
    if samples.numel() < target_length:
        return F.pad(samples, (0, target_length - samples.numel()))
    return samples[:target_length]


def resample_vector(values: torch.Tensor, target_length: int) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0 or target_length <= 0:
        raise ValueError("Некорректный одномерный вектор")
    if values.numel() == target_length:
        return values
    return F.interpolate(
        values[None, None],
        size=target_length,
        mode="linear",
        align_corners=False,
    )[0, 0]


def pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.numel() == 0:
        raise ValueError("Pearson требует непустые векторы одинаковой формы")
    first_centered = first.float() - first.float().mean()
    second_centered = second.float() - second.float().mean()
    denominator = first_centered.square().sum().sqrt() * second_centered.square().sum().sqrt()
    if float(denominator) <= EPSILON:
        return 1.0 if torch.allclose(first, second) else 0.0
    return float((first_centered * second_centered).sum() / denominator)


def _power_spectrum(samples: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_fft = min(2_048, max(256, 2 ** int(np.floor(np.log2(samples.numel())))))
    padded = samples.float()
    if padded.numel() < n_fft:
        padded = F.pad(padded, (0, n_fft - padded.numel()))
    window = torch.hann_window(n_fft, dtype=padded.dtype)
    spectrum = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    power = spectrum.abs().square().mean(dim=-1).clamp_min(EPSILON)
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return power, frequencies


def spectral_features(samples: torch.Tensor, sample_rate: int) -> dict[str, float]:
    power, frequencies = _power_spectrum(samples, sample_rate)
    total = power.sum().clamp_min(EPSILON)
    centroid = (power * frequencies).sum() / total
    audible = frequencies >= 20
    flatness = torch.exp(torch.log(power[audible]).mean()) / power[audible].mean()
    high = power[frequencies >= min(4_000, sample_rate / 2)].sum() / total
    return {
        "spectral_centroid_hz": float(centroid),
        "spectral_flatness": float(flatness),
        "high_frequency_fraction_4khz": float(high),
    }


def log_spectral_distance_db(
    first: torch.Tensor,
    second: torch.Tensor,
    sample_rate: int,
) -> float:
    first_power, _ = _power_spectrum(first, sample_rate)
    second_power, _ = _power_spectrum(second, sample_rate)
    target_length = min(first_power.numel(), second_power.numel())
    first_normalized = first_power[:target_length] / first_power[:target_length].sum()
    second_normalized = second_power[:target_length] / second_power[:target_length].sum()
    first_db = 10 * torch.log10(first_normalized.clamp_min(EPSILON))
    second_db = 10 * torch.log10(second_normalized.clamp_min(EPSILON))
    return float((first_db - second_db).square().mean().sqrt())


def envelope_peak_times(
    envelope: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int = 512,
    threshold: float = 0.15,
    minimum_distance_seconds: float = 0.08,
    max_peaks: int = 8,
) -> list[float]:
    if envelope.ndim != 1 or envelope.numel() == 0:
        raise ValueError("Ожидается непустая одномерная огибающая")
    if float(envelope.max()) <= 0:
        return []
    minimum_frames = max(1, int(round(minimum_distance_seconds * sample_rate / hop_length)))
    kernel_size = 2 * minimum_frames + 1
    local_maximum = F.max_pool1d(
        envelope[None, None],
        kernel_size=kernel_size,
        stride=1,
        padding=minimum_frames,
    )[0, 0]
    candidate_indices = torch.nonzero(
        (envelope >= local_maximum - 1e-7) & (envelope >= threshold),
        as_tuple=False,
    ).flatten()
    ranked = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: float(envelope[index]),
        reverse=True,
    )
    selected: list[int] = []
    for index in ranked:
        if all(abs(index - previous) >= minimum_frames for previous in selected):
            selected.append(index)
        if len(selected) >= max_peaks:
            break
    return sorted(index * hop_length / sample_rate for index in selected)


def peak_timing_distance_seconds(first: list[float], second: list[float], duration: float) -> float:
    if not first and not second:
        return 0.0
    if not first or not second:
        return float(duration)

    def directed(source: list[float], target: list[float]) -> float:
        return float(np.mean([min(abs(value - other) for other in target) for value in source]))

    return 0.5 * (directed(first, second) + directed(second, first))


def _onset_seconds(
    envelope: torch.Tensor,
    sample_rate: int,
    *,
    hop_length: int = 512,
    threshold: float = 0.1,
) -> float:
    active = torch.nonzero(envelope >= threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return envelope.numel() * hop_length / sample_rate
    return int(active[0]) * hop_length / sample_rate


def compare_to_reference(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    sample_rate: int,
) -> dict[str, float | int]:
    """Посчитать одинаковый набор метрик для одного candidate."""
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("Метрики принимают mono-сигналы")
    candidate_original_length = candidate.numel()
    aligned = align_length(candidate, reference.numel())
    reference_envelope = extract_rms_envelope(reference)
    candidate_envelope = extract_rms_envelope(aligned)
    candidate_envelope = resample_vector(candidate_envelope, reference_envelope.numel())
    reference_peaks = envelope_peak_times(reference_envelope, sample_rate=sample_rate)
    candidate_peaks = envelope_peak_times(candidate_envelope, sample_rate=sample_rate)

    reference_float = reference.float()
    aligned_float = aligned.float()
    scale_denominator = reference_float.square().sum().clamp_min(EPSILON)
    optimal_scale = (reference_float * aligned_float).sum() / scale_denominator
    residual = aligned_float - optimal_scale * reference_float
    residual_ratio = residual.square().mean().sqrt() / aligned_float.square().mean().sqrt().clamp_min(EPSILON)
    copy_residual_db = max(-120.0, float(20 * torch.log10(residual_ratio.clamp_min(EPSILON))))

    reference_spectral = spectral_features(reference_float, sample_rate)
    candidate_spectral = spectral_features(aligned_float, sample_rate)
    duration_seconds = reference.numel() / sample_rate
    return {
        "duration_error_ms": 1_000 * (candidate_original_length - reference.numel()) / sample_rate,
        "envelope_mse": float((reference_envelope - candidate_envelope).square().mean()),
        "envelope_pearson": pearson_correlation(reference_envelope, candidate_envelope),
        "onset_error_ms": 1_000
        * (
            _onset_seconds(candidate_envelope, sample_rate)
            - _onset_seconds(reference_envelope, sample_rate)
        ),
        "peak_timing_mae_ms": 1_000
        * peak_timing_distance_seconds(reference_peaks, candidate_peaks, duration_seconds),
        "reference_peak_count": len(reference_peaks),
        "candidate_peak_count": len(candidate_peaks),
        "peak_count_abs_error": abs(len(reference_peaks) - len(candidate_peaks)),
        "waveform_pearson": pearson_correlation(reference_float, aligned_float),
        "optimal_reference_scale": float(optimal_scale),
        "copy_residual_db": copy_residual_db,
        "log_spectral_distance_db": log_spectral_distance_db(
            reference_float,
            aligned_float,
            sample_rate,
        ),
        **candidate_spectral,
        "spectral_centroid_delta_hz": (
            candidate_spectral["spectral_centroid_hz"]
            - reference_spectral["spectral_centroid_hz"]
        ),
        "spectral_flatness_delta": (
            candidate_spectral["spectral_flatness"]
            - reference_spectral["spectral_flatness"]
        ),
        "high_frequency_fraction_delta": (
            candidate_spectral["high_frequency_fraction_4khz"]
            - reference_spectral["high_frequency_fraction_4khz"]
        ),
    }


def pairwise_diversity(
    items: list[tuple[int, torch.Tensor]],
    sample_rate: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for (first_seed, first), (second_seed, second) in combinations(items, 2):
        target_length = max(first.numel(), second.numel())
        first_aligned = align_length(first, target_length)
        second_aligned = align_length(second, target_length)
        first_envelope = extract_rms_envelope(first_aligned)
        second_envelope = extract_rms_envelope(second_aligned)
        rows.append(
            {
                "seed_a": first_seed,
                "seed_b": second_seed,
                "waveform_pearson": pearson_correlation(first_aligned, second_aligned),
                "envelope_pearson": pearson_correlation(first_envelope, second_envelope),
                "log_spectral_distance_db": log_spectral_distance_db(
                    first_aligned,
                    second_aligned,
                    sample_rate,
                ),
            }
        )
    return rows
