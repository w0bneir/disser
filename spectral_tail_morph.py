"""Reference-preserving spectral-temporal morph of a late SFX tail."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpectralTailMorphParameters:
    protected_energy_quantile: float = 0.90
    transition_ms: float = 100.0
    n_fft: int = 2_048
    hop_length: int = 512
    frequency_smoothing_bins: int = 9
    time_smoothing_frames: int = 9
    max_modulation_db: float = 6.0
    modulation_depth: float = 1.0
    phase_mix: float = 0.0

    def validate(self, sample_rate: int, num_frames: int) -> None:
        if sample_rate <= 0 or num_frames <= 0:
            raise ValueError("sample_rate и num_frames должны быть положительными")
        if not 0.5 < self.protected_energy_quantile < 1.0:
            raise ValueError("protected_energy_quantile должен лежать в (0.5, 1.0)")
        if self.transition_ms <= 0:
            raise ValueError("transition_ms должен быть положительным")
        if self.n_fft < 256 or self.n_fft & (self.n_fft - 1):
            raise ValueError("n_fft должен быть степенью двойки не меньше 256")
        if not 0 < self.hop_length <= self.n_fft:
            raise ValueError("hop_length должен лежать в (0, n_fft]")
        for value, name in (
            (self.frequency_smoothing_bins, "frequency_smoothing_bins"),
            (self.time_smoothing_frames, "time_smoothing_frames"),
        ):
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} должен быть положительным нечётным числом")
        if not 0 < self.max_modulation_db <= 12:
            raise ValueError("max_modulation_db должен лежать в (0, 12]")
        if not 0 < self.modulation_depth <= 1:
            raise ValueError("modulation_depth должен лежать в (0, 1]")
        if not 0 <= self.phase_mix <= 1:
            raise ValueError("phase_mix должен лежать в [0, 1]")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_frames_channels(audio: np.ndarray, name: str) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError(f"{name}: ожидается непустой массив [frames, channels]")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name}: обнаружены NaN или бесконечность")
    return samples


def _match_channels(generated: np.ndarray, reference_channels: int) -> np.ndarray:
    if generated.shape[1] == reference_channels:
        return generated
    if generated.shape[1] == 1:
        return np.repeat(generated, reference_channels, axis=1)
    if reference_channels == 1:
        return generated.mean(axis=1, keepdims=True)
    raise ValueError("Число каналов reference и generated несовместимо")


def _protected_frames(reference: np.ndarray, quantile: float) -> int:
    frame_energy = np.mean(np.square(reference, dtype=np.float64), axis=1)
    total = float(frame_energy.sum())
    if total <= 0:
        raise ValueError("Reference не содержит энергии")
    cumulative = np.cumsum(frame_energy)
    return min(reference.shape[0] - 1, int(np.searchsorted(cumulative, quantile * total)) + 1)


def _time_mask(
    num_frames: int,
    protected_frames: int,
    transition_frames: int,
) -> np.ndarray:
    mask = np.ones(num_frames, dtype=np.float32)
    mask[:protected_frames] = 0.0
    available = min(transition_frames, num_frames - protected_frames)
    if available > 0:
        phase = np.linspace(0.0, np.pi, available, endpoint=False, dtype=np.float32)
        mask[protected_frames : protected_frames + available] = 0.5 - 0.5 * np.cos(phase)
    return mask


def _safe_residual_scale(reference: np.ndarray, residual: np.ndarray) -> float:
    if float(np.max(np.abs(reference))) > 1.000001:
        raise ValueError("Reference выходит за допустимый диапазон PCM")
    if float(np.max(np.abs(reference + residual))) <= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if float(np.max(np.abs(reference + middle * residual))) <= 1.0:
            low = middle
        else:
            high = middle
    return low


def generate_spectral_tail_morph(
    reference: np.ndarray,
    generated: np.ndarray,
    sample_rate: int,
    *,
    parameters: SpectralTailMorphParameters,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Morph only late-tail spectral evolution while retaining reference phase."""
    reference_samples = _as_frames_channels(reference, "reference")
    generated_samples = _as_frames_channels(generated, "generated")
    if generated_samples.shape[0] != reference_samples.shape[0]:
        raise ValueError("Reference и generated должны иметь одинаковую длину")
    generated_samples = _match_channels(generated_samples, reference_samples.shape[1])
    parameters.validate(sample_rate, reference_samples.shape[0])

    protected_frames = _protected_frames(
        reference_samples,
        parameters.protected_energy_quantile,
    )
    reference_tensor = torch.from_numpy(reference_samples.T.copy())
    generated_tensor = torch.from_numpy(generated_samples.T.copy())
    window = torch.hann_window(parameters.n_fft, dtype=reference_tensor.dtype)
    stft_kwargs = {
        "n_fft": parameters.n_fft,
        "hop_length": parameters.hop_length,
        "win_length": parameters.n_fft,
        "window": window,
        "center": True,
        "return_complex": True,
    }
    reference_stft = torch.stft(reference_tensor, **stft_kwargs)
    generated_stft = torch.stft(generated_tensor, **stft_kwargs)
    reference_magnitude = reference_stft.abs().clamp_min(1e-7)
    generated_magnitude = generated_stft.abs().clamp_min(1e-7)
    delta_db = 20 * torch.log10(generated_magnitude / reference_magnitude)

    tail_start_frame = min(
        delta_db.shape[-1] - 1,
        int(np.ceil(protected_frames / parameters.hop_length)),
    )
    delta_db = delta_db - delta_db[..., tail_start_frame:].mean(dim=-1, keepdim=True)
    smoothed = F.avg_pool2d(
        delta_db[:, None],
        kernel_size=(parameters.frequency_smoothing_bins, parameters.time_smoothing_frames),
        stride=1,
        padding=(parameters.frequency_smoothing_bins // 2, parameters.time_smoothing_frames // 2),
    )[:, 0]
    smoothed = smoothed - smoothed[..., tail_start_frame:].mean(dim=-1, keepdim=True)
    modulation_db = torch.clamp(
        parameters.modulation_depth * smoothed,
        -parameters.max_modulation_db,
        parameters.max_modulation_db,
    )
    gain = torch.pow(10.0, modulation_db / 20.0)
    morphed_magnitude = reference_magnitude * gain

    reference_frame_energy = reference_magnitude.square().sum(dim=-2, keepdim=True).sqrt()
    morphed_frame_energy = morphed_magnitude.square().sum(dim=-2, keepdim=True).sqrt()
    morphed_magnitude = morphed_magnitude * (
        reference_frame_energy / morphed_frame_energy.clamp_min(1e-7)
    )
    reference_phase = reference_stft / reference_magnitude
    generated_phase = generated_stft / generated_magnitude
    phase_vector = (
        (1.0 - parameters.phase_mix) * reference_phase
        + parameters.phase_mix * generated_phase
    )
    phase_vector = phase_vector / phase_vector.abs().clamp_min(1e-7)
    morphed_stft = morphed_magnitude * phase_vector
    morphed = torch.istft(
        morphed_stft,
        n_fft=parameters.n_fft,
        hop_length=parameters.hop_length,
        win_length=parameters.n_fft,
        window=window,
        center=True,
        length=reference_samples.shape[0],
    ).T.numpy()

    transition_frames = max(1, int(round(parameters.transition_ms * sample_rate / 1_000)))
    mask = _time_mask(reference_samples.shape[0], protected_frames, transition_frames)
    residual = mask[:, None] * (morphed - reference_samples)
    effective_scale = _safe_residual_scale(reference_samples, residual)
    variation = reference_samples + effective_scale * residual
    tail_reference_rms = float(
        np.sqrt(np.mean(np.square(reference_samples[protected_frames:], dtype=np.float64)))
    )
    tail_residual_rms = float(
        np.sqrt(np.mean(np.square(variation[protected_frames:] - reference_samples[protected_frames:])))
    )
    tail_residual_db = float(
        20 * np.log10(max(tail_residual_rms, 1e-12) / max(tail_reference_rms, 1e-12))
    )
    diagnostics: dict[str, float | int] = {
        "protected_frames": protected_frames,
        "protected_ms": 1_000 * protected_frames / sample_rate,
        "effective_residual_scale": effective_scale,
        "core_max_abs_error": float(
            np.max(np.abs(variation[:protected_frames] - reference_samples[:protected_frames]))
        ),
        "tail_residual_db": tail_residual_db,
        "modulation_rms_db": float(
            modulation_db[..., tail_start_frame:].square().mean().sqrt()
        ),
        "output_peak": float(np.max(np.abs(variation))),
    }
    return variation.astype(np.float32), diagnostics
