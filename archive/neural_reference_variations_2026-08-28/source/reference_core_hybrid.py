"""Reference-preserving hybrid variation for impulsive sound effects.

The method keeps the reference waveform untouched during a protected event
core. After a raised-cosine transition it adds a locally level-matched,
peak-limited generative residual. The generator may change tail texture, but it
cannot replace the attack that identifies the event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HybridParameters:
    core_ms: float = 120.0
    transition_ms: float = 80.0
    envelope_window_ms: float = 20.0
    local_gain_min: float = 0.5
    local_gain_max: float = 2.0
    residual_peak_multiple: float = 3.0
    residual_mix: float = 0.15

    def validate(self, sample_rate: int, num_frames: int) -> None:
        if sample_rate <= 0 or num_frames <= 0:
            raise ValueError("sample_rate и num_frames должны быть положительными")
        if self.core_ms < 0 or self.transition_ms <= 0:
            raise ValueError("core_ms >= 0, transition_ms > 0")
        if self.envelope_window_ms <= 0:
            raise ValueError("envelope_window_ms должен быть положительным")
        if not 0 < self.local_gain_min <= self.local_gain_max:
            raise ValueError("Некорректные пределы локального gain")
        if self.residual_peak_multiple <= 0:
            raise ValueError("residual_peak_multiple должен быть положительным")
        if not 0 < self.residual_mix <= 1:
            raise ValueError("residual_mix должен лежать в диапазоне (0, 1]")
        duration_ms = 1_000 * num_frames / sample_rate
        if self.core_ms >= duration_ms:
            raise ValueError("Защищённое ядро должно быть короче аудиофайла")

    def to_dict(self) -> dict[str, float]:
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


def _moving_rms(audio: np.ndarray, window_frames: int) -> np.ndarray:
    power = np.mean(np.square(audio, dtype=np.float64), axis=1)
    kernel = np.full(window_frames, 1.0 / window_frames, dtype=np.float64)
    return np.sqrt(np.convolve(power, kernel, mode="same")).astype(np.float32)


def _tail_mask(
    num_frames: int,
    sample_rate: int,
    *,
    core_ms: float,
    transition_ms: float,
) -> np.ndarray:
    core_frames = int(round(core_ms * sample_rate / 1_000))
    transition_frames = max(1, int(round(transition_ms * sample_rate / 1_000)))
    mask = np.ones(num_frames, dtype=np.float32)
    mask[:core_frames] = 0.0
    available = min(transition_frames, num_frames - core_frames)
    if available > 0:
        phase = np.linspace(0.0, np.pi, available, endpoint=False, dtype=np.float32)
        mask[core_frames : core_frames + available] = 0.5 - 0.5 * np.cos(phase)
    return mask


def _safe_effective_mix(
    reference: np.ndarray,
    residual: np.ndarray,
    requested_mix: float,
) -> float:
    if float(np.max(np.abs(reference))) > 1.000001:
        raise ValueError("Reference выходит за допустимый диапазон PCM")

    def peak(mix: float) -> float:
        return float(np.max(np.abs(reference + mix * residual)))

    if peak(requested_mix) <= 1.0:
        return requested_mix
    low, high = 0.0, requested_mix
    for _ in range(40):
        middle = 0.5 * (low + high)
        if peak(middle) <= 1.0:
            low = middle
        else:
            high = middle
    return low


def generate_reference_core_hybrid(
    reference: np.ndarray,
    generated: np.ndarray,
    sample_rate: int,
    *,
    parameters: HybridParameters,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Add a controlled generative tail while preserving the reference core."""
    reference_samples = _as_frames_channels(reference, "reference")
    generated_samples = _as_frames_channels(generated, "generated")
    if generated_samples.shape[0] != reference_samples.shape[0]:
        raise ValueError("Reference и generated должны иметь одинаковую длину")
    generated_samples = _match_channels(generated_samples, reference_samples.shape[1])
    parameters.validate(sample_rate, reference_samples.shape[0])

    window_frames = max(1, int(round(parameters.envelope_window_ms * sample_rate / 1_000)))
    reference_rms = _moving_rms(reference_samples, window_frames)
    generated_rms = _moving_rms(generated_samples, window_frames)
    local_gain = reference_rms / np.maximum(generated_rms, 1e-7)
    local_gain = np.clip(local_gain, parameters.local_gain_min, parameters.local_gain_max)
    shaped = generated_samples * local_gain[:, None]
    instantaneous_limit = parameters.residual_peak_multiple * np.maximum(reference_rms, 1e-6)
    shaped = np.clip(shaped, -instantaneous_limit[:, None], instantaneous_limit[:, None])

    mask = _tail_mask(
        reference_samples.shape[0],
        sample_rate,
        core_ms=parameters.core_ms,
        transition_ms=parameters.transition_ms,
    )
    residual = shaped * mask[:, None]
    effective_mix = _safe_effective_mix(reference_samples, residual, parameters.residual_mix)
    variation = reference_samples + effective_mix * residual
    core_frames = int(round(parameters.core_ms * sample_rate / 1_000))
    diagnostics: dict[str, float | int] = {
        "core_frames": core_frames,
        "requested_mix": float(parameters.residual_mix),
        "effective_mix": float(effective_mix),
        "reference_peak": float(np.max(np.abs(reference_samples))),
        "output_peak": float(np.max(np.abs(variation))),
        "core_max_abs_error": float(
            np.max(np.abs(variation[:core_frames] - reference_samples[:core_frames]))
        ),
        "tail_residual_rms": float(
            np.sqrt(np.mean(np.square(variation[core_frames:] - reference_samples[core_frames:])))
        ),
    }
    return variation.astype(np.float32), diagnostics
