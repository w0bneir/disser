"""Transient-locked stochastic microstructure synthesis for one-shot SFX.

The module is deliberately CPU-only.  It estimates an admissible
microstructure corridor from *within-group* differences between natural takes,
then creates a new stochastic carrier for the body and tail of one reference.
The leading transient is copied sample-for-sample and all interpolation is
smooth, so the renderer cannot create a second onset by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, signal


EPSILON = 1e-10
METHOD_VERSION = "transient_locked_microstructure_v1"
ONSET_S = 0.020
PROTECT_UNTIL_S = 0.055
FULL_EFFECT_S = 0.150
ANALYSIS_REGIONS = (
    ("body_early", 0.055, 0.180),
    ("body_late", 0.180, 0.650),
    ("tail", 0.650, 1.600),
)
ANALYSIS_BANDS_HZ = np.asarray(
    [45.0, 100.0, 220.0, 480.0, 1_000.0, 2_100.0, 4_400.0, 9_000.0, 19_500.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MicrostructureProfile:
    method_version: str
    sample_rate: int
    names: tuple[str, ...]
    groups: tuple[str, ...]
    descriptors: np.ndarray
    group_centers: Mapping[str, np.ndarray]
    descriptor_scale: np.ndarray
    pairwise_distances: np.ndarray
    pairwise_groups: tuple[str, ...]
    corridor_low: float
    corridor_median: float
    corridor_high: float

    def json_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "method_version": self.method_version,
            "sample_rate": self.sample_rate,
            "files": len(self.names),
            "groups": {
                group: sum(value == group for value in self.groups)
                for group in sorted(set(self.groups), key=_natural_group_key)
            },
            "descriptor_dimensions": int(self.descriptors.shape[1]),
            "normalization_scope": "within_group_only",
            "within_group_pair_count": int(self.pairwise_distances.size),
            "natural_microstructure_distance": {
                "minimum": float(np.min(self.pairwise_distances)),
                "q25": self.corridor_low,
                "median": self.corridor_median,
                "q75": self.corridor_high,
                "maximum": float(np.max(self.pairwise_distances)),
            },
        }
        if include_arrays:
            result["names"] = list(self.names)
            result["file_groups"] = list(self.groups)
            result["descriptors"] = self.descriptors.tolist()
            result["group_centers"] = {
                group: values.tolist() for group, values in self.group_centers.items()
            }
            result["descriptor_scale"] = self.descriptor_scale.tolist()
            result["within_group_pairs"] = [
                {"group": group, "distance": float(distance)}
                for group, distance in zip(self.pairwise_groups, self.pairwise_distances)
            ]
        return result


def _natural_group_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value.casefold())


def _as_audio(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 256 or values.shape[1] < 1:
        raise ValueError("Audio must have shape [frames, channels]")
    if not np.isfinite(values).all():
        raise ValueError("Audio contains NaN or Inf")
    return values


def _frame_slice(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, min(audio.shape[0] - 1, int(round(start_s * sample_rate))))
    end = max(start + 1, min(audio.shape[0], int(round(end_s * sample_rate))))
    return audio[start:end]


def _mono_energy(audio: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(audio), axis=1) + EPSILON)


def _region_descriptor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = _as_audio(audio)
    nperseg = min(1024, max(128, 2 ** int(np.floor(np.log2(values.shape[0])))))
    hop = max(1, nperseg // 4)
    channel_powers = []
    frequencies = None
    for channel in range(values.shape[1]):
        channel_frequencies, _, spectrum = signal.spectrogram(
            values[:, channel],
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg - hop,
            mode="complex",
        )
        frequencies = channel_frequencies
        channel_powers.append(np.square(np.abs(spectrum)))
    power = np.mean(np.stack(channel_powers), axis=0) + EPSILON
    assert frequencies is not None
    band_power: list[np.ndarray] = []
    for low, high in zip(ANALYSIS_BANDS_HZ[:-1], ANALYSIS_BANDS_HZ[1:]):
        selected = (frequencies >= low) & (frequencies < min(high, sample_rate * 0.49))
        if np.any(selected):
            band_power.append(np.mean(power[selected], axis=0))
        else:
            band_power.append(np.full(power.shape[1], EPSILON))
    bands = np.stack(band_power)
    mean_log_power = 10.0 * np.log10(np.mean(bands, axis=1) + EPSILON)
    mean_log_power -= float(np.mean(mean_log_power))
    log_envelopes = 10.0 * np.log10(bands + EPSILON)
    modulation_depth = np.std(log_envelopes, axis=1)
    if log_envelopes.shape[1] >= 3:
        modulation_rate = np.sqrt(np.mean(np.square(np.diff(log_envelopes, axis=1)), axis=1))
    else:
        modulation_rate = np.zeros(log_envelopes.shape[0], dtype=np.float64)
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    normalized = power / np.maximum(np.sum(power, axis=0, keepdims=True), EPSILON)
    if normalized.shape[1] >= 2:
        flux = np.sqrt(np.sum(np.square(np.diff(normalized, axis=1)), axis=0))
        flux_stats = np.asarray([np.mean(flux), np.std(flux)], dtype=np.float64)
    else:
        flux_stats = np.zeros(2, dtype=np.float64)
    if values.shape[1] >= 2:
        left = values[:, 0] - float(np.mean(values[:, 0]))
        right = values[:, 1] - float(np.mean(values[:, 1]))
        correlation = float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), EPSILON))
        mid = 0.5 * (values[:, 0] + values[:, 1])
        side = 0.5 * (values[:, 0] - values[:, 1])
        side_mid = 20.0 * np.log10(
            max(np.sqrt(np.mean(np.square(side)) + EPSILON), EPSILON)
            / max(np.sqrt(np.mean(np.square(mid)) + EPSILON), EPSILON)
        )
    else:
        correlation, side_mid = 1.0, -80.0
    return np.concatenate(
        (
            mean_log_power,
            modulation_depth,
            modulation_rate,
            np.asarray([np.mean(flatness), np.std(flatness)], dtype=np.float64),
            flux_stats,
            np.asarray([np.clip(correlation, -1.0, 1.0), np.clip(side_mid, -80.0, 20.0)]),
        )
    )


def microstructure_descriptor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Describe body/tail texture while intentionally excluding the attack."""
    values = _as_audio(audio)
    descriptors = [
        _region_descriptor(_frame_slice(values, sample_rate, start_s, end_s), sample_rate)
        for _, start_s, end_s in ANALYSIS_REGIONS
    ]
    return np.concatenate(descriptors).astype(np.float64)


def fit_microstructure_profile(
    audio_bank: Sequence[np.ndarray],
    sample_rate: int,
    *,
    groups: Sequence[str],
    names: Sequence[str] | None = None,
) -> MicrostructureProfile:
    """Fit pooled within-group statistics without treating groups as takes."""
    if len(audio_bank) < 6:
        raise ValueError("At least six natural takes are required")
    if len(groups) != len(audio_bank):
        raise ValueError("groups length must match audio_bank")
    counts = {group: groups.count(group) for group in set(groups)}
    if any(count < 3 for count in counts.values()):
        raise ValueError("Every group must contain at least three natural takes")
    values = [_as_audio(audio) for audio in audio_bank]
    if len({audio.shape for audio in values}) != 1:
        raise ValueError("Natural takes must have the same prepared shape")
    resolved_names = tuple(names) if names is not None else tuple(f"take_{i + 1}" for i in range(len(values)))
    if len(resolved_names) != len(values):
        raise ValueError("names length must match audio_bank")
    descriptors = np.stack([microstructure_descriptor(audio, sample_rate) for audio in values])
    group_array = np.asarray(tuple(str(group) for group in groups), dtype=object)
    group_centers: dict[str, np.ndarray] = {}
    centered = np.empty_like(descriptors)
    for group in sorted(set(group_array), key=_natural_group_key):
        indices = np.flatnonzero(group_array == group)
        center = np.median(descriptors[indices], axis=0)
        group_centers[str(group)] = center
        centered[indices] = descriptors[indices] - center
    mad = np.median(np.abs(centered), axis=0) * 1.4826
    standard = np.std(centered, axis=0)
    floor = np.maximum(np.median(np.abs(centered), axis=0) * 0.10, 1e-4)
    scale = np.maximum(np.where(mad > floor, mad, standard), floor)
    pairwise: list[float] = []
    pair_groups: list[str] = []
    dimension_scale = math.sqrt(float(descriptors.shape[1]))
    for group in sorted(set(group_array), key=_natural_group_key):
        indices = np.flatnonzero(group_array == group)
        for left, right in combinations(indices, 2):
            distance = np.linalg.norm((descriptors[left] - descriptors[right]) / scale) / dimension_scale
            pairwise.append(float(distance))
            pair_groups.append(str(group))
    distances = np.asarray(pairwise, dtype=np.float64)
    if not np.isfinite(distances).all() or not np.any(distances > 0.0):
        raise ValueError("Natural microstructure statistics are degenerate")
    return MicrostructureProfile(
        method_version=METHOD_VERSION,
        sample_rate=int(sample_rate),
        names=resolved_names,
        groups=tuple(str(group) for group in groups),
        descriptors=descriptors,
        group_centers=group_centers,
        descriptor_scale=scale,
        pairwise_distances=distances,
        pairwise_groups=tuple(pair_groups),
        corridor_low=float(np.quantile(distances, 0.25)),
        corridor_median=float(np.median(distances)),
        corridor_high=float(np.quantile(distances, 0.75)),
    )


def microstructure_distance(left: np.ndarray, right: np.ndarray, profile: MicrostructureProfile) -> float:
    left_descriptor = microstructure_descriptor(left, profile.sample_rate)
    right_descriptor = microstructure_descriptor(right, profile.sample_rate)
    delta = (left_descriptor - right_descriptor) / profile.descriptor_scale
    return float(np.linalg.norm(delta) / math.sqrt(float(delta.size)))


def calibrate_microstructure_strength(
    reference: np.ndarray,
    profile: MicrostructureProfile,
    *,
    seed: int,
    target_distance: float,
    maximum_strength: float = 0.75,
    iterations: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Find a wet mix whose descriptor distance approaches a natural target."""
    if target_distance <= 0.0:
        raise ValueError("target_distance must be positive")
    if not 0.0 < maximum_strength <= 1.0:
        raise ValueError("maximum_strength must be in (0, 1]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    low, high = 0.0, float(maximum_strength)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for _ in range(iterations):
        strength = 0.5 * (low + high)
        audio = synthesize_microstructure(
            reference,
            profile.sample_rate,
            seed=seed,
            strength=strength,
        )
        distance = microstructure_distance(reference, audio, profile)
        candidates.append((abs(distance - target_distance), strength, audio))
        if distance < target_distance:
            low = strength
        else:
            high = strength
    _, selected_strength, selected_audio = min(candidates, key=lambda item: item[0])
    selected_distance = microstructure_distance(reference, selected_audio, profile)
    return selected_audio, float(selected_strength), float(selected_distance)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    phase = np.clip(values, 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def _activity_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    energy = _mono_energy(audio)
    window = max(8, int(round(0.012 * sample_rate)))
    smoothed = ndimage.uniform_filter1d(energy, size=window, mode="nearest")
    early_start = int(round(ONSET_S * sample_rate))
    early_end = min(audio.shape[0], int(round(0.180 * sample_rate)))
    reference = max(float(np.max(smoothed[early_start:early_end])), EPSILON)
    relative_db = 20.0 * np.log10(np.maximum(smoothed, EPSILON) / reference)
    return _smoothstep((relative_db + 58.0) / 12.0)


def _stochastic_carrier(channel: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    nperseg = min(2048, max(256, 2 ** int(np.floor(np.log2(channel.size)))))
    hop = max(1, nperseg // 8)
    noverlap = nperseg - hop
    _, _, reference_spectrum = signal.stft(
        channel,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    noise = rng.standard_normal(channel.size)
    _, _, noise_spectrum = signal.stft(
        noise,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    reference_log = np.log(np.abs(reference_spectrum) + EPSILON)
    noise_log = np.log(np.abs(noise_spectrum) + EPSILON)
    # Coarse time-frequency envelope: resonant colour and decay are retained,
    # but phase and fine bin-level magnitude fluctuations come from new noise.
    target_log = ndimage.gaussian_filter(reference_log, sigma=(3.0, 1.2), mode="nearest")
    normalizer_log = ndimage.gaussian_filter(noise_log, sigma=(3.0, 1.2), mode="nearest")
    gain = np.exp(np.clip(target_log - normalizer_log, -8.0, 8.0))
    shaped_spectrum = noise_spectrum * gain
    _, reconstructed = signal.istft(
        shaped_spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    if reconstructed.size < channel.size:
        reconstructed = np.pad(reconstructed, (0, channel.size - reconstructed.size))
    return reconstructed[: channel.size]


def synthesize_microstructure(
    reference: np.ndarray,
    sample_rate: int,
    *,
    seed: int,
    strength: float,
) -> np.ndarray:
    """Create a new body/tail carrier while keeping the leading attack exact.

    ``strength`` is a bounded wet mix, not an arbitrary effect amount.  The
    carrier uses new random samples for each seed and is shaped by a smoothed
    time-frequency envelope of the reference.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    values = _as_audio(reference)
    if strength == 0.0:
        return values.astype(np.float32, copy=True)
    rng = np.random.default_rng(int(seed))
    if values.shape[1] >= 2:
        mid = 0.5 * (values[:, 0] + values[:, 1])
        side = 0.5 * (values[:, 0] - values[:, 1])
        mid_carrier = _stochastic_carrier(mid, sample_rate, rng)
        side_carrier = _stochastic_carrier(side, sample_rate, rng)
        carrier = np.stack((mid_carrier + side_carrier, mid_carrier - side_carrier), axis=1)
        if values.shape[1] > 2:
            extras = [
                _stochastic_carrier(values[:, channel], sample_rate, rng)
                for channel in range(2, values.shape[1])
            ]
            carrier = np.column_stack((carrier, *extras))
    else:
        carrier = _stochastic_carrier(values[:, 0], sample_rate, rng)[:, None]
    # Match body energy globally; the carrier already follows local energy in
    # the time-frequency plane.  Conservative clipping prevents silent-tail
    # amplification from becoming audible hiss.
    body_start = int(round(FULL_EFFECT_S * sample_rate))
    source_rms = np.sqrt(np.mean(np.square(values[body_start:])) + EPSILON)
    carrier_rms = np.sqrt(np.mean(np.square(carrier[body_start:])) + EPSILON)
    carrier *= float(np.clip(source_rms / max(carrier_rms, EPSILON), 0.25, 4.0))
    time_s = np.arange(values.shape[0], dtype=np.float64) / sample_rate
    onset_mix = _smoothstep((time_s - PROTECT_UNTIL_S) / (FULL_EFFECT_S - PROTECT_UNTIL_S))
    mix = strength * onset_mix * _activity_mask(values, sample_rate)
    output = values * (1.0 - mix[:, None]) + carrier * mix[:, None]
    protect_frames = min(values.shape[0], int(round(PROTECT_UNTIL_S * sample_rate)))
    output[:protect_frames] = values[:protect_frames]
    fade_frames = min(output.shape[0], max(1, int(round(0.060 * sample_rate))))
    output[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float64)[:, None]
    if not np.isfinite(output).all():
        raise RuntimeError("Microstructure synthesis produced NaN or Inf")
    return output.astype(np.float32)


def leading_attack_error(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    reference_values = _as_audio(reference)
    candidate_values = _as_audio(candidate)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("Reference and candidate must have equal shape")
    frames = min(reference_values.shape[0], int(round(PROTECT_UNTIL_S * sample_rate)))
    return float(np.max(np.abs(reference_values[:frames] - candidate_values[:frames])))


def common_peak_safe(audio_bank: Sequence[np.ndarray], *, peak_dbfs: float = -1.0) -> list[np.ndarray]:
    values = [_as_audio(audio) for audio in audio_bank]
    if not values:
        raise ValueError("audio_bank is empty")
    peak = max(float(np.max(np.abs(audio))) for audio in values)
    limit = 10.0 ** (float(peak_dbfs) / 20.0)
    scale = min(1.0, limit / max(peak, EPSILON))
    return [(audio * scale).astype(np.float32) for audio in values]
