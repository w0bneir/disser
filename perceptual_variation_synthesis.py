"""Natural-take-calibrated, perceptually bounded SFX variation synthesis.

The first draft deliberately uses smooth, explainable DSP rather than a free
generative model.  A bank of natural takes defines the scale and shape of
admissible variation.  A single reference is then transformed toward one of
the observed natural deltas while its leading transient and phase are kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Sequence

import numpy as np
from scipy import signal


EPSILON = 1e-10
METHOD_VERSION = "natural_statistics_synthesis_v0"
DESCRIPTOR_SEGMENTS = (
    (0.020, 0.065),
    (0.065, 0.140),
    (0.140, 0.340),
    (0.340, 0.760),
    (0.760, 1.350),
    (1.350, 2.250),
)
SPECTRAL_SEGMENTS = ((0.020, 0.120), (0.120, 0.420), (0.420, 1.600))
SPECTRAL_CENTERS_S = np.asarray([0.070, 0.270, 1.010], dtype=np.float64)
BAND_CENTERS_HZ = np.geomspace(55.0, 19_000.0, 12)


@dataclass(frozen=True)
class NaturalVariationProfile:
    method_version: str
    sample_rate: int
    names: tuple[str, ...]
    reference_index: int
    descriptor_center: np.ndarray
    descriptor_scale: np.ndarray
    descriptors: np.ndarray
    pairwise_distances: np.ndarray
    corridor_low: float
    corridor_median: float
    corridor_high: float

    def json_dict(self) -> dict[str, object]:
        return {
            "method_version": self.method_version,
            "sample_rate": self.sample_rate,
            "names": list(self.names),
            "reference_index": self.reference_index,
            "reference_name": self.names[self.reference_index],
            "descriptor_dimensions": int(self.descriptors.shape[1]),
            "natural_pair_distance": {
                "minimum": float(np.min(self.pairwise_distances)),
                "q25": self.corridor_low,
                "median": self.corridor_median,
                "q75": self.corridor_high,
                "maximum": float(np.max(self.pairwise_distances)),
            },
        }


@dataclass(frozen=True)
class VariationTransform:
    temporal_gain_db: np.ndarray
    spectral_gain_db: np.ndarray
    side_gain_db: np.ndarray

    def json_dict(self) -> dict[str, object]:
        return {
            "temporal_gain_db": self.temporal_gain_db.tolist(),
            "spectral_gain_db": self.spectral_gain_db.tolist(),
            "side_gain_db": self.side_gain_db.tolist(),
        }


def _as_audio(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 32 or values.shape[1] < 1:
        raise ValueError("Audio must have shape [frames, channels]")
    if not np.isfinite(values).all():
        raise ValueError("Audio contains NaN or Inf")
    return values


def _segment(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, min(audio.shape[0] - 1, int(round(start_s * sample_rate))))
    end = max(start + 1, min(audio.shape[0], int(round(end_s * sample_rate))))
    return audio[start:end]


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64))) + EPSILON))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), EPSILON)))


def _energy(audio: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(audio), axis=1) + EPSILON)


def _band_profile_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = _as_audio(audio)
    if values.shape[0] < 64:
        values = np.pad(values, ((0, 64 - values.shape[0]), (0, 0)))
    windowed = values * signal.windows.hann(values.shape[0], sym=False)[:, None]
    spectrum = np.mean(np.square(np.abs(np.fft.rfft(windowed, axis=0))), axis=1) + EPSILON
    frequencies = np.fft.rfftfreq(values.shape[0], 1.0 / sample_rate)
    edges = np.geomspace(38.0, min(21_000.0, sample_rate * 0.49), BAND_CENTERS_HZ.size + 1)
    levels = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = spectrum[(frequencies >= low) & (frequencies < high)]
        levels.append(10.0 * np.log10(float(np.mean(selected)) + EPSILON) if selected.size else -100.0)
    result = np.asarray(levels, dtype=np.float64)
    result -= float(np.mean(result))
    return result


def _side_mid_db(audio: np.ndarray) -> float:
    values = _as_audio(audio)
    if values.shape[1] < 2:
        return -80.0
    mid = 0.5 * (values[:, 0] + values[:, 1])
    side = 0.5 * (values[:, 0] - values[:, 1])
    return float(np.clip(_db(_rms(side) / max(_rms(mid), EPSILON)), -80.0, 20.0))


def event_descriptor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a gain-aware but phase-insensitive event descriptor."""
    values = _as_audio(audio)
    levels = np.asarray(
        [_db(_rms(_segment(values, sample_rate, start, end))) for start, end in DESCRIPTOR_SEGMENTS],
        dtype=np.float64,
    )
    spectral = np.concatenate(
        [_band_profile_db(_segment(values, sample_rate, start, end), sample_rate) for start, end in SPECTRAL_SEGMENTS]
    )
    spatial = np.asarray(
        [_side_mid_db(_segment(values, sample_rate, start, end)) for start, end in SPECTRAL_SEGMENTS],
        dtype=np.float64,
    )
    return np.concatenate((levels, spectral, spatial))


def _robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0) * 1.4826
    standard = np.std(matrix, axis=0)
    scale = np.where(mad > 1e-5, mad, np.where(standard > 1e-5, standard, 1.0))
    return center, scale


def fit_natural_variation_profile(
    audio_bank: Sequence[np.ndarray],
    sample_rate: int,
    *,
    names: Sequence[str] | None = None,
) -> NaturalVariationProfile:
    if len(audio_bank) < 3:
        raise ValueError("At least three natural takes are required")
    values = [_as_audio(audio) for audio in audio_bank]
    shapes = {(audio.shape, int(sample_rate)) for audio in values}
    if len(shapes) != 1:
        raise ValueError("Natural takes must have the same prepared shape and sample rate")
    descriptors = np.stack([event_descriptor(audio, sample_rate) for audio in values])
    center, scale = _robust_scale(descriptors)
    standardized = (descriptors - center) / scale
    dimension_scale = math.sqrt(float(standardized.shape[1]))
    distance_matrix = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=2) / dimension_scale
    pairwise = np.asarray(
        [distance_matrix[left, right] for left, right in combinations(range(len(values)), 2)],
        dtype=np.float64,
    )
    reference_index = int(np.argmin(np.mean(distance_matrix, axis=1)))
    resolved_names = tuple(names) if names is not None else tuple(f"take_{i + 1}" for i in range(len(values)))
    if len(resolved_names) != len(values):
        raise ValueError("names length must match audio_bank")
    return NaturalVariationProfile(
        method_version=METHOD_VERSION,
        sample_rate=int(sample_rate),
        names=resolved_names,
        reference_index=reference_index,
        descriptor_center=center,
        descriptor_scale=scale,
        descriptors=descriptors,
        pairwise_distances=pairwise,
        corridor_low=float(np.quantile(pairwise, 0.25)),
        corridor_median=float(np.median(pairwise)),
        corridor_high=float(np.quantile(pairwise, 0.75)),
    )


def profile_distance(
    left_audio: np.ndarray,
    right_audio: np.ndarray,
    profile: NaturalVariationProfile,
) -> float:
    left = event_descriptor(left_audio, profile.sample_rate)
    right = event_descriptor(right_audio, profile.sample_rate)
    difference = (left - right) / profile.descriptor_scale
    return float(np.linalg.norm(difference) / math.sqrt(float(difference.size)))


def estimate_transform(reference: np.ndarray, donor: np.ndarray, sample_rate: int) -> VariationTransform:
    reference = _as_audio(reference)
    donor = _as_audio(donor)
    if reference.shape != donor.shape:
        raise ValueError("Reference and donor must have the same prepared shape")
    temporal = []
    temporal_limits = (1.25, 1.75, 2.5, 3.0, 3.5, 4.0)
    for (start, end), limit in zip(DESCRIPTOR_SEGMENTS, temporal_limits):
        delta = _db(_rms(_segment(donor, sample_rate, start, end))) - _db(
            _rms(_segment(reference, sample_rate, start, end))
        )
        temporal.append(float(np.clip(delta, -limit, limit)))
    spectral = []
    for start, end in SPECTRAL_SEGMENTS:
        delta = _band_profile_db(_segment(donor, sample_rate, start, end), sample_rate) - _band_profile_db(
            _segment(reference, sample_rate, start, end), sample_rate
        )
        spectral.append(np.clip(delta, -3.5, 3.5))
    side = []
    for start, end in SPECTRAL_SEGMENTS:
        delta = _side_mid_db(_segment(donor, sample_rate, start, end)) - _side_mid_db(
            _segment(reference, sample_rate, start, end)
        )
        side.append(float(np.clip(delta, -2.5, 2.5)))
    return VariationTransform(
        temporal_gain_db=np.asarray(temporal, dtype=np.float64),
        spectral_gain_db=np.stack(spectral),
        side_gain_db=np.asarray(side, dtype=np.float64),
    )


def _transient_protection(time_s: np.ndarray) -> np.ndarray:
    # Prepared clips place the detected onset at 20 ms.  Keep the first 15 ms
    # after onset intact and smoothly enable variation during the next 55 ms.
    phase = np.clip((time_s - 0.035) / 0.055, 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def _apply_spectral_transform(
    audio: np.ndarray,
    sample_rate: int,
    spectral_gain_db: np.ndarray,
    strength: float,
) -> np.ndarray:
    nperseg = min(2048, audio.shape[0])
    noverlap = nperseg - max(1, nperseg // 4)
    output = np.zeros_like(audio, dtype=np.float64)
    for channel in range(audio.shape[1]):
        frequencies, times, spectrum = signal.stft(
            audio[:, channel],
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )
        gains_by_band = np.vstack(
            [np.interp(times, SPECTRAL_CENTERS_S, spectral_gain_db[:, band]) for band in range(BAND_CENTERS_HZ.size)]
        )
        protection = _transient_protection(times)
        gain_matrix = np.empty_like(spectrum.real)
        for frame in range(times.size):
            curve = np.interp(
                frequencies,
                BAND_CENTERS_HZ,
                gains_by_band[:, frame],
                left=gains_by_band[0, frame],
                right=gains_by_band[-1, frame],
            )
            gain_matrix[:, frame] = np.power(10.0, strength * protection[frame] * curve / 20.0)
        _, reconstructed = signal.istft(
            spectrum * gain_matrix,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True,
        )
        output[:, channel] = reconstructed[: audio.shape[0]]
    return output


def _interpolated_segment_curve(values: np.ndarray, frames: int, sample_rate: int) -> np.ndarray:
    centers = np.asarray([(start + end) * 0.5 for start, end in DESCRIPTOR_SEGMENTS], dtype=np.float64)
    time_s = np.arange(frames, dtype=np.float64) / sample_rate
    curve = np.interp(time_s, centers, values, left=0.0, right=float(values[-1]))
    return curve * _transient_protection(time_s)


def synthesize_variation(
    reference: np.ndarray,
    donor: np.ndarray,
    sample_rate: int,
    *,
    strength: float,
) -> tuple[np.ndarray, VariationTransform]:
    """Transform one reference toward an observed donor delta.

    ``strength=0`` is sample-identical to the reference.  Values in [0, 1]
    interpolate toward the bounded natural delta; extrapolation is rejected.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    reference = _as_audio(reference)
    donor = _as_audio(donor)
    transform = estimate_transform(reference, donor, sample_rate)
    if strength == 0.0:
        return reference.astype(np.float32, copy=True), transform
    output = _apply_spectral_transform(reference, sample_rate, transform.spectral_gain_db, strength)
    temporal_db = _interpolated_segment_curve(transform.temporal_gain_db, output.shape[0], sample_rate)
    output *= np.power(10.0, strength * temporal_db / 20.0)[:, None]
    if output.shape[1] >= 2:
        side_centers = SPECTRAL_CENTERS_S
        time_s = np.arange(output.shape[0], dtype=np.float64) / sample_rate
        side_db = np.interp(
            time_s,
            side_centers,
            transform.side_gain_db,
            left=0.0,
            right=float(transform.side_gain_db[-1]),
        )
        side_factor = np.power(10.0, strength * _transient_protection(time_s) * side_db / 20.0)
        left, right = output[:, 0].copy(), output[:, 1].copy()
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right) * side_factor
        output[:, 0] = mid + side
        output[:, 1] = mid - side
    fade_frames = min(output.shape[0], max(1, int(round(0.04 * sample_rate))))
    output[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float64)[:, None]
    return output.astype(np.float32), transform


def waveform_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values = _as_audio(left).reshape(-1)
    right_values = _as_audio(right).reshape(-1)
    left_values = left_values - float(np.mean(left_values))
    right_values = right_values - float(np.mean(right_values))
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    return float(np.dot(left_values, right_values) / max(denominator, EPSILON))


def common_peak_safe(audio_bank: Sequence[np.ndarray], *, peak_dbfs: float = -1.0) -> list[np.ndarray]:
    if not audio_bank:
        raise ValueError("audio_bank is empty")
    values = [_as_audio(audio) for audio in audio_bank]
    peak = max(float(np.max(np.abs(audio))) for audio in values)
    limit = 10.0 ** (float(peak_dbfs) / 20.0)
    scale = min(1.0, limit / max(peak, EPSILON))
    return [(audio * scale).astype(np.float32) for audio in values]
