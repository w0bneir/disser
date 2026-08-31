"""Content-aware analysis and history-aware playback for natural SFX take pools.

The module is deliberately CPU-only.  It does not synthesize or modify the
spectral content of a take: the only rendering-time operations are onset
alignment, bounded gain matching, zero padding, and a short end fade.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy import signal


EPSILON = 1e-12
METHOD_VERSION = "natural_pool_optimizer_v1"
DEFAULT_COMPONENT_WEIGHTS = {
    "attack": 0.30,
    "timbre": 0.30,
    "envelope": 0.25,
    "spatial": 0.10,
    "level": 0.05,
}
SELECTION_OBJECTIVE_WEIGHTS = {
    "coverage_mean": 1.0,
    "coverage_max": 0.35,
    "diversity": -0.12,
    "centrality": 0.03,
}
DEFAULT_SCHEDULER_HISTORY = 4
DEFAULT_SCHEDULER_TEMPERATURE = 0.12
SCHEDULER_SCORE_WEIGHTS = {
    "target_alignment": 1.10,
    "separation_floor": 0.25,
    "recurrence_penalty": -0.30,
    "balance_penalty": -0.20,
    "transition_penalty": -0.75,
    "trigram_penalty": -0.55,
    "bounce_penalty": -0.15,
    "identity_penalty": -0.20,
}


@dataclass(frozen=True)
class ClipMetrics:
    file: str
    name: str
    group: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    onset_s: float
    peak_dbfs: float
    rms_dbfs: float
    early_rms_dbfs: float
    crest_db: float
    spectral_centroid_hz: float
    decay_20_db_s: float
    decay_40_db_s: float
    stereo_correlation: float
    side_to_mid_db: float
    near_full_scale_samples: int

    def json_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None
        return result


@dataclass
class AnalyzedClip:
    path: Path
    group: str
    sample_rate: int
    audio: np.ndarray
    onset_frame: int
    prepared: np.ndarray
    metrics: ClipMetrics
    features: dict[str, np.ndarray]


@dataclass(frozen=True)
class GroupRecommendation:
    group: str
    count: int
    medoid_index: int
    selected_indices: tuple[int, ...]
    median_distance: float
    maximum_distance: float
    coverage_mean: float
    coverage_max: float
    utility_score: float

    def json_dict(self, clips: Sequence[AnalyzedClip]) -> dict[str, object]:
        return {
            "group": self.group,
            "count": self.count,
            "medoid": clips[self.medoid_index].metrics.name,
            "recommended_pool": [clips[index].metrics.name for index in self.selected_indices],
            "median_distance": self.median_distance,
            "maximum_distance": self.maximum_distance,
            "coverage_mean": self.coverage_mean,
            "coverage_max": self.coverage_max,
            "utility_score": self.utility_score,
        }


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def discover_wav_files(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Каталог с дублями не найден: {directory}")
    paths = [path for path in directory.rglob("*.wav") if path.is_file()]
    paths.sort(key=lambda path: _natural_key(str(path.relative_to(directory))))
    if not paths:
        raise FileNotFoundError(f"В каталоге нет WAV-файлов: {directory}")
    return paths


def infer_group(path: Path) -> str:
    match = re.search(r"(?:shot|group)[ _-]*(\d+)(?:[._ -]|$)", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    number = re.search(r"(?:^|\D)(\d+)(?:[._ -]|$)", path.stem)
    if number:
        return number.group(1)
    return path.parent.name or "all"


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise ValueError(f"Пустой WAV: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"NaN/Inf в WAV: {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _energy_waveform(audio: np.ndarray) -> np.ndarray:
    """Return channel-energy amplitude without destructive stereo summation."""
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        return np.abs(values)
    return np.sqrt(np.mean(np.square(values), axis=1) + EPSILON)


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    if values.size == 0:
        return float(np.sqrt(EPSILON))
    return float(np.sqrt(np.mean(np.square(values)) + EPSILON))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), EPSILON)))


def _moving_rms(audio: np.ndarray, window: int) -> np.ndarray:
    values = np.square(np.asarray(audio, dtype=np.float64))
    window = max(1, min(int(window), values.size))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    means = (cumulative[window:] - cumulative[:-window]) / window
    return np.sqrt(np.maximum(means, 0.0) + EPSILON)


def detect_onset(audio: np.ndarray, sample_rate: int) -> int:
    energy = _energy_waveform(audio)
    envelope = _moving_rms(energy, max(8, int(round(sample_rate * 0.0015))))
    peak_index = int(np.argmax(envelope))
    peak = float(envelope[peak_index])
    noise_floor = float(np.percentile(envelope, 10.0))
    threshold = max(peak * 10.0 ** (-30.0 / 20.0), noise_floor * 6.0)
    crossings = np.flatnonzero(envelope[: peak_index + 1] >= threshold)
    if crossings.size == 0:
        return max(0, peak_index - int(round(0.003 * sample_rate)))
    return max(0, int(crossings[0]) - int(round(0.002 * sample_rate)))


def prepare_event_clip(
    audio: np.ndarray,
    sample_rate: int,
    onset_frame: int,
    *,
    duration_s: float = 2.5,
    pre_roll_s: float = 0.02,
    fade_out_s: float = 0.06,
) -> np.ndarray:
    if duration_s <= pre_roll_s + fade_out_s:
        raise ValueError("Слишком малая длительность подготовленного события")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    target_frames = int(round(duration_s * sample_rate))
    pre_roll = int(round(pre_roll_s * sample_rate))
    source_start = int(onset_frame) - pre_roll
    destination_start = max(0, -source_start)
    source_start = max(0, source_start)
    available = min(values.shape[0] - source_start, target_frames - destination_start)
    output = np.zeros((target_frames, values.shape[1]), dtype=np.float32)
    if available > 0:
        output[destination_start : destination_start + available] = values[
            source_start : source_start + available
        ]
    fade_frames = min(target_frames, max(1, int(round(fade_out_s * sample_rate))))
    output[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)[:, None]
    return output


def _segment(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, min(audio.shape[0], int(round(start_s * sample_rate))))
    if start >= audio.shape[0]:
        shape = (1,) if audio.ndim == 1 else (1, audio.shape[1])
        return np.zeros(shape, dtype=np.float64)
    end = min(audio.shape[0], max(start + 1, int(round(end_s * sample_rate))))
    return np.asarray(audio[start:end], dtype=np.float64)


def _log_band_profile(audio: np.ndarray, sample_rate: int, bands: int = 10) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] < 16:
        values = np.pad(values, ((0, 16 - values.shape[0]), (0, 0)))
    windowed = values * signal.windows.hann(values.shape[0], sym=False)[:, None]
    channel_power = np.square(np.abs(np.fft.rfft(windowed, axis=0)))
    spectrum = np.mean(channel_power, axis=1) + EPSILON
    frequencies = np.fft.rfftfreq(values.shape[0], 1.0 / sample_rate)
    upper = min(20_000.0, sample_rate * 0.49)
    edges = np.geomspace(40.0, upper, bands + 1)
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = spectrum[(frequencies >= low) & (frequencies < high)]
        result.append(np.log(float(selected.mean()) + EPSILON) if selected.size else np.log(EPSILON))
    profile = np.asarray(result, dtype=np.float64)
    profile -= float(profile.mean())
    return profile


def _decay_time(mono: np.ndarray, sample_rate: int, drop_db: float) -> float:
    frame = max(8, int(round(0.020 * sample_rate)))
    hop = max(1, int(round(0.005 * sample_rate)))
    if mono.size < frame:
        return float("nan")
    frames = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
    envelope = np.sqrt(np.mean(np.square(frames), axis=1) + EPSILON)
    early_frames = max(1, min(envelope.size, int(round(0.25 / (hop / sample_rate)))))
    peak_index = int(np.argmax(envelope[:early_frames]))
    peak = float(envelope[peak_index])
    threshold = peak * 10.0 ** (-drop_db / 20.0)
    future_max = np.maximum.accumulate(envelope[::-1])[::-1]
    crossings = np.flatnonzero(future_max[peak_index:] <= threshold)
    if crossings.size == 0:
        return float("nan")
    return float(crossings[0] * hop / sample_rate)


def _stereo_stats(audio: np.ndarray) -> tuple[float, float]:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1 or values.shape[1] < 2:
        return 1.0, -120.0
    left, right = values[:, 0], values[:, 1]
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    correlation = float(np.dot(left_centered, right_centered) / max(denominator, EPSILON))
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    ratio_db = _db(_rms(side) / max(_rms(mid), EPSILON))
    return float(np.clip(correlation, -1.0, 1.0)), ratio_db


def _extract_features(
    prepared: np.ndarray,
    sample_rate: int,
    *,
    pre_roll_s: float,
    early_rms_dbfs: float,
    peak_dbfs: float,
    decay_20_db_s: float,
    decay_40_db_s: float,
) -> dict[str, np.ndarray]:
    onset = pre_roll_s

    attack_profiles = []
    for start, end in ((0.0, 0.015), (0.015, 0.050), (0.050, 0.120)):
        attack_profiles.append(
            _log_band_profile(_segment(prepared, sample_rate, onset + start, onset + end), sample_rate)
        )
    attack_wave = _energy_waveform(_segment(prepared, sample_rate, onset, onset + 0.120))
    attack_envelope = _moving_rms(attack_wave, max(4, int(round(0.001 * sample_rate))))
    attack_envelope = np.maximum(signal.resample(attack_envelope, 24), EPSILON)
    attack_envelope = 20.0 * np.log10(attack_envelope / max(float(attack_envelope.max()), EPSILON) + EPSILON)
    attack_envelope = np.clip(attack_envelope, -80.0, 0.0)

    timbre_profiles = []
    for start, end in ((0.0, 0.080), (0.080, 0.300), (0.300, 1.200)):
        timbre_profiles.append(
            _log_band_profile(_segment(prepared, sample_rate, onset + start, onset + end), sample_rate)
        )

    envelope_edges = np.concatenate(([0.0], np.geomspace(0.004, 2.0, 17)))
    envelope_values = []
    reference_energy = _rms(_segment(prepared, sample_rate, onset, onset + 0.080))
    for start, end in zip(envelope_edges[:-1], envelope_edges[1:]):
        value = _rms(_segment(prepared, sample_rate, onset + float(start), onset + float(end)))
        envelope_values.append(
            float(np.clip(_db(value / max(reference_energy, EPSILON)), -80.0, 20.0))
        )
    envelope_values.extend(
        [
            decay_20_db_s if math.isfinite(decay_20_db_s) else 3.0,
            decay_40_db_s if math.isfinite(decay_40_db_s) else 3.0,
        ]
    )

    spatial_values = []
    for start, end in ((0.0, 0.080), (0.080, 0.300), (0.300, 1.200)):
        correlation, ratio = _stereo_stats(_segment(prepared, sample_rate, onset + start, onset + end))
        spatial_values.extend([correlation, np.clip(ratio, -60.0, 20.0)])

    return {
        "attack": np.concatenate((*attack_profiles, attack_envelope)).astype(np.float64),
        "timbre": np.concatenate(timbre_profiles).astype(np.float64),
        "envelope": np.asarray(envelope_values, dtype=np.float64),
        "spatial": np.asarray(spatial_values, dtype=np.float64),
        "level": np.asarray([early_rms_dbfs, peak_dbfs], dtype=np.float64),
    }


def analyze_file(
    path: Path,
    *,
    clip_duration_s: float = 2.5,
    pre_roll_s: float = 0.02,
) -> AnalyzedClip:
    audio, sample_rate = read_audio(path)
    onset_frame = detect_onset(audio, sample_rate)
    prepared = prepare_event_clip(
        audio,
        sample_rate,
        onset_frame,
        duration_s=clip_duration_s,
        pre_roll_s=pre_roll_s,
    )
    energy_waveform = _energy_waveform(audio)
    analysis_end = min(audio.shape[0], onset_frame + int(round(2.0 * sample_rate)))
    active_energy = energy_waveform[onset_frame:analysis_end]
    early = audio[onset_frame : min(audio.shape[0], onset_frame + int(round(0.5 * sample_rate)))]
    peak = float(np.max(np.abs(audio)))
    full_rms = _rms(audio)
    early_rms = _rms(early)

    spectrum_source = audio[onset_frame:analysis_end]
    if spectrum_source.size:
        windowed = spectrum_source * signal.windows.hann(spectrum_source.shape[0], sym=False)[:, None]
        power = np.mean(np.square(np.abs(np.fft.rfft(windowed, axis=0))), axis=1)
        frequencies = np.fft.rfftfreq(windowed.shape[0], 1.0 / sample_rate)
        centroid = float(np.sum(frequencies * power) / max(float(np.sum(power)), EPSILON))
    else:
        centroid = 0.0
    correlation, side_mid_db = _stereo_stats(audio[onset_frame:analysis_end])
    decay_20 = _decay_time(active_energy, sample_rate, 20.0)
    decay_40 = _decay_time(active_energy, sample_rate, 40.0)
    metrics = ClipMetrics(
        file=str(Path(path).resolve()),
        name=Path(path).name,
        group=infer_group(Path(path)),
        sample_rate=sample_rate,
        channels=int(audio.shape[1]),
        frames=int(audio.shape[0]),
        duration_s=float(audio.shape[0] / sample_rate),
        onset_s=float(onset_frame / sample_rate),
        peak_dbfs=_db(peak),
        rms_dbfs=_db(full_rms),
        early_rms_dbfs=_db(early_rms),
        crest_db=_db(peak / max(full_rms, EPSILON)),
        spectral_centroid_hz=centroid,
        decay_20_db_s=decay_20,
        decay_40_db_s=decay_40,
        stereo_correlation=correlation,
        side_to_mid_db=side_mid_db,
        near_full_scale_samples=int(np.count_nonzero(np.abs(audio) >= 0.9999)),
    )
    features = _extract_features(
        prepared,
        sample_rate,
        pre_roll_s=pre_roll_s,
        early_rms_dbfs=metrics.early_rms_dbfs,
        peak_dbfs=metrics.peak_dbfs,
        decay_20_db_s=metrics.decay_20_db_s,
        decay_40_db_s=metrics.decay_40_db_s,
    )
    return AnalyzedClip(
        path=Path(path),
        group=metrics.group,
        sample_rate=sample_rate,
        audio=audio,
        onset_frame=onset_frame,
        prepared=prepared,
        metrics=metrics,
        features=features,
    )


def analyze_directory(
    directory: Path,
    *,
    clip_duration_s: float = 2.5,
    pre_roll_s: float = 0.02,
) -> list[AnalyzedClip]:
    clips = [
        analyze_file(path, clip_duration_s=clip_duration_s, pre_roll_s=pre_roll_s)
        for path in discover_wav_files(directory)
    ]
    sample_rates = {clip.sample_rate for clip in clips}
    if len(sample_rates) != 1:
        raise ValueError(f"Смешанные частоты дискретизации не поддерживаются: {sorted(sample_rates)}")
    return clips


def _robust_standardize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - median), axis=0)
    standard = np.std(values, axis=0)
    scale = np.where(mad > 1e-7, mad, np.where(standard > 1e-7, standard, 1.0))
    return np.clip((values - median) / scale, -8.0, 8.0)


def _pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    differences = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.mean(np.square(differences), axis=2))


def build_distance_matrices(
    clips: Sequence[AnalyzedClip],
    *,
    weights: Mapping[str, float] = DEFAULT_COMPONENT_WEIGHTS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(clips) < 2:
        raise ValueError("Для матрицы различий нужны минимум два дубля")
    unknown = set(weights) - set(clips[0].features)
    if unknown:
        raise ValueError(f"Неизвестные компоненты расстояния: {sorted(unknown)}")
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0:
        raise ValueError("Сумма весов должна быть положительной")
    component_matrices: dict[str, np.ndarray] = {}
    total = np.zeros((len(clips), len(clips)), dtype=np.float64)
    off_diagonal = ~np.eye(len(clips), dtype=bool)
    for component, weight in weights.items():
        feature_matrix = np.stack([clip.features[component] for clip in clips])
        distances = _pairwise_euclidean(_robust_standardize(feature_matrix))
        positive = distances[off_diagonal & (distances > EPSILON)]
        scale = float(np.median(positive)) if positive.size else 1.0
        normalized = distances / max(scale, EPSILON)
        np.fill_diagonal(normalized, 0.0)
        component_matrices[component] = normalized
        total += (float(weight) / weight_sum) * normalized
    total = 0.5 * (total + total.T)
    np.fill_diagonal(total, 0.0)
    return total, component_matrices


def build_groupwise_distance_matrices(
    clips: Sequence[AnalyzedClip],
    *,
    weights: Mapping[str, float] = DEFAULT_COMPONENT_WEIGHTS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit feature scaling separately inside each acoustic context.

    Between-group cells are NaN by design: the matrix is intended for pool
    selection and scheduling, not for claiming cross-context similarity.
    """
    if len(clips) < 2:
        raise ValueError("Для матрицы различий нужны минимум два дубля")
    total = np.full((len(clips), len(clips)), np.nan, dtype=np.float64)
    components = {
        component: np.full((len(clips), len(clips)), np.nan, dtype=np.float64)
        for component in weights
    }
    for indices in group_index_map(clips).values():
        if len(indices) == 1:
            total[indices[0], indices[0]] = 0.0
            for matrix in components.values():
                matrix[indices[0], indices[0]] = 0.0
            continue
        local_clips = [clips[index] for index in indices]
        local_total, local_components = build_distance_matrices(local_clips, weights=weights)
        target = np.ix_(indices, indices)
        total[target] = local_total
        for component, matrix in local_components.items():
            components[component][target] = matrix
    return total, components


def group_index_map(clips: Sequence[AnalyzedClip]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, clip in enumerate(clips):
        result.setdefault(clip.group, []).append(index)
    return dict(sorted(result.items(), key=lambda item: _natural_key(item[0])))


def medoid_index(indices: Sequence[int], distance_matrix: np.ndarray) -> int:
    values = list(map(int, indices))
    if not values:
        raise ValueError("Пустая группа")
    submatrix = distance_matrix[np.ix_(values, values)]
    return values[int(np.argmin(np.mean(submatrix, axis=1)))]


def pool_coverage(
    group_indices: Sequence[int],
    selected_indices: Sequence[int],
    distance_matrix: np.ndarray,
) -> tuple[float, float]:
    distances = distance_matrix[np.ix_(list(group_indices), list(selected_indices))]
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest)), float(np.max(nearest))


def select_representative_pool(
    indices: Sequence[int],
    distance_matrix: np.ndarray,
    *,
    size: int,
) -> list[int]:
    group = list(map(int, indices))
    if not 1 <= size <= len(group):
        raise ValueError("Размер пула вне допустимого диапазона")
    center = medoid_index(group, distance_matrix)
    if size == 1:
        return [center]
    if size == len(group):
        return group.copy()

    candidate_count = math.comb(len(group) - 1, size - 1)
    if candidate_count > 200_000:
        selected = [center]
        while len(selected) < size:
            remaining = [index for index in group if index not in selected]
            best = min(
                remaining,
                key=lambda index: (
                    pool_coverage(group, [*selected, index], distance_matrix)[0],
                    pool_coverage(group, [*selected, index], distance_matrix)[1],
                ),
            )
            selected.append(best)
        return selected

    group_order = {value: position for position, value in enumerate(group)}
    remaining_without_center = [value for value in group if value != center]
    candidate_sets = (
        tuple(sorted((center, *others), key=group_order.__getitem__))
        for others in combinations(remaining_without_center, size - 1)
    )
    best_selection: tuple[int, ...] | None = None
    best_objective = float("inf")
    for selected in candidate_sets:
        coverage_mean, coverage_max = pool_coverage(group, selected, distance_matrix)
        pair_values = [distance_matrix[left, right] for left, right in combinations(selected, 2)]
        diversity = float(np.mean(pair_values)) if pair_values else 0.0
        centrality = float(np.mean([distance_matrix[center, index] for index in selected]))
        objective = (
            SELECTION_OBJECTIVE_WEIGHTS["coverage_mean"] * coverage_mean
            + SELECTION_OBJECTIVE_WEIGHTS["coverage_max"] * coverage_max
            + SELECTION_OBJECTIVE_WEIGHTS["diversity"] * diversity
            + SELECTION_OBJECTIVE_WEIGHTS["centrality"] * centrality
        )
        if objective < best_objective - 1e-12:
            best_objective = objective
            best_selection = selected
    if best_selection is None:
        raise RuntimeError("Не удалось выбрать репрезентативный пул")
    return list(best_selection)


def recommend_groups(
    clips: Sequence[AnalyzedClip],
    distance_matrix: np.ndarray,
    *,
    pool_size: int = 3,
) -> list[GroupRecommendation]:
    recommendations = []
    for group, indices in group_index_map(clips).items():
        if len(indices) < 2:
            continue
        selected = select_representative_pool(indices, distance_matrix, size=min(pool_size, len(indices)))
        submatrix = distance_matrix[np.ix_(indices, indices)]
        values = submatrix[np.triu_indices(len(indices), k=1)]
        coverage_mean, coverage_max = pool_coverage(indices, selected, distance_matrix)
        median_distance = float(np.median(values))
        maximum_distance = float(np.max(values))
        outlier_ratio = maximum_distance / max(median_distance, EPSILON)
        coherence_penalty = 1.0 + max(0.0, outlier_ratio - 2.5)
        utility = median_distance * math.log1p(len(indices)) / coherence_penalty
        recommendations.append(
            GroupRecommendation(
                group=group,
                count=len(indices),
                medoid_index=medoid_index(indices, distance_matrix),
                selected_indices=tuple(selected),
                median_distance=median_distance,
                maximum_distance=maximum_distance,
                coverage_mean=coverage_mean,
                coverage_max=coverage_max,
                utility_score=utility,
            )
        )
    return recommendations


def choose_experiment_group(recommendations: Sequence[GroupRecommendation]) -> GroupRecommendation:
    eligible = [item for item in recommendations if item.count >= 5]
    if not eligible:
        eligible = [item for item in recommendations if item.count >= 3]
    if not eligible:
        raise ValueError("Нет группы минимум из трёх дублей")
    maximum_count = max(item.count for item in eligible)
    largest = [item for item in eligible if item.count == maximum_count]
    return sorted(largest, key=lambda item: _natural_key(item.group))[0]


def random_schedule(pool: Sequence[int], *, count: int, seed: int) -> list[int]:
    values = list(map(int, pool))
    if not values or count < 1:
        raise ValueError("Пустой пул или некорректная длина")
    rng = np.random.default_rng(seed)
    return [values[int(index)] for index in rng.integers(0, len(values), size=count)]


def shuffle_schedule(pool: Sequence[int], *, count: int, seed: int) -> list[int]:
    values = list(map(int, pool))
    if len(values) < 2 or count < 1:
        raise ValueError("Shuffle требует минимум два дубля")
    rng = np.random.default_rng(seed)
    result: list[int] = []
    while len(result) < count:
        block = [values[int(index)] for index in rng.permutation(len(values))]
        if result and block[0] == result[-1]:
            block[0], block[1] = block[1], block[0]
        result.extend(block)
    return result[:count]


def perceptual_schedule(
    pool: Sequence[int],
    distance_matrix: np.ndarray,
    *,
    count: int,
    seed: int,
    history: int = DEFAULT_SCHEDULER_HISTORY,
    temperature: float = DEFAULT_SCHEDULER_TEMPERATURE,
) -> list[int]:
    values = list(map(int, pool))
    if len(values) < 2 or count < 1:
        raise ValueError("Content-aware scheduler требует минимум два дубля")
    if history < 1 or temperature <= 0:
        raise ValueError("Некорректные параметры scheduler")
    rng = np.random.default_rng(seed)
    pool_distances = distance_matrix[np.ix_(values, values)]
    positive = pool_distances[pool_distances > EPSILON]
    scale = float(np.median(positive)) if positive.size else 1.0
    normalized = distance_matrix / max(scale, EPSILON)
    target_distance = (
        float(np.percentile(positive / max(scale, EPSILON), 70.0))
        if positive.size
        else 1.0
    )
    center = medoid_index(values, distance_matrix)
    center_distances = np.asarray([normalized[value, center] for value in values], dtype=np.float64)
    identity_threshold = float(np.percentile(center_distances, 80.0))
    quotas = {value: count // len(values) for value in values}
    remainder_order = [values[int(index)] for index in rng.permutation(len(values))]
    for value in remainder_order[: count % len(values)]:
        quotas[value] += 1
    start_candidates = [value for value in values if quotas[value] > 0]
    start = int(rng.choice(start_candidates))
    result = [start]
    usage: Counter[int] = Counter(result)
    transitions: Counter[tuple[int, int]] = Counter()
    trigrams: Counter[tuple[int, int, int]] = Counter()

    while len(result) < count:
        recent = result[-history:]
        candidates = [
            candidate
            for candidate in values
            if candidate != result[-1] and usage[candidate] < quotas[candidate]
        ]
        feasible = []
        for candidate in candidates:
            remaining = {
                value: quotas[value] - usage[value] - (1 if value == candidate else 0)
                for value in values
            }
            total_remaining = sum(remaining.values())
            candidate_can_be_separated = remaining[candidate] <= total_remaining - remaining[candidate]
            largest_other = max(remaining.values()) if remaining else 0
            globally_arrangeable = largest_other <= total_remaining - largest_other + 1
            if candidate_can_be_separated and globally_arrangeable:
                feasible.append(candidate)
        if feasible:
            candidates = feasible
        if not candidates:
            candidates = [candidate for candidate in values if candidate != result[-1]]
        outgoing_minimum = min(transitions[(result[-1], candidate)] for candidate in candidates)
        raw_scores = []
        for candidate in candidates:
            recency_weights = np.exp(-0.65 * np.arange(len(recent) - 1, -1, -1, dtype=np.float64))
            novelty_values = np.asarray([normalized[candidate, item] for item in recent], dtype=np.float64)
            target_alignment = float(
                np.average(
                    1.0 - np.abs(novelty_values - target_distance) / max(target_distance, EPSILON),
                    weights=recency_weights,
                )
            )
            separation_floor = min(1.0, float(novelty_values[-1]) / max(target_distance, EPSILON))
            recurrence = float(
                np.average(
                    np.asarray([1.0 if candidate == item else 0.0 for item in recent]),
                    weights=recency_weights,
                )
            )
            expected_usage = len(result) / len(values)
            balance_penalty = max(0.0, usage[candidate] - expected_usage) / max(1.0, expected_usage)
            transition_penalty = transitions[(result[-1], candidate)] - outgoing_minimum
            trigram_penalty = (
                trigrams[(result[-2], result[-1], candidate)] if len(result) >= 2 else 0
            )
            bounce_penalty = 1.0 if len(result) >= 2 and result[-2] == candidate else 0.0
            identity_penalty = max(0.0, normalized[candidate, center] - identity_threshold)
            raw_scores.append(
                SCHEDULER_SCORE_WEIGHTS["target_alignment"] * target_alignment
                + SCHEDULER_SCORE_WEIGHTS["separation_floor"] * separation_floor
                + SCHEDULER_SCORE_WEIGHTS["recurrence_penalty"] * recurrence
                + SCHEDULER_SCORE_WEIGHTS["balance_penalty"] * balance_penalty
                + SCHEDULER_SCORE_WEIGHTS["transition_penalty"] * transition_penalty
                + SCHEDULER_SCORE_WEIGHTS["trigram_penalty"] * trigram_penalty
                + SCHEDULER_SCORE_WEIGHTS["bounce_penalty"] * bounce_penalty
                + SCHEDULER_SCORE_WEIGHTS["identity_penalty"] * identity_penalty
            )
        scores = np.asarray(raw_scores, dtype=np.float64)
        scores -= float(np.max(scores))
        probabilities = np.exp(scores / temperature)
        probabilities /= float(np.sum(probabilities))
        selected = int(rng.choice(candidates, p=probabilities))
        transitions[(result[-1], selected)] += 1
        if len(result) >= 2:
            trigrams[(result[-2], result[-1], selected)] += 1
        result.append(selected)
        usage[selected] += 1
    return result


def schedule_diagnostics(schedule: Sequence[int], distance_matrix: np.ndarray, *, history: int = 4) -> dict[str, float | int]:
    values = list(map(int, schedule))
    if not values:
        raise ValueError("Пустое расписание")
    adjacent = [float(distance_matrix[left, right]) for left, right in zip(values, values[1:])]
    history_values = []
    for position, candidate in enumerate(values):
        recent = values[max(0, position - history) : position]
        if recent:
            history_values.append(float(np.mean([distance_matrix[candidate, item] for item in recent])))
    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    probabilities = counts / float(np.sum(counts))
    entropy = float(-np.sum(probabilities * np.log(probabilities + EPSILON)))
    normalized_entropy = entropy / max(math.log(len(counts)), EPSILON) if len(counts) > 1 else 0.0
    transition_counts = np.asarray(list(Counter(zip(values, values[1:])).values()), dtype=np.float64)
    if transition_counts.size:
        transition_probabilities = transition_counts / float(np.sum(transition_counts))
        transition_entropy = float(-np.sum(transition_probabilities * np.log(transition_probabilities + EPSILON)))
    else:
        transition_entropy = 0.0
    return {
        "events": len(values),
        "unique_takes": len(set(values)),
        "immediate_repeats": sum(left == right for left, right in zip(values, values[1:])),
        "aba_patterns": sum(values[index] == values[index - 2] for index in range(2, len(values))),
        "mean_adjacent_distance": float(np.mean(adjacent)) if adjacent else 0.0,
        "minimum_adjacent_distance": float(np.min(adjacent)) if adjacent else 0.0,
        "mean_history_distance": float(np.mean(history_values)) if history_values else 0.0,
        "usage_entropy_normalized": normalized_entropy,
        "transition_entropy": transition_entropy,
    }


def normalize_prepared_bank(
    clips: Sequence[AnalyzedClip],
    indices: Sequence[int],
    *,
    pre_roll_s: float = 0.02,
    early_window_s: float = 0.5,
    max_gain_change_db: float = 4.0,
    peak_limit: float = 10.0 ** (-1.0 / 20.0),
) -> dict[int, np.ndarray]:
    selected = list(map(int, indices))
    if not selected:
        raise ValueError("Пустой банк")
    sample_rates = {clips[index].sample_rate for index in selected}
    if len(sample_rates) != 1:
        raise ValueError("Смешанные sample rate в банке")
    sample_rate = next(iter(sample_rates))
    start = int(round(pre_roll_s * sample_rate))
    end = start + int(round(early_window_s * sample_rate))
    energies = {
        index: _rms(clips[index].prepared[start:end])
        for index in selected
    }
    target = float(np.median(list(energies.values())))
    minimum_gain = 10.0 ** (-max_gain_change_db / 20.0)
    maximum_gain = 10.0 ** (max_gain_change_db / 20.0)
    bank = {
        index: np.asarray(clips[index].prepared, dtype=np.float64)
        * float(np.clip(target / max(energies[index], EPSILON), minimum_gain, maximum_gain))
        for index in selected
    }
    largest_peak = max(float(np.max(np.abs(values))) for values in bank.values())
    common_scale = min(1.0, peak_limit / max(largest_peak, EPSILON))
    return {index: (values * common_scale).astype(np.float32) for index, values in bank.items()}


def assemble_sequence(
    bank: Mapping[int, np.ndarray],
    schedule: Sequence[int],
    sample_rate: int,
    *,
    interval_ms: float,
    lead_ms: float = 250.0,
) -> np.ndarray:
    if not schedule or interval_ms <= 0:
        raise ValueError("Пустое расписание или некорректный интервал")
    first = np.asarray(bank[int(schedule[0])])
    if first.ndim == 1:
        first = first[:, None]
    interval = int(round(sample_rate * interval_ms / 1000.0))
    lead = int(round(sample_rate * lead_ms / 1000.0))
    event_frames = max(np.asarray(bank[int(index)]).shape[0] for index in schedule)
    output = np.zeros((lead + interval * (len(schedule) - 1) + event_frames, first.shape[1]), dtype=np.float64)
    for position, index in enumerate(schedule):
        event = np.asarray(bank[int(index)], dtype=np.float64)
        if event.ndim == 1:
            event = event[:, None]
        if event.shape[1] != output.shape[1]:
            raise ValueError("Несовместимое число каналов в банке")
        start = lead + position * interval
        output[start : start + event.shape[0]] += event
    return output.astype(np.float32)


def rms_match_sequences(
    sequences: Mapping[str, np.ndarray],
    *,
    anchor: str,
    peak_limit: float = 10.0 ** (-1.0 / 20.0),
) -> dict[str, np.ndarray]:
    if anchor not in sequences:
        raise ValueError("Anchor отсутствует")
    target = _rms(sequences[anchor])
    matched = {
        name: np.asarray(values, dtype=np.float64) * target / max(_rms(values), EPSILON)
        for name, values in sequences.items()
    }
    largest_peak = max(float(np.max(np.abs(values))) for values in matched.values())
    common_scale = min(1.0, peak_limit / max(largest_peak, EPSILON))
    return {name: (values * common_scale).astype(np.float32) for name, values in matched.items()}


def technical_audio_gate(audio: np.ndarray) -> tuple[bool, list[str]]:
    values = np.asarray(audio)
    failures = []
    if values.size == 0:
        failures.append("пустой сигнал")
    if not np.isfinite(values).all():
        failures.append("NaN/Inf")
    if values.size and _rms(values) < 1e-7:
        failures.append("тишина")
    if values.size and float(np.max(np.abs(values))) > 1.0 + 1e-7:
        failures.append("peak выше 0 dBFS")
    return not failures, failures


def names_for_schedule(schedule: Iterable[int], clips: Sequence[AnalyzedClip]) -> list[str]:
    return [clips[int(index)].metrics.name for index in schedule]
