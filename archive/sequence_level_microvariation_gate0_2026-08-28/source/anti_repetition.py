"""CPU-only primitives for a blind sequence-level anti-repetition test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
from scipy import signal


EPSILON = 1e-10


@dataclass(frozen=True)
class VNFParameters:
    length_ms: float
    pulses: int
    decay_db: float
    wet_mix: float
    highpass_hz: float
    seed: int


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Пустой аудиофайл: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"NaN/Inf в аудиофайле: {path}")
    return audio.mean(axis=1, dtype=np.float32), int(sample_rate)


def rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)) + EPSILON))


def energy_match(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    output = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if output.size != target.size:
        raise ValueError("Длина candidate не совпадает с reference")
    output *= rms(target) / max(rms(output), EPSILON)
    return output.astype(np.float32)


def peak_safe(audio: np.ndarray, limit: float = 0.999) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float64).copy()
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > limit:
        output *= limit / peak
    return output.astype(np.float32)


def pitch_gain_variant(
    reference: np.ndarray,
    *,
    semitones: float,
    gain_db: float,
) -> np.ndarray:
    """Approximate the common game-engine playback-rate and gain randomisation."""
    source = np.asarray(reference, dtype=np.float64).reshape(-1)
    ratio = 2.0 ** (float(semitones) / 12.0)
    changed_length = max(1, int(round(source.size / ratio)))
    shifted = signal.resample(source, changed_length)
    if shifted.size < source.size:
        shifted = np.pad(shifted, (0, source.size - shifted.size))
    else:
        shifted = shifted[: source.size]
    shifted *= 10.0 ** (float(gain_db) / 20.0)
    return peak_safe(shifted)


def velvet_noise_kernel(
    sample_rate: int,
    parameters: VNFParameters,
) -> np.ndarray:
    """Create a sparse exponentially decaying FIR inspired by short VN filters."""
    if sample_rate < 8_000:
        raise ValueError("Слишком низкая частота дискретизации")
    if not 0.5 <= parameters.length_ms <= 10.0:
        raise ValueError("Небезопасная длина velvet-noise filter")
    if not 3 <= parameters.pulses <= 32:
        raise ValueError("Небезопасное число velvet-noise impulses")
    length = max(parameters.pulses, int(round(sample_rate * parameters.length_ms / 1000.0)))
    rng = np.random.default_rng(parameters.seed)
    kernel = np.zeros(length, dtype=np.float64)
    grid = length / parameters.pulses
    for index in range(parameters.pulses):
        start = int(np.floor(index * grid))
        end = min(length, max(start + 1, int(np.floor((index + 1) * grid))))
        location = int(rng.integers(start, end))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        progress = index / max(1, parameters.pulses - 1)
        decay = 10.0 ** (-parameters.decay_db * progress / 20.0)
        kernel[location] += sign * decay * float(rng.uniform(0.5, 1.5))
    kernel /= max(float(np.sqrt(np.sum(np.square(kernel)))), EPSILON)
    return kernel


def velvet_spectral_variant(
    reference: np.ndarray,
    sample_rate: int,
    parameters: VNFParameters,
) -> np.ndarray:
    """Add a short high-frequency-targeted decorrelated component."""
    if not 0.01 <= parameters.wet_mix <= 0.30:
        raise ValueError("Небезопасная wet_mix")
    nyquist = sample_rate / 2.0
    if not 80.0 <= parameters.highpass_hz < nyquist * 0.9:
        raise ValueError("Небезопасная highpass frequency")
    source = np.asarray(reference, dtype=np.float64).reshape(-1)
    kernel = velvet_noise_kernel(sample_rate, parameters)
    wet = signal.fftconvolve(source, kernel, mode="full")[: source.size]
    sos = signal.butter(
        1,
        parameters.highpass_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    wet = signal.sosfilt(sos, wet)
    wet *= rms(source) / max(rms(wet), EPSILON)
    output = source + parameters.wet_mix * wet
    return peak_safe(energy_match(source, output))


def spectral_profile(audio: np.ndarray, sample_rate: int, bands: int = 10) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64).reshape(-1)
    frequencies, power = signal.welch(
        source,
        fs=sample_rate,
        window="hann",
        nperseg=min(2048, source.size),
        noverlap=min(1536, max(0, source.size - 1)),
    )
    edges = np.geomspace(80.0, sample_rate / 2.0, bands + 1)
    values = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = power[(frequencies >= low) & (frequencies < high)]
        values.append(np.log(max(float(selected.mean()) if selected.size else EPSILON, EPSILON)))
    profile = np.asarray(values, dtype=np.float64)
    profile -= float(profile.mean())
    profile /= max(float(np.linalg.norm(profile)), EPSILON)
    return profile


def adaptive_schedule(
    candidates: Sequence[np.ndarray],
    reference: np.ndarray,
    sample_rate: int,
    *,
    count: int,
    history: int = 4,
) -> list[int]:
    """Choose each hit to differ from recent history while staying reference-derived."""
    if count < 1 or len(candidates) < 2:
        raise ValueError("Недостаточно кандидатов для adaptive schedule")
    profiles = np.stack([spectral_profile(item, sample_rate) for item in candidates])
    reference_profile = spectral_profile(reference, sample_rate)
    reference_distance = np.linalg.norm(profiles - reference_profile[None, :], axis=1)
    selected: list[int] = [int(np.argmax(reference_distance))]
    while len(selected) < count:
        recent = selected[-history:]
        distances = np.stack(
            [np.linalg.norm(profiles - profiles[index][None, :], axis=1) for index in recent]
        )
        novelty = distances.min(axis=0) + 0.25 * distances.mean(axis=0)
        identity_penalty = 0.15 * reference_distance
        score = novelty - identity_penalty
        score[selected[-1]] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def no_repeat_schedule(item_count: int, count: int, seed: int) -> list[int]:
    if item_count < 2 or count < 1:
        raise ValueError("Недостаточно элементов для no-repeat schedule")
    rng = np.random.default_rng(seed)
    result: list[int] = []
    while len(result) < count:
        block = list(map(int, rng.permutation(item_count)))
        if result and block[0] == result[-1]:
            block[0], block[1] = block[1], block[0]
        result.extend(block)
    return result[:count]


def assemble_sequence(
    hits: Sequence[np.ndarray],
    sample_rate: int,
    *,
    interval_ms: float,
    lead_ms: float = 250.0,
) -> np.ndarray:
    if not hits:
        raise ValueError("Пустая последовательность")
    interval = int(round(sample_rate * interval_ms / 1000.0))
    lead = int(round(sample_rate * lead_ms / 1000.0))
    if interval < 1:
        raise ValueError("Некорректный интервал")
    frames = lead + interval * (len(hits) - 1) + max(np.asarray(hit).size for hit in hits)
    output = np.zeros(frames, dtype=np.float64)
    for index, hit in enumerate(hits):
        values = np.asarray(hit, dtype=np.float64).reshape(-1)
        start = lead + index * interval
        output[start : start + values.size] += values
    return output.astype(np.float32)


def loudness_match_sequences(
    sequences: dict[str, np.ndarray],
    *,
    anchor_name: str,
    peak_limit: float = 10.0 ** (-1.0 / 20.0),
) -> dict[str, np.ndarray]:
    if anchor_name not in sequences:
        raise ValueError("Anchor sequence отсутствует")
    target_rms = rms(sequences[anchor_name])
    matched = {
        name: np.asarray(values, dtype=np.float64) * target_rms / max(rms(values), EPSILON)
        for name, values in sequences.items()
    }
    largest_peak = max(float(np.max(np.abs(values))) for values in matched.values())
    common_scale = min(1.0, peak_limit / max(largest_peak, EPSILON))
    return {name: (values * common_scale).astype(np.float32) for name, values in matched.items()}


def technical_gate(reference: np.ndarray, candidate: np.ndarray) -> tuple[bool, list[str]]:
    target = np.asarray(reference).reshape(-1)
    output = np.asarray(candidate).reshape(-1)
    failures: list[str] = []
    if output.size != target.size:
        failures.append("длина не совпадает")
    if not np.isfinite(output).all():
        failures.append("обнаружены NaN/Inf")
    if output.size == 0 or rms(output) < 1e-6:
        failures.append("тишина")
    if output.size and float(np.max(np.abs(output))) > 1.0:
        failures.append("peak превышает 0 dBFS")
    return not failures, failures
