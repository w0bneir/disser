"""CPU-only building blocks for structure-aware single-reference SFX variation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal


EPSILON = 1e-10


@dataclass(frozen=True)
class EventRegions:
    sample_rate: int
    frames: int
    peak_sample: int
    attack_end_sample: int
    body_end_sample: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate

    def to_dict(self) -> dict[str, float | int]:
        output = asdict(self)
        output.update(
            {
                "peak_seconds": self.peak_sample / self.sample_rate,
                "attack_end_seconds": self.attack_end_sample / self.sample_rate,
                "body_end_seconds": self.body_end_sample / self.sample_rate,
                "duration_seconds": self.duration_seconds,
            }
        )
        return output


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Пустой аудиофайл: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"NaN/Inf в аудиофайле: {path}")
    return audio.mean(axis=1, dtype=np.float32), int(sample_rate)


def rms_envelope(
    audio: np.ndarray,
    *,
    frame_length: int = 256,
    hop_length: int = 128,
) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if values.size < frame_length:
        values = np.pad(values, (0, frame_length - values.size))
    frame_count = 1 + int(np.ceil((values.size - frame_length) / hop_length))
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(values, (0, max(0, padded_length - values.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    return np.sqrt(np.mean(np.square(frames), axis=1) + EPSILON)


def _first_sustained_below(
    values_db: np.ndarray,
    *,
    start: int,
    threshold_db: float,
    sustained_frames: int,
) -> int:
    below = values_db <= threshold_db
    for index in range(start, max(start, below.size - sustained_frames + 1)):
        if bool(np.all(below[index : index + sustained_frames])):
            return index
    return below.size - 1


def analyze_event_regions(audio: np.ndarray, sample_rate: int) -> EventRegions:
    """Estimate attack/body/tail boundaries from the post-peak RMS decay."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size < int(0.15 * sample_rate):
        raise ValueError("Референс слишком короткий для attack/body/tail анализа")
    envelope = rms_envelope(values)
    normalized_db = 20.0 * np.log10(envelope / max(float(envelope.max()), EPSILON) + EPSILON)
    peak_frame = int(np.argmax(envelope))
    attack_frame = _first_sustained_below(
        normalized_db,
        start=peak_frame,
        threshold_db=-6.0,
        sustained_frames=4,
    )
    body_frame = _first_sustained_below(
        normalized_db,
        start=max(attack_frame + 1, peak_frame),
        threshold_db=-18.0,
        sustained_frames=8,
    )
    peak_sample = min(values.size - 1, peak_frame * 128 + 128)
    raw_attack_end = attack_frame * 128 + 128
    raw_body_end = body_frame * 128 + 128
    attack_end = int(
        np.clip(
            raw_attack_end,
            peak_sample + int(0.03 * sample_rate),
            min(values.size - 1, peak_sample + int(0.12 * sample_rate)),
        )
    )
    body_end = int(
        np.clip(
            raw_body_end,
            attack_end + int(0.10 * sample_rate),
            min(values.size - 1, attack_end + int(0.60 * sample_rate)),
        )
    )
    return EventRegions(
        sample_rate=sample_rate,
        frames=values.size,
        peak_sample=peak_sample,
        attack_end_sample=attack_end,
        body_end_sample=body_end,
    )


def region_mask(
    frames: int,
    *,
    start: int,
    end: int,
    fade_in: int,
    fade_out: int,
) -> np.ndarray:
    if frames < 1 or not 0 <= start < end <= frames:
        raise ValueError("Некорректные границы region mask")
    mask = np.zeros(frames, dtype=np.float32)
    mask[start:end] = 1.0
    if start > 0 and fade_in > 0:
        count = min(fade_in, end - start)
        phase = np.linspace(0.0, np.pi / 2.0, count, endpoint=True)
        mask[start : start + count] = np.square(np.sin(phase)).astype(np.float32)
    if end < frames and fade_out > 0:
        count = min(fade_out, end - start)
        phase = np.linspace(np.pi / 2.0, 0.0, count, endpoint=True)
        mask[end - count : end] = np.square(np.sin(phase)).astype(np.float32)
    return mask


def build_component_masks(regions: EventRegions) -> dict[str, np.ndarray]:
    sr = regions.sample_rate
    attack_fade = max(1, int(round(0.02 * sr)))
    body_fade_in = max(1, int(round(0.02 * sr)))
    body_fade_out = max(1, int(round(0.05 * sr)))
    tail_fade = max(1, int(round(0.06 * sr)))
    body_start = max(0, regions.attack_end_sample - body_fade_in)
    tail_start = max(0, regions.body_end_sample - tail_fade)
    return {
        "attack": region_mask(
            regions.frames,
            start=0,
            end=regions.attack_end_sample,
            fade_in=0,
            fade_out=attack_fade,
        ),
        "body": region_mask(
            regions.frames,
            start=body_start,
            end=regions.body_end_sample,
            fade_in=body_fade_in,
            fade_out=body_fade_out,
        ),
        "tail": region_mask(
            regions.frames,
            start=tail_start,
            end=regions.frames,
            fade_in=tail_fade,
            fade_out=0,
        ),
    }


def _fix_length(audio: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if values.size >= frames:
        return values[:frames]
    return np.pad(values, (0, frames - values.size))


def _stft(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, spectrum = signal.stft(
        np.asarray(audio, dtype=np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=1024,
        noverlap=768,
        boundary="zeros",
        padded=True,
    )
    return frequencies, times, spectrum


def _istft(spectrum: np.ndarray, sample_rate: int, frames: int) -> np.ndarray:
    _, audio = signal.istft(
        spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=1024,
        noverlap=768,
        input_onesided=True,
        boundary=True,
    )
    return _fix_length(audio, frames)


def multiband_microdelay(
    audio: np.ndarray,
    sample_rate: int,
    *,
    delay_ms: tuple[float, ...],
    gain_db: tuple[float, ...],
) -> np.ndarray:
    """Apply a smooth frequency-dependent sub-millisecond group delay."""
    if len(delay_ms) != len(gain_db) or len(delay_ms) < 3:
        raise ValueError("delay_ms и gain_db должны иметь одинаковые anchor points")
    frequencies, _, spectrum = _stft(audio, sample_rate)
    anchors = np.geomspace(30.0, sample_rate / 2.0, len(delay_ms) - 1)
    anchors = np.concatenate(([0.0], anchors))
    delays = np.interp(frequencies, anchors, np.asarray(delay_ms)) / 1000.0
    gains = np.power(10.0, np.interp(frequencies, anchors, np.asarray(gain_db)) / 20.0)
    phase = np.exp(-2j * np.pi * frequencies * delays)
    processed = spectrum * (gains * phase)[:, None]
    return _istft(processed, sample_rate, np.asarray(audio).size)


def spectral_residual_transfer(
    audio: np.ndarray,
    sample_rate: int,
    *,
    strength: float,
    frequency_shift_bins: float,
    time_shift_frames: float,
    maximum_delta_db: float,
) -> np.ndarray:
    """Move local spectral residual while keeping the smooth envelope and phase."""
    if not 0.0 < strength <= 1.0 or maximum_delta_db <= 0:
        raise ValueError("Некорректные spectral residual параметры")
    _, _, spectrum = _stft(audio, sample_rate)
    magnitude = np.abs(spectrum).clip(EPSILON)
    log_magnitude = np.log(magnitude)
    smooth = ndimage.gaussian_filter(log_magnitude, sigma=(8.0, 2.0), mode="nearest")
    residual = log_magnitude - smooth
    shifted = ndimage.shift(
        residual,
        shift=(frequency_shift_bins, time_shift_frames),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    proposed_residual = (1.0 - strength) * residual + strength * shifted
    maximum_delta = maximum_delta_db * np.log(10.0) / 20.0
    delta = np.clip(proposed_residual - residual, -maximum_delta, maximum_delta)
    processed = spectrum * np.exp(delta)
    return _istft(processed, sample_rate, np.asarray(audio).size)


def cascade_allpass(
    audio: np.ndarray,
    *,
    delays_samples: tuple[int, ...],
    coefficients: tuple[float, ...],
) -> np.ndarray:
    """Decorrelate phase with stable Schroeder all-pass sections."""
    if len(delays_samples) != len(coefficients) or not delays_samples:
        raise ValueError("Нужны парные all-pass delays и coefficients")
    output = np.asarray(audio, dtype=np.float64).reshape(-1)
    for delay, coefficient in zip(delays_samples, coefficients):
        if delay < 1 or not -0.8 < coefficient < 0.8:
            raise ValueError("Небезопасный all-pass параметр")
        numerator = np.zeros(delay + 1, dtype=np.float64)
        denominator = np.zeros(delay + 1, dtype=np.float64)
        numerator[0] = coefficient
        numerator[-1] = 1.0
        denominator[0] = 1.0
        denominator[-1] = coefficient
        output = signal.lfilter(numerator, denominator, output)
    return output


def blend_component(
    reference: np.ndarray,
    wet: np.ndarray,
    mask: np.ndarray,
    *,
    peak_limit: float = 0.999,
) -> np.ndarray:
    """Energy-match one transformed component and crossfade it into the reference."""
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    processed = _fix_length(wet, target.size)
    weights = np.square(np.asarray(mask, dtype=np.float64).reshape(-1))
    if weights.size != target.size or float(weights.sum()) <= EPSILON:
        raise ValueError("Некорректная component mask")
    target_energy = float(np.sum(np.square(target) * weights))
    wet_energy = float(np.sum(np.square(processed) * weights))
    if wet_energy <= EPSILON:
        raise ValueError("Преобразованный компонент является тишиной")
    processed *= np.sqrt(target_energy / wet_energy)
    output = target * (1.0 - mask) + processed * mask
    peak = float(np.max(np.abs(output)))
    if peak > peak_limit:
        output *= peak_limit / peak
    return output.astype(np.float32)


def generate_causal_variations(
    reference: np.ndarray,
    sample_rate: int,
    regions: EventRegions,
) -> dict[str, np.ndarray]:
    masks = build_component_masks(regions)
    attack_mild = multiband_microdelay(
        reference,
        sample_rate,
        delay_ms=(0.0, 0.06, -0.10, 0.15, -0.08, 0.04),
        gain_db=(0.0, 0.25, -0.35, 0.40, -0.25, 0.0),
    )
    attack_medium = multiband_microdelay(
        reference,
        sample_rate,
        delay_ms=(0.0, -0.12, 0.20, -0.28, 0.18, -0.08),
        gain_db=(0.0, -0.45, 0.55, -0.60, 0.45, -0.15),
    )
    body_mild = spectral_residual_transfer(
        reference,
        sample_rate,
        strength=0.45,
        frequency_shift_bins=1.0,
        time_shift_frames=1.0,
        maximum_delta_db=2.0,
    )
    body_medium = spectral_residual_transfer(
        reference,
        sample_rate,
        strength=0.75,
        frequency_shift_bins=-2.0,
        time_shift_frames=1.5,
        maximum_delta_db=3.5,
    )
    tail_mild = cascade_allpass(
        reference,
        delays_samples=(17, 31, 47),
        coefficients=(0.22, -0.17, 0.20),
    )
    tail_medium = cascade_allpass(
        reference,
        delays_samples=(23, 43, 67),
        coefficients=(0.35, -0.26, 0.30),
    )
    return {
        "A1_attack_mild": blend_component(reference, attack_mild, masks["attack"]),
        "A2_attack_medium": blend_component(reference, attack_medium, masks["attack"]),
        "B1_body_mild": blend_component(reference, body_mild, masks["body"]),
        "B2_body_medium": blend_component(reference, body_medium, masks["body"]),
        "T1_tail_mild": blend_component(reference, tail_mild, masks["tail"]),
        "T2_tail_medium": blend_component(reference, tail_medium, masks["tail"]),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    count = min(left.size, right.size)
    left = left[:count]
    right = right[:count]
    if count < 2 or np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spectral_features(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    frequencies, power = signal.welch(
        values,
        fs=sample_rate,
        window="hann",
        nperseg=min(2048, values.size),
        noverlap=min(1536, max(0, values.size - 1)),
    )
    power = np.maximum(power, EPSILON)
    total = float(power.sum())
    centroid = float(np.sum(frequencies * power) / total)
    high_fraction = float(power[frequencies >= 4000.0].sum() / total)
    return centroid, high_fraction


def _strong_peak_count(audio: np.ndarray, sample_rate: int) -> int:
    envelope = rms_envelope(audio)
    normalized = envelope / max(float(envelope.max()), EPSILON)
    minimum_distance = max(1, int(round(0.08 * sample_rate / 128)))
    peaks, _ = signal.find_peaks(normalized, height=0.25, distance=minimum_distance)
    return int(peaks.size)


def diagnostic_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    regions: EventRegions,
) -> dict[str, float | int]:
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    output = _fix_length(candidate, target.size)
    target_envelope = rms_envelope(target)
    output_envelope = rms_envelope(output)
    target_centroid, target_high = _spectral_features(target, sample_rate)
    output_centroid, output_high = _spectral_features(output, sample_rate)
    scale = float(np.dot(target, output) / max(np.dot(target, target), EPSILON))
    residual = output - scale * target
    copy_ratio = np.sqrt(np.mean(np.square(residual))) / max(
        np.sqrt(np.mean(np.square(output))), EPSILON
    )
    boundaries = {
        "attack": (0, regions.attack_end_sample),
        "body": (regions.attack_end_sample, regions.body_end_sample),
        "tail": (regions.body_end_sample, regions.frames),
    }
    result: dict[str, float | int] = {
        "waveform_correlation": _correlation(target, output),
        "envelope_correlation": _correlation(target_envelope, output_envelope),
        "rms_ratio": float(
            np.sqrt(np.mean(np.square(output))) / max(np.sqrt(np.mean(np.square(target))), EPSILON)
        ),
        "copy_residual_db": float(max(-120.0, 20.0 * np.log10(max(copy_ratio, EPSILON)))),
        "spectral_centroid_delta_hz": output_centroid - target_centroid,
        "high_frequency_fraction_delta": output_high - target_high,
        "reference_strong_peak_count": _strong_peak_count(target, sample_rate),
        "candidate_strong_peak_count": _strong_peak_count(output, sample_rate),
    }
    for name, (start, end) in boundaries.items():
        result[f"{name}_waveform_correlation"] = _correlation(
            target[start:end], output[start:end]
        )
    return result


def technical_gate(reference: np.ndarray, candidate: np.ndarray) -> tuple[bool, list[str]]:
    target = np.asarray(reference).reshape(-1)
    output = np.asarray(candidate).reshape(-1)
    failures: list[str] = []
    if output.size != target.size:
        failures.append("длина не совпадает с референсом")
    if not np.isfinite(output).all():
        failures.append("обнаружены NaN/Inf")
    if output.size and float(np.max(np.abs(output))) > 1.0:
        failures.append("peak превышает 0 dBFS")
    if output.size == 0 or float(np.sqrt(np.mean(np.square(output)))) < 1e-6:
        failures.append("результат является тишиной")
    return not failures, failures
