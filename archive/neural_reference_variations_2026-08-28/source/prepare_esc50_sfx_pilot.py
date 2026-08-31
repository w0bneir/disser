"""Build a deterministic transient-SFX pilot corpus from the official ESC-50 archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from scipy import signal


ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC50_ARCHIVE_BYTES = 645_832_161
ESC50_ARCHIVE_SHA256 = "661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f"
ESC50_LICENSE = "CC BY-NC 3.0"
SAMPLE_RATE = 44_100
DEFAULT_WINDOW_SECONDS = 1.75
DEFAULT_PRE_ATTACK_SECONDS = 0.08

# The pilot deliberately excludes long soundscapes and obvious music-like classes.
# Each source is reduced to one window around its strongest transient.
TRANSIENT_SFX_CATEGORIES = (
    "can_opening",
    "clapping",
    "coughing",
    "door_wood_creaks",
    "door_wood_knock",
    "fireworks",
    "footsteps",
    "glass_breaking",
    "mouse_click",
    "sneezing",
    "water_drops",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден ESC-50 archive: {path}")
    actual_size = path.stat().st_size
    if actual_size != ESC50_ARCHIVE_BYTES:
        raise ValueError(
            f"Неверный размер ESC-50 archive: {actual_size}; ожидалось {ESC50_ARCHIVE_BYTES}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != ESC50_ARCHIVE_SHA256:
        raise ValueError(f"Неверный SHA256 ESC-50 archive: {actual_hash}")
    return {
        "path": str(path.resolve()),
        "bytes": actual_size,
        "sha256": actual_hash,
        "source_commit": ESC50_COMMIT,
    }


def _rms_envelope(audio: np.ndarray, frame: int = 1024, hop: int = 256) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if values.size < frame:
        values = np.pad(values, (0, frame - values.size))
    count = 1 + int(np.ceil((values.size - frame) / hop))
    padded = np.pad(values, (0, max(0, (count - 1) * hop + frame - values.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def dominant_event_crop(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    pre_attack_seconds: float = DEFAULT_PRE_ATTACK_SECONDS,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Crop one deterministic window around the strongest RMS transient."""
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Ожидался непустой mono/stereo audio array")
    if sample_rate != SAMPLE_RATE:
        divisor = int(np.gcd(sample_rate, SAMPLE_RATE))
        values = signal.resample_poly(
            values,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
        sample_rate = SAMPLE_RATE

    target_frames = int(round(window_seconds * sample_rate))
    source_envelope = _rms_envelope(values)
    peak_frame = int(np.argmax(source_envelope))
    peak_sample = peak_frame * 256 + 512
    start = peak_sample - int(round(pre_attack_seconds * sample_rate))
    start = min(max(0, start), max(0, values.size - target_frames))
    crop = values[start : start + target_frames]
    if crop.size < target_frames:
        crop = np.pad(crop, (0, target_frames - crop.size))
    crop = crop.astype(np.float32, copy=True)

    # Prevent a crop-boundary click without changing the event body.
    fade_frames = min(int(round(0.005 * sample_rate)), crop.size // 2)
    if fade_frames:
        ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        crop[:fade_frames] *= ramp
        crop[-fade_frames:] *= ramp[::-1]

    crop_envelope = _rms_envelope(crop)
    normalized = crop_envelope / max(float(crop_envelope.max()), 1e-12)
    minimum_peak_distance = max(1, int(round(0.08 * sample_rate / 256)))
    strong_peaks, properties = signal.find_peaks(
        normalized,
        height=0.25,
        distance=minimum_peak_distance,
    )
    heights = np.sort(properties.get("peak_heights", np.asarray([], dtype=np.float64)))[::-1]
    second_peak_ratio = float(heights[1] / heights[0]) if heights.size > 1 else 0.0
    return crop, {
        "source_peak_seconds": float(peak_sample / sample_rate),
        "crop_start_seconds": float(start / sample_rate),
        "strong_peak_count": int(strong_peaks.size),
        "second_to_first_peak_ratio": second_peak_ratio,
        "active_envelope_fraction": float(np.mean(normalized >= 0.1)),
        "peak": float(np.max(np.abs(crop))),
        "rms": float(np.sqrt(np.mean(np.square(crop, dtype=np.float64)))),
    }


def prepare_corpus(
    *,
    archive_path: Path,
    output_dir: Path,
    validation_fold: int = 5,
    categories: tuple[str, ...] = TRANSIENT_SFX_CATEGORIES,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Каталог pilot corpus не пуст: {output_dir}")
    if not 1 <= validation_fold <= 5:
        raise ValueError("validation_fold должен быть в диапазоне 1..5")
    archive_report = validate_archive(archive_path)

    root = f"ESC-50-{ESC50_COMMIT}/"
    metadata_member = root + "meta/esc50.csv"
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        metadata_text = archive.read(metadata_member).decode("utf-8")
        metadata = list(csv.DictReader(io.StringIO(metadata_text)))
        selected = [row for row in metadata if row["category"] in categories]
        expected = 40 * len(categories)
        if len(selected) != expected:
            raise ValueError(f"Выбрано {len(selected)} ESC-50 rows вместо {expected}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(selected, start=1):
            source_member = root + "audio/" + row["filename"]
            audio_bytes = archive.read(source_member)
            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=True,
            )
            crop, diagnostics = dominant_event_crop(audio, int(sample_rate))
            split = "val" if int(row["fold"]) == validation_fold else "train"
            category_dir = output_dir / split / str(row["category"])
            category_dir.mkdir(parents=True, exist_ok=True)
            filename = PurePosixPath(str(row["filename"])).stem + "_event.wav"
            output_path = category_dir / filename
            sf.write(output_path, crop, SAMPLE_RATE, subtype="PCM_24")
            rows.append(
                {
                    "source_file": row["filename"],
                    "output_file": str(output_path.relative_to(output_dir)),
                    "fold": int(row["fold"]),
                    "split": split,
                    "category": row["category"],
                    **diagnostics,
                }
            )
            if index % 100 == 0:
                print(f"    prepared {index}/{len(selected)}", flush=True)

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    for split in counts:
        for category in categories:
            counts[split][category] = sum(
                row["split"] == split and row["category"] == category for row in rows
            )
    report = {
        "stage": "esc50_transient_sfx_pilot_preparation",
        "source": archive_report,
        "license": ESC50_LICENSE,
        "official_repository": "https://github.com/karolpiczak/ESC-50",
        "selection_policy": {
            "categories": list(categories),
            "validation_fold": validation_fold,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "pre_attack_seconds": DEFAULT_PRE_ATTACK_SECONDS,
            "crop_rule": "strongest 1024-sample RMS frame; deterministic 80 ms pre-roll",
            "note": "No source is accepted/rejected using the eventual thesis outcome metrics.",
        },
        "counts": counts,
        "items": rows,
    }
    report_path = output_dir / "corpus_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "SOURCE_AND_LICENSE.txt").write_text(
        "ESC-50\n"
        f"Source commit: {ESC50_COMMIT}\n"
        "Official repository: https://github.com/karolpiczak/ESC-50\n"
        f"Dataset license: {ESC50_LICENSE}\n"
        "This directory contains deterministic event crops for non-commercial research.\n",
        encoding="utf-8",
    )
    print(f"[+] ESC-50 SFX pilot: {output_dir.resolve()}", flush=True)
    print(
        f"[+] train={sum(counts['train'].values())}; val={sum(counts['val'].values())}",
        flush=True,
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare transient SFX pilot from ESC-50")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        prepare_corpus(
            archive_path=args.archive,
            output_dir=args.output_dir,
            validation_fold=args.validation_fold,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"[!] ESC-50 preparation blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
