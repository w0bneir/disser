"""Создать детерминированные pitch/time/EQ-вариации референсных SFX."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import soundfile as sf
import torch
import torchaudio.functional as AF

from audio_io import save_wav
from dsp_baseline import DspRanges, generate_dsp_variation, parameters_from_seed


MANIFEST_COLUMNS = [
    "protocol_id",
    "case_id",
    "seed",
    "reference_path",
    "output_path",
    "pitch_cents",
    "time_stretch_factor",
    "eq_gain_db",
    "eq_center_hz",
    "eq_width_octaves",
    "elapsed_seconds",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id",
        "sample_rate",
        "seeds",
        "pitch_cents",
        "time_stretch_fraction",
        "eq_gain_db",
        "eq_center_min_hz",
        "eq_center_max_hz",
        "eq_width_octaves",
        "cases",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"В DSP-конфигурации отсутствуют поля: {sorted(missing)}")
    if not config["seeds"] or not config["cases"]:
        raise ValueError("DSP-конфигурация должна содержать seeds и cases")
    return config


def _selection(
    config: dict[str, Any],
    case_id: str | None,
    seed: int | None,
) -> tuple[list[dict[str, str]], list[int]]:
    cases = list(config["cases"])
    seeds = [int(value) for value in config["seeds"]]
    if case_id is not None:
        cases = [case for case in cases if case["id"] == case_id]
        if not cases:
            raise ValueError(f"Неизвестный case-id: {case_id}")
    if seed is not None:
        if seed not in seeds:
            raise ValueError(f"Seed {seed} отсутствует в конфигурации")
        seeds = [seed]
    return cases, seeds


def _write_manifest(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(
    config_path: Path,
    results_dir: Path,
    *,
    case_id: str | None,
    seed: int | None,
    resume: bool,
) -> None:
    config = read_config(config_path)
    cases, seeds = _selection(config, case_id, seed)
    ranges = DspRanges(
        pitch_cents=float(config["pitch_cents"]),
        time_stretch_fraction=float(config["time_stretch_fraction"]),
        eq_gain_db=float(config["eq_gain_db"]),
        eq_center_min_hz=float(config["eq_center_min_hz"]),
        eq_center_max_hz=float(config["eq_center_max_hz"]),
        eq_width_octaves=float(config["eq_width_octaves"]),
    )
    manifest_path = results_dir / "manifest.csv"
    rows: list[dict[str, str | int | float]] = []
    if resume and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))

    for case in cases:
        reference_path = Path(case["reference_path"])
        audio, sample_rate = sf.read(
            reference_path,
            dtype="float32",
            always_2d=True,
        )
        original_sample_rate = int(sample_rate)
        original_num_samples = int(audio.shape[0])
        target_sample_rate = int(config["sample_rate"])
        if sample_rate != target_sample_rate:
            audio = (
                AF.resample(
                    torch.from_numpy(audio.T.copy()),
                    int(sample_rate),
                    target_sample_rate,
                )
                .T.contiguous()
                .numpy()
            )
            sample_rate = target_sample_rate
        ranges.validate(sample_rate)
        for current_seed in seeds:
            run_dir = results_dir / case["id"] / f"seed_{current_seed}"
            output_path = run_dir / "dsp.wav"
            metadata_path = run_dir / "metadata.json"
            if resume and output_path.is_file() and metadata_path.is_file():
                print(f"[=] {case['id']} seed={current_seed}: уже готово")
                continue

            parameters = parameters_from_seed(current_seed, ranges)
            started = perf_counter()
            variation = generate_dsp_variation(
                audio,
                sample_rate,
                parameters=parameters,
            )
            elapsed_seconds = perf_counter() - started
            save_wav(output_path, variation, sample_rate)
            metadata = {
                "method": "dsp_pitch_time_eq",
                "protocol_id": str(config["protocol_id"]),
                "case_id": case["id"],
                "seed": current_seed,
                "reference_path": str(reference_path),
                "reference_sha256": _sha256(reference_path),
                "original_sample_rate": original_sample_rate,
                "original_num_samples": original_num_samples,
                "sample_rate": sample_rate,
                "num_samples": int(audio.shape[0]),
                "num_channels": int(audio.shape[1]),
                "ranges": ranges.__dict__,
                "parameters": parameters.to_dict(),
                "elapsed_seconds": elapsed_seconds,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rows = [
                row
                for row in rows
                if not (
                    str(row["case_id"]) == case["id"]
                    and int(row["seed"]) == current_seed
                )
            ]
            rows.append(
                {
                    "protocol_id": str(config["protocol_id"]),
                    "case_id": case["id"],
                    "seed": current_seed,
                    "reference_path": str(reference_path),
                    "output_path": str(output_path),
                    **parameters.to_dict(),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            _write_manifest(manifest_path, rows)
            print(
                f"[+] {case['id']} seed={current_seed}: {output_path} | "
                f"pitch={parameters.pitch_cents:+.1f} ct, "
                f"time={parameters.time_stretch_factor:.4f}, "
                f"EQ={parameters.eq_gain_db:+.2f} dB @ {parameters.eq_center_hz:.0f} Hz"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dsp_baseline.json"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/dsp_baseline_v1"),
    )
    parser.add_argument("--case-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config,
        arguments.results_dir,
        case_id=arguments.case_id,
        seed=arguments.seed,
        resume=arguments.resume,
    )
