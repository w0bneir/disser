"""Prepare a blinded listening calibration from a SAME latent sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import soundfile as sf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_audio(path: Path, expected_rate: int, expected_frames: int) -> None:
    info = sf.info(path)
    if info.samplerate != expected_rate or info.frames != expected_frames:
        raise ValueError(
            f"Incompatible audio {path}: {info.samplerate} Hz, {info.frames} frames"
        )


def run(arguments: argparse.Namespace) -> None:
    source_dir = arguments.source_dir.resolve()
    reference_source = source_dir / "reference.wav"
    if not reference_source.is_file():
        raise FileNotFoundError(reference_source)
    reference_info = sf.info(reference_source)

    metrics_by_file: dict[str, dict[str, str]] = {}
    with (source_dir / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            metrics_by_file[row["file"]] = row

    candidates: list[Path] = []
    for filename in arguments.candidates:
        path = source_dir / filename
        if not path.is_file() or filename not in metrics_by_file:
            raise FileNotFoundError(f"Candidate or metrics missing: {path}")
        _validate_audio(path, reference_info.samplerate, reference_info.frames)
        candidates.append(path)

    randomizer = random.Random(arguments.randomization_seed)
    randomizer.shuffle(candidates)
    public_dir = arguments.output_dir.resolve() / "public"
    private_dir = arguments.output_dir.resolve() / "private_do_not_open_before_scoring"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_source, public_dir / "reference.wav")

    answer_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, str]] = []
    for index, source in enumerate(candidates, start=1):
        candidate_id = f"L{index:02d}"
        destination = public_dir / f"{candidate_id}.wav"
        shutil.copy2(source, destination)
        answer_rows.append(
            {
                "candidate_id": candidate_id,
                "source_file": source.name,
                "sha256": _sha256(destination),
                "objective_metrics": metrics_by_file[source.name],
            }
        )
        score_rows.append(
            {
                "candidate_id": candidate_id,
                "same_shot_identity_1_5": "",
                "naturalness_1_5": "",
                "audible_useful_difference_1_5": "",
                "brightness_match_1_5": "",
                "extra_shots_1_5": "",
                "usefulness_1_5": "",
                "comments": "",
            }
        )

    with (public_dir / "scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    (public_dir / "INSTRUCTIONS.md").write_text(
        "# Слепая калибровка SAME latent neighbourhood\n\n"
        "1. Сначала один раз прослушайте `reference.wav` и зафиксируйте громкость.\n"
        "2. Прослушайте `L01.wav`–`L03.wav` в случайном порядке, каждый не более двух раз.\n"
        "3. Оценивайте полезное отличие в теле и хвосте, а не только разницу waveform.\n"
        "4. `extra_shots`: 1 — лишних выстрелов нет, 5 — явно слышны.\n"
        "5. Не открывайте private-папку до передачи всех трёх оценок.\n",
        encoding="utf-8",
    )
    (private_dir / "answer_key.json").write_text(
        json.dumps(
            {
                "protocol_id": "sa3_same_tangent_listening_calibration_v1",
                "source_dir": str(source_dir),
                "randomization_seed": arguments.randomization_seed,
                "reference_sha256": _sha256(public_dir / "reference.wav"),
                "candidates": answer_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Blind package prepared: {public_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--randomization-seed", type=int, default=230823)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
