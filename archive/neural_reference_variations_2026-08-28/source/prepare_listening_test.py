"""Собрать воспроизводимый слепой listening-пакет для SFX-вариаций."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from audio_io import peak_normalize, save_wav


TARGET_SAMPLE_RATE = 44_100
LISTENING_PROTOCOL_ID = "pilot_blind_listening_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def load_standard_audio(path: Path) -> np.ndarray:
    """Загрузить WAV как peak-matched stereo 44.1 kHz ``[frames, 2]``."""
    samples, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Некорректный аудиофайл: {path}")
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    elif samples.shape[1] != 2:
        raise ValueError(f"Ожидался mono/stereo WAV: {path}")
    if sample_rate != TARGET_SAMPLE_RATE:
        samples = (
            AF.resample(
                torch.from_numpy(samples.T.copy()),
                int(sample_rate),
                TARGET_SAMPLE_RATE,
            )
            .T.contiguous()
            .numpy()
        )
    return peak_normalize(samples, target_peak=0.95)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _balanced_order(
    entries: list[tuple[int | None, np.ndarray]],
    repetitions: int,
    rng: random.Random,
) -> list[tuple[int | None, np.ndarray]]:
    if repetitions <= 0 or not entries:
        raise ValueError("Нужны положительные repetitions и непустой список")
    result: list[tuple[int | None, np.ndarray]] = []
    for _ in range(repetitions):
        cycle = list(entries)
        rng.shuffle(cycle)
        if len(cycle) > 1 and result and cycle[0][0] == result[-1][0]:
            cycle = cycle[1:] + cycle[:1]
        result.extend(cycle)
    return result


def _loop_audio(
    entries: list[tuple[int | None, np.ndarray]],
    *,
    repetitions: int,
    silence_seconds: float,
    rng: random.Random,
) -> tuple[np.ndarray, list[int | None]]:
    if silence_seconds < 0:
        raise ValueError("silence_seconds не может быть отрицательным")
    order = _balanced_order(entries, repetitions, rng)
    silence = np.zeros(
        (round(TARGET_SAMPLE_RATE * silence_seconds), 2),
        dtype=np.float32,
    )
    segments: list[np.ndarray] = []
    for index, (_, audio) in enumerate(order):
        if index:
            segments.append(silence)
        segments.append(audio)
    return np.concatenate(segments, axis=0), [seed for seed, _ in order]


def _source_records(
    *,
    case_id: str,
    seeds: list[int],
    dsp_results_dir: Path,
    generation_results_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        sources = {
            "dsp": dsp_results_dir / case_id / f"seed_{seed}" / "dsp.wav",
            "reference_sde": (
                generation_results_dir / case_id / f"seed_{seed}" / "guided.wav"
            ),
        }
        for method, path in sources.items():
            if not path.is_file():
                raise FileNotFoundError(f"Не найден {method}, seed={seed}: {path}")
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "source_path": path,
                    "source_sha256": _sha256(path),
                    "audio": load_standard_audio(path),
                }
            )
    return records


def prepare_listening_package(
    *,
    case_id: str,
    reference_path: Path,
    dsp_results_dir: Path,
    generation_results_dir: Path,
    output_dir: Path,
    seeds: list[int],
    randomization_seed: int,
    loop_repetitions: int,
    silence_seconds: float,
) -> None:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds должны быть непустыми и уникальными")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Listening-каталог уже непустой: {output_dir}")
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    public_dir = output_dir / "public"
    private_dir = output_dir / "private_do_not_open_before_scoring"
    stimuli_dir = public_dir / "stimuli"
    packages_dir = public_dir / "packages"
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    packages_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(randomization_seed))
    reference_audio = load_standard_audio(reference_path)
    save_wav(public_dir / "reference.wav", reference_audio, TARGET_SAMPLE_RATE)
    records = _source_records(
        case_id=case_id,
        seeds=seeds,
        dsp_results_dir=dsp_results_dir,
        generation_results_dir=generation_results_dir,
    )

    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    individual_key_rows: list[dict[str, Any]] = []
    individual_score_rows: list[dict[str, Any]] = []
    for index, record in enumerate(shuffled_records, start=1):
        stimulus_id = f"S{index:02d}"
        stimulus_path = stimuli_dir / f"{stimulus_id}.wav"
        save_wav(stimulus_path, record["audio"], TARGET_SAMPLE_RATE)
        individual_key_rows.append(
            {
                "stimulus_id": stimulus_id,
                "method": record["method"],
                "source_seed": record["seed"],
                "source_path": str(record["source_path"]),
                "source_sha256": record["source_sha256"],
            }
        )
        individual_score_rows.append(
            {
                "listener_id": "",
                "stimulus_id": stimulus_id,
                "same_event_identity_1_5": "",
                "naturalness_1_5": "",
                "artifact_severity_1_5": "",
                "sound_design_usefulness_1_5": "",
                "comments": "",
            }
        )

    by_method = {
        method: [
            (int(record["seed"]), record["audio"])
            for record in records
            if record["method"] == method
        ]
        for method in ("dsp", "reference_sde")
    }
    by_method["repeat"] = [(seed, reference_audio) for seed in seeds]
    package_methods = list(by_method)
    rng.shuffle(package_methods)
    package_key_rows: list[dict[str, Any]] = []
    package_score_rows: list[dict[str, Any]] = []
    for index, method in enumerate(package_methods, start=1):
        package_id = f"P{index:02d}"
        loop, source_order = _loop_audio(
            by_method[method],
            repetitions=loop_repetitions,
            silence_seconds=silence_seconds,
            rng=rng,
        )
        save_wav(packages_dir / f"{package_id}_loop.wav", loop, TARGET_SAMPLE_RATE)
        package_key_rows.append(
            {
                "package_id": package_id,
                "method": method,
                "source_seed_order": "|".join(
                    str(seed) for seed in source_order
                ),
            }
        )
        package_score_rows.append(
            {
                "listener_id": "",
                "package_id": package_id,
                "same_event_consistency_1_5": "",
                "naturalness_1_5": "",
                "useful_diversity_1_5": "",
                "artifact_severity_1_5": "",
                "listening_fatigue_1_5": "",
                "sound_design_usefulness_1_5": "",
                "comments": "",
            }
        )

    _write_csv(
        public_dir / "individual_scores.csv",
        list(individual_score_rows[0]),
        individual_score_rows,
    )
    _write_csv(
        public_dir / "package_scores.csv",
        list(package_score_rows[0]),
        package_score_rows,
    )
    _write_csv(
        private_dir / "individual_answer_key.csv",
        list(individual_key_rows[0]),
        individual_key_rows,
    )
    _write_csv(
        private_dir / "package_answer_key.csv",
        list(package_key_rows[0]),
        package_key_rows,
    )
    instructions = """# Пилотное слепое прослушивание SFX

Не открывайте каталог `private_do_not_open_before_scoring` до заполнения обеих
таблиц. Не смотрите waveform, спектрограммы и графики во время теста.

1. Выберите комфортную громкость по `reference.wav` и больше её не меняйте.
2. Прослушайте reference два раза.
3. Прослушайте файлы `stimuli/S01.wav` ... по имени, максимум по два раза.
4. Заполните `individual_scores.csv`.
5. Прослушайте каждый файл из `packages/` целиком, максимум по два раза.
6. Заполните `package_scores.csv` и сохраните таблицы.

Шкалы identity, naturalness, usefulness и useful diversity:
`1` — очень низко, `5` — очень высоко.

Шкалы artifact severity и listening fatigue:
`1` — артефактов/утомления нет, `5` — очень сильные.

Same-event identity означает «это новый дубль того же конкретного события», а
не просто звук из той же широкой категории. Не пытайтесь угадать метод.
"""
    (public_dir / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    context = {
        "listening_protocol_id": LISTENING_PROTOCOL_ID,
        "case_id": case_id,
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "seeds": seeds,
        "randomization_seed": int(randomization_seed),
        "loop_repetitions": int(loop_repetitions),
        "silence_seconds": float(silence_seconds),
        "git_commit": _git_commit(),
    }
    (private_dir / "context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[+] Listening-пакет готов: {public_dir.resolve()}")
    print(f"[!] Ключ не открывать до оценки: {private_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--dsp-results-dir",
        type=Path,
        default=Path("results/dsp_baseline_v1"),
    )
    parser.add_argument("--generation-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--randomization-seed", type=int, default=20_260_822)
    parser.add_argument("--loop-repetitions", type=int, default=3)
    parser.add_argument("--silence-seconds", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_listening_package(
        case_id=arguments.case_id,
        reference_path=arguments.reference,
        dsp_results_dir=arguments.dsp_results_dir,
        generation_results_dir=arguments.generation_results_dir,
        output_dir=arguments.output_dir,
        seeds=arguments.seeds,
        randomization_seed=arguments.randomization_seed,
        loop_repetitions=arguments.loop_repetitions,
        silence_seconds=arguments.silence_seconds,
    )
