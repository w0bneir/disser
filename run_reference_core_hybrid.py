"""Build a blind CPU calibration package for reference-core hybrid audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from reference_core_hybrid import HybridParameters, generate_reference_core_hybrid
from sfx_metrics import compare_to_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_pcm24(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if float(np.max(np.abs(audio))) > 1.000001:
        raise ValueError(f"Небезопасный peak перед записью: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reference_path = Path(config["reference_path"])
    generated_path = Path(config["generated_path"])
    reference, reference_rate = sf.read(reference_path, dtype="float32", always_2d=True)
    generated, generated_rate = sf.read(generated_path, dtype="float32", always_2d=True)
    if reference_rate != generated_rate:
        raise ValueError("Reference и generated должны иметь одинаковый sample rate")

    public_dir = output_dir / "public"
    private_dir = output_dir / "private_do_not_open_before_scoring"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    reference_public_path = public_dir / "reference.wav"
    _write_pcm24(reference_public_path, reference, reference_rate)

    candidates: list[dict[str, object]] = []
    for residual_mix in config["residual_mixes"]:
        parameters = HybridParameters(
            core_ms=float(config["core_ms"]),
            transition_ms=float(config["transition_ms"]),
            envelope_window_ms=float(config["envelope_window_ms"]),
            local_gain_min=float(config["local_gain_min"]),
            local_gain_max=float(config["local_gain_max"]),
            residual_peak_multiple=float(config["residual_peak_multiple"]),
            residual_mix=float(residual_mix),
        )
        audio, diagnostics = generate_reference_core_hybrid(
            reference,
            generated,
            reference_rate,
            parameters=parameters,
        )
        candidates.append(
            {"parameters": parameters.to_dict(), "diagnostics": diagnostics, "audio": audio}
        )

    rng = np.random.default_rng(int(config["randomization_seed"]))
    order = list(rng.permutation(len(candidates)))
    answer_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    reference_mono = torch.from_numpy(reference.mean(axis=1).copy())
    candidate_prefix = str(config.get("candidate_prefix", "H"))
    if not candidate_prefix.isalpha() or len(candidate_prefix) > 3:
        raise ValueError("candidate_prefix должен содержать от одной до трёх букв")
    for public_index, source_index in enumerate(order, start=1):
        candidate_id = f"{candidate_prefix.upper()}{public_index:02d}"
        item = candidates[source_index]
        candidate_path = public_dir / f"{candidate_id}.wav"
        audio = np.asarray(item["audio"], dtype=np.float32)
        _write_pcm24(candidate_path, audio, reference_rate)
        parameters = dict(item["parameters"])
        diagnostics = dict(item["diagnostics"])
        answer_rows.append(
            {
                "candidate_id": candidate_id,
                "residual_mix": parameters["residual_mix"],
                "sha256": _sha256(candidate_path),
            }
        )
        candidate_mono = torch.from_numpy(audio.mean(axis=1).copy())
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                **parameters,
                **diagnostics,
                **compare_to_reference(reference_mono, candidate_mono, reference_rate),
            }
        )

    _write_csv(private_dir / "objective_metrics.csv", metric_rows)
    _write_csv(
        public_dir / "scores.csv",
        [
            {
                "listener_id": "",
                "candidate_id": row["candidate_id"],
                "same_shot_identity_1_5": "",
                "naturalness_1_5": "",
                "audible_useful_difference_1_5": "",
                "brightness_match_1_5": "",
                "extra_shots_1_5": "",
                "usefulness_1_5": "",
                "comments": "",
            }
            for row in answer_rows
        ],
    )
    candidate_list = " и ".join(str(row["candidate_id"]) for row in answer_rows)
    (public_dir / "INSTRUCTIONS.md").write_text(
        "# Слепая CPU-калибровка reference-core hybrid\n\n"
        "1. Настройте громкость по `reference.wav` и не меняйте её.\n"
        f"2. Прослушайте {candidate_list}; не более двух воспроизведений каждого файла.\n"
        "3. Оцените, остался ли это тот же выстрел, звучит ли файл естественно "
        "и слышно ли полезное отличие от reference.\n"
        "4. Для `extra_shots_1_5`: 1 — лишних выстрелов нет, 5 — явно слышны.\n"
        "5. Не открывайте private-папку до заполнения `scores.csv`.\n",
        encoding="utf-8",
    )
    (private_dir / "answer_key.json").write_text(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "config_path": str(config_path),
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "reference_sha256": _sha256(reference_public_path),
                "candidates": answer_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Слепой CPU-пакет сохранён: {public_dir.resolve()}")
    for row in metric_rows:
        print(
            f"    {row['candidate_id']}: core error={float(row['core_max_abs_error']):.1e}, "
            f"Pearson={float(row['envelope_pearson']):.4f}, "
            f"HF delta={float(row['high_frequency_fraction_delta']):+.4f}, "
            f"copy residual={float(row['copy_residual_db']):.1f} dB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reference_core_hybrid.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config, arguments.output_dir)
