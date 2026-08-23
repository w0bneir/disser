"""Build one blind CPU candidate for spectral late-tail morphing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sfx_metrics import compare_to_reference
from spectral_tail_morph import SpectralTailMorphParameters, generate_spectral_tail_morph


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
    fieldnames = list(rows[0])
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

    parameters = SpectralTailMorphParameters(
        protected_energy_quantile=float(config["protected_energy_quantile"]),
        transition_ms=float(config["transition_ms"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        frequency_smoothing_bins=int(config["frequency_smoothing_bins"]),
        time_smoothing_frames=int(config["time_smoothing_frames"]),
        max_modulation_db=float(config["max_modulation_db"]),
        modulation_depth=float(config["modulation_depth"]),
        phase_mix=float(config.get("phase_mix", 0.0)),
    )
    audio, diagnostics = generate_spectral_tail_morph(
        reference,
        generated,
        reference_rate,
        parameters=parameters,
    )
    public_dir = output_dir / "public"
    private_dir = output_dir / "private_do_not_open_before_scoring"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = str(config.get("candidate_id", "S01"))
    reference_public = public_dir / "reference.wav"
    candidate_public = public_dir / f"{candidate_id}.wav"
    _write_pcm24(reference_public, reference, reference_rate)
    _write_pcm24(candidate_public, audio, reference_rate)

    reference_mono = torch.from_numpy(reference.mean(axis=1).copy())
    candidate_mono = torch.from_numpy(audio.mean(axis=1).copy())
    metrics = {
        "candidate_id": candidate_id,
        **parameters.to_dict(),
        **diagnostics,
        **compare_to_reference(reference_mono, candidate_mono, reference_rate),
    }
    _write_csv(private_dir / "objective_metrics.csv", [metrics])
    _write_csv(
        public_dir / "scores.csv",
        [
            {
                "listener_id": "",
                "candidate_id": candidate_id,
                "same_shot_identity_1_5": "",
                "naturalness_1_5": "",
                "audible_useful_difference_1_5": "",
                "brightness_match_1_5": "",
                "extra_shots_1_5": "",
                "usefulness_1_5": "",
                "comments": "",
            }
        ],
    )
    (public_dir / "INSTRUCTIONS.md").write_text(
        "# Слепая CPU-проверка spectral tail morph\n\n"
        "1. Настройте громкость по `reference.wav` и не меняйте её.\n"
        f"2. Прослушайте `{candidate_id}.wav` не более двух раз.\n"
        "3. Оцените идентичность события, естественность и полезное отличие хвоста.\n"
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
                "candidate_id": candidate_id,
                "reference_sha256": _sha256(reference_public),
                "candidate_sha256": _sha256(candidate_public),
                "parameters": parameters.to_dict(),
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Слепой spectral-tail пакет сохранён: {public_dir.resolve()}")
    print(
        f"    protected={float(diagnostics['protected_ms']):.1f} ms, "
        f"core error={float(diagnostics['core_max_abs_error']):.1e}, "
        f"tail residual={float(diagnostics['tail_residual_db']):.1f} dB, "
        f"Pearson={float(metrics['envelope_pearson']):.4f}, "
        f"HF delta={float(metrics['high_frequency_fraction_delta']):+.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/spectral_tail_morph_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config, arguments.output_dir)
