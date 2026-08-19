"""Сравнить Repeat, DSP, text-only и Reference SDEdit для одного SFX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyzer import extract_rms_envelope
from sfx_metrics import (
    METRIC_PROTOCOL_ID,
    compare_to_reference,
    load_mono_audio,
    pairwise_diversity,
    resample_vector,
)


METHODS = ("repeat", "dsp", "text_only", "reference_sde")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_paths(
    *,
    case_id: str,
    reference_path: Path,
    dsp_results_dir: Path,
    generation_results_dir: Path,
    seeds: list[int],
    allow_missing: bool,
) -> list[dict[str, str | int | Path]]:
    rows: list[dict[str, str | int | Path]] = []
    for seed in seeds:
        candidates = {
            "repeat": reference_path,
            "dsp": dsp_results_dir / case_id / f"seed_{seed}" / "dsp.wav",
            "text_only": generation_results_dir / case_id / f"seed_{seed}" / "baseline.wav",
            "reference_sde": generation_results_dir / case_id / f"seed_{seed}" / "guided.wav",
        }
        for method, path in candidates.items():
            if not path.is_file():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"Не найден {method}, seed={seed}: {path}")
            rows.append({"method": method, "seed": seed, "path": path})
    return rows


def _summary_rows(file_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    identity_columns = {
        "method",
        "seed",
        "path",
    }
    summaries: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in file_rows if row["method"] == method]
        if not selected:
            continue
        summary: dict[str, object] = {"method": method, "count": len(selected)}
        for column in selected[0]:
            if column in identity_columns:
                continue
            values = [float(row[column]) for row in selected]
            summary[f"median_{column}"] = median(values)
        summaries.append(summary)
    return summaries


def _diversity_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        summaries.append(
            {
                "method": method,
                "pair_count": len(selected),
                "median_waveform_pearson": median(
                    float(row["waveform_pearson"]) for row in selected
                ),
                "median_envelope_pearson": median(
                    float(row["envelope_pearson"]) for row in selected
                ),
                "median_log_spectral_distance_db": median(
                    float(row["log_spectral_distance_db"]) for row in selected
                ),
            }
        )
    return summaries


def _plot_envelopes(
    path: Path,
    reference: torch.Tensor,
    sample_rate: int,
    audio_by_method: dict[str, list[tuple[int, torch.Tensor]]],
) -> None:
    reference_envelope = extract_rms_envelope(reference)
    duration = reference.numel() / sample_rate
    time_axis = np.linspace(0, duration, reference_envelope.numel())
    colors = {
        "repeat": "#777777",
        "dsp": "#2f6f9f",
        "text_only": "#d28c22",
        "reference_sde": "#b5292e",
    }
    figure, axes = plt.subplots(len(METHODS), 1, figsize=(11, 10), sharex=True)
    for axis, method in zip(axes, METHODS):
        axis.plot(time_axis, reference_envelope.numpy(), color="black", linewidth=2, label="reference")
        for seed, audio in audio_by_method.get(method, []):
            envelope = extract_rms_envelope(audio)
            envelope = resample_vector(envelope, reference_envelope.numel())
            axis.plot(time_axis, envelope.numpy(), color=colors[method], alpha=0.65, label=f"seed {seed}")
        axis.set_ylabel(method)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Время, с")
    figure.suptitle("RMS-огибающие: reference и сравниваемые методы")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    *,
    case_id: str,
    reference_path: Path,
    dsp_results_dir: Path,
    generation_results_dir: Path,
    output_dir: Path,
    seeds: list[int],
    allow_missing: bool,
    analysis_sample_rate: int,
) -> None:
    reference, sample_rate = load_mono_audio(
        reference_path,
        target_sample_rate=analysis_sample_rate,
    )
    inputs = collect_paths(
        case_id=case_id,
        reference_path=reference_path,
        dsp_results_dir=dsp_results_dir,
        generation_results_dir=generation_results_dir,
        seeds=seeds,
        allow_missing=allow_missing,
    )
    file_rows: list[dict[str, object]] = []
    audio_by_method: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for item in inputs:
        audio, _ = load_mono_audio(
            item["path"],
            target_sample_rate=sample_rate,
        )
        method = str(item["method"])
        seed = int(item["seed"])
        audio_by_method.setdefault(method, []).append((seed, audio))
        file_rows.append(
            {
                "method": method,
                "seed": seed,
                "path": str(item["path"]),
                **compare_to_reference(reference, audio, sample_rate),
            }
        )

    diversity_rows: list[dict[str, object]] = []
    for method, items in audio_by_method.items():
        for row in pairwise_diversity(items, sample_rate):
            diversity_rows.append({"method": method, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _summary_rows(file_rows)
    diversity_summary = _diversity_summary(diversity_rows)
    _write_csv(output_dir / "file_metrics.csv", file_rows)
    _write_csv(output_dir / "method_summary.csv", summary_rows)
    _write_csv(output_dir / "pairwise_diversity.csv", diversity_rows)
    _write_csv(output_dir / "diversity_summary.csv", diversity_summary)
    _plot_envelopes(
        output_dir / "envelope_overlay.png",
        reference,
        sample_rate,
        audio_by_method,
    )
    (output_dir / "evaluation_context.json").write_text(
        json.dumps(
            {
                "metric_protocol_id": METRIC_PROTOCOL_ID,
                "case_id": case_id,
                "reference_path": str(reference_path),
                "sample_rate": sample_rate,
                "seeds": seeds,
                "dsp_results_dir": str(dsp_results_dir),
                "generation_results_dir": str(generation_results_dir),
                "allow_missing": allow_missing,
                "available_methods": sorted(audio_by_method),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Оценка сохранена: {output_dir.resolve()}")
    for row in summary_rows:
        print(
            f"    {row['method']}: n={row['count']}, "
            f"Envelope Pearson={float(row['median_envelope_pearson']):.4f}, "
            f"MSE={float(row['median_envelope_mse']):.4f}, "
            f"copy residual={float(row['median_copy_residual_db']):.1f} dB"
        )


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
    parser.add_argument("--analysis-sample-rate", type=int, default=44_100)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        case_id=arguments.case_id,
        reference_path=arguments.reference,
        dsp_results_dir=arguments.dsp_results_dir,
        generation_results_dir=arguments.generation_results_dir,
        output_dir=arguments.output_dir,
        seeds=arguments.seeds,
        allow_missing=arguments.allow_missing,
        analysis_sample_rate=arguments.analysis_sample_rate,
    )
