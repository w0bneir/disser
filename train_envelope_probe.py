"""CPU-обучение signed ridge envelope probe по latent_diagnostics.npz."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from envelope_probe import WaveformEnvelopeProbe, envelope_training_loss, normalize_envelope


@dataclass(frozen=True)
class ProbeSample:
    name: str
    group: str
    latents: torch.Tensor
    waveform_envelope: torch.Tensor


def _resample_envelope(envelope: np.ndarray, length: int) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(envelope, dtype=np.float32)).reshape(1, 1, -1)
    return F.interpolate(values, size=length, mode="linear", align_corners=True).reshape(-1)


def load_probe_samples(paths: Iterable[Path]) -> list[ProbeSample]:
    """Загрузить baseline/guided как две записи одной split-группы."""
    samples: list[ProbeSample] = []
    for path in sorted({Path(item).resolve() for item in paths}):
        with np.load(path, allow_pickle=False) as archive:
            if archive["format_version"].item() != 1:
                raise ValueError(f"Неподдерживаемая версия diagnostics: {path}")
            for mode in ("baseline", "guided"):
                latents_array = archive[f"{mode}_active_latents"].astype(np.float32)
                waveform_array = archive[f"{mode}_waveform_envelope"].astype(np.float32)
                if latents_array.ndim != 2 or latents_array.shape[-1] < 2:
                    raise ValueError(f"Некорректные latents в {path}: {latents_array.shape}")
                if waveform_array.ndim != 1 or waveform_array.size < 2:
                    raise ValueError(f"Некорректная waveform-огибающая в {path}")
                if not np.isfinite(latents_array).all() or not np.isfinite(waveform_array).all():
                    raise FloatingPointError(f"NaN/Inf в {path}")
                samples.append(
                    ProbeSample(
                        name=f"{path.parent.parent.name}/{path.parent.name}/{mode}",
                        group=str(path),
                        latents=torch.from_numpy(latents_array),
                        waveform_envelope=_resample_envelope(
                            waveform_array, latents_array.shape[-1]
                        ),
                    )
                )
    if not samples:
        raise ValueError("Не найдено ни одного latent_diagnostics.npz")
    channel_counts = {sample.latents.shape[0] for sample in samples}
    if len(channel_counts) != 1:
        raise ValueError(f"В diagnostics разное число latent-каналов: {channel_counts}")
    return samples


def split_probe_samples(
    samples: list[ProbeSample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ProbeSample], list[ProbeSample]]:
    """Разделить только по парам, не допуская baseline/guided leakage."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction должен быть в диапазоне (0, 1)")
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        raise ValueError("Для train/validation нужны минимум две независимые пары")
    random.Random(seed).shuffle(groups)
    validation_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:validation_count])
    training = [sample for sample in samples if sample.group not in validation_groups]
    validation = [sample for sample in samples if sample.group in validation_groups]
    return training, validation


def _pearson(predicted: torch.Tensor, target: torch.Tensor) -> float:
    _, metrics = envelope_training_loss(predicted, target, correlation_weight=0.0)
    return float(metrics["pearson_correlation"].detach())


def evaluate_probe(
    probe: WaveformEnvelopeProbe,
    samples: list[ProbeSample],
) -> dict[str, object]:
    probe.eval()
    per_sample: list[dict[str, float | str]] = []
    with torch.no_grad():
        for sample in samples:
            latents = sample.latents.unsqueeze(0)
            target = sample.waveform_envelope.unsqueeze(0)
            predicted = probe(latents)
            rms = normalize_envelope(torch.sqrt(latents.square().mean(dim=1) + 1e-8))
            per_sample.append(
                {
                    "name": sample.name,
                    "probe_mse": float(F.mse_loss(predicted, target)),
                    "probe_pearson": _pearson(predicted, target),
                    "latent_rms_mse": float(F.mse_loss(rms, target)),
                    "latent_rms_pearson": _pearson(rms, target),
                }
            )
    numeric_keys = ("probe_mse", "probe_pearson", "latent_rms_mse", "latent_rms_pearson")
    report: dict[str, object] = {
        "sample_count": len(samples),
        **{
            key: float(np.mean([float(row[key]) for row in per_sample]))
            for key in numeric_keys
        },
        "per_sample": per_sample,
    }
    return report


def _fit_ridge(samples: list[ProbeSample], alpha: float) -> WaveformEnvelopeProbe:
    """Closed-form weighted ridge; каждая аудиозапись имеет одинаковый вес."""
    if not samples:
        raise ValueError("Ridge train-набор не может быть пустым")
    if alpha < 0:
        raise ValueError("Ridge alpha не может быть отрицательным")
    feature_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    sample_weight_chunks: list[np.ndarray] = []
    for sample in samples:
        features = sample.latents.T.double().numpy()
        target = sample.waveform_envelope.double().numpy()
        feature_chunks.append(features)
        target_chunks.append(target)
        sample_weight_chunks.append(np.full(target.shape, 1.0 / target.size, dtype=np.float64))

    features = np.concatenate(feature_chunks, axis=0)
    target = np.concatenate(target_chunks)
    sample_weights = np.concatenate(sample_weight_chunks)
    feature_mean = np.average(features, axis=0, weights=sample_weights)
    feature_variance = np.average(
        (features - feature_mean) ** 2,
        axis=0,
        weights=sample_weights,
    )
    feature_scale = np.sqrt(feature_variance)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    square_root_weights = np.sqrt(sample_weights).reshape(-1, 1)
    weighted_design = design * square_root_weights
    weighted_target = target * square_root_weights.reshape(-1)
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + alpha * regularizer,
        weighted_design.T @ weighted_target,
    )
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Ridge создал NaN/Inf")

    probe = WaveformEnvelopeProbe(features.shape[1], ridge_alpha=alpha)
    probe.set_ridge_state(
        feature_mean=torch.from_numpy(feature_mean),
        feature_scale=torch.from_numpy(feature_scale),
        channel_weights=torch.from_numpy(coefficients[1:]),
        bias=float(coefficients[0]),
    )
    return probe


def select_ridge_alpha(
    training: list[ProbeSample],
    alphas: list[float],
) -> tuple[float, list[dict[str, float]]]:
    """Выбрать alpha leave-one-pair-out только внутри train-раздела."""
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("Нужен непустой список неотрицательных ridge alpha")
    groups = sorted({sample.group for sample in training})
    if len(groups) < 2:
        raise ValueError("Для внутреннего cross-validation нужны минимум две train-пары")
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        fold_correlations: list[float] = []
        fold_mse: list[float] = []
        for held_out_group in groups:
            fold_training = [sample for sample in training if sample.group != held_out_group]
            fold_validation = [sample for sample in training if sample.group == held_out_group]
            report = evaluate_probe(_fit_ridge(fold_training, alpha), fold_validation)
            fold_correlations.append(float(report["probe_pearson"]))
            fold_mse.append(float(report["probe_mse"]))
        rows.append(
            {
                "alpha": float(alpha),
                "mean_pearson": float(np.mean(fold_correlations)),
                "mean_mse": float(np.mean(fold_mse)),
            }
        )
    # Pearson — основной критерий формы; MSE используется только при равенстве.
    best = max(rows, key=lambda row: (row["mean_pearson"], -row["mean_mse"]))
    return best["alpha"], rows


def train_probe(
    training: list[ProbeSample],
    validation: list[ProbeSample],
    *,
    ridge_alphas: list[float],
) -> tuple[WaveformEnvelopeProbe, dict[str, object]]:
    selected_alpha, cross_validation = select_ridge_alpha(training, ridge_alphas)
    probe = _fit_ridge(training, selected_alpha)
    report: dict[str, object] = {
        "selected_alpha": selected_alpha,
        "train_cross_validation": cross_validation,
        "training": evaluate_probe(probe, training),
        "validation": evaluate_probe(probe, validation),
    }
    return probe, report


def discover_diagnostics(roots: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(root.rglob("latent_diagnostics.npz"))
        else:
            raise FileNotFoundError(root)
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path, nargs="+", help="NPZ-файлы или каталоги")
    parser.add_argument("--output", type=Path, default=Path("models/envelope_probe.safetensors"))
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--validation-fraction", type=float, default=0.33)
    parser.add_argument("--minimum-groups", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    paths = discover_diagnostics(arguments.diagnostics)
    samples = load_probe_samples(paths)
    group_count = len({sample.group for sample in samples})
    if group_count < arguments.minimum_groups:
        raise ValueError(
            f"Нужно минимум {arguments.minimum_groups} независимых diagnostics-пар, "
            f"найдено {group_count}"
        )
    training, validation = split_probe_samples(
        samples,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    probe, report = train_probe(
        training,
        validation,
        ridge_alphas=arguments.ridge_alphas,
    )

    from safetensors.torch import save_file

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in probe.state_dict().items()},
        str(arguments.output),
    )
    metadata = {
        "format_version": 2,
        "probe": probe.config(),
        "training_arguments": {
            "ridge_alphas": arguments.ridge_alphas,
            "validation_fraction": arguments.validation_fraction,
            "seed": arguments.seed,
        },
        "diagnostic_files": [str(path) for path in paths],
        "group_count": group_count,
        **report,
    }
    metadata_path = arguments.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[+] Probe сохранён: {arguments.output}")
    print(f"[+] Отчёт сохранён: {metadata_path}")


if __name__ == "__main__":
    main()
