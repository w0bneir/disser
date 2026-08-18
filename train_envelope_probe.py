"""CPU-обучение waveform-aware envelope probe по latent_diagnostics.npz."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from envelope_probe import (
    WaveformEnvelopeProbe,
    envelope_training_loss,
    normalize_envelope,
)


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
    """Загрузить baseline/guided как две записи, сохранив pair как одну split-группу."""
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
                        name=f"{path.parent.name}/{mode}",
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
) -> dict[str, float]:
    probe.eval()
    probe_mse: list[float] = []
    probe_correlation: list[float] = []
    rms_mse: list[float] = []
    rms_correlation: list[float] = []
    with torch.no_grad():
        for sample in samples:
            latents = sample.latents.unsqueeze(0)
            target = sample.waveform_envelope.unsqueeze(0)
            predicted = probe(latents)
            rms = normalize_envelope(torch.sqrt(latents.square().mean(dim=1) + 1e-8))
            probe_mse.append(float(F.mse_loss(predicted, target)))
            probe_correlation.append(_pearson(predicted, target))
            rms_mse.append(float(F.mse_loss(rms, target)))
            rms_correlation.append(_pearson(rms, target))
    return {
        "sample_count": len(samples),
        "probe_mse": float(np.mean(probe_mse)),
        "probe_pearson": float(np.mean(probe_correlation)),
        "latent_rms_mse": float(np.mean(rms_mse)),
        "latent_rms_pearson": float(np.mean(rms_correlation)),
    }


def train_probe(
    training: list[ProbeSample],
    validation: list[ProbeSample],
    *,
    epochs: int,
    learning_rate: float,
    correlation_weight: float,
    temporal_kernel_size: int,
    seed: int,
) -> tuple[WaveformEnvelopeProbe, dict[str, object]]:
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs и learning_rate должны быть положительными")
    torch.manual_seed(seed)
    latent_channels = training[0].latents.shape[0]
    probe = WaveformEnvelopeProbe(latent_channels, temporal_kernel_size)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    best_state = deepcopy(probe.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        probe.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        for index in torch.randperm(len(training), generator=generator).tolist():
            sample = training[index]
            predicted = probe(sample.latents.unsqueeze(0))
            target = sample.waveform_envelope.unsqueeze(0)
            loss, _ = envelope_training_loss(
                predicted,
                target,
                correlation_weight=correlation_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probe.eval()
        validation_losses = []
        with torch.no_grad():
            for sample in validation:
                predicted = probe(sample.latents.unsqueeze(0))
                loss, _ = envelope_training_loss(
                    predicted,
                    sample.waveform_envelope.unsqueeze(0),
                    correlation_weight=correlation_weight,
                )
                validation_losses.append(float(loss))
        validation_loss = float(np.mean(validation_losses))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(probe.state_dict())

    probe.load_state_dict(best_state)
    report: dict[str, object] = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--correlation-weight", type=float, default=0.25)
    parser.add_argument("--temporal-kernel-size", type=int, default=5)
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
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        correlation_weight=arguments.correlation_weight,
        temporal_kernel_size=arguments.temporal_kernel_size,
        seed=arguments.seed,
    )

    from safetensors.torch import save_file

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in probe.state_dict().items()},
        str(arguments.output),
    )
    metadata = {
        "format_version": 1,
        "probe": probe.config(),
        "training_arguments": {
            "epochs": arguments.epochs,
            "learning_rate": arguments.learning_rate,
            "correlation_weight": arguments.correlation_weight,
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
