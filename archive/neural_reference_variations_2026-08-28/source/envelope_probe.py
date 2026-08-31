"""Лёгкий differentiable surrogate waveform-огибающей для latent guidance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_envelope(values: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Нормировать последнюю ось в [0, 1], не создавая NaN на константе."""
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    span = maximum - minimum
    normalized = (values - minimum) / span.clamp_min(eps)
    return torch.where(span > eps, normalized, torch.zeros_like(normalized))


class WaveformEnvelopeProbe(nn.Module):
    """Знаковая ridge-проекция Stable Audio latents в waveform-огибающую.

    VAE декодирует разные latent-каналы со знаком, поэтому усреднение энергии
    каналов теряет информацию. Probe хранит только стандартизацию, 64 веса и
    bias. Все они фиксированы после CPU-обучения, но градиент по входным latents
    остаётся доступен для будущего guidance.
    """

    def __init__(self, latent_channels: int, *, ridge_alpha: float = 0.0) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels должен быть положительным")
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha не может быть отрицательным")
        self.latent_channels = latent_channels
        self.ridge_alpha = ridge_alpha
        initial_weights = torch.linspace(-0.5, 0.5, latent_channels, dtype=torch.float32)
        self.register_buffer("feature_mean", torch.zeros(latent_channels, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(latent_channels, dtype=torch.float32))
        self.register_buffer("channel_weights", initial_weights)
        self.register_buffer("bias", torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        latents: torch.Tensor,
        *,
        active_length: int | None = None,
    ) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("Probe ожидает latents формы [batch, channels, time]")
        if latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"Probe ожидает {self.latent_channels} каналов, получено {latents.shape[1]}"
            )
        if active_length is None:
            active_length = latents.shape[-1]
        if not 1 <= active_length <= latents.shape[-1]:
            raise ValueError("active_length выходит за временную ось latents")

        if torch.any(self.feature_scale <= 0):
            raise ValueError("feature_scale probe должен быть положительным")
        # FP32 нужен и при обучении, и в будущем guidance: ранние sigma могут
        # переполнять FP16, а сам probe занимает лишь несколько сотен байт.
        active = latents[:, :, :active_length].float()
        standardized = (active - self.feature_mean.reshape(1, -1, 1)) / self.feature_scale.reshape(
            1, -1, 1
        )
        projected = (
            standardized * self.channel_weights.reshape(1, -1, 1)
        ).sum(dim=1) + self.bias
        return normalize_envelope(projected)

    def set_ridge_state(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        channel_weights: torch.Tensor,
        bias: torch.Tensor | float,
    ) -> None:
        """Установить проверенное closed-form ridge-решение."""
        expected = (self.latent_channels,)
        values = (feature_mean, feature_scale, channel_weights)
        if any(tuple(value.shape) != expected for value in values):
            raise ValueError(f"Ridge-векторы должны иметь форму {expected}")
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("NaN/Inf в ridge-состоянии")
        if torch.any(feature_scale <= 0):
            raise ValueError("feature_scale должен быть положительным")
        bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
        if bias_tensor.numel() != 1 or not torch.isfinite(bias_tensor).all():
            raise ValueError("bias должен быть конечным скаляром")
        self.feature_mean.copy_(feature_mean.float())
        self.feature_scale.copy_(feature_scale.float())
        self.channel_weights.copy_(channel_weights.float())
        self.bias.copy_(bias_tensor.reshape(()))

    def config(self) -> dict[str, Any]:
        return {
            "architecture": "signed_latent_ridge_v2",
            "latent_channels": self.latent_channels,
            "ridge_alpha": self.ridge_alpha,
        }


def envelope_training_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    correlation_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MSE + корреляционный штраф для нормированных batch-огибающих."""
    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape != target.shape:
        raise ValueError("Predicted и target должны иметь одинаковую форму [batch, time]")
    if predicted.shape[-1] < 2:
        raise ValueError("Для корреляции нужны минимум две временные точки")
    if correlation_weight < 0:
        raise ValueError("correlation_weight не может быть отрицательным")

    mse = F.mse_loss(predicted, target)
    predicted_centered = predicted - predicted.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = (predicted_centered * target_centered).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(predicted_centered, dim=-1)
        * torch.linalg.vector_norm(target_centered, dim=-1)
    ).clamp_min(1e-8)
    correlation = numerator / denominator
    correlation_penalty = 1.0 - correlation.mean()
    loss = mse + correlation_weight * correlation_penalty
    return loss, {"mse": mse, "pearson_correlation": correlation.mean()}


def load_waveform_envelope_probe(
    weights_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> WaveformEnvelopeProbe:
    """Строго загрузить совместимую пару safetensors + JSON."""
    weights_path = Path(weights_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else weights_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2:
        raise ValueError(f"Неподдерживаемая версия probe: {metadata.get('format_version')}")
    config = metadata.get("probe", {})
    if config.get("architecture") != "signed_latent_ridge_v2":
        raise ValueError(f"Неподдерживаемая архитектура probe: {config.get('architecture')}")
    latent_channels = int(config["latent_channels"])
    ridge_alpha = float(config["ridge_alpha"])
    probe = WaveformEnvelopeProbe(latent_channels, ridge_alpha=ridge_alpha)

    from safetensors.torch import load_file

    state = load_file(str(weights_path), device=str(device))
    probe.to(device=device)
    probe.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in probe.state_dict().values()):
        raise FloatingPointError("NaN/Inf в сохранённом probe")
    probe.eval()
    probe.requires_grad_(False)
    return probe
