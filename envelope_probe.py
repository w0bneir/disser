"""Лёгкий differentiable surrogate waveform-огибающей для latent guidance."""

from __future__ import annotations

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
    """Предсказать слышимую RMS-огибающую из Stable Audio latents.

    Probe намеренно мал: он обучает положительные веса 64 latent-каналов и
    короткий временной фильтр. Это оставляет guidance-граф дешёвым и снижает
    риск переобучения на небольшом диагностическом наборе.
    """

    def __init__(self, latent_channels: int, temporal_kernel_size: int = 5) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels должен быть положительным")
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size должен быть положительным нечётным")
        self.latent_channels = latent_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.channel_logits = nn.Parameter(torch.zeros(latent_channels, dtype=torch.float32))
        self.temporal_logits = nn.Parameter(
            torch.zeros(temporal_kernel_size, dtype=torch.float32)
        )

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

        # FP32 нужен и при обучении, и в будущем guidance: ранние sigma могут
        # переполнять FP16, а сам probe занимает лишь несколько десятков чисел.
        energy = torch.log1p(latents[:, :, :active_length].float().square())
        channel_weights = torch.softmax(self.channel_logits, dim=0).reshape(1, -1, 1)
        combined = (energy * channel_weights).sum(dim=1, keepdim=True)

        temporal_kernel = torch.softmax(self.temporal_logits, dim=0).reshape(1, 1, -1)
        padding = self.temporal_kernel_size // 2
        combined = F.pad(combined, (padding, padding), mode="replicate")
        smoothed = F.conv1d(combined, temporal_kernel).squeeze(1)
        return normalize_envelope(smoothed)

    def config(self) -> dict[str, Any]:
        return {
            "architecture": "weighted_latent_energy_v1",
            "latent_channels": self.latent_channels,
            "temporal_kernel_size": self.temporal_kernel_size,
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
