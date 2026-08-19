"""Reference-guided denoising и абляционные guidance-режимы Stable Audio."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from envelope_probe import WaveformEnvelopeProbe


@dataclass
class StableAudioGenerationResult:
    """Результат одного baseline или guided прогона."""

    audio: np.ndarray
    sample_rate: int
    latent_envelope: torch.Tensor
    guidance_envelope: torch.Tensor
    active_latents: torch.Tensor | None
    guidance_loss: float | None
    guidance_trace: list[dict[str, float | int]]
    initialization: dict[str, float | int | str]
    elapsed_seconds: float
    peak_vram_mb: float


def normalize_per_sample(values: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Нормировать последние измерение каждого объекта к диапазону [0, 1]."""
    if values.ndim < 1:
        raise ValueError("Для нормализации нужен хотя бы один размер")
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    span = maximum - minimum
    normalized = (values - minimum) / span.clamp_min(eps)
    return torch.where(span > eps, normalized, torch.zeros_like(normalized))


def latent_rms_envelope(latents: torch.Tensor, *, active_length: int | None = None) -> torch.Tensor:
    """RMS-огибающая Stable Audio латента [batch, channels, time]."""
    if latents.ndim != 3:
        raise ValueError(
            "Латент Stable Audio должен иметь форму [batch, channels, time], "
            f"получено {tuple(latents.shape)}"
        )
    if active_length is None:
        active_length = latents.shape[-1]
    if not 1 <= active_length <= latents.shape[-1]:
        raise ValueError(
            f"active_length должен быть в [1, {latents.shape[-1]}], получено {active_length}"
        )
    energy = torch.sqrt(latents[:, :, :active_length].square().mean(dim=1) + 1e-8)
    return normalize_per_sample(energy)


def predict_guidance_envelope(
    latents: torch.Tensor,
    *,
    active_length: int,
    envelope_probe: WaveformEnvelopeProbe | None,
) -> torch.Tensor:
    """Получить огибающую старым RMS или валидированным waveform-probe."""
    if envelope_probe is None:
        envelope = latent_rms_envelope(latents, active_length=active_length)
    else:
        envelope = envelope_probe(latents, active_length=active_length)
    expected_shape = (latents.shape[0], active_length)
    if tuple(envelope.shape) != expected_shape:
        raise ValueError(
            f"Guidance envelope должна иметь форму {expected_shape}, "
            f"получено {tuple(envelope.shape)}"
        )
    if not torch.isfinite(envelope).all():
        raise FloatingPointError("Guidance envelope содержит NaN/Inf")
    return envelope


def waveform_rms_envelope(
    waveform: torch.Tensor,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Differentiable RMS-огибающая waveform [batch, channels, samples]."""
    if waveform.ndim != 3:
        raise ValueError("Waveform должна иметь форму [batch, channels, samples]")
    if waveform.shape[-1] == 0:
        raise ValueError("Waveform не должна быть пустой")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length и hop_length должны быть положительными")
    mono = waveform.float().mean(dim=1)
    if mono.shape[-1] < frame_length:
        mono = F.pad(mono, (0, frame_length - mono.shape[-1]))
    frames = mono.unfold(-1, frame_length, hop_length)
    rms = torch.sqrt(frames.square().mean(dim=-1) + 1e-8)
    return normalize_per_sample(rms)


def resample_target_envelope(target: torch.Tensor, target_length: int) -> torch.Tensor:
    """Интерполировать одномерную RMS-огибающую до длины латента."""
    if target.ndim != 1:
        raise ValueError(f"E_target должна быть одномерной, получено {tuple(target.shape)}")
    if target.numel() == 0:
        raise ValueError("E_target не должна быть пустой")
    if target_length <= 0:
        raise ValueError("Длина целевой огибающей должна быть положительной")
    return F.interpolate(
        target.reshape(1, 1, -1), size=target_length, mode="linear", align_corners=True
    ).reshape(-1)


def envelope_metrics(target: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    """MSE и корреляция Пирсона для двух одномерных нормированных огибающих."""
    if target.ndim != 1 or generated.ndim != 1:
        raise ValueError("Метрики принимают две одномерные огибающие")
    if target.numel() != generated.numel() or target.numel() < 2:
        raise ValueError("Огибающие должны иметь одинаковую длину не меньше двух")
    target = target.float()
    generated = generated.float()
    mse = F.mse_loss(generated, target).item()
    target_centered = target - target.mean()
    generated_centered = generated - generated.mean()
    denominator = torch.linalg.vector_norm(target_centered) * torch.linalg.vector_norm(generated_centered)
    correlation = 0.0 if denominator <= 1e-8 else (target_centered @ generated_centered / denominator).item()
    return {"mse": float(mse), "pearson_correlation": float(correlation)}


def envelope_shape_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    correlation_weight: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Дифференцируемая MSE + Pearson-цель для batch waveform-огибающих."""
    if generated.ndim != 2:
        raise ValueError("Generated envelope должна иметь форму [batch, time]")
    if target.ndim != 1 or target.numel() != generated.shape[-1]:
        raise ValueError("Target envelope должна совпадать с временной длиной generated")
    if correlation_weight < 0:
        raise ValueError("correlation_weight не может быть отрицательным")
    if eps <= 0:
        raise ValueError("eps должен быть положительным")

    target_batch = target.unsqueeze(0).expand_as(generated)
    mse = F.mse_loss(generated, target_batch)
    generated_centered = generated - generated.mean(dim=-1, keepdim=True)
    target_centered = target_batch - target_batch.mean(dim=-1, keepdim=True)
    numerator = (generated_centered * target_centered).sum(dim=-1)
    denominator = torch.linalg.vector_norm(generated_centered, dim=-1) * torch.linalg.vector_norm(
        target_centered, dim=-1
    )
    pearson = numerator / denominator.clamp_min(eps)
    mean_pearson = pearson.mean()
    loss = mse + correlation_weight * (1.0 - mean_pearson)
    return loss, mse, mean_pearson


def active_latent_length(pipe: Any, duration_seconds: float) -> int:
    """Число latent-позиций, соответствующее обрезаемому итоговому WAV."""
    if duration_seconds <= 0:
        raise ValueError("Длительность должна быть положительной")
    sample_rate = int(pipe.vae.config.sampling_rate)
    hop_length = int(pipe.vae.hop_length)
    maximum = int(pipe.transformer.config.sample_size)
    return max(1, min(maximum, ceil(duration_seconds * sample_rate / hop_length)))


def reference_sde_start_index(num_inference_steps: int, noise_strength: float) -> int:
    """Начальный индекс хвоста расписания для reference SDEdit.

    ``noise_strength=1`` использует полное расписание, а малые значения оставляют
    только поздние шаги и сильнее сохраняют VAE-latent референса. ``ceil`` нужен,
    чтобы даже 4-шаговый smoke-test проходил минимум два шага при strength=0.3.
    """
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps должен быть положительным")
    if not 0 < noise_strength <= 1:
        raise ValueError("reference noise strength должен быть в диапазоне (0, 1]")
    effective_steps = min(
        num_inference_steps,
        max(1, ceil(num_inference_steps * noise_strength)),
    )
    return num_inference_steps - effective_steps


def _x0_from_v_prediction(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    sigma_data: float,
    prediction_type: str,
) -> torch.Tensor:
    """Точная preconditioning-формула CosineDPMSolver из Diffusers 0.39."""
    sigma = torch.as_tensor(sigma, device=latents.device, dtype=latents.dtype)
    c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
    if prediction_type == "epsilon":
        c_out = sigma * sigma_data / torch.sqrt(sigma.square() + sigma_data**2)
    elif prediction_type == "v_prediction":
        c_out = -sigma * sigma_data / torch.sqrt(sigma.square() + sigma_data**2)
    else:
        raise ValueError(f"Неподдерживаемый prediction_type: {prediction_type}")
    return c_skip * latents + c_out * model_output


def _clip_gradient_norm(gradient: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    if maximum_norm <= 0:
        raise ValueError("maximum_norm должен быть положительным")
    norms = torch.linalg.vector_norm(gradient.flatten(start_dim=1), dim=1, keepdim=True)
    scale = torch.clamp(maximum_norm / norms.clamp_min(1e-8), max=1.0)
    return gradient * scale.reshape(-1, *([1] * (gradient.ndim - 1)))


def guide_latents(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    *,
    sigma: torch.Tensor | float,
    target_envelope: torch.Tensor,
    active_length: int,
    gamma: float,
    gradient_clip_norm: float,
    max_relative_step: float,
    envelope_probe: WaveformEnvelopeProbe | None = None,
    reference_active_length: int | None = None,
    sigma_data: float = 1.0,
    prediction_type: str = "v_prediction",
) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float]]:
    """Один малый шаг Direct Latent Guidance.

    Transformer намеренно не входит в граф autograd: его предсказание считается
    константой. Градиент проходит только через текущий latent и формулу z-hat_0,
    поэтому VAE и весь DiT не держат активации в VRAM.
    """
    if gamma < 0:
        raise ValueError("gamma не может быть отрицательной")
    if not 0 < max_relative_step <= 1:
        raise ValueError("max_relative_step должен быть в диапазоне (0, 1]")
    if latents.shape != model_output.shape:
        raise ValueError("Латент и выход transformer должны иметь одинаковую форму")
    if reference_active_length is None:
        reference_active_length = active_length
    if reference_active_length <= 0:
        raise ValueError("reference_active_length должен быть положительным")
    duration_scale = max(1.0, active_length / reference_active_length)
    if gamma == 0:
        envelope = predict_guidance_envelope(
            latents,
            active_length=active_length,
            envelope_probe=envelope_probe,
        )
        return latents, envelope.detach(), 0.0, {
            "gradient_norm": 0.0,
            "correction_norm": 0.0,
            "active_latent_norm": float(torch.linalg.vector_norm(latents[:, :, :active_length].float()).cpu()),
            "relative_correction": 0.0,
            "duration_scale": duration_scale,
            "loss_after": 0.0,
        }

    # Stable Audio работает в FP16, но при ранних шагах sigma достигает 500.
    # В FP16 sigma**2 переполняется (max ≈ 65504) и даёт NaN в формуле x0.
    # Guidance-граф очень мал, поэтому безопасно и дёшево считать его в FP32.
    working_latents = latents.detach().float().requires_grad_(True)
    working_output = model_output.detach().float()
    target = resample_target_envelope(target_envelope.to(latents.device, torch.float32), active_length)
    predicted_x0 = _x0_from_v_prediction(
        working_latents,
        working_output,
        sigma,
        sigma_data=sigma_data,
        prediction_type=prediction_type,
    )
    envelope = predict_guidance_envelope(
        predicted_x0,
        active_length=active_length,
        envelope_probe=envelope_probe,
    )
    loss = F.mse_loss(envelope, target.unsqueeze(0).expand_as(envelope))
    gradient = torch.autograd.grad(loss, working_latents, only_inputs=True)[0]
    raw_gradient_norm = torch.linalg.vector_norm(gradient[:, :, :active_length])
    gradient = _clip_gradient_norm(gradient, gradient_clip_norm)

    # При reduction="mean" относительная L2-сила градиента убывает примерно
    # обратно пропорционально числу временных позиций. Компенсируем это, чтобы
    # один gamma имел сопоставимый смысл для коротких и длинных SFX. Масштаб не
    # ослабляет короткие записи и по-прежнему ограничен max_relative_step.
    correction = -gamma * duration_scale * gradient
    correction_norm = torch.linalg.vector_norm(correction[:, :, :active_length])
    active_latent_norm = torch.linalg.vector_norm(working_latents[:, :, :active_length]).clamp_min(1e-8)
    maximum_correction_norm = max_relative_step * active_latent_norm
    correction_scale = torch.clamp(maximum_correction_norm / correction_norm.clamp_min(1e-8), max=1.0)
    correction = correction * correction_scale
    correction_norm = torch.linalg.vector_norm(correction[:, :, :active_length])
    corrected_fp32 = (working_latents + correction).detach()
    corrected = corrected_fp32.to(dtype=latents.dtype)
    if not torch.isfinite(corrected).all():
        raise FloatingPointError("Guidance создал NaN/Inf в латенте; запуск безопасно остановлен")

    with torch.no_grad():
        corrected_x0 = _x0_from_v_prediction(
            corrected_fp32,
            working_output,
            sigma,
            sigma_data=sigma_data,
            prediction_type=prediction_type,
        )
        corrected_envelope = predict_guidance_envelope(
            corrected_x0,
            active_length=active_length,
            envelope_probe=envelope_probe,
        )
        loss_after = F.mse_loss(corrected_envelope, target.unsqueeze(0).expand_as(corrected_envelope))
    diagnostics = {
        "gradient_norm": float(raw_gradient_norm.detach().cpu()),
        "correction_norm": float(correction_norm.detach().cpu()),
        "active_latent_norm": float(active_latent_norm.detach().cpu()),
        "relative_correction": float((correction_norm / active_latent_norm).detach().cpu()),
        "duration_scale": duration_scale,
        "loss_after": float(loss_after.detach().cpu()),
    }
    return corrected, envelope.detach(), float(loss.detach().cpu()), diagnostics


def guide_final_latents(
    latents: torch.Tensor,
    *,
    target_envelope: torch.Tensor,
    active_length: int,
    envelope_probe: WaveformEnvelopeProbe,
    gamma: float,
    gradient_clip_norm: float,
    max_relative_step: float,
    steps: int,
    reference_active_length: int | None = None,
) -> tuple[torch.Tensor, float, list[dict[str, float | int]]]:
    """Оптимизировать только final latent внутри суммарного per-frame trust region."""
    if gamma <= 0:
        raise ValueError("Final probe guidance требует gamma > 0")
    if steps <= 0:
        raise ValueError("Число final guidance шагов должно быть положительным")
    if not 0 < max_relative_step <= 1:
        raise ValueError("max_relative_step должен быть в диапазоне (0, 1]")
    if reference_active_length is None:
        reference_active_length = active_length
    if reference_active_length <= 0:
        raise ValueError("reference_active_length должен быть положительным")

    anchor = latents.detach().float()
    working = anchor.clone()
    target = resample_target_envelope(
        target_envelope.to(latents.device, torch.float32), active_length
    )
    duration_scale = max(1.0, active_length / reference_active_length)
    anchor_active = anchor[:, :, :active_length]
    active_latent_norm = torch.linalg.vector_norm(anchor_active).clamp_min(1e-8)
    anchor_frame_norm = torch.linalg.vector_norm(anchor_active, dim=1)
    trace: list[dict[str, float | int]] = []
    last_loss = float("nan")

    for index in range(steps):
        candidate_input = working.detach().requires_grad_(True)
        envelope = predict_guidance_envelope(
            candidate_input,
            active_length=active_length,
            envelope_probe=envelope_probe,
        )
        loss = F.mse_loss(envelope, target.unsqueeze(0).expand_as(envelope))
        gradient = torch.autograd.grad(loss, candidate_input, only_inputs=True)[0]
        raw_gradient_norm = torch.linalg.vector_norm(gradient[:, :, :active_length])
        gradient = _clip_gradient_norm(gradient, gradient_clip_norm)
        proposed = candidate_input - gamma * duration_scale * gradient

        proposed_delta = proposed[:, :, :active_length].detach() - anchor_active
        proposed_frame_norm = torch.linalg.vector_norm(proposed_delta, dim=1)
        maximum_frame_norm = max_relative_step * anchor_frame_norm
        frame_scale = torch.clamp(
            maximum_frame_norm / proposed_frame_norm.clamp_min(1e-8),
            max=1.0,
        )
        projected_delta = proposed_delta * frame_scale.unsqueeze(1)
        candidate = anchor.clone()
        candidate[:, :, :active_length] = anchor_active + projected_delta

        with torch.no_grad():
            corrected_envelope = predict_guidance_envelope(
                candidate,
                active_length=active_length,
                envelope_probe=envelope_probe,
            )
            loss_after = F.mse_loss(
                corrected_envelope,
                target.unsqueeze(0).expand_as(corrected_envelope),
            )
        if loss_after > loss:
            break
        working = candidate
        correction_norm = torch.linalg.vector_norm(projected_delta)
        frame_relative = torch.linalg.vector_norm(projected_delta, dim=1) / anchor_frame_norm.clamp_min(
            1e-8
        )
        last_loss = float(loss_after.detach().cpu())
        trace.append(
            {
                "step": index,
                "sigma": 0.0,
                "loss_before": float(loss.detach().cpu()),
                "gradient_norm": float(raw_gradient_norm.detach().cpu()),
                "correction_norm": float(correction_norm.detach().cpu()),
                "active_latent_norm": float(active_latent_norm.detach().cpu()),
                "relative_correction": float(
                    (correction_norm / active_latent_norm).detach().cpu()
                ),
                "max_frame_relative_correction": float(frame_relative.max().detach().cpu()),
                "duration_scale": duration_scale,
                "loss_after": last_loss,
            }
        )

    if not trace:
        raise RuntimeError("Final probe guidance не смог уменьшить loss внутри trust region")
    corrected = working.to(dtype=latents.dtype)
    if not torch.isfinite(corrected).all():
        raise FloatingPointError("Final probe guidance создал NaN/Inf")
    return corrected, last_loss, trace


def guide_final_latents_with_decoder(
    latents: torch.Tensor,
    *,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    target_envelope: torch.Tensor,
    active_length: int,
    waveform_length: int,
    gamma: float,
    gradient_clip_norm: float,
    max_relative_step: float,
    steps: int,
    decoder_context_frames: int = 64,
    reference_active_length: int | None = None,
) -> tuple[torch.Tensor, float, list[dict[str, float | int]]]:
    """Один или несколько точных final-latent шагов через короткий VAE decode."""
    if gamma <= 0:
        raise ValueError("Decoder guidance требует gamma > 0")
    if steps <= 0:
        raise ValueError("Число decoder guidance шагов должно быть положительным")
    if waveform_length <= 0:
        raise ValueError("waveform_length должен быть положительным")
    if decoder_context_frames < 0:
        raise ValueError("decoder_context_frames не может быть отрицательным")
    if not 1 <= active_length <= latents.shape[-1]:
        raise ValueError("Некорректный active_length")
    if not 0 < max_relative_step <= 1:
        raise ValueError("max_relative_step должен быть в диапазоне (0, 1]")
    if reference_active_length is None:
        reference_active_length = active_length
    if reference_active_length <= 0:
        raise ValueError("reference_active_length должен быть положительным")

    context_length = min(latents.shape[-1], active_length + decoder_context_frames)
    anchor = latents.detach().float()
    anchor_active = anchor[:, :, :active_length]
    context_tail = latents.detach()[:, :, active_length:context_length]
    working_active = anchor_active.clone()
    active_latent_norm = torch.linalg.vector_norm(anchor_active).clamp_min(1e-8)
    anchor_frame_norm = torch.linalg.vector_norm(anchor_active, dim=1)
    duration_scale = max(1.0, active_length / reference_active_length)
    trace: list[dict[str, float | int]] = []
    last_loss = float("nan")

    def decode_envelope(active: torch.Tensor) -> torch.Tensor:
        decoder_input = torch.cat(
            [active.to(dtype=latents.dtype), context_tail],
            dim=-1,
        )
        waveform = decode_fn(decoder_input)
        if waveform.ndim != 3 or waveform.shape[-1] < waveform_length:
            raise ValueError("VAE decoder вернул слишком короткую waveform")
        return waveform_rms_envelope(waveform[:, :, :waveform_length])

    for index in range(steps):
        candidate_input = working_active.detach().to(dtype=latents.dtype).requires_grad_(True)
        envelope = decode_envelope(candidate_input)
        target = resample_target_envelope(
            target_envelope.to(latents.device, torch.float32), envelope.shape[-1]
        )
        loss = F.mse_loss(envelope, target.unsqueeze(0).expand_as(envelope))
        gradient = torch.autograd.grad(loss, candidate_input, only_inputs=True)[0].float()
        raw_gradient_norm = torch.linalg.vector_norm(gradient)
        gradient = _clip_gradient_norm(gradient, gradient_clip_norm)
        proposed_active = working_active - gamma * duration_scale * gradient

        proposed_delta = proposed_active.detach() - anchor_active
        proposed_frame_norm = torch.linalg.vector_norm(proposed_delta, dim=1)
        maximum_frame_norm = max_relative_step * anchor_frame_norm
        frame_scale = torch.clamp(
            maximum_frame_norm / proposed_frame_norm.clamp_min(1e-8),
            max=1.0,
        )
        projected_delta = proposed_delta * frame_scale.unsqueeze(1)
        candidate_active = anchor_active + projected_delta
        with torch.no_grad():
            corrected_envelope = decode_envelope(candidate_active)
            loss_after = F.mse_loss(
                corrected_envelope,
                target.unsqueeze(0).expand_as(corrected_envelope),
            )
        if loss_after > loss:
            break
        working_active = candidate_active
        correction_norm = torch.linalg.vector_norm(projected_delta)
        frame_relative = torch.linalg.vector_norm(projected_delta, dim=1) / anchor_frame_norm.clamp_min(
            1e-8
        )
        last_loss = float(loss_after.detach().cpu())
        trace.append(
            {
                "step": index,
                "sigma": 0.0,
                "loss_before": float(loss.detach().cpu()),
                "gradient_norm": float(raw_gradient_norm.detach().cpu()),
                "correction_norm": float(correction_norm.detach().cpu()),
                "active_latent_norm": float(active_latent_norm.detach().cpu()),
                "relative_correction": float(
                    (correction_norm / active_latent_norm).detach().cpu()
                ),
                "max_frame_relative_correction": float(frame_relative.max().detach().cpu()),
                "decoder_context_frames": context_length - active_length,
                "duration_scale": duration_scale,
                "loss_after": last_loss,
            }
        )

    if not trace:
        raise RuntimeError("Decoder guidance не смог уменьшить waveform loss внутри trust region")
    corrected = anchor.clone()
    corrected[:, :, :active_length] = working_active
    corrected = corrected.to(dtype=latents.dtype)
    if not torch.isfinite(corrected).all():
        raise FloatingPointError("Decoder guidance создал NaN/Inf")
    return corrected, last_loss, trace


def select_decoder_guidance_indices(
    total_steps: int,
    *,
    start_fraction: float,
    guidance_steps: int,
) -> list[int]:
    """Равномерно выбрать поздние denoising-шаги, обязательно включая последний."""
    if total_steps <= 0:
        raise ValueError("Общее число denoising-шагов должно быть положительным")
    if not 0.5 <= start_fraction < 1:
        raise ValueError("Decoder denoising start fraction должна быть в диапазоне [0.5, 1)")
    if guidance_steps <= 0:
        raise ValueError("Число decoder denoising шагов должно быть положительным")

    start_index = min(total_steps - 1, int(total_steps * start_fraction))
    available_steps = total_steps - start_index
    selected_count = min(guidance_steps, available_steps)
    if selected_count == 1:
        return [total_steps - 1]
    if selected_count == available_steps:
        return list(range(start_index, total_steps))
    span = total_steps - 1 - start_index
    return [
        round(start_index + position * span / (selected_count - 1))
        for position in range(selected_count)
    ]


def guide_denoising_latents_with_decoder(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    *,
    sigma: torch.Tensor | float,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    target_envelope: torch.Tensor,
    active_length: int,
    waveform_length: int,
    gamma: float,
    gradient_clip_norm: float,
    max_relative_step: float,
    decoder_context_frames: int = 64,
    reference_active_length: int | None = None,
    sigma_data: float = 1.0,
    prediction_type: str = "v_prediction",
    max_backtracking_steps: int = 3,
    correlation_weight: float = 0.1,
) -> tuple[torch.Tensor, float, dict[str, float | int]]:
    """Скорректировать noisy latent по точной waveform-loss его x0-прогноза.

    Transformer остаётся вне autograd. Градиент проходит от короткого VAE
    decode через аналитическую формулу predicted x0 только к текущему latent.
    Коррекция ограничивается отдельно для каждой активной временной позиции.
    """
    if gamma <= 0:
        raise ValueError("Decoder denoising guidance требует gamma > 0")
    if latents.shape != model_output.shape:
        raise ValueError("Латент и выход transformer должны иметь одинаковую форму")
    if waveform_length <= 0:
        raise ValueError("waveform_length должен быть положительным")
    if decoder_context_frames < 0:
        raise ValueError("decoder_context_frames не может быть отрицательным")
    if not 1 <= active_length <= latents.shape[-1]:
        raise ValueError("Некорректный active_length")
    if not 0 < max_relative_step <= 1:
        raise ValueError("max_relative_step должен быть в диапазоне (0, 1]")
    if max_backtracking_steps < 0:
        raise ValueError("max_backtracking_steps не может быть отрицательным")
    if correlation_weight < 0:
        raise ValueError("correlation_weight не может быть отрицательным")
    if reference_active_length is None:
        reference_active_length = active_length
    if reference_active_length <= 0:
        raise ValueError("reference_active_length должен быть положительным")

    context_length = min(latents.shape[-1], active_length + decoder_context_frames)
    working = latents.detach().float().requires_grad_(True)
    fixed_output = model_output.detach().float()
    duration_scale = max(1.0, active_length / reference_active_length)

    def decode_envelope(values: torch.Tensor) -> torch.Tensor:
        predicted_x0 = _x0_from_v_prediction(
            values,
            fixed_output,
            sigma,
            sigma_data=sigma_data,
            prediction_type=prediction_type,
        )
        waveform = decode_fn(predicted_x0[:, :, :context_length].to(dtype=latents.dtype))
        if waveform.ndim != 3 or waveform.shape[-1] < waveform_length:
            raise ValueError("VAE decoder вернул слишком короткую waveform")
        return waveform_rms_envelope(waveform[:, :, :waveform_length])

    envelope = decode_envelope(working)
    if not torch.isfinite(envelope).all():
        raise FloatingPointError("Decoder denoising envelope содержит NaN/Inf")
    target = resample_target_envelope(
        target_envelope.to(latents.device, torch.float32), envelope.shape[-1]
    )
    loss, mse_before, pearson_before = envelope_shape_loss(
        envelope,
        target,
        correlation_weight=correlation_weight,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Decoder denoising loss содержит NaN/Inf")
    gradient = torch.autograd.grad(loss, working, only_inputs=True)[0].float()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Decoder denoising gradient содержит NaN/Inf")
    active_gradient = gradient[:, :, :active_length]
    raw_gradient_norm = torch.linalg.vector_norm(active_gradient)
    active_gradient = _clip_gradient_norm(active_gradient, gradient_clip_norm)

    anchor = latents.detach().float()
    anchor_active = anchor[:, :, :active_length]
    anchor_frame_norm = torch.linalg.vector_norm(anchor_active, dim=1)
    proposed_delta = -gamma * duration_scale * active_gradient
    proposed_frame_norm = torch.linalg.vector_norm(proposed_delta, dim=1)
    maximum_frame_norm = max_relative_step * anchor_frame_norm
    frame_scale = torch.clamp(
        maximum_frame_norm / proposed_frame_norm.clamp_min(1e-8),
        max=1.0,
    )
    projected_delta = proposed_delta * frame_scale.unsqueeze(1)

    accepted = False
    accepted_scale = 0.0
    loss_after = loss.detach()
    corrected_fp32 = anchor
    for backtracking_index in range(max_backtracking_steps + 1):
        candidate_scale = 0.5**backtracking_index
        candidate = anchor.clone()
        candidate[:, :, :active_length] = anchor_active + candidate_scale * projected_delta
        with torch.no_grad():
            candidate_envelope = decode_envelope(candidate)
            candidate_loss, candidate_mse, candidate_pearson = envelope_shape_loss(
                candidate_envelope,
                target,
                correlation_weight=correlation_weight,
            )
        if not torch.isfinite(candidate_loss):
            raise FloatingPointError("Decoder denoising candidate loss содержит NaN/Inf")
        improves_joint_loss = candidate_loss < loss.detach()
        preserves_mse = candidate_mse <= mse_before.detach() + 1e-8
        preserves_pearson = candidate_pearson >= pearson_before.detach() - 1e-6
        if improves_joint_loss and preserves_mse and preserves_pearson:
            corrected_fp32 = candidate
            loss_after = candidate_loss
            mse_after = candidate_mse
            pearson_after = candidate_pearson
            accepted = True
            accepted_scale = candidate_scale
            break

    accepted_delta = accepted_scale * projected_delta
    if not accepted:
        mse_after = mse_before.detach()
        pearson_after = pearson_before.detach()
    correction_norm = torch.linalg.vector_norm(accepted_delta)
    active_latent_norm = torch.linalg.vector_norm(anchor_active).clamp_min(1e-8)
    frame_relative = torch.linalg.vector_norm(accepted_delta, dim=1) / anchor_frame_norm.clamp_min(
        1e-8
    )
    corrected = corrected_fp32.to(dtype=latents.dtype)
    if not torch.isfinite(corrected).all():
        raise FloatingPointError("Decoder denoising guidance создал NaN/Inf")
    diagnostics: dict[str, float | int] = {
        "gradient_norm": float(raw_gradient_norm.detach().cpu()),
        "correction_norm": float(correction_norm.detach().cpu()),
        "active_latent_norm": float(active_latent_norm.detach().cpu()),
        "relative_correction": float((correction_norm / active_latent_norm).detach().cpu()),
        "max_frame_relative_correction": float(frame_relative.max().detach().cpu()),
        "decoder_context_frames": context_length - active_length,
        "duration_scale": duration_scale,
        "accepted": int(accepted),
        "backtracking_scale": accepted_scale,
        "correlation_weight": correlation_weight,
        "mse_before": float(mse_before.detach().cpu()),
        "mse_after": float(mse_after.detach().cpu()),
        "pearson_before": float(pearson_before.detach().cpu()),
        "pearson_after": float(pearson_after.detach().cpu()),
        "loss_after": float(loss_after.detach().cpu()),
    }
    return corrected, float(loss.detach().cpu()), diagnostics


def _prepare_conditions(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None,
    duration_seconds: float,
    guidance_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Повторить штатную подготовку text + duration conditioning StableAudioPipeline."""
    do_cfg = guidance_scale > 1.0
    prompt_embeds = pipe.encode_prompt(prompt, device, do_cfg, negative_prompt)
    start_states, end_states = pipe.encode_duration(
        0.0,
        duration_seconds,
        device,
        do_cfg and negative_prompt is not None,
        batch_size=1,
    )
    text_duration = torch.cat([prompt_embeds, start_states, end_states], dim=1)
    global_duration = torch.cat([start_states, end_states], dim=2)

    if do_cfg and negative_prompt is None:
        text_duration = torch.cat([torch.zeros_like(text_duration), text_duration], dim=0)
        global_duration = torch.cat([global_duration, global_duration], dim=0)
    return text_duration, global_duration, do_cfg


def predict_noise_sequential_cfg(
    pipe: Any,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    text_duration: torch.Tensor,
    global_duration: torch.Tensor,
    rotary_embedding: torch.Tensor,
    do_cfg: bool,
    guidance_scale: float,
) -> torch.Tensor:
    """Предсказать шум без удвоения batch при CFG.

    Официальный pipeline объединяет unconditional и conditional ветви в batch=2.
    Это быстрее, но на видеокартах с ограниченной VRAM заметно повышает пик
    памяти. Здесь ветви выполняются последовательно, по одной за раз.
    """
    model_input = pipe.scheduler.scale_model_input(latents, timestep)
    with torch.no_grad():
        if not do_cfg:
            return pipe.transformer(
                model_input,
                timestep.unsqueeze(0),
                encoder_hidden_states=text_duration,
                global_hidden_states=global_duration,
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]

        unconditional = pipe.transformer(
            model_input,
            timestep.unsqueeze(0),
            encoder_hidden_states=text_duration[:1],
            global_hidden_states=global_duration[:1],
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        conditional = pipe.transformer(
            model_input,
            timestep.unsqueeze(0),
            encoder_hidden_states=text_duration[1:2],
            global_hidden_states=global_duration[1:2],
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
    return unconditional + guidance_scale * (conditional - unconditional)


def prepare_initial_latents(pipe: Any, *, seed: int, device: torch.device) -> torch.Tensor:
    """Получить исходный шум штатной функцией pipeline, один раз на пару режимов."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe.prepare_latents(
        batch_size=1,
        num_channels_vae=int(pipe.transformer.config.in_channels),
        sample_size=int(pipe.transformer.config.sample_size),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )


def encode_reference_audio_latents(
    pipe: Any,
    waveform: torch.Tensor,
    *,
    duration_seconds: float,
    device: torch.device,
    context_frames: int = 64,
) -> torch.Tensor:
    """Детерминированно кодировать референс и дополнить latent до размера DiT.

    VAE получает только активный фрагмент плюс ограниченный контекст, а не
    максимальные 47 секунд Stable Audio. Это существенно снижает VRAM при
    сохранении контекста декодера у правой границы целевого WAV.
    """
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Reference waveform должна быть непустым mono-тензором")
    if not torch.isfinite(waveform).all():
        raise FloatingPointError("Reference waveform содержит NaN/Inf")
    if context_frames < 0:
        raise ValueError("context_frames не может быть отрицательным")

    maximum_frames = int(pipe.transformer.config.sample_size)
    latent_channels = int(pipe.transformer.config.in_channels)
    active_frames = active_latent_length(pipe, duration_seconds)
    encoded_frames = min(maximum_frames, active_frames + context_frames)
    hop_length = int(pipe.vae.hop_length)
    audio_channels = int(pipe.vae.config.audio_channels)
    encoded_samples = encoded_frames * hop_length
    vae_dtype = getattr(pipe.vae, "dtype", torch.float16)

    audio = torch.zeros(
        (1, audio_channels, encoded_samples),
        device=device,
        dtype=vae_dtype,
    )
    copy_length = min(int(waveform.numel()), encoded_samples)
    mono = waveform[:copy_length].to(device=device, dtype=vae_dtype)
    audio[:, :, :copy_length] = mono.reshape(1, 1, -1)
    pipe.vae.requires_grad_(False)
    with torch.no_grad():
        distribution = pipe.vae.encode(audio).latent_dist
        encoded = distribution.mode()
    if encoded.ndim != 3 or encoded.shape[0] != 1:
        raise RuntimeError(f"VAE вернул некорректную форму latent: {tuple(encoded.shape)}")
    if encoded.shape[1] != latent_channels:
        raise RuntimeError(
            f"VAE вернул {encoded.shape[1]} каналов, transformer ожидает {latent_channels}"
        )
    if not torch.isfinite(encoded).all():
        raise FloatingPointError("VAE reference latent содержит NaN/Inf")

    full_latents = torch.zeros(
        (1, latent_channels, maximum_frames),
        device=device,
        dtype=encoded.dtype,
    )
    copy_frames = min(maximum_frames, encoded.shape[-1])
    full_latents[:, :, :copy_frames] = encoded[:, :, :copy_frames]
    return full_latents.detach().to(device="cpu", dtype=torch.float16)


def prepare_reference_sde_latents(
    scheduler: Any,
    *,
    reference_latents: torch.Tensor,
    initial_noise_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_strength: float,
) -> tuple[torch.Tensor, int, float]:
    """Зашумить reference latent тем же шумом, что использует paired baseline."""
    if reference_latents.shape != initial_noise_latents.shape:
        raise ValueError("Reference и initial noise latents должны иметь одинаковую форму")
    if timesteps.ndim != 1 or timesteps.numel() == 0:
        raise ValueError("Scheduler timesteps должны быть непустым одномерным тензором")
    start_index = reference_sde_start_index(int(timesteps.numel()), noise_strength)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(start_index)
    init_sigma = float(scheduler.init_noise_sigma)
    if not np.isfinite(init_sigma) or init_sigma <= 0:
        raise ValueError("Scheduler init_noise_sigma должен быть положительным")
    raw_noise = initial_noise_latents / init_sigma
    timestep = timesteps[start_index].reshape(1)
    noisy = scheduler.add_noise(reference_latents, raw_noise, timestep)
    if not torch.isfinite(noisy).all():
        raise FloatingPointError("Reference SDEdit initialization содержит NaN/Inf")
    sigma = float(scheduler.sigmas[start_index])
    return noisy, start_index, sigma


def generate_sfx(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None,
    duration_seconds: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    initial_latents: torch.Tensor,
    reference_latents: torch.Tensor | None = None,
    reference_noise_strength: float | None = None,
    target_envelope: torch.Tensor | None = None,
    gamma: float = 0.0,
    gradient_clip_norm: float = 0.05,
    guidance_start_fraction: float = 0.5,
    max_relative_step: float = 0.03,
    guidance_reference_duration_seconds: float = 0.5,
    envelope_probe: WaveformEnvelopeProbe | None = None,
    guidance_mode: str = "denoising",
    final_guidance_steps: int = 10,
    decoder_guidance_start_fraction: float = 0.7,
    decoder_correlation_weight: float = 0.1,
    return_active_latents: bool = False,
) -> StableAudioGenerationResult:
    """Выполнить baseline (gamma=0) или Direct Latent Guidance.

    В обычных режимах VAE вызывается после denoising. Экспериментальный
    ``decoder_denoising`` дополнительно декодирует только выбранные x0-прогнозы.
    """
    if not 0 <= guidance_start_fraction < 1:
        raise ValueError("guidance_start_fraction должен быть в [0, 1)")
    if gamma > 0 and target_envelope is None:
        raise ValueError("Для guidance нужен target_envelope")
    if initial_latents.ndim != 3:
        raise ValueError("initial_latents должен иметь форму [batch, channels, time]")
    if (reference_latents is None) != (reference_noise_strength is None):
        raise ValueError(
            "reference_latents и reference_noise_strength должны задаваться вместе"
        )
    if reference_latents is not None and gamma > 0:
        raise ValueError(
            "Reference SDEdit сначала проверяется изолированно и не совмещается с gradient guidance"
        )
    if duration_seconds <= 0 or num_inference_steps <= 0:
        raise ValueError("Длительность и число шагов должны быть положительными")
    if guidance_reference_duration_seconds <= 0:
        raise ValueError("guidance_reference_duration_seconds должен быть положительным")
    if guidance_mode not in {"denoising", "final", "decoder", "decoder_denoising"}:
        raise ValueError(
            "guidance_mode должен быть 'denoising', 'final', 'decoder' "
            "или 'decoder_denoising'"
        )
    if gamma > 0 and guidance_mode == "final" and envelope_probe is None:
        raise ValueError("Final guidance требует waveform-aware envelope probe")
    if final_guidance_steps <= 0:
        raise ValueError("final_guidance_steps должен быть положительным")
    if not 0.5 <= decoder_guidance_start_fraction < 1:
        raise ValueError("decoder_guidance_start_fraction должен быть в [0.5, 1)")
    if not 0 <= decoder_correlation_weight <= 1:
        raise ValueError("decoder_correlation_weight должен быть в [0, 1]")

    device = torch.device("cuda")
    latents = initial_latents.detach().clone().to(device)
    if latents.shape[0] != 1:
        raise ValueError("Демонстратор поддерживает batch_size=1")
    active_length = active_latent_length(pipe, duration_seconds)
    reference_active_length = active_latent_length(pipe, guidance_reference_duration_seconds)
    waveform_length = int(duration_seconds * int(pipe.vae.config.sampling_rate))
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    initialization: dict[str, float | int | str] = {
        "mode": "gaussian_noise",
        "start_index": 0,
        "effective_inference_steps": int(timesteps.numel()),
        "initial_sigma": float(pipe.scheduler.sigmas[0]),
    }
    if reference_latents is not None:
        reference_on_device = reference_latents.detach().to(
            device=device,
            dtype=latents.dtype,
        )
        latents, start_index, initial_sigma = prepare_reference_sde_latents(
            pipe.scheduler,
            reference_latents=reference_on_device,
            initial_noise_latents=latents,
            timesteps=timesteps,
            noise_strength=float(reference_noise_strength),
        )
        timesteps = timesteps[start_index:]
        initialization = {
            "mode": "reference_sde",
            "noise_strength": float(reference_noise_strength),
            "start_index": start_index,
            "effective_inference_steps": int(timesteps.numel()),
            "initial_sigma": initial_sigma,
        }
    text_duration, global_duration, do_cfg = _prepare_conditions(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration_seconds=duration_seconds,
        guidance_scale=guidance_scale,
        device=device,
    )
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim,
        latents.shape[-1] + global_duration.shape[1],
        use_real=True,
        repeat_interleave_real=False,
    )
    scheduler_generator = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = perf_counter()
    last_loss: float | None = None
    guidance_trace: list[dict[str, float | int]] = []
    decoder_guidance_indices: set[int] = set()
    if gamma > 0 and guidance_mode == "decoder_denoising":
        pipe.vae.requires_grad_(False)
        decoder_guidance_indices = set(
            select_decoder_guidance_indices(
                len(timesteps),
                start_fraction=decoder_guidance_start_fraction,
                guidance_steps=final_guidance_steps,
            )
        )

    for index, timestep in enumerate(timesteps):
        model_output = predict_noise_sequential_cfg(
            pipe,
            latents=latents,
            timestep=timestep,
            text_duration=text_duration,
            global_duration=global_duration,
            rotary_embedding=rotary_embedding,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale,
        )

        if (
            gamma > 0
            and guidance_mode == "denoising"
            and index >= int(len(timesteps) * guidance_start_fraction)
        ):
            latents, _, last_loss, step_diagnostics = guide_latents(
                latents,
                model_output,
                sigma=pipe.scheduler.sigmas[pipe.scheduler.step_index],
                target_envelope=target_envelope,
                active_length=active_length,
                gamma=gamma,
                gradient_clip_norm=gradient_clip_norm,
                max_relative_step=max_relative_step,
                envelope_probe=envelope_probe,
                reference_active_length=reference_active_length,
                sigma_data=float(pipe.scheduler.config.sigma_data),
                prediction_type=str(pipe.scheduler.config.prediction_type),
            )
            guidance_trace.append(
                {
                    "step": index,
                    "sigma": float(pipe.scheduler.sigmas[pipe.scheduler.step_index]),
                    "loss_before": last_loss,
                    **step_diagnostics,
                }
            )

        if gamma > 0 and guidance_mode == "decoder_denoising" and index in decoder_guidance_indices:
            latents, loss_before, step_diagnostics = guide_denoising_latents_with_decoder(
                latents,
                model_output,
                sigma=pipe.scheduler.sigmas[pipe.scheduler.step_index],
                decode_fn=lambda values: pipe.vae.decode(values).sample,
                target_envelope=target_envelope,
                active_length=active_length,
                waveform_length=waveform_length,
                gamma=gamma,
                gradient_clip_norm=gradient_clip_norm,
                max_relative_step=max_relative_step,
                reference_active_length=reference_active_length,
                sigma_data=float(pipe.scheduler.config.sigma_data),
                prediction_type=str(pipe.scheduler.config.prediction_type),
                correlation_weight=decoder_correlation_weight,
            )
            last_loss = float(step_diagnostics["loss_after"])
            guidance_trace.append(
                {
                    "step": index,
                    "sigma": float(pipe.scheduler.sigmas[pipe.scheduler.step_index]),
                    "loss_before": loss_before,
                    **step_diagnostics,
                }
            )

        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"NaN/Inf в латенте до шага scheduler #{index}; запись WAV отменена")

        latents = pipe.scheduler.step(
            model_output,
            timestep,
            latents,
            generator=scheduler_generator,
        ).prev_sample
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"NaN/Inf после шага scheduler #{index}; запись WAV отменена")

    if gamma > 0 and guidance_mode == "final":
        if envelope_probe is None:
            raise AssertionError("Final guidance probe был проверен до denoising")
        latents, last_loss, guidance_trace = guide_final_latents(
            latents,
            target_envelope=target_envelope,
            active_length=active_length,
            envelope_probe=envelope_probe,
            gamma=gamma,
            gradient_clip_norm=gradient_clip_norm,
            max_relative_step=max_relative_step,
            steps=final_guidance_steps,
            reference_active_length=reference_active_length,
        )
    elif gamma > 0 and guidance_mode == "decoder":
        pipe.vae.requires_grad_(False)
        latents, last_loss, guidance_trace = guide_final_latents_with_decoder(
            latents,
            decode_fn=lambda values: pipe.vae.decode(values).sample,
            target_envelope=target_envelope,
            active_length=active_length,
            waveform_length=waveform_length,
            gamma=gamma,
            gradient_clip_norm=gradient_clip_norm,
            max_relative_step=max_relative_step,
            steps=final_guidance_steps,
            reference_active_length=reference_active_length,
        )

    with torch.no_grad():
        waveform = pipe.vae.decode(latents).sample
    if not torch.isfinite(waveform).all():
        raise FloatingPointError("VAE вернул NaN/Inf; запись WAV отменена")
    waveform = waveform[:, :, :waveform_length]
    final_envelope = latent_rms_envelope(latents, active_length=active_length)[0].detach().cpu()
    with torch.no_grad():
        final_guidance_envelope = predict_guidance_envelope(
            latents,
            active_length=active_length,
            envelope_probe=envelope_probe,
        )[0].detach().cpu()
    active_latents_cpu = None
    if return_active_latents:
        active_latents_cpu = latents[0, :, :active_length].detach().to(device="cpu", dtype=torch.float16)
    audio = waveform[0].detach().float().cpu().numpy().T
    elapsed_seconds = perf_counter() - started_at
    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    pipe.maybe_free_model_hooks()
    return StableAudioGenerationResult(
        audio=audio,
        sample_rate=int(pipe.vae.config.sampling_rate),
        latent_envelope=final_envelope,
        guidance_envelope=final_guidance_envelope,
        active_latents=active_latents_cpu,
        guidance_loss=last_loss,
        guidance_trace=guidance_trace,
        initialization=initialization,
        elapsed_seconds=elapsed_seconds,
        peak_vram_mb=peak_vram_mb,
    )
