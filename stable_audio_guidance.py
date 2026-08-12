"""Direct Latent Guidance для Stable Audio Open без VAE в цикле денойзинга."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class StableAudioGenerationResult:
    """Результат одного baseline или guided прогона."""

    audio: np.ndarray
    sample_rate: int
    latent_envelope: torch.Tensor
    guidance_loss: float | None
    guidance_trace: list[dict[str, float | int]]
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


def active_latent_length(pipe: Any, duration_seconds: float) -> int:
    """Число latent-позиций, соответствующее обрезаемому итоговому WAV."""
    if duration_seconds <= 0:
        raise ValueError("Длительность должна быть положительной")
    sample_rate = int(pipe.vae.config.sampling_rate)
    hop_length = int(pipe.vae.hop_length)
    maximum = int(pipe.transformer.config.sample_size)
    return max(1, min(maximum, ceil(duration_seconds * sample_rate / hop_length)))


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
    if gamma == 0:
        envelope = latent_rms_envelope(latents, active_length=active_length)
        return latents, envelope.detach(), 0.0, {
            "gradient_norm": 0.0,
            "correction_norm": 0.0,
            "active_latent_norm": float(torch.linalg.vector_norm(latents[:, :, :active_length].float()).cpu()),
            "relative_correction": 0.0,
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
    envelope = latent_rms_envelope(predicted_x0, active_length=active_length)
    loss = F.mse_loss(envelope, target.unsqueeze(0).expand_as(envelope))
    gradient = torch.autograd.grad(loss, working_latents, only_inputs=True)[0]
    raw_gradient_norm = torch.linalg.vector_norm(gradient[:, :, :active_length])
    gradient = _clip_gradient_norm(gradient, gradient_clip_norm)

    correction = -gamma * gradient
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
        corrected_envelope = latent_rms_envelope(corrected_x0, active_length=active_length)
        loss_after = F.mse_loss(corrected_envelope, target.unsqueeze(0).expand_as(corrected_envelope))
    diagnostics = {
        "gradient_norm": float(raw_gradient_norm.detach().cpu()),
        "correction_norm": float(correction_norm.detach().cpu()),
        "active_latent_norm": float(active_latent_norm.detach().cpu()),
        "relative_correction": float((correction_norm / active_latent_norm).detach().cpu()),
        "loss_after": float(loss_after.detach().cpu()),
    }
    return corrected, envelope.detach(), float(loss.detach().cpu()), diagnostics


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
    target_envelope: torch.Tensor | None = None,
    gamma: float = 0.0,
    gradient_clip_norm: float = 0.05,
    guidance_start_fraction: float = 0.5,
    max_relative_step: float = 0.03,
) -> StableAudioGenerationResult:
    """Выполнить baseline (gamma=0) или Direct Latent Guidance.

    ВАЖНО: VAE вызывается единожды, после полного цикла denoising.
    """
    if not 0 <= guidance_start_fraction < 1:
        raise ValueError("guidance_start_fraction должен быть в [0, 1)")
    if gamma > 0 and target_envelope is None:
        raise ValueError("Для guidance нужен target_envelope")
    if initial_latents.ndim != 3:
        raise ValueError("initial_latents должен иметь форму [batch, channels, time]")
    if duration_seconds <= 0 or num_inference_steps <= 0:
        raise ValueError("Длительность и число шагов должны быть положительными")

    device = torch.device("cuda")
    latents = initial_latents.detach().clone().to(device)
    if latents.shape[0] != 1:
        raise ValueError("Демонстратор поддерживает batch_size=1")
    active_length = active_latent_length(pipe, duration_seconds)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
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

        if gamma > 0 and index >= int(len(timesteps) * guidance_start_fraction):
            latents, _, last_loss, step_diagnostics = guide_latents(
                latents,
                model_output,
                sigma=pipe.scheduler.sigmas[pipe.scheduler.step_index],
                target_envelope=target_envelope,
                active_length=active_length,
                gamma=gamma,
                gradient_clip_norm=gradient_clip_norm,
                max_relative_step=max_relative_step,
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

    with torch.no_grad():
        waveform = pipe.vae.decode(latents).sample
    if not torch.isfinite(waveform).all():
        raise FloatingPointError("VAE вернул NaN/Inf; запись WAV отменена")
    waveform_length = int(duration_seconds * int(pipe.vae.config.sampling_rate))
    waveform = waveform[:, :, :waveform_length]
    final_envelope = latent_rms_envelope(latents, active_length=active_length)[0].detach().cpu()
    audio = waveform[0].detach().float().cpu().numpy().T
    elapsed_seconds = perf_counter() - started_at
    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    pipe.maybe_free_model_hooks()
    return StableAudioGenerationResult(
        audio=audio,
        sample_rate=int(pipe.vae.config.sampling_rate),
        latent_envelope=final_envelope,
        guidance_loss=last_loss,
        guidance_trace=guidance_trace,
        elapsed_seconds=elapsed_seconds,
        peak_vram_mb=peak_vram_mb,
    )
