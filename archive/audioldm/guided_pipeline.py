"""Генерация SFX с Direct Latent Guidance для AudioLDM 1.

Модуль не обучает и не изменяет веса AudioLDM. Он корректирует только
текущий латент в ходе одного цикла обратной диффузии, чтобы временная
энергия предсказанного чистого латента приближалась к RMS-огибающей
пользовательского референса.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_MODEL_ID = "cvssp/audioldm-s-full-v2"
DEFAULT_NEGATIVE_PROMPT = "low quality, noise, distortion, hiss"


@dataclass
class GenerationResult:
    """Результат одного запуска baseline или guided-варианта."""

    mode: str
    seed: int
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    elapsed_seconds: float
    peak_vram_mb: float
    guidance_losses: list[float]


def normalize_per_sample(values: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Нормировать ``[batch, time]`` независимо для каждого примера."""
    if values.ndim != 2:
        raise ValueError(f"Ожидалась форма [batch, time], получено {tuple(values.shape)}")
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    span = maximum - minimum
    normalized = (values - minimum) / span.clamp_min(eps)
    return torch.where(span > eps, normalized, torch.zeros_like(normalized))


def latent_rms_envelope(latents: torch.Tensor) -> torch.Tensor:
    """Вычислить Direct Latent Energy Envelope.

    ``z`` имеет форму ``[batch, channels, latent_time, latent_frequency]``.
    RMS по channel и frequency оставляет одну временную кривую на пример.
    """
    if latents.ndim != 4:
        raise ValueError(
            "Латенты AudioLDM должны иметь форму "
            f"[batch, channels, time, frequency], получено {tuple(latents.shape)}"
        )
    energy = torch.sqrt(latents.square().mean(dim=(1, 3)) + 1e-8)
    return normalize_per_sample(energy)


def resample_target_envelope(target_envelope: torch.Tensor, target_length: int) -> torch.Tensor:
    """Интерполировать одномерную целевую огибающую к длине латента."""
    if target_envelope.ndim != 1:
        raise ValueError(f"E_target должен иметь форму [time], получено {tuple(target_envelope.shape)}")
    if target_envelope.numel() == 0 or target_length <= 0:
        raise ValueError("Огибающая и целевая длина должны быть непустыми")
    return F.interpolate(
        target_envelope.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1)


def envelope_metrics(target: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    """Вернуть MSE и корреляцию Пирсона между двумя нормированными кривыми."""
    if target.ndim != 1 or generated.ndim != 1:
        raise ValueError("Для метрик требуются две одномерные огибающие")
    generated = resample_target_envelope(generated, target.numel())
    target_cpu = target.detach().float().cpu()
    generated_cpu = generated.detach().float().cpu()
    mse = F.mse_loss(generated_cpu, target_cpu).item()

    centered_target = target_cpu - target_cpu.mean()
    centered_generated = generated_cpu - generated_cpu.mean()
    denominator = torch.linalg.vector_norm(centered_target) * torch.linalg.vector_norm(centered_generated)
    correlation = 0.0 if denominator <= 1e-8 else float((centered_target * centered_generated).sum() / denominator)
    return {"mse": float(mse), "pearson_correlation": correlation}


def encode_class_labels(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """Получить совместимые с AudioLDM 1 class_labels для CFG.

    В diffusers 0.39 используется ``_encode_prompt``. Изоляция этого
    приватного вызова в одной функции упрощает адаптацию к следующей версии.
    """
    if hasattr(pipe, "_encode_prompt"):
        embeddings = pipe._encode_prompt(
            prompt=prompt,
            device=device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
    elif hasattr(pipe, "encode_prompt"):
        encoded = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        if not isinstance(encoded, tuple) or len(encoded) < 2:
            raise RuntimeError("encode_prompt вернул неожиданное значение; ожидались prompt и negative embeddings")
        prompt_embeddings, negative_embeddings = encoded[:2]
        embeddings = torch.cat([negative_embeddings, prompt_embeddings], dim=0)
    else:
        raise RuntimeError("В установленном AudioLDM нет метода кодирования prompt")

    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)
    if embeddings.ndim != 2 or embeddings.shape[0] != 2:
        raise RuntimeError(
            "AudioLDM 1 ожидает class_labels формы [2, embedding_dim] для CFG, "
            f"получено {tuple(embeddings.shape)}"
        )
    return embeddings


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def prepare_initial_latents(
    pipe: Any,
    duration_seconds: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Создать стартовый шум штатным методом AudioLDM.

    Возвращает также требуемую длину итоговой волны. Внутреннее округление
    высоты и размер частотной оси делегируются ``AudioLDMPipeline``.
    """
    if duration_seconds <= 0:
        raise ValueError("Длительность референса должна быть положительной")

    upsample_factor = float(np.prod(pipe.vocoder.config.upsample_rates)) / float(
        pipe.vocoder.config.sampling_rate
    )
    height = int(duration_seconds / upsample_factor)
    vae_scale_factor = int(pipe.vae_scale_factor)
    height = max(vae_scale_factor, int(np.ceil(height / vae_scale_factor)) * vae_scale_factor)
    original_waveform_length = int(round(duration_seconds * pipe.vocoder.config.sampling_rate))

    initial_latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        dtype=torch.float32,
        device=device,
        generator=_make_generator(device, seed),
        latents=None,
    )
    if initial_latents.ndim != 4:
        raise RuntimeError(f"prepare_latents вернул не 4D-латенты: {tuple(initial_latents.shape)}")
    return initial_latents, original_waveform_length


def _predict_x0(
    scheduler: Any,
    timestep: torch.Tensor | int,
    latents: torch.Tensor,
    noise_prediction: torch.Tensor,
) -> torch.Tensor:
    """Аналитически оценить чистый латент z-hat-0 для DDPM-совместимого scheduler."""
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError("Direct Latent Guidance требует scheduler с alphas_cumprod")
    index = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    alpha = scheduler.alphas_cumprod[index].to(device=latents.device, dtype=latents.dtype)
    beta = 1.0 - alpha
    return (latents - beta.sqrt() * noise_prediction) / alpha.sqrt()


def _clip_gradient_norm(gradient: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Ограничить L2-норму градиента отдельно для каждого элемента batch."""
    if max_norm <= 0:
        raise ValueError("gradient_clip_norm должен быть больше нуля")
    norms = torch.linalg.vector_norm(gradient.flatten(start_dim=1), dim=1, keepdim=True)
    scale = torch.clamp(max_norm / norms.clamp_min(1e-8), max=1.0)
    return gradient * scale.view(-1, 1, 1, 1)


def _decode_waveform(pipe: Any, latents: torch.Tensor, original_waveform_length: int) -> np.ndarray:
    """Один раз декодировать готовый латент и привести его к mono numpy."""
    with torch.no_grad():
        mel_spectrogram = pipe.decode_latents(latents)
        if mel_spectrogram.ndim != 4 or mel_spectrogram.shape[1] != 1:
            raise RuntimeError(
                "VAE AudioLDM должен вернуть [batch, 1, time, mel_bins], "
                f"получено {tuple(mel_spectrogram.shape)}"
            )
        waveform = pipe.mel_spectrogram_to_waveform(mel_spectrogram)

    if waveform.ndim == 2:
        audio = waveform[0, :original_waveform_length]
    elif waveform.ndim == 3 and waveform.shape[1] == 1:
        audio = waveform[0, 0, :original_waveform_length]
    else:
        raise RuntimeError(f"Вокодер вернул неожиданную форму: {tuple(waveform.shape)}")
    return audio.detach().float().cpu().numpy()


def generate_sfx(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    initial_latents: torch.Tensor,
    original_waveform_length: int,
    target_envelope: torch.Tensor | None,
    seed: int,
    mode: str,
    gamma: float = 20.0,
    num_inference_steps: int = 30,
    cfg_scale: float = 7.5,
    gradient_clip_norm: float = 0.1,
    eta: float = 0.0,
) -> GenerationResult:
    """Сгенерировать один вариант SFX.

    ``mode='baseline'`` отключает коррекцию. ``mode='guided'`` использует
    Direct Latent Guidance; VAE внутри денойзинга не вызывается.
    """
    if mode not in {"baseline", "guided"}:
        raise ValueError("mode должен быть 'baseline' или 'guided'")
    if mode == "guided" and target_envelope is None:
        raise ValueError("Для guided-режима требуется target_envelope")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps должен быть больше нуля")

    device = initial_latents.device
    class_labels = encode_class_labels(pipe, prompt, negative_prompt, device)
    latents = initial_latents.detach().clone()
    target_on_device = None if target_envelope is None else target_envelope.to(device=device, dtype=latents.dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(_make_generator(device, seed + 10_000), eta)
    guidance_losses: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = perf_counter()

    for timestep in timesteps:
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        with torch.no_grad():
            noise_prediction = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=None,
                class_labels=class_labels,
            ).sample

        noise_unconditional, noise_text = noise_prediction.chunk(2)
        noise_prediction = noise_unconditional + cfg_scale * (noise_text - noise_unconditional)

        if mode == "guided" and gamma > 0:
            differentiable_latents = latents.detach().requires_grad_(True)
            predicted_x0 = _predict_x0(
                pipe.scheduler,
                timestep,
                differentiable_latents,
                noise_prediction.detach(),
            )
            current_envelope = latent_rms_envelope(predicted_x0)
            expected_envelope = resample_target_envelope(target_on_device, current_envelope.shape[-1])
            loss = F.mse_loss(current_envelope, expected_envelope.unsqueeze(0).expand_as(current_envelope))
            gradient = torch.autograd.grad(loss, differentiable_latents, only_inputs=True)[0]
            gradient = _clip_gradient_norm(gradient, gradient_clip_norm)
            latents = (differentiable_latents - gamma * gradient).detach()
            guidance_losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            latents = pipe.scheduler.step(noise_prediction, timestep, latents, **extra_step_kwargs).prev_sample

    audio = _decode_waveform(pipe, latents, original_waveform_length)
    elapsed_seconds = perf_counter() - started_at
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    return GenerationResult(
        mode=mode,
        seed=seed,
        audio=audio,
        sample_rate=int(pipe.vocoder.config.sampling_rate),
        duration_seconds=len(audio) / float(pipe.vocoder.config.sampling_rate),
        elapsed_seconds=elapsed_seconds,
        peak_vram_mb=float(peak_vram_mb),
        guidance_losses=guidance_losses,
    )


def load_audioldm_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    device: torch.device | None = None,
) -> Any:
    """Загрузить AudioLDM 1 в float32 для совместимости с GTX 1070."""
    from diffusers import AudioLDMPipeline

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
