"""Guarded AudioX reference-variation runner for the RTX 5070 workstation.

Only ``--preflight-only`` is allowed before the official checkpoint passes the
read-only gate. ``--smoke-test`` performs one two-step, one-seed generation.
One 50-step ``--full-test`` is enabled only by the successful smoke metadata.
Both GPU modes use sequential classifier-free guidance and component offload.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

from tools.preflight_audiox import (
    PreflightError,
    configure_windows_console,
    run_preflight,
)


def _right_pad(audio, length: int):
    import torch

    if audio.shape[-1] >= length:
        return audio[..., :length]
    return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))


def _load_reference(path: Path, sample_rate: int):
    import torch
    import torchaudio

    audio, source_rate = torchaudio.load(path)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if not torch.isfinite(audio).all():
        raise RuntimeError("Reference содержит NaN или Inf")
    peak = float(audio.abs().max())
    if peak > 1.0:
        audio = audio / peak
    return audio.contiguous()


def _deterministic_vae_encode(pretransform, audio):
    """Use the posterior mean instead of adding hidden VAE sampling noise."""

    autoencoder = pretransform.model
    encoded = autoencoder.encoder(audio)
    mean, _scale = encoded.chunk(2, dim=1)
    return mean / pretransform.scale


def _enable_sequential_cfg(diffusion_transformer) -> None:
    """Replace batch-doubled CFG with two sequential DiT forwards."""

    original_forward = diffusion_transformer.forward

    def sequential_forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs,
    ):
        import torch

        if cfg_scale == 1.0 or (cross_attn_cond is None and prepend_cond is None):
            return original_forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                negative_cross_attn_cond=negative_cross_attn_cond,
                negative_cross_attn_mask=negative_cross_attn_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                cfg_scale=cfg_scale,
                cfg_dropout_prob=cfg_dropout_prob,
                scale_phi=scale_phi,
                mask=mask,
                return_info=return_info,
                **kwargs,
            )

        # AudioX disables the cross-attention mask in its own forward because
        # of a flash-attention kernel issue.  Keep the audited behaviour.
        cross_attn_cond_mask = None
        cond_result = self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=None,
            mask=mask,
            input_concat_cond=input_concat_cond,
            global_embed=global_embed,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            return_info=return_info,
            **kwargs,
        )
        if return_info:
            cond_output, info = cond_result
        else:
            cond_output = cond_result

        if cross_attn_cond is not None:
            null_cross = torch.zeros_like(cross_attn_cond)
            uncond_cross = (
                negative_cross_attn_cond
                if negative_cross_attn_cond is not None
                else null_cross
            )
            if negative_cross_attn_mask is not None:
                valid = negative_cross_attn_mask.bool().unsqueeze(2)
                uncond_cross = torch.where(valid, uncond_cross, null_cross)
        else:
            uncond_cross = None
        uncond_prepend = torch.zeros_like(prepend_cond) if prepend_cond is not None else None

        uncond_output = self._forward(
            x,
            t,
            cross_attn_cond=uncond_cross,
            cross_attn_cond_mask=None,
            mask=mask,
            input_concat_cond=input_concat_cond,
            global_embed=global_embed,
            prepend_cond=uncond_prepend,
            prepend_cond_mask=prepend_cond_mask,
            return_info=False,
            **kwargs,
        )
        output = uncond_output + (cond_output - uncond_output) * cfg_scale
        if scale_phi:
            cond_std = cond_output.std(dim=1, keepdim=True)
            output_std = output.std(dim=1, keepdim=True).clamp_min(1e-6)
            output = scale_phi * output * (cond_std / output_std) + (1 - scale_phi) * output
        return (output, info) if return_info else output

    diffusion_transformer.forward = types.MethodType(
        sequential_forward, diffusion_transformer
    )


def _load_model(config: dict[str, Any], checkpoint: Path):
    import torch
    from audiox.models.factory import create_model_from_config

    started = time.perf_counter()
    print("[+] AudioX: сборка архитектуры на CPU...")
    model = create_model_from_config(config)
    print("[+] AudioX: memory-mapped загрузка checkpoint на CPU...")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = payload["state_dict"]
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict, payload
    gc.collect()

    model.eval().requires_grad_(False)
    model.to(dtype=torch.float16)
    text_conditioner = model.conditioner.conditioners["text_prompt"]
    text_model = text_conditioner.__dict__.get("model")
    if text_model is not None:
        text_model.to(dtype=torch.float16)
    _enable_sequential_cfg(model.model.model)
    print(f"[+] AudioX загружен за {time.perf_counter() - started:.1f} с")
    return model


def _conditioning_inputs(model, reference, prompt: str, device: str, sample_rate: int):
    import torch

    dtype = torch.float16
    conditioners = model.conditioner.conditioners

    video_conditioner = conditioners["video_prompt"]
    video_embedding = video_conditioner.empty_visual_feat.to(device=device, dtype=dtype)
    video_tensors = [
        video_embedding.expand(1, -1, -1),
        torch.ones((1, 1), device=device, dtype=dtype),
    ]

    text_conditioner = conditioners["text_prompt"]
    text_tensors = text_conditioner([prompt], device)

    audio_conditioner = conditioners["audio_prompt"]
    audio_conditioner.pretransform.to(device=device, dtype=dtype)
    audio_conditioner.proj_out.to(device=device, dtype=dtype)
    audio_prompt = _right_pad(reference, sample_rate * 10).unsqueeze(0)
    audio_prompt = audio_prompt.to(device=device, dtype=dtype)
    audio_latents = _deterministic_vae_encode(
        audio_conditioner.pretransform, audio_prompt
    ).permute(0, 2, 1)
    audio_embedding = audio_conditioner.proj_out(audio_latents)
    audio_tensors = [
        audio_embedding,
        torch.ones(
            (audio_embedding.shape[0], audio_embedding.shape[1]),
            device=device,
            dtype=dtype,
        ),
    ]

    inputs = model.get_conditioning_inputs(
        {
            "video_prompt": video_tensors,
            "text_prompt": text_tensors,
            "audio_prompt": audio_tensors,
        }
    )
    inputs["cross_attn_mask"] = None

    model.conditioner.to("cpu")
    text_model = text_conditioner.__dict__.get("model")
    if text_model is not None:
        text_model.to("cpu")
    del audio_prompt, audio_latents, audio_embedding
    gc.collect()
    torch.cuda.empty_cache()
    return inputs


def _encode_reference(model, reference, sample_size: int, device: str):
    import torch

    pretransform = model.pretransform.to(device=device, dtype=torch.float16)
    padded = _right_pad(reference, sample_size).unsqueeze(0)
    padded = padded.to(device=device, dtype=torch.float16)
    latents = _deterministic_vae_encode(pretransform, padded)
    latent_size = sample_size // pretransform.downsampling_ratio
    latents = _right_pad(latents, latent_size)
    codec = pretransform.decode(latents).float()
    pretransform.to("cpu")
    del padded
    gc.collect()
    torch.cuda.empty_cache()
    return latents, codec


def _decode_latents(model, latents, device: str):
    import torch

    model.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    pretransform = model.pretransform.to(device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        audio = pretransform.decode(latents.to(device=device, dtype=torch.float32))
    pretransform.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return audio.float().cpu()


def _save_audio(path: Path, audio, sample_rate: int, length: int | None = None):
    import torch
    import torchaudio

    value = audio.detach().float().cpu()
    if value.dim() == 3:
        value = value[0]
    if length is not None:
        value = _right_pad(value, length)
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Невалидный audio tensor для {path}")
    peak = float(value.abs().max())
    if peak > 0.999:
        value = value * (0.999 / peak)
    torchaudio.save(path, value, sample_rate, encoding="PCM_S", bits_per_sample=16)


def _validate_smoke_gate(path: Path, checkpoint: Path) -> dict[str, Any]:
    metadata_path = path.resolve() / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Не найден metadata успешного smoke: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": "audiox_reference_variation_smoke_v1",
        "steps": 2,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "sequential_cfg": True,
        "deterministic_vae_mean": True,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r}, ожидалось {value!r}"
        for key, value in required.items()
        if metadata.get(key) != value
    ]
    reserved = float(metadata.get("max_cuda_reserved_mib", float("inf")))
    if reserved > 8_000:
        mismatches.append(f"smoke peak reserved VRAM {reserved:.1f} MiB > 8000 MiB")
    if mismatches:
        raise RuntimeError("Smoke gate не пройден: " + "; ".join(mismatches))
    return metadata


def generate(arguments: argparse.Namespace, checkpoint: Path) -> None:
    import torch
    from audiox.inference.sampling import sample_k

    if arguments.full_test:
        smoke_metadata = _validate_smoke_gate(arguments.smoke_results_dir, checkpoint)
        if arguments.seed != int(smoke_metadata["seed"]):
            raise RuntimeError("Первый full-test обязан использовать seed успешного smoke")
        if arguments.init_noise_level != float(smoke_metadata["init_noise_level"]):
            raise RuntimeError("Первый full-test обязан использовать noise успешного smoke")
        if arguments.cfg_scale != float(smoke_metadata["cfg_scale"]):
            raise RuntimeError("Первый full-test обязан использовать CFG успешного smoke")
        steps = 50
        protocol_id = "audiox_reference_variation_50step_v1"
    else:
        steps = 2
        protocol_id = "audiox_reference_variation_smoke_v1"

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    sample_rate = int(config["sample_rate"])
    sample_size = int(config["sample_size"])
    reference = _load_reference(arguments.reference, sample_rate)
    original_length = reference.shape[-1]
    results_dir = arguments.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=False)
    _save_audio(results_dir / "reference.wav", reference, sample_rate)

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.cuda.reset_peak_memory_stats()
    model = _load_model(config, checkpoint)

    print("[+] AudioX: reference/text conditioning...")
    conditioning = _conditioning_inputs(
        model, reference, arguments.prompt, "cuda", sample_rate
    )
    print("[+] AudioX: deterministic reference encode...")
    init_latents, codec = _encode_reference(model, reference, sample_size, "cuda")
    _save_audio(
        results_dir / "codec_roundtrip.wav",
        codec,
        sample_rate,
        length=original_length,
    )

    model.model.to(device="cuda", dtype=torch.float16)
    latent_size = sample_size // model.pretransform.downsampling_ratio
    init_latents = _right_pad(init_latents, latent_size).to("cuda", torch.float16)
    noise = torch.randn_like(init_latents)
    print(
        f"[+] AudioX {'full-test' if arguments.full_test else 'smoke'}: "
        f"{steps} шаг., sequential CFG, "
        f"seed={arguments.seed}, noise={arguments.init_noise_level}, "
        f"cfg={arguments.cfg_scale}"
    )
    with torch.inference_mode():
        sampled = sample_k(
            model.model,
            noise,
            init_data=init_latents,
            steps=steps,
            sampler_type="dpmpp-2m-sde",
            sigma_min=min(0.03, arguments.init_noise_level / 3),
            sigma_max=arguments.init_noise_level,
            rho=1.0,
            device="cuda",
            cfg_scale=arguments.cfg_scale,
            batch_cfg=True,
            rescale_cfg=True,
            **conditioning,
        )
    del noise, init_latents, conditioning
    generated = _decode_latents(model, sampled, "cuda")
    _save_audio(results_dir / "candidate_full.wav", generated, sample_rate)
    _save_audio(
        results_dir / "candidate_cropped.wav",
        generated,
        sample_rate,
        length=original_length,
    )

    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    metadata = {
        "protocol_id": protocol_id,
        "reference": str(arguments.reference.resolve()),
        "prompt": arguments.prompt,
        "seed": arguments.seed,
        "steps": steps,
        "init_noise_level": arguments.init_noise_level,
        "cfg_scale": arguments.cfg_scale,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "audiox_source_commit": "3bdfb7081636b9e62224039e37dadaa264dc781f",
        "sample_rate": sample_rate,
        "model_sample_size": sample_size,
        "reference_samples": original_length,
        "sequential_cfg": True,
        "deterministic_vae_mean": True,
        "max_cuda_allocated_mib": round(allocated, 1),
        "max_cuda_reserved_mib": round(reserved, 1),
    }
    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[+] AudioX {'full-test' if arguments.full_test else 'smoke'} готов: {results_dir}; "
        f"peak VRAM allocated/reserved {allocated:.0f}/{reserved:.0f} MiB"
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "artifacts" / "AudioX_source" / "config.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=project_root / "artifacts" / "audiox_hf_cache",
    )
    parser.add_argument(
        "--reference", type=Path, default=project_root / "references" / "shot_sound.wav"
    )
    parser.add_argument(
        "--prompt",
        default="one natural gunshot with decay, same acoustic event, no additional shots",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--init-noise-level", type=float, default=0.10)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument(
        "--smoke-results-dir",
        type=Path,
        default=project_root / "results" / "2026-08-23_audiox_shot_smoke_01",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "results" / "audiox_reference_smoke",
    )
    return parser


def main() -> int:
    configure_windows_console()
    arguments = build_parser().parse_args()
    selected_modes = sum(
        bool(mode)
        for mode in (arguments.preflight_only, arguments.smoke_test, arguments.full_test)
    )
    if selected_modes != 1:
        print(
            "[!] Укажите ровно один режим: --preflight-only, --smoke-test или --full-test",
            file=sys.stderr,
        )
        return 2
    if not 0.03 < arguments.init_noise_level <= 1.0:
        print("[!] Для первой серии noise должен быть в диапазоне (0.03, 1.0]", file=sys.stderr)
        return 2
    if not 1.0 <= arguments.cfg_scale <= 7.0:
        print("[!] CFG должен быть в диапазоне [1, 7]", file=sys.stderr)
        return 2

    os.environ["HF_HOME"] = str(arguments.cache_root.resolve())
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        checkpoint = run_preflight(arguments)
        if arguments.preflight_only:
            return 0
        generate(arguments, checkpoint)
    except (PreflightError, RuntimeError, ValueError) as error:
        print(f"[!] AudioX запуск остановлен: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
