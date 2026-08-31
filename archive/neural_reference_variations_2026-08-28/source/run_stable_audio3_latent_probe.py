"""Safe SAME latent-neighbourhood probe for Stable Audio 3 Small-SFX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sfx_metrics import compare_to_reference, pairwise_diversity
from stable_audio3_latent_variations import (
    TangentPerturbationParameters,
    tangent_covariance_rotation,
)


MINIMUM_TOTAL_MIB = 12_000
MINIMUM_FREE_MIB = 10_000
TARGET_SAMPLE_RATE = 44_100


def _configure_local_hf_cache() -> Path | None:
    candidate = Path(sys.prefix) / "hf-cache"
    if candidate.exists():
        os.environ.setdefault("HF_HOME", str(candidate))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return candidate
    return None


def _local_small_sfx_files(cache: Path) -> tuple[Path, Path]:
    repository = cache / "hub" / "models--stabilityai--stable-audio-3-small-sfx"
    config_files = sorted(repository.glob("snapshots/*/model_config.json"))
    checkpoint_files = sorted(repository.glob("snapshots/*/model.safetensors"))
    if not config_files or not checkpoint_files:
        raise RuntimeError("Stable Audio 3 Small-SFX checkpoint is not present in the local cache")
    return config_files[-1], checkpoint_files[-1]


def _query_gpu() -> dict[str, str | int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(parts) < 4:
        raise RuntimeError(f"Could not parse nvidia-smi output: {completed.stdout!r}")
    return {
        "name": parts[0],
        "driver": parts[1],
        "total_mib": int(parts[2]),
        "free_mib": int(parts[3]),
    }


def preflight(reference: Path) -> dict[str, object]:
    if not reference.is_file():
        raise FileNotFoundError(reference)
    info = _query_gpu()
    print(
        f"[+] GPU: {info['name']}; VRAM: {info['total_mib']} MiB total, "
        f"{info['free_mib']} MiB free"
    )
    if int(info["total_mib"]) < MINIMUM_TOTAL_MIB or int(info["free_mib"]) < MINIMUM_FREE_MIB:
        raise RuntimeError(
            f"VRAM guard blocked the run: need >= {MINIMUM_TOTAL_MIB} MiB total and "
            f">= {MINIMUM_FREE_MIB} MiB free"
        )
    cache = _configure_local_hf_cache()
    if cache is None:
        raise RuntimeError(f"Local HF cache not found under environment: {sys.prefix}")
    config_path, checkpoint_path = _local_small_sfx_files(cache)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    info.update(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version.split()[0],
            "environment": sys.prefix,
            "hf_cache": str(cache),
            "model_config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "reference": str(reference.resolve()),
        }
    )
    return info


def _no_boost_peak_protect(audio: torch.Tensor, ceiling_dbfs: float) -> tuple[torch.Tensor, float]:
    if not torch.isfinite(audio).all():
        raise FloatingPointError("Decoded audio contains NaN or Inf")
    target_peak = 10 ** (ceiling_dbfs / 20)
    peak = float(audio.abs().max())
    gain = min(1.0, target_peak / max(peak, 1e-12))
    return audio.float() * gain, gain


def _write_pcm24(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    samples = audio.detach().float().cpu().numpy().T
    if not np.isfinite(samples).all() or float(np.max(np.abs(samples))) > 1.000001:
        raise ValueError(f"Unsafe audio before write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype="PCM_24")


def _tail_rms(audio: torch.Tensor, sample_rate: int, start_seconds: float = 1.0) -> float:
    start = min(audio.shape[-1] - 1, int(round(start_seconds * sample_rate)))
    return float(audio[..., start:].float().square().mean().sqrt())


def _candidate_metrics(
    reference: torch.Tensor,
    codec: torch.Tensor,
    candidate: torch.Tensor,
    sample_rate: int,
) -> dict[str, float | int | bool]:
    reference_mono = reference.mean(dim=0).cpu()
    codec_mono = codec.mean(dim=0).cpu()
    candidate_mono = candidate.mean(dim=0).cpu()
    ref_metrics = compare_to_reference(reference_mono, candidate_mono, sample_rate)
    codec_metrics = compare_to_reference(codec_mono, candidate_mono, sample_rate)
    reference_tail = _tail_rms(reference, sample_rate)
    candidate_tail = _tail_rms(candidate, sample_rate)
    tail_delta_db = 20 * math.log10(max(candidate_tail, 1e-12) / max(reference_tail, 1e-12))
    objective_pass = bool(
        ref_metrics["envelope_pearson"] >= 0.90
        and ref_metrics["peak_count_abs_error"] <= 1
        and abs(ref_metrics["spectral_centroid_delta_hz"]) <= 2_000
        and abs(ref_metrics["high_frequency_fraction_delta"]) <= 0.12
        and abs(tail_delta_db) <= 12
    )
    non_copy_signal = bool(
        codec_metrics["waveform_pearson"] < 0.999
        and codec_metrics["copy_residual_db"] > -35
    )
    return {
        **{f"ref_{key}": value for key, value in ref_metrics.items()},
        **{f"codec_{key}": value for key, value in codec_metrics.items()},
        "reference_tail_rms_after_1s": reference_tail,
        "candidate_tail_rms_after_1s": candidate_tail,
        "tail_rms_delta_db": tail_delta_db,
        "objective_identity_pass": objective_pass,
        "objective_non_copy_signal": non_copy_signal,
    }


def _disable_decoder_noise(autoencoder: object) -> bool:
    bottleneck = getattr(autoencoder, "bottleneck", None)
    if bottleneck is None or not hasattr(bottleneck, "noise_regularize"):
        return False
    previous = bool(bottleneck.noise_regularize)
    bottleneck.noise_regularize = False
    return previous


def run(arguments: argparse.Namespace) -> None:
    context = preflight(arguments.reference)
    if arguments.preflight_only:
        print("[+] Preflight complete; model was not loaded.")
        return

    from stable_audio_3.loading_utils import load_autoencoder
    from stable_audio_3.model import AutoencoderModel

    reference_np, sample_rate = sf.read(
        arguments.reference,
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz reference, got {sample_rate}")
    reference = torch.from_numpy(reference_np.T.copy())
    output_dir = arguments.results_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    config_path = Path(str(context["model_config"]))
    checkpoint_path = Path(str(context["checkpoint"]))
    autoencoder = load_autoencoder(
        str(config_path),
        str(checkpoint_path),
        device="cuda",
    )
    autoencoder.eval().requires_grad_(False)
    model = AutoencoderModel(autoencoder, TARGET_SAMPLE_RATE, "cuda")
    load_seconds = time.perf_counter() - load_started
    decoder_noise_was_enabled = _disable_decoder_noise(autoencoder)
    print(
        f"[+] SAME-S loaded in {load_seconds:.2f}s; decoder noise disabled="
        f"{decoder_noise_was_enabled}"
    )

    torch.manual_seed(0)
    with torch.inference_mode():
        latents = model.encode(reference, sample_rate, chunked=True)
        codec_raw = model.decode(latents, chunked=True)[0, :, : reference.shape[-1]]
    codec, codec_gain = _no_boost_peak_protect(codec_raw, arguments.peak_ceiling_dbfs)
    _write_pcm24(output_dir / "reference.wav", reference, sample_rate)
    _write_pcm24(output_dir / "codec_deterministic.wav", codec, sample_rate)
    print(f"[+] Encoded latent shape: {tuple(latents.shape)}")

    protect_frames = max(
        1,
        int(math.ceil(arguments.protect_seconds * sample_rate / autoencoder.downsampling_ratio)),
    )
    strengths = list(arguments.angle_degrees)
    seeds = list(arguments.seeds)
    if arguments.smoke_test:
        strengths = strengths[:1]
        seeds = seeds[:1]
    requested = [(angle, seed) for angle in strengths for seed in seeds]
    requested = requested[: arguments.max_new_candidates]
    if not requested:
        raise ValueError("No candidates requested")

    rows: list[dict[str, object]] = []
    candidate_groups: dict[float, list[tuple[int, torch.Tensor]]] = {}
    for angle, seed in requested:
        parameters = TangentPerturbationParameters(
            angle_degrees=float(angle),
            protect_frames=protect_frames,
            transition_frames=arguments.transition_frames,
            smoothing_frames=arguments.smoothing_frames,
            covariance_rank=arguments.covariance_rank,
        )
        perturbed, diagnostics = tangent_covariance_rotation(
            latents,
            parameters=parameters,
            seed=int(seed),
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            decoded_raw = model.decode(perturbed, chunked=True)[0, :, : reference.shape[-1]]
        generation_seconds = time.perf_counter() - started
        decoded, gain = _no_boost_peak_protect(decoded_raw, arguments.peak_ceiling_dbfs)
        filename = f"tangent_angle{angle:g}_seed{seed}.wav"
        _write_pcm24(output_dir / filename, decoded, sample_rate)
        metrics = _candidate_metrics(reference, codec, decoded, sample_rate)
        row: dict[str, object] = {
            "file": filename,
            **diagnostics,
            "protect_seconds_effective": protect_frames * autoencoder.downsampling_ratio / sample_rate,
            "decode_seconds": generation_seconds,
            "raw_peak": float(decoded_raw.abs().max()),
            "applied_gain": gain,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            **metrics,
        }
        rows.append(row)
        candidate_groups.setdefault(float(angle), []).append(
            (int(seed), decoded.mean(dim=0).detach().cpu())
        )
        print(
            f"    {filename}: env={float(metrics['ref_envelope_pearson']):.4f}, "
            f"wave(codec)={float(metrics['codec_waveform_pearson']):.4f}, "
            f"LSD={float(metrics['ref_log_spectral_distance_db']):.2f} dB, "
            f"tail={float(metrics['tail_rms_delta_db']):+.1f} dB, "
            f"gate={bool(metrics['objective_identity_pass'])}, "
            f"VRAM={float(row['peak_allocated_mib']):.0f} MiB"
        )
        del perturbed, decoded_raw, decoded
        torch.cuda.empty_cache()

    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    diversity_rows: list[dict[str, object]] = []
    for angle, items in candidate_groups.items():
        for diversity in pairwise_diversity(items, sample_rate):
            diversity_rows.append({"angle_degrees": angle, **diversity})
    if diversity_rows:
        with (output_dir / "diversity.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diversity_rows[0]))
            writer.writeheader()
            writer.writerows(diversity_rows)
    metadata = {
        "protocol_id": "sa3_same_tangent_probe_v1",
        "reference": str(arguments.reference.resolve()),
        "latent_shape": list(latents.shape),
        "decoder_noise_disabled": decoder_noise_was_enabled,
        "codec_applied_gain": codec_gain,
        "model_load_seconds": load_seconds,
        "parameters": vars(arguments),
        "candidates": rows,
        "pairwise_diversity": diversity_rows,
    }
    metadata["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["parameters"].items()
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[+] Probe complete: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("references/shot_sound.wav"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/sa3_same_tangent_probe"))
    parser.add_argument("--angle-degrees", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--protect-seconds", type=float, default=0.12)
    parser.add_argument("--transition-frames", type=int, default=2)
    parser.add_argument("--smoothing-frames", type=int, default=3)
    parser.add_argument("--covariance-rank", type=int, default=8)
    parser.add_argument("--peak-ceiling-dbfs", type=float, default=-1.0)
    parser.add_argument("--max-new-candidates", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
