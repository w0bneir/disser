"""Контрольная генерация SFX моделью Stable Audio Open 1.0."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from audio_io import save_wav


DEFAULT_MODEL_ID = "stabilityai/stable-audio-open-1.0"


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    required = {"duration_seconds", "num_inference_steps", "cfg_scale", "seeds", "cases"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"В Stable Audio конфиге отсутствуют поля: {', '.join(sorted(missing))}")
    if config["duration_seconds"] <= 0 or config["num_inference_steps"] <= 0:
        raise ValueError("Длительность и число шагов должны быть положительными")
    if not config["seeds"] or not config["cases"]:
        raise ValueError("В конфиге нужны хотя бы один seed и один prompt")
    for case in config["cases"]:
        if not {"id", "prompt"}.issubset(case):
            raise ValueError("Каждый пример должен содержать id и prompt")
    return config


def load_stable_audio(model_id: str, *, local_files_only: bool) -> Any:
    """Загрузить модель в FP16 с выгрузкой компонентов в CPU.

    На GTX 1070 весь Stable Audio одновременно в VRAM не помещается.
    Model CPU offload оставляет в видеопамяти только активный компонент.
    """
    from diffusers import StableAudioPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Stable Audio Open в этом демонстраторе требует CUDA-видеокарту")
    pipe = StableAudioPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def release_gpu() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def generate_one(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    duration_seconds: float,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
) -> tuple[np.ndarray, int, float, float]:
    """Выполнить штатный StableAudioPipeline без пользовательского guidance."""
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    started_at = perf_counter()
    with torch.inference_mode():
        output = pipe(
            prompt,
            audio_start_in_s=0.0,
            audio_end_in_s=duration_seconds,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            output_type="pt",
        ).audios[0]
    elapsed_seconds = perf_counter() - started_at
    # Stable Audio возвращает [channels, samples]; soundfile ожидает [samples, channels].
    audio = output.detach().float().cpu().numpy().T
    sample_rate = int(pipe.vae.sampling_rate)
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
    return audio, sample_rate, elapsed_seconds, peak_vram_mb


def run(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    cooldown_seconds: float = 8.0,
    max_new_runs: int | None = None,
    allow_download: bool = False,
) -> None:
    if max_new_runs is not None and max_new_runs <= 0:
        raise ValueError("max_new_runs должен быть положительным")
    config = read_config(config_path)
    print("[+] Загрузка Stable Audio Open с model CPU offload...")
    pipe = load_stable_audio(config.get("model_id", DEFAULT_MODEL_ID), local_files_only=not allow_download)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0

    for case in config["cases"]:
        for seed in config["seeds"]:
            run_dir = output_dir / case["id"] / f"seed_{seed}"
            audio_path = run_dir / "audio.wav"
            metadata_path = run_dir / "metadata.json"
            if resume and audio_path.is_file() and metadata_path.is_file():
                print(f"    {case['id']}, seed={seed}: уже готово (--resume).")
                continue

            print(f"    {case['id']}, seed={seed}: генерация...")
            audio, sample_rate, elapsed_seconds, peak_vram_mb = generate_one(
                pipe,
                prompt=case["prompt"],
                negative_prompt=config.get("negative_prompt", ""),
                duration_seconds=float(config["duration_seconds"]),
                num_inference_steps=int(config["num_inference_steps"]),
                cfg_scale=float(config["cfg_scale"]),
                seed=int(seed),
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            raw_peak = float(np.max(np.abs(audio)))
            raw_rms = float(np.sqrt(np.mean(np.square(audio))))
            save_wav(audio_path, audio, sample_rate)
            metadata_path.write_text(
                json.dumps(
                    {
                        "case_id": case["id"],
                        "prompt": case["prompt"],
                        "seed": int(seed),
                        "duration_seconds": float(config["duration_seconds"]),
                        "num_inference_steps": int(config["num_inference_steps"]),
                        "cfg_scale": float(config["cfg_scale"]),
                        "sample_rate": sample_rate,
                        "channels": int(audio.shape[1]),
                        "raw_peak": raw_peak,
                        "raw_rms": raw_rms,
                        "elapsed_seconds": elapsed_seconds,
                        "peak_vram_mb": peak_vram_mb,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            del audio
            release_gpu()
            completed += 1
            print(f"      сохранено: {audio_path} | VRAM: {peak_vram_mb:.0f} МБ | время: {elapsed_seconds:.1f} с")
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)
            if max_new_runs is not None and completed >= max_new_runs:
                print("[+] Безопасная остановка. Повторите команду с --resume для следующего варианта.")
                return

    print(f"[+] Контрольные образцы Stable Audio готовы: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("stable_audio_probe.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stable_audio_probe"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=8.0)
    parser.add_argument("--max-new-runs", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true", help="Разрешить загрузку модели, если её нет в локальном кеше")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.config,
        args.output_dir,
        resume=args.resume,
        cooldown_seconds=args.cooldown_seconds,
        max_new_runs=args.max_new_runs,
        allow_download=args.allow_download,
    )
