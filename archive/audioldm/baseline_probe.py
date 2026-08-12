"""Контрольная проверка качества штатной AudioLDM без Direct Latent Guidance."""

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
from guided_pipeline import DEFAULT_MODEL_ID, load_audioldm_pipeline


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    required = {"duration_seconds", "num_inference_steps", "cfg_scale", "seeds", "cases"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"В baseline-конфиге отсутствуют поля: {', '.join(sorted(missing))}")
    if config["duration_seconds"] <= 0 or config["num_inference_steps"] <= 0:
        raise ValueError("Длительность и число шагов должны быть положительными")
    if not config["seeds"] or not config["cases"]:
        raise ValueError("В конфиге нужны хотя бы один seed и один prompt")
    for case in config["cases"]:
        if not {"id", "prompt"}.issubset(case):
            raise ValueError("Каждый пример должен содержать id и prompt")
    return config


def release_gpu(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def generate_baseline(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    duration_seconds: float,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Выполнить только официальный ``AudioLDMPipeline.__call__``."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    started = perf_counter()
    with torch.inference_mode():
        audio = pipe(
            prompt,
            audio_length_in_s=duration_seconds,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        ).audios[0]
    elapsed_seconds = perf_counter() - started
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return np.asarray(audio, dtype=np.float32), elapsed_seconds, float(peak_vram_mb)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    cooldown_seconds: float = 5.0,
    max_new_runs: int | None = None,
) -> None:
    if max_new_runs is not None and max_new_runs <= 0:
        raise ValueError("max_new_runs должен быть положительным")
    config = read_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Загрузка штатной AudioLDM на {device}...")
    pipe = load_audioldm_pipeline(config.get("model_id", DEFAULT_MODEL_ID), device)
    sample_rate = int(pipe.vocoder.config.sampling_rate)
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
            audio, elapsed_seconds, peak_vram_mb = generate_baseline(
                pipe,
                prompt=case["prompt"],
                negative_prompt=config.get("negative_prompt", ""),
                duration_seconds=float(config["duration_seconds"]),
                num_inference_steps=int(config["num_inference_steps"]),
                cfg_scale=float(config["cfg_scale"]),
                seed=int(seed),
                device=device,
            )
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
            release_gpu(device)
            completed += 1
            print(f"      сохранено: {audio_path} | VRAM: {peak_vram_mb:.0f} МБ")
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)
            if max_new_runs is not None and completed >= max_new_runs:
                print("[+] Безопасная остановка. Повторите команду с --resume для следующего варианта.")
                return

    print(f"[+] Контрольные образцы готовы: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("baseline_probe.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_probe"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--max-new-runs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.config,
        args.output_dir,
        resume=args.resume,
        cooldown_seconds=args.cooldown_seconds,
        max_new_runs=args.max_new_runs,
    )
