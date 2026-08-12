"""Запуск воспроизводимой демонстрации baseline против Direct Latent Guidance."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audio_io import save_wav
from analyzer import extract_rms_envelope, load_audio
from guided_pipeline import (
    DEFAULT_MODEL_ID,
    envelope_metrics,
    generate_sfx,
    load_audioldm_pipeline,
    prepare_initial_latents,
)


METRIC_FIELDS = [
    "case_id",
    "seed",
    "mode",
    "gamma",
    "mse",
    "pearson_correlation",
    "elapsed_seconds",
    "peak_vram_mb",
    "audio_duration_seconds",
]


def read_config(config_path: Path) -> dict[str, Any]:
    """Прочитать и проверить минимальную структуру JSON-конфига."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_top_level = {"seeds", "num_inference_steps", "cfg_scale", "gamma", "cases"}
    missing = required_top_level.difference(config)
    if missing:
        raise ValueError(f"В конфиге отсутствуют поля: {', '.join(sorted(missing))}")
    if len(config["cases"]) != 2:
        raise ValueError("Для демонстратора требуется ровно два референсных примера")
    if len(config["seeds"]) != 3:
        raise ValueError("Для демонстратора требуются ровно три seed")

    case_ids: set[str] = set()
    for case in config["cases"]:
        required_case_fields = {"id", "reference_wav", "prompt", "negative_prompt"}
        missing_case = required_case_fields.difference(case)
        if missing_case:
            raise ValueError(f"В описании примера отсутствуют поля: {', '.join(sorted(missing_case))}")
        if case["id"] in case_ids:
            raise ValueError(f"Идентификатор примера повторяется: {case['id']}")
        case_ids.add(case["id"])
    return config


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    save_wav(path, audio, sample_rate)


def plot_envelope_comparison(
    output_path: Path,
    duration_seconds: float,
    target: torch.Tensor,
    baseline: torch.Tensor,
    guided: torch.Tensor,
    *,
    case_id: str,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    def time_axis(values: torch.Tensor) -> np.ndarray:
        return np.linspace(0, duration_seconds, values.numel())

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis(target), target.numpy(), color="black", linewidth=2.5, label="Целевая E_target")
    plt.plot(time_axis(baseline), baseline.numpy(), color="#777777", linestyle="--", label="Baseline")
    plt.plot(time_axis(guided), guided.numpy(), color="#cc2222", linestyle="-", label="Direct Latent Guidance")
    plt.title(f"{case_id}, seed={seed}: сравнение огибающих")
    plt.xlabel("Время, с")
    plt.ylabel("Нормированная RMS-энергия")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_summary(output_path: Path, metrics: list[dict[str, Any]]) -> None:
    """Сохранить график средних метрик по трём seed для каждого примера."""
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(str(row["case_id"]), str(row["mode"]))].append(row)

    case_ids = sorted({str(row["case_id"]) for row in metrics})
    modes = ["baseline", "guided"]
    mse_values = [[np.mean([float(row["mse"]) for row in grouped[(case_id, mode)]]) for case_id in case_ids] for mode in modes]
    corr_values = [
        [np.mean([float(row["pearson_correlation"]) for row in grouped[(case_id, mode)]]) for case_id in case_ids]
        for mode in modes
    ]

    positions = np.arange(len(case_ids))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["Baseline", "Guided"]
    colors = ["#777777", "#cc2222"]
    for index, (values, label, color) in enumerate(zip(mse_values, labels, colors)):
        axes[0].bar(positions + (index - 0.5) * width, values, width, label=label, color=color)
    for index, (values, label, color) in enumerate(zip(corr_values, labels, colors)):
        axes[1].bar(positions + (index - 0.5) * width, values, width, label=label, color=color)

    axes[0].set_title("Средняя MSE огибающей")
    axes[0].set_ylabel("Меньше — лучше")
    axes[1].set_title("Средняя корреляция Пирсона")
    axes[1].set_ylabel("Больше — лучше")
    for axis in axes:
        axis.set_xticks(positions, case_ids)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_from_existing_pair(
    run_dir: Path,
    target_envelope: torch.Tensor,
    model_sample_rate: int,
    *,
    case_id: str,
    seed: int,
    gamma: float,
) -> list[dict[str, Any]]:
    """Восстановить метрики готовой пары при продолжении прерванного запуска."""
    rows: list[dict[str, Any]] = []
    for mode, gamma_value in (("baseline", 0.0), ("guided", gamma)):
        audio, sample_rate = load_audio(run_dir / f"{mode}.wav", target_sr=model_sample_rate)
        metric = envelope_metrics(target_envelope, extract_rms_envelope(audio))
        rows.append(
            {
                "case_id": case_id,
                "seed": seed,
                "mode": mode,
                "gamma": gamma_value,
                "mse": metric["mse"],
                "pearson_correlation": metric["pearson_correlation"],
                "elapsed_seconds": "",
                "peak_vram_mb": "",
                "audio_duration_seconds": len(audio) / sample_rate,
            }
        )
    return rows


def _release_cuda_memory(device: torch.device) -> None:
    """Освободить временные объекты между парами запусков на экранной GPU."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def run(
    config_path: Path,
    results_dir: Path,
    *,
    smoke_test: bool = False,
    resume: bool = False,
    cooldown_seconds: float = 2.0,
    max_new_pairs: int | None = None,
) -> None:
    if max_new_pairs is not None and max_new_pairs <= 0:
        raise ValueError("max_new_pairs должен быть положительным")
    config = read_config(config_path)
    if smoke_test:
        config = {**config, "cases": config["cases"][:1], "seeds": config["seeds"][:1], "num_inference_steps": 1}

    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Загрузка AudioLDM на {device}...")
    pipe = load_audioldm_pipeline(config.get("model_id", DEFAULT_MODEL_ID), device)
    model_sample_rate = int(pipe.vocoder.config.sampling_rate)
    all_metrics: list[dict[str, Any]] = []
    completed_new_pairs = 0

    for case in config["cases"]:
        reference_path = (config_path.parent / case["reference_wav"]).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Не найден референс для '{case['id']}': {reference_path}")

        reference_audio, reference_sr = load_audio(reference_path, target_sr=model_sample_rate)
        if reference_sr != model_sample_rate:
            raise RuntimeError("Референс был загружен с неверной частотой дискретизации")
        duration_seconds = len(reference_audio) / reference_sr
        target_envelope = extract_rms_envelope(reference_audio)
        print(f"[+] Пример '{case['id']}': {duration_seconds:.2f} с, {len(target_envelope)} точек E_target")

        for seed in config["seeds"]:
            run_dir = results_dir / case["id"] / f"seed_{seed}"
            baseline_path = run_dir / "baseline.wav"
            guided_path = run_dir / "guided.wav"
            if resume and baseline_path.is_file() and guided_path.is_file():
                print(f"    seed={seed}: готовые WAV найдены, метрики восстанавливаются (--resume).")
                all_metrics.extend(
                    _metrics_from_existing_pair(
                        run_dir,
                        target_envelope,
                        model_sample_rate,
                        case_id=case["id"],
                        seed=int(seed),
                        gamma=float(config["gamma"]),
                    )
                )
                continue

            initial_latents, original_waveform_length = prepare_initial_latents(
                pipe, duration_seconds, int(seed), device
            )
            common_kwargs = {
                "pipe": pipe,
                "prompt": case["prompt"],
                "negative_prompt": case["negative_prompt"],
                "initial_latents": initial_latents,
                "original_waveform_length": original_waveform_length,
                "seed": int(seed),
                "num_inference_steps": int(config["num_inference_steps"]),
                "cfg_scale": float(config["cfg_scale"]),
                "gradient_clip_norm": float(config.get("gradient_clip_norm", 0.1)),
                "eta": float(config.get("eta", 0.0)),
            }

            print(f"    seed={seed}: baseline...")
            baseline = generate_sfx(
                **common_kwargs,
                target_envelope=None,
                mode="baseline",
                gamma=0.0,
            )
            print(f"    seed={seed}: guided (gamma={config['gamma']})...")
            guided = generate_sfx(
                **common_kwargs,
                target_envelope=target_envelope,
                mode="guided",
                gamma=float(config["gamma"]),
            )

            save_audio(baseline_path, baseline.audio, baseline.sample_rate)
            save_audio(guided_path, guided.audio, guided.sample_rate)
            baseline_envelope = extract_rms_envelope(torch.from_numpy(baseline.audio))
            guided_envelope = extract_rms_envelope(torch.from_numpy(guided.audio))
            plot_envelope_comparison(
                run_dir / "envelope_comparison.png",
                duration_seconds,
                target_envelope,
                baseline_envelope,
                guided_envelope,
                case_id=case["id"],
                seed=int(seed),
            )

            for result, envelope, gamma in (
                (baseline, baseline_envelope, 0.0),
                (guided, guided_envelope, float(config["gamma"])),
            ):
                metric = envelope_metrics(target_envelope, envelope)
                all_metrics.append(
                    {
                        "case_id": case["id"],
                        "seed": result.seed,
                        "mode": result.mode,
                        "gamma": gamma,
                        "mse": metric["mse"],
                        "pearson_correlation": metric["pearson_correlation"],
                        "elapsed_seconds": result.elapsed_seconds,
                        "peak_vram_mb": result.peak_vram_mb,
                        "audio_duration_seconds": result.duration_seconds,
                    }
                )

            # Метрики сохраняются после каждой пары: прерывание не уничтожит
            # уже завершенную часть эксперимента.
            write_metrics(results_dir / "metrics.csv", all_metrics)
            del common_kwargs, initial_latents, baseline, guided, baseline_envelope, guided_envelope
            _release_cuda_memory(device)
            if cooldown_seconds > 0:
                print(f"    освобождение GPU-памяти; пауза {cooldown_seconds:.1f} с...")
                time.sleep(cooldown_seconds)
            completed_new_pairs += 1
            if max_new_pairs is not None and completed_new_pairs >= max_new_pairs:
                plot_summary(results_dir / "summary_metrics.png", all_metrics)
                print(
                    f"[+] Безопасная остановка после {completed_new_pairs} новой пары. "
                    "Для продолжения повторите команду с --resume."
                )
                return

    write_metrics(results_dir / "metrics.csv", all_metrics)
    plot_summary(results_dir / "summary_metrics.png", all_metrics)
    print(f"[+] Готово. Результаты сохранены в: {results_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments.json"), help="JSON-конфиг эксперимента")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Каталог результатов")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Один пример, один seed и один шаг денойзинга для проверки GPU-конвейера",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Пропустить уже готовые пары WAV и восстановить их метрики",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help="Пауза с очисткой CUDA-кеша между парами seed (по умолчанию: 2)",
    )
    parser.add_argument(
        "--max-new-pairs",
        type=int,
        default=None,
        help="Остановиться после указанного числа новых пар baseline/guided; используйте вместе с --resume",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config,
        arguments.results_dir,
        smoke_test=arguments.smoke_test,
        resume=arguments.resume,
        cooldown_seconds=arguments.cooldown_seconds,
        max_new_pairs=arguments.max_new_pairs,
    )
