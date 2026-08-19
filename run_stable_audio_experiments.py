"""Сравнить Stable Audio baseline и Direct Latent Guidance на WAV-референсах."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyzer import extract_rms_envelope, load_audio
from audio_io import save_wav
from envelope_probe import load_waveform_envelope_probe
from stable_audio_guidance import (
    envelope_metrics,
    generate_sfx,
    prepare_initial_latents,
    resample_target_envelope,
)
from stable_audio_probe import DEFAULT_MODEL_ID, load_stable_audio

# nvidia-smi reports MiB. A marketed 12 GB RTX 5070 exposes 12_227 MiB,
# which is 11.94 GiB and must not be rejected against a binary 12.0 GiB limit.
# Keep this launcher aligned with tools/preflight_gpu.ps1.
MINIMUM_GUIDANCE_VRAM_GB = 12_000 / 1024
MINIMUM_GUIDANCE_FREE_VRAM_GB = 10_000 / 1024


METRIC_COLUMNS = [
    "case_id",
    "seed",
    "mode",
    "gamma",
    "mse",
    "pearson_correlation",
    "elapsed_seconds",
    "peak_vram_mb",
    "duration_seconds",
]

PAIR_OUTPUT_FILES = (
    "baseline.wav",
    "guided.wav",
    "metadata.json",
    "envelope_comparison.png",
    "guidance_trace.csv",
    "guidance_diagnostics.png",
)
LATENT_DIAGNOSTICS_FILE = "latent_diagnostics.npz"


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    required = {
        "seeds",
        "num_inference_steps",
        "cfg_scale",
        "gamma",
        "gradient_clip_norm",
        "guidance_start_fraction",
        "max_relative_step",
        "guidance_reference_duration_seconds",
        "cases",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"В конфиге нет полей: {', '.join(sorted(missing))}")
    if not config["seeds"] or not config["cases"]:
        raise ValueError("В конфиге должны быть хотя бы один seed и один пример")
    for case in config["cases"]:
        if not {"id", "reference_path", "prompt"}.issubset(case):
            raise ValueError("Каждый пример должен иметь id, reference_path и prompt")
        if not Path(case["reference_path"]).is_file():
            raise FileNotFoundError(f"Не найден референс: {case['reference_path']}")
    return config


def resolve_inference_steps(
    *,
    configured_steps: int,
    smoke_test: bool,
    requested_steps: int | None,
    max_new_pairs: int | None,
) -> int:
    """Выбрать безопасное число шагов denoising для одного запуска."""
    if smoke_test:
        if requested_steps is not None:
            raise ValueError("--num-inference-steps нельзя совмещать с --smoke-test")
        # Один шаг — крайний случай стохастического scheduler и невалидный smoke-test.
        # Четыре шага по-прежнему лёгкие, но проходят нормальный denoising-маршрут.
        return 4
    if requested_steps is None:
        return configured_steps
    if not 2 <= requested_steps <= configured_steps:
        raise ValueError(
            f"--num-inference-steps должен быть в диапазоне [2, {configured_steps}]"
        )
    if max_new_pairs != 1:
        raise ValueError(
            "Промежуточный запуск требует --max-new-pairs 1, "
            "чтобы не запускать несколько GPU-пар подряд"
        )
    return requested_steps


def resolve_guidance_gamma(
    *,
    configured_gamma: float,
    requested_gamma: float | None,
    max_new_pairs: int | None,
) -> float:
    """Выбрать gamma для одиночного воспроизводимого диагностического запуска."""
    if requested_gamma is None:
        return configured_gamma
    if not np.isfinite(requested_gamma) or not 0 < requested_gamma <= 50:
        raise ValueError("--gamma должен быть в диапазоне (0, 50]")
    if max_new_pairs != 1:
        raise ValueError(
            "Переопределение gamma требует --max-new-pairs 1, "
            "чтобы не запустить серию GPU-пар"
        )
    return requested_gamma


def resolve_experiment_selection(
    *,
    configured_cases: list[dict[str, Any]],
    configured_seeds: list[int],
    requested_case_id: str | None,
    requested_seed: int | None,
    smoke_test: bool,
    max_new_pairs: int | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Выбрать один case/seed для диагностического запуска без правки JSON."""
    if (requested_case_id is not None or requested_seed is not None) and max_new_pairs != 1:
        raise ValueError(
            "Переопределение case/seed требует --max-new-pairs 1, "
            "чтобы не запустить серию GPU-пар"
        )

    cases = configured_cases
    if requested_case_id is not None:
        cases = [case for case in configured_cases if case["id"] == requested_case_id]
        if not cases:
            available = ", ".join(str(case["id"]) for case in configured_cases)
            raise ValueError(f"Неизвестный --case-id {requested_case_id!r}; доступны: {available}")
    elif smoke_test:
        cases = configured_cases[:1]

    seeds = configured_seeds
    if requested_seed is not None:
        if not 0 <= requested_seed <= 2**63 - 1:
            raise ValueError("--seed должен быть в диапазоне [0, 2^63 - 1]")
        seeds = [requested_seed]
    elif smoke_test:
        seeds = configured_seeds[:1]
    return cases, seeds


def pair_is_complete(run_dir: Path, *, require_latent_diagnostics: bool = False) -> bool:
    """Считать пару готовой только при наличии всех обязательных артефактов."""
    required = PAIR_OUTPUT_FILES
    if require_latent_diagnostics:
        required = (*required, LATENT_DIAGNOSTICS_FILE)
    return all((run_dir / name).is_file() for name in required)


def validate_latent_diagnostics_request(
    *,
    export_latent_diagnostics: bool,
    max_new_pairs: int | None,
) -> None:
    if export_latent_diagnostics and max_new_pairs != 1:
        raise ValueError(
            "--export-latent-diagnostics требует --max-new-pairs 1, "
            "чтобы raw latent экспортировался только для одной GPU-пары"
        )


def validate_envelope_probe_request(
    *,
    envelope_probe_path: Path | None,
    max_new_pairs: int | None,
) -> None:
    if envelope_probe_path is None:
        return
    if max_new_pairs != 1:
        raise ValueError(
            "--envelope-probe требует --max-new-pairs 1 до завершения GPU smoke-test"
        )
    if not envelope_probe_path.is_file():
        raise FileNotFoundError(f"Не найдены веса envelope probe: {envelope_probe_path}")
    metadata_path = envelope_probe_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Не найден JSON envelope probe: {metadata_path}")


def validate_probe_guidance_mode(
    *,
    probe_guidance_mode: str,
    final_guidance_steps: int,
    envelope_probe_path: Path | None,
    decoder_guidance_start_fraction: float = 0.7,
    decoder_correlation_weight: float = 0.1,
) -> None:
    valid_modes = {"denoising", "final", "decoder", "decoder_denoising"}
    if probe_guidance_mode not in valid_modes:
        raise ValueError("Неизвестный probe guidance mode")
    if not 1 <= final_guidance_steps <= 100:
        raise ValueError("--final-guidance-steps должен быть в диапазоне [1, 100]")
    if (
        probe_guidance_mode in {"final", "decoder", "decoder_denoising"}
        and envelope_probe_path is None
    ):
        raise ValueError(
            f"--probe-guidance-mode {probe_guidance_mode} требует --envelope-probe"
        )
    if probe_guidance_mode in {"decoder", "decoder_denoising"} and final_guidance_steps > 3:
        raise ValueError(
            "Экспериментальные decoder modes допускают --final-guidance-steps "
            "в диапазоне [1, 3]"
        )
    if probe_guidance_mode == "decoder_denoising" and not (
        0.5 <= decoder_guidance_start_fraction < 1
    ):
        raise ValueError(
            "--decoder-guidance-start-fraction должен быть в диапазоне [0.5, 1)"
        )
    if not 0 <= decoder_correlation_weight <= 1:
        raise ValueError("--decoder-correlation-weight должен быть в диапазоне [0, 1]")


def load_reference_for_analysis(
    path: str | Path,
    *,
    analysis_sample_rate: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Загрузить референс на той же частоте, на которой оценивается результат.

    ``extract_rms_envelope`` принимает размеры окна и шага в отсчётах. Поэтому
    две записи с разными sample rate получили бы разные физические размеры RMS-
    окна и несопоставимые метрики, даже при одинаковой форме сигнала.
    """
    waveform, sample_rate = load_audio(path, target_sr=analysis_sample_rate)
    if sample_rate != analysis_sample_rate:
        raise RuntimeError(
            "Частота анализа референса не совпала с частотой модели: "
            f"{sample_rate} != {analysis_sample_rate}"
        )
    return waveform, sample_rate, extract_rms_envelope(waveform).cpu()


def save_latent_diagnostics(
    path: Path,
    *,
    target_envelope: torch.Tensor,
    baseline_latent_envelope: torch.Tensor,
    guided_latent_envelope: torch.Tensor,
    baseline_waveform_envelope: torch.Tensor,
    guided_waveform_envelope: torch.Tensor,
    baseline_active_latents: torch.Tensor | None,
    guided_active_latents: torch.Tensor | None,
    sample_rate: int,
    duration_seconds: float,
    latent_hop_length: int,
    activity_threshold: float = 0.1,
) -> dict[str, Any]:
    """Сохранить компактную пару latent/waveform для обучения envelope-probe."""
    if baseline_active_latents is None or guided_active_latents is None:
        raise ValueError("Для диагностического экспорта нужны active latents обоих режимов")
    if baseline_active_latents.ndim != 2 or guided_active_latents.ndim != 2:
        raise ValueError("Active latents должны иметь форму [channels, time]")
    if baseline_active_latents.shape != guided_active_latents.shape:
        raise ValueError("Baseline и guided active latents должны иметь одинаковую форму")
    if latent_hop_length <= 0:
        raise ValueError("latent_hop_length должен быть положительным")
    if not 0 < activity_threshold < 1:
        raise ValueError("activity_threshold должен быть в диапазоне (0, 1)")

    latent_length = baseline_active_latents.shape[-1]
    if baseline_latent_envelope.numel() != latent_length:
        raise ValueError("Baseline latent-огибающая не совпадает с active latent length")
    if guided_latent_envelope.numel() != latent_length:
        raise ValueError("Guided latent-огибающая не совпадает с active latent length")
    waveform_length = baseline_waveform_envelope.numel()
    if guided_waveform_envelope.numel() != waveform_length:
        raise ValueError("Waveform-огибающие должны иметь одинаковую длину")
    target_for_latent = resample_target_envelope(target_envelope, latent_length).cpu()
    target_for_waveform = resample_target_envelope(target_envelope, waveform_length).cpu()
    baseline_latent_for_waveform = resample_target_envelope(
        baseline_latent_envelope.cpu(), waveform_length
    )
    guided_latent_for_waveform = resample_target_envelope(
        guided_latent_envelope.cpu(), waveform_length
    )

    arrays = {
        "baseline_active_latents": baseline_active_latents.cpu().numpy().astype(np.float16),
        "guided_active_latents": guided_active_latents.cpu().numpy().astype(np.float16),
        "target_envelope_latent": target_for_latent.numpy().astype(np.float32),
        "baseline_latent_envelope": baseline_latent_envelope.cpu().numpy().astype(np.float32),
        "guided_latent_envelope": guided_latent_envelope.cpu().numpy().astype(np.float32),
        "target_envelope_waveform": target_for_waveform.numpy().astype(np.float32),
        "baseline_waveform_envelope": baseline_waveform_envelope.cpu().numpy().astype(np.float32),
        "guided_waveform_envelope": guided_waveform_envelope.cpu().numpy().astype(np.float32),
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("NaN/Inf в latent diagnostics; NPZ не сохранён")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        sample_rate=np.asarray(sample_rate, dtype=np.int32),
        duration_seconds=np.asarray(duration_seconds, dtype=np.float64),
        latent_hop_length=np.asarray(latent_hop_length, dtype=np.int32),
        activity_threshold=np.asarray(activity_threshold, dtype=np.float32),
        **arrays,
    )

    def activity_fraction(values: torch.Tensor) -> float:
        return float((values > activity_threshold).float().mean().item())

    return {
        "file": path.name,
        "format_version": 1,
        "active_latent_shape": list(baseline_active_latents.shape),
        "latent_hop_length": latent_hop_length,
        "target_vs_baseline_latent": envelope_metrics(
            target_for_latent, baseline_latent_envelope.float().cpu()
        ),
        "target_vs_guided_latent": envelope_metrics(
            target_for_latent, guided_latent_envelope.float().cpu()
        ),
        "baseline_latent_vs_waveform": envelope_metrics(
            baseline_latent_for_waveform, baseline_waveform_envelope.float().cpu()
        ),
        "guided_latent_vs_waveform": envelope_metrics(
            guided_latent_for_waveform, guided_waveform_envelope.float().cpu()
        ),
        "activity_fraction": {
            "target": activity_fraction(target_for_waveform),
            "baseline_waveform": activity_fraction(baseline_waveform_envelope),
            "guided_waveform": activity_fraction(guided_waveform_envelope),
        },
    }


def release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def require_safe_gpu(
    *,
    minimum_vram_gb: float,
    minimum_free_vram_gb: float,
    allow_unsafe_vram: bool,
) -> None:
    """Не допустить ручной Stable Audio guidance на заведомо недостаточной GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("Для Stable Audio guidance требуется CUDA-видеокарта")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info(0)
    free_vram_gb = free_vram_bytes / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f"[+] GPU: {gpu_name}; VRAM: {total_vram_gb * 1024:.0f} MiB всего, "
        f"{free_vram_gb * 1024:.0f} MiB свободно"
    )
    if (
        total_vram_gb < minimum_vram_gb or free_vram_gb < minimum_free_vram_gb
    ) and not allow_unsafe_vram:
        raise RuntimeError(
            "Ручной Stable Audio guidance заблокирован: "
            f"всего {total_vram_gb * 1024:.0f} MiB, свободно {free_vram_gb * 1024:.0f} MiB; "
            f"требуется не менее {minimum_vram_gb * 1024:.0f} MiB всего и "
            f"{minimum_free_vram_gb * 1024:.0f} MiB свободно. "
            "Это предотвращает зависание Windows."
        )
    if total_vram_gb < minimum_vram_gb or free_vram_gb < minimum_free_vram_gb:
        print("[!] Запущен небезопасный режим по явному флагу; система может зависнуть.")


def write_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def plot_envelopes(
    path: Path,
    *,
    target: torch.Tensor,
    baseline: torch.Tensor,
    guided: torch.Tensor,
    duration_seconds: float,
    case_id: str,
    seed: int,
) -> None:
    def axis(values: torch.Tensor) -> np.ndarray:
        return np.linspace(0, duration_seconds, values.numel(), endpoint=True)

    plt.figure(figsize=(10, 4.5))
    plt.plot(axis(target), target.numpy(), color="black", linewidth=2.2, label="E_target (референс)")
    plt.plot(axis(baseline), baseline.numpy(), color="#6f6f6f", linestyle="--", label="Stable Audio baseline")
    plt.plot(axis(guided), guided.numpy(), color="#bf2424", linewidth=1.8, label="Direct Latent Guidance")
    plt.title(f"{case_id}, seed={seed}")
    plt.xlabel("Время, с")
    plt.ylabel("Нормированная RMS-огибающая")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def update_summary_plot(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    cases = sorted({str(row["case_id"]) for row in rows})
    modes = ["baseline", "guided"]
    values = np.full((len(cases), len(modes)), np.nan)
    for case_index, case_id in enumerate(cases):
        for mode_index, mode in enumerate(modes):
            current = [float(row["mse"]) for row in rows if row["case_id"] == case_id and row["mode"] == mode]
            if current:
                values[case_index, mode_index] = float(np.mean(current))
    positions = np.arange(len(cases))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(positions - width / 2, values[:, 0], width, label="Baseline", color="#7b7b7b")
    plt.bar(positions + width / 2, values[:, 1], width, label="Guided", color="#bf2424")
    plt.xticks(positions, cases)
    plt.ylabel("Средняя MSE RMS-огибающей (меньше — лучше)")
    plt.title("Stable Audio: baseline против Direct Latent Guidance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_guidance_trace(
    csv_path: Path,
    plot_path: Path,
    trace: list[dict[str, float | int]],
) -> None:
    """Сохранить диагностику силы коррекции и latent-loss по шагам."""
    if not trace:
        return
    columns = list(trace[0])
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(trace)

    steps = [int(row["step"]) for row in trace]
    loss_before = [float(row["loss_before"]) for row in trace]
    loss_after = [float(row["loss_after"]) for row in trace]
    relative = [100.0 * float(row["relative_correction"]) for row in trace]
    figure, loss_axis = plt.subplots(figsize=(9, 4.5))
    loss_axis.plot(steps, loss_before, label="Loss до коррекции", color="#777777", linestyle="--")
    loss_axis.plot(steps, loss_after, label="Loss после коррекции", color="#bf2424")
    loss_axis.set_xlabel("Шаг guidance")
    loss_axis.set_ylabel("MSE guidance-огибающей")
    loss_axis.grid(alpha=0.25)
    correction_axis = loss_axis.twinx()
    correction_axis.plot(steps, relative, label="Размер коррекции", color="#2457a6", alpha=0.55)
    correction_axis.set_ylabel("Коррекция относительно нормы latent, %")
    lines = loss_axis.lines + correction_axis.lines
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _result_row(
    *,
    case_id: str,
    seed: int,
    mode: str,
    gamma: float,
    metrics: dict[str, float],
    elapsed_seconds: float,
    peak_vram_mb: float,
    duration_seconds: float,
) -> dict[str, float | int | str]:
    return {
        "case_id": case_id,
        "seed": seed,
        "mode": mode,
        "gamma": gamma,
        "mse": metrics["mse"],
        "pearson_correlation": metrics["pearson_correlation"],
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_mb": peak_vram_mb,
        "duration_seconds": duration_seconds,
    }


def run(
    config_path: Path,
    results_dir: Path,
    *,
    resume: bool,
    cooldown_seconds: float,
    max_new_pairs: int | None,
    allow_download: bool,
    smoke_test: bool,
    requested_num_inference_steps: int | None,
    requested_gamma: float | None,
    requested_case_id: str | None,
    requested_seed: int | None,
    export_latent_diagnostics: bool,
    envelope_probe_path: Path | None,
    probe_guidance_mode: str,
    final_guidance_steps: int,
    decoder_guidance_start_fraction: float,
    decoder_correlation_weight: float,
    minimum_vram_gb: float,
    minimum_free_vram_gb: float,
    allow_unsafe_vram: bool,
    preflight_only: bool,
) -> None:
    if max_new_pairs is not None and max_new_pairs <= 0:
        raise ValueError("max_new_pairs должен быть положительным")
    validate_latent_diagnostics_request(
        export_latent_diagnostics=export_latent_diagnostics,
        max_new_pairs=max_new_pairs,
    )
    validate_envelope_probe_request(
        envelope_probe_path=envelope_probe_path,
        max_new_pairs=max_new_pairs,
    )
    validate_probe_guidance_mode(
        probe_guidance_mode=probe_guidance_mode,
        final_guidance_steps=final_guidance_steps,
        envelope_probe_path=envelope_probe_path,
        decoder_guidance_start_fraction=decoder_guidance_start_fraction,
        decoder_correlation_weight=decoder_correlation_weight,
    )
    envelope_probe = None
    if envelope_probe_path is not None:
        envelope_probe = load_waveform_envelope_probe(envelope_probe_path, device="cpu")
        print("[+] Envelope probe checkpoint проверен на CPU.")
    config = read_config(config_path)
    require_safe_gpu(
        minimum_vram_gb=minimum_vram_gb,
        minimum_free_vram_gb=minimum_free_vram_gb,
        allow_unsafe_vram=allow_unsafe_vram,
    )
    if preflight_only:
        print("[+] GPU preflight завершён. Модель не загружалась.")
        return
    cases, seeds = resolve_experiment_selection(
        configured_cases=config["cases"],
        configured_seeds=config["seeds"],
        requested_case_id=requested_case_id,
        requested_seed=requested_seed,
        smoke_test=smoke_test,
        max_new_pairs=max_new_pairs,
    )
    steps = resolve_inference_steps(
        configured_steps=int(config["num_inference_steps"]),
        smoke_test=smoke_test,
        requested_steps=requested_num_inference_steps,
        max_new_pairs=max_new_pairs,
    )
    guidance_gamma = resolve_guidance_gamma(
        configured_gamma=float(config["gamma"]),
        requested_gamma=requested_gamma,
        max_new_pairs=max_new_pairs,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.csv"
    rows: list[dict[str, float | int | str]] = []
    if resume and metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
            rows = list(csv.DictReader(metrics_file))

    print("[+] Загрузка Stable Audio Open с CPU offload...")
    pipe = load_stable_audio(config.get("model_id", DEFAULT_MODEL_ID), local_files_only=not allow_download)
    print("[+] Pipeline готов; подготавливаются референс и начальный latent.")
    device = torch.device("cuda")
    if envelope_probe is not None:
        expected_channels = int(pipe.transformer.config.in_channels)
        if envelope_probe.latent_channels != expected_channels:
            raise ValueError(
                f"Probe ожидает {envelope_probe.latent_channels} каналов, "
                f"модель использует {expected_channels}"
            )
        envelope_probe.to(device=device)
        print(
            "[+] Waveform-aware envelope probe загружен: "
            f"{envelope_probe.config()['architecture']}"
        )
    guidance_envelope_mode = "latent_rms"
    if envelope_probe is not None:
        guidance_envelope_mode = f"waveform_probe_{probe_guidance_mode}"
    if probe_guidance_mode == "decoder_denoising" and decoder_correlation_weight > 0:
        guidance_loss_mode = "decoder_waveform_mse_pearson"
    elif probe_guidance_mode in {"decoder", "decoder_denoising"}:
        guidance_loss_mode = "decoder_waveform"
    else:
        guidance_loss_mode = guidance_envelope_mode
    analysis_sample_rate = int(pipe.vae.config.sampling_rate)
    print(f"[+] Параметры guidance: gamma={guidance_gamma:g}")
    completed_pairs = 0

    for case in cases:
        waveform, reference_sample_rate, target = load_reference_for_analysis(
            case["reference_path"],
            analysis_sample_rate=analysis_sample_rate,
        )
        duration_seconds = waveform.numel() / reference_sample_rate
        print(
            f"[+] {case['id']}: {duration_seconds:.2f} с, "
            f"{target.numel()} точек E_target при {reference_sample_rate} Гц"
        )

        for seed in seeds:
            run_dir = results_dir / case["id"] / f"seed_{seed}"
            baseline_path = run_dir / "baseline.wav"
            guided_path = run_dir / "guided.wav"
            if resume and pair_is_complete(
                run_dir,
                require_latent_diagnostics=export_latent_diagnostics,
            ):
                print(f"    seed={seed}: уже готово (--resume).")
                continue
            if resume and run_dir.exists():
                print(f"    seed={seed}: найдены неполные артефакты; пара будет пересоздана.")

            print(f"    seed={seed}: baseline и guidance ({steps} шаг.)...")
            initial_latents = prepare_initial_latents(pipe, seed=int(seed), device=device)
            baseline = generate_sfx(
                pipe,
                prompt=case["prompt"],
                negative_prompt=config.get("negative_prompt"),
                duration_seconds=duration_seconds,
                num_inference_steps=steps,
                guidance_scale=float(config["cfg_scale"]),
                seed=int(seed),
                initial_latents=initial_latents,
                gamma=0.0,
                envelope_probe=envelope_probe,
                guidance_mode=probe_guidance_mode,
                final_guidance_steps=final_guidance_steps,
                decoder_guidance_start_fraction=decoder_guidance_start_fraction,
                decoder_correlation_weight=decoder_correlation_weight,
                return_active_latents=export_latent_diagnostics,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            save_wav(baseline_path, baseline.audio, baseline.sample_rate)
            release_gpu()
            guided = generate_sfx(
                pipe,
                prompt=case["prompt"],
                negative_prompt=config.get("negative_prompt"),
                duration_seconds=duration_seconds,
                num_inference_steps=steps,
                guidance_scale=float(config["cfg_scale"]),
                seed=int(seed),
                initial_latents=initial_latents,
                target_envelope=target,
                gamma=guidance_gamma,
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                guidance_start_fraction=float(config["guidance_start_fraction"]),
                max_relative_step=float(config["max_relative_step"]),
                guidance_reference_duration_seconds=float(
                    config["guidance_reference_duration_seconds"]
                ),
                envelope_probe=envelope_probe,
                guidance_mode=probe_guidance_mode,
                final_guidance_steps=final_guidance_steps,
                decoder_guidance_start_fraction=decoder_guidance_start_fraction,
                decoder_correlation_weight=decoder_correlation_weight,
                return_active_latents=export_latent_diagnostics,
            )
            save_wav(guided_path, guided.audio, guided.sample_rate)
            save_guidance_trace(
                run_dir / "guidance_trace.csv",
                run_dir / "guidance_diagnostics.png",
                guided.guidance_trace,
            )

            baseline_envelope = extract_rms_envelope(torch.from_numpy(baseline.audio.mean(axis=1))).cpu()
            guided_envelope = extract_rms_envelope(torch.from_numpy(guided.audio.mean(axis=1))).cpu()
            target_for_baseline = resample_target_envelope(target, baseline_envelope.numel())
            target_for_guided = resample_target_envelope(target, guided_envelope.numel())
            baseline_metrics = envelope_metrics(target_for_baseline, baseline_envelope)
            guided_metrics = envelope_metrics(target_for_guided, guided_envelope)
            target_for_baseline_guidance = resample_target_envelope(
                target, baseline.guidance_envelope.numel()
            )
            target_for_guided_guidance = resample_target_envelope(
                target, guided.guidance_envelope.numel()
            )
            baseline_guidance_for_waveform = resample_target_envelope(
                baseline.guidance_envelope, baseline_envelope.numel()
            )
            guided_guidance_for_waveform = resample_target_envelope(
                guided.guidance_envelope, guided_envelope.numel()
            )
            guidance_envelope_metadata = {
                "mode": guidance_envelope_mode,
                "probe": envelope_probe.config() if envelope_probe is not None else None,
                "weights_path": str(envelope_probe_path) if envelope_probe_path is not None else None,
                "target_vs_baseline": envelope_metrics(
                    target_for_baseline_guidance, baseline.guidance_envelope
                ),
                "target_vs_guided": envelope_metrics(
                    target_for_guided_guidance, guided.guidance_envelope
                ),
                "baseline_vs_waveform": envelope_metrics(
                    baseline_guidance_for_waveform, baseline_envelope
                ),
                "guided_vs_waveform": envelope_metrics(
                    guided_guidance_for_waveform, guided_envelope
                ),
            }
            latent_diagnostics_metadata = None
            if export_latent_diagnostics:
                latent_diagnostics_metadata = save_latent_diagnostics(
                    run_dir / LATENT_DIAGNOSTICS_FILE,
                    target_envelope=target,
                    baseline_latent_envelope=baseline.latent_envelope,
                    guided_latent_envelope=guided.latent_envelope,
                    baseline_waveform_envelope=baseline_envelope,
                    guided_waveform_envelope=guided_envelope,
                    baseline_active_latents=baseline.active_latents,
                    guided_active_latents=guided.active_latents,
                    sample_rate=guided.sample_rate,
                    duration_seconds=duration_seconds,
                    latent_hop_length=int(pipe.vae.hop_length),
                )
            plot_envelopes(
                run_dir / "envelope_comparison.png",
                target=target,
                baseline=baseline_envelope,
                guided=guided_envelope,
                duration_seconds=duration_seconds,
                case_id=case["id"],
                seed=int(seed),
            )
            metadata = {
                "case_id": case["id"],
                "reference_path": case["reference_path"],
                "prompt": case["prompt"],
                "seed": int(seed),
                "duration_seconds": duration_seconds,
                "analysis_sample_rate": reference_sample_rate,
                "reference_envelope_points": int(target.numel()),
                "generated_envelope_points": int(guided_envelope.numel()),
                "num_inference_steps": steps,
                "cfg_scale": float(config["cfg_scale"]),
                "gamma": guidance_gamma,
                "gradient_clip_norm": float(config["gradient_clip_norm"]),
                "guidance_start_fraction": float(config["guidance_start_fraction"]),
                "max_relative_step": float(config["max_relative_step"]),
                "probe_guidance_mode": probe_guidance_mode,
                "final_guidance_steps": final_guidance_steps,
                "decoder_guidance_start_fraction": decoder_guidance_start_fraction,
                "decoder_correlation_weight": decoder_correlation_weight,
                "guidance_reference_duration_seconds": float(
                    config["guidance_reference_duration_seconds"]
                ),
                "baseline": baseline_metrics,
                "guided": guided_metrics,
                "guided_final_guidance_loss": guided.guidance_loss,
                "guided_final_latent_loss": (
                    guided.guidance_loss if envelope_probe is None else None
                ),
                "guidance_trace": guided.guidance_trace,
                "guidance_envelope": guidance_envelope_metadata,
            }
            if latent_diagnostics_metadata is not None:
                metadata["latent_diagnostics"] = latent_diagnostics_metadata
            (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            rows = [
                row
                for row in rows
                if not (str(row["case_id"]) == case["id"] and int(row["seed"]) == int(seed))
            ]
            rows.extend(
                [
                    _result_row(
                        case_id=case["id"],
                        seed=int(seed),
                        mode="baseline",
                        gamma=0.0,
                        metrics=baseline_metrics,
                        elapsed_seconds=baseline.elapsed_seconds,
                        peak_vram_mb=baseline.peak_vram_mb,
                        duration_seconds=duration_seconds,
                    ),
                    _result_row(
                        case_id=case["id"],
                        seed=int(seed),
                        mode="guided",
                        gamma=guidance_gamma,
                        metrics=guided_metrics,
                        elapsed_seconds=guided.elapsed_seconds,
                        peak_vram_mb=guided.peak_vram_mb,
                        duration_seconds=duration_seconds,
                    ),
                ]
            )
            write_metrics(metrics_path, rows)
            update_summary_plot(results_dir / "summary_mse.png", rows)
            print(
                "      MSE baseline/guided: "
                f"{baseline_metrics['mse']:.4f} / {guided_metrics['mse']:.4f}; "
                f"VRAM: {max(baseline.peak_vram_mb, guided.peak_vram_mb):.0f} МБ"
            )
            if guided.guidance_trace:
                first_loss = float(guided.guidance_trace[0]["loss_before"])
                final_loss = float(guided.guidance_trace[-1]["loss_after"])
                print(
                    f"      guidance loss ({guidance_loss_mode}): "
                    f"{first_loss:.4f} -> {final_loss:.4f}"
                )
            del baseline, guided, initial_latents
            release_gpu()
            completed_pairs += 1
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)
            if max_new_pairs is not None and completed_pairs >= max_new_pairs:
                print("[+] Безопасная остановка. Продолжайте той же командой с --resume.")
                return

    print(f"[+] Эксперимент завершён: {results_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("stable_audio_experiments.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/stable_audio_guidance"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    parser.add_argument("--max-new-pairs", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "Промежуточный режим на 2–50 шагов; требует --max-new-pairs 1. "
            "Не совмещается с --smoke-test."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "Переопределить силу Direct Latent Guidance в диапазоне (0, 50]; "
            "требует --max-new-pairs 1."
        ),
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Запустить только указанный case из конфигурации; требует --max-new-pairs 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Запустить только указанный seed; требует --max-new-pairs 1.",
    )
    parser.add_argument(
        "--export-latent-diagnostics",
        action="store_true",
        help=(
            "Сохранить active latents и aligned envelopes в NPZ; "
            "требует --max-new-pairs 1."
        ),
    )
    parser.add_argument(
        "--envelope-probe",
        type=Path,
        default=None,
        help=(
            "Использовать validated waveform-aware probe вместо latent RMS; "
            "требует --max-new-pairs 1."
        ),
    )
    parser.add_argument(
        "--probe-guidance-mode",
        choices=("denoising", "final", "decoder", "decoder_denoising"),
        default="denoising",
        help=(
            "denoising применяет probe на каждом шаге; final выполняет "
            "projected probe-guidance; decoder выполняет точную финальную "
            "VAE-aware коррекцию; decoder_denoising применяет точный градиент "
            "к выбранным поздним x0-прогнозам."
        ),
    )
    parser.add_argument(
        "--final-guidance-steps",
        type=int,
        default=10,
        help=(
            "Число projected-gradient итераций для final/decoder или число "
            "выбранных поздних шагов для decoder_denoising."
        ),
    )
    parser.add_argument(
        "--decoder-guidance-start-fraction",
        type=float,
        default=0.7,
        help=(
            "Начало окна decoder_denoising в диапазоне [0.5, 1); "
            "по умолчанию последние 30%% траектории."
        ),
    )
    parser.add_argument(
        "--decoder-correlation-weight",
        type=float,
        default=0.1,
        help=(
            "Вес дифференцируемого штрафа 1-Pearson в decoder_denoising "
            "в диапазоне [0, 1]."
        ),
    )
    parser.add_argument("--minimum-vram-gb", type=float, default=MINIMUM_GUIDANCE_VRAM_GB)
    parser.add_argument("--minimum-free-vram-gb", type=float, default=MINIMUM_GUIDANCE_FREE_VRAM_GB)
    parser.add_argument(
        "--allow-unsafe-vram",
        action="store_true",
        help="Отключить защиту VRAM. На GPU с менее чем 12 000 MiB может зависнуть Windows.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Проверить GPU и завершиться без загрузки модели.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(
            arguments.config,
            arguments.results_dir,
            resume=arguments.resume,
            cooldown_seconds=arguments.cooldown_seconds,
            max_new_pairs=arguments.max_new_pairs,
            allow_download=arguments.allow_download,
            smoke_test=arguments.smoke_test,
            requested_num_inference_steps=arguments.num_inference_steps,
            requested_gamma=arguments.gamma,
            requested_case_id=arguments.case_id,
            requested_seed=arguments.seed,
            export_latent_diagnostics=arguments.export_latent_diagnostics,
            envelope_probe_path=arguments.envelope_probe,
            probe_guidance_mode=arguments.probe_guidance_mode,
            final_guidance_steps=arguments.final_guidance_steps,
            decoder_guidance_start_fraction=arguments.decoder_guidance_start_fraction,
            decoder_correlation_weight=arguments.decoder_correlation_weight,
            minimum_vram_gb=arguments.minimum_vram_gb,
            minimum_free_vram_gb=arguments.minimum_free_vram_gb,
            allow_unsafe_vram=arguments.allow_unsafe_vram,
            preflight_only=arguments.preflight_only,
        )
    finally:
        release_gpu()
