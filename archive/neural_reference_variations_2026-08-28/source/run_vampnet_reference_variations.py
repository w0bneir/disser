"""Защищённый launcher prompt-free VampNet-прототипа.

На первом этапе launcher проверяет официальный LAC codec отдельно от
генеративных моделей. Это обязательный quality gate перед скачиванием и
загрузкой ещё ~2.4 GB весов.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sfx_metrics import compare_to_reference
from vampnet_reference_variations import (
    MINIMUM_FREE_VRAM_MIB,
    MINIMUM_TOTAL_VRAM_MIB,
    SAMPLE_RATE,
    build_reference_mask,
    build_tiered_reference_mask,
    comparison_metrics,
    envelope,
    load_reference_mono,
    serializable_description,
    technical_audio_gate,
    validate_model_assets,
)


DEFAULT_MODEL_DIR = Path("artifacts/vampnet_models")
VAMPNET_SOURCE_COMMIT = "72e2675790091fe28ecfd8391303a46b25a703db"
LAC_SOURCE_COMMIT = "7761206878d1fba79aad314a38f975e9589af0a4"


def codec_relative_metrics(
    codec_reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    """Measure generation beyond codec loss; still not a perceptual verdict."""
    return compare_to_reference(
        torch.from_numpy(np.asarray(codec_reference, dtype=np.float32)),
        torch.from_numpy(np.asarray(candidate, dtype=np.float32)),
        SAMPLE_RATE,
    )


def load_coarse_lora(model: torch.nn.Module, checkpoint: Path) -> dict[str, object]:
    """Safely load a local LoRA-only state into an already constructed model."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Не найден coarse LoRA checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Coarse LoRA checkpoint не является непустым state dict")
    if any("lora_" not in str(key) for key in state):
        raise ValueError("Coarse LoRA checkpoint содержит не-LoRA параметры")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Coarse LoRA checkpoint содержит не-tensor значения")

    # loralib merges adapters into the base weight on eval().  Temporarily
    # switching to train() guarantees the newly loaded adapter is merged once.
    model.train()
    incompatible = model.load_state_dict(state, strict=False)
    model.eval()
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Неожиданные LoRA keys: {incompatible.unexpected_keys[:3]}"
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": digest.hexdigest(),
        "tensor_count": len(state),
        "parameter_count": int(sum(value.numel() for value in state.values())),
    }


def prepare_reference_on_gpu(reference: np.ndarray):
    """Повторить официальную -24 LUFS подготовку, но считать её на GPU."""
    from audiotools import AudioSignal

    signal = AudioSignal(
        torch.from_numpy(reference)[None, None, :],
        SAMPLE_RATE,
    ).to("cuda")
    original_loudness = signal.loudness().detach()
    prepared = signal.clone().normalize(-24.0).ensure_max_of_audio(1.0)
    return prepared.audio_data, original_loudness


def restore_reference_loudness(
    audio: torch.Tensor,
    *,
    frames: int,
    original_loudness: torch.Tensor,
) -> np.ndarray:
    """Согласовать integrated loudness и безопасно ограничить peak на GPU."""
    from audiotools import AudioSignal

    values = audio[..., :frames]
    if values.shape[-1] < frames:
        values = torch.nn.functional.pad(values, (0, frames - values.shape[-1]))
    signal = AudioSignal(values.float(), SAMPLE_RATE)
    signal.normalize(original_loudness).ensure_max_of_audio(0.99)
    return signal.audio_data.detach().float().cpu().numpy().reshape(-1)


def write_supervisor_demo(results_dir: Path, report: dict[str, object]) -> Path:
    """Создать автономную локальную страницу для последовательного A/B."""
    variation_cards = []
    for index, row in enumerate(report["variations"], start=1):
        filename = html.escape(str(row["file"]))
        variation_cards.append(
            f"""
            <section class="card">
              <h2>Вариация {index}</h2>
              <audio controls preload="metadata" src="{filename}"></audio>
              <p>Seed {row['seed']}; генерация {float(row['seconds']):.2f} с.</p>
            </section>
            """
        )
    config = report["configuration"]
    profile = str(config.get("mask_profile", "conservative"))
    profile_description = {
        "conservative": "Три нижних акустических codebook-а сохранены",
        "tiered-mid": "Codebook 0–1 сохранены; изменения перенесены в 2–3",
        "tiered-event": "Codebook 0 сохранён; изменения перенесены в 1–3 после атаки",
    }[profile]
    page = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Черновик reference-guided SFX variations</title>
  <style>
    body {{ font: 17px/1.5 system-ui, sans-serif; max-width: 920px; margin: 40px auto;
            padding: 0 22px; color: #1d232a; background: #f4f5f7; }}
    h1 {{ line-height: 1.15; }}
    .note {{ border-left: 5px solid #4263eb; padding: 12px 16px; background: white; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
             gap: 16px; margin: 24px 0; }}
    .card {{ background: white; border-radius: 12px; padding: 18px;
             box-shadow: 0 2px 10px #0001; }}
    audio {{ width: 100%; }}
    code {{ background: #e9ecef; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Reference-guided SFX variations — первый черновик</h1>
  <p class="note"><strong>Задача:</strong> автоматически получить новые дубли одного
  звукового события без текстового промпта. Сначала слушайте референс, затем codec
  round-trip, после него три вариации. Метрики не заменяют слуховую оценку.</p>

  <div class="grid">
    <section class="card">
      <h2>Исходный референс</h2>
      <audio controls preload="metadata" src="reference_mono_44100.wav"></audio>
    </section>
    <section class="card">
      <h2>Контроль кодека</h2>
      <audio controls preload="metadata" src="codec_roundtrip.wav"></audio>
      <p>Не вариация: показывает только потери encode/decode.</p>
    </section>
    {''.join(variation_cards)}
  </div>

  <h2>Что оценивать</h2>
  <ol>
    <li>Все ли три результата воспринимаются как тот же самый одиночный выстрел?</li>
    <li>Есть ли между ними полезное различие при последовательном воспроизведении?</li>
    <li>Нет ли металлических артефактов, морфинга или дополнительных выстрелов?</li>
  </ol>

  <h2>Сохранение временной структуры</h2>
  <img src="envelope_overview.png" alt="Нормированные RMS-огибающие" style="width:100%;background:white;border-radius:12px">

  <h2>Зафиксированный метод</h2>
  <p>Prompt отсутствует. {profile_description}, верхние уровни
  пересэмплированы разреженно с периодическими временными якорями; первые
  {float(config['attack_ms']):.0f} мс защищены. Частота 44,1 кГц, mono, одинаковая
  длина. Пик VRAM {float(report['peak_vram_mib']):.0f} MiB.</p>

  <p><small>Это исследовательский прототип. До слепого прослушивания он не считается
  доказательством гипотезы.</small></p>
</body>
</html>
"""
    path = results_dir / "demo.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_envelope_overview(
    path: Path,
    *,
    reference: np.ndarray,
    variations: list[np.ndarray],
) -> Path:
    """Сохранить наглядный график формы, не выдавая его за слуховую оценку."""
    cache_dir = (Path("artifacts") / "matplotlib_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration = reference.size / SAMPLE_RATE

    def normalized(values: np.ndarray) -> np.ndarray:
        curve = envelope(values)
        return curve / max(float(np.max(curve)), 1e-12)

    target = normalized(reference)
    plt.figure(figsize=(10, 4.5))
    plt.plot(
        np.linspace(0, duration, target.size),
        target,
        color="black",
        linewidth=2.4,
        label="reference",
    )
    colors = ("#4263eb", "#e8590c", "#2b8a3e", "#9c36b5")
    for index, waveform in enumerate(variations):
        curve = normalized(waveform)
        plt.plot(
            np.linspace(0, duration, curve.size),
            curve,
            color=colors[index % len(colors)],
            linewidth=1.3,
            alpha=0.82,
            label=f"variation {index + 1}",
        )
    plt.xlabel("Время, с")
    plt.ylabel("Нормированная RMS-огибающая")
    plt.title("Временная структура: reference и token-вариации")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def require_safe_gpu(*, allow_unsafe_vram: bool = False) -> dict[str, int | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("Для VampNet-прототипа требуется CUDA-видеокарта")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_mib = int(total_bytes // (1024**2))
    free_mib = int(free_bytes // (1024**2))
    name = torch.cuda.get_device_name(0)
    print(f"[+] GPU: {name}; VRAM: {total_mib} MiB всего, {free_mib} MiB свободно")
    if (
        total_mib < MINIMUM_TOTAL_VRAM_MIB
        or free_mib < MINIMUM_FREE_VRAM_MIB
    ) and not allow_unsafe_vram:
        raise RuntimeError(
            "VampNet-прототип заблокирован предохранителем VRAM: "
            f"получено {total_mib}/{free_mib} MiB, требуется не менее "
            f"{MINIMUM_TOTAL_VRAM_MIB}/{MINIMUM_FREE_VRAM_MIB} MiB "
            "(всего/свободно)"
        )
    return {"name": name, "total_mib": total_mib, "free_mib": free_mib}


def run_codec_gate(
    *,
    reference_path: Path,
    model_dir: Path,
    results_dir: Path,
    allow_unsafe_vram: bool,
) -> Path:
    assets = validate_model_assets(model_dir, required=("codec.pth",))
    gpu = require_safe_gpu(allow_unsafe_vram=allow_unsafe_vram)

    try:
        from lac.model.lac import LAC
    except ImportError as error:
        raise RuntimeError(
            "LAC/VampNet зависимости не найдены. Запускайте через "
            "artifacts\\vampnet_env\\Scripts\\python.exe"
        ) from error

    reference = load_reference_mono(reference_path)
    if float(np.sqrt(np.mean(np.square(reference)))) <= 1e-6:
        raise ValueError("Референс является тишиной")

    started = time.perf_counter()
    print("[+] Загрузка только LAC codec; генеративные модели не загружаются...", flush=True)
    codec = LAC.load(model_dir / "codec.pth", map_location="cpu")
    codec.eval().requires_grad_(False).to("cuda")
    print(f"[+] Codec загружен за {time.perf_counter() - started:.1f} с", flush=True)
    torch.cuda.reset_peak_memory_stats()

    stage_started = time.perf_counter()
    prepared, original_loudness = prepare_reference_on_gpu(reference)
    print(
        f"[+] Аудио подготовлено на GPU за {time.perf_counter() - stage_started:.1f} с",
        flush=True,
    )
    with torch.inference_mode():
        stage_started = time.perf_counter()
        encoded = codec.encode(prepared, SAMPLE_RATE)
        torch.cuda.synchronize()
        print(
            f"[+] Codec encode завершён за {time.perf_counter() - stage_started:.1f} с; "
            f"codes={tuple(encoded['codes'].shape)}",
            flush=True,
        )
        stage_started = time.perf_counter()
        decoded = codec.decode(encoded["z"], encoded["length"])["audio"]
        torch.cuda.synchronize()
        print(f"[+] Codec decode завершён за {time.perf_counter() - stage_started:.1f} с", flush=True)
    torch.cuda.synchronize()
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / (1024**2))
    reconstructed = restore_reference_loudness(
        decoded,
        frames=reference.size,
        original_loudness=original_loudness,
    )

    passed, failures = technical_audio_gate(reference, reconstructed)
    metrics = comparison_metrics(reference, reconstructed)

    results_dir.mkdir(parents=True, exist_ok=True)
    reference_output = results_dir / "reference_mono_44100.wav"
    codec_output = results_dir / "codec_roundtrip.wav"
    sf.write(reference_output, reference, SAMPLE_RATE, subtype="PCM_24")
    sf.write(codec_output, reconstructed, SAMPLE_RATE, subtype="PCM_24")

    metadata = {
        "stage": "codec_roundtrip_gate",
        "method": "official VampNet LAC codec, no generation",
        "codec_input_normalization": "official mono -24 integrated LUFS on GPU",
        "source": str(reference_path.resolve()),
        "model_assets": assets,
        "gpu": gpu,
        "peak_vram_mib": peak_vram_mib,
        "reference": serializable_description(reference, SAMPLE_RATE),
        "codec_roundtrip": serializable_description(reconstructed, SAMPLE_RATE),
        "diagnostic_metrics_not_a_listening_test": metrics,
        "technical_gate_passed": passed,
        "technical_gate_failures": failures,
        "requires_human_listening": True,
        "known_limitations": [
            "mono prototype",
            "codec reconstruction is not yet a generated variation",
            "objective metrics do not prove perceptual identity or naturalness",
        ],
    }
    metadata_path = results_dir / "codec_gate.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[+] Codec gate: {'PASS' if passed else 'FAIL'}; "
        f"envelope corr={metrics['envelope_correlation']:.4f}; "
        f"peak VRAM={peak_vram_mib:.0f} MiB"
    )
    print(f"[+] Результаты: {results_dir.resolve()}")
    return metadata_path


def run_generation(
    *,
    reference_path: Path,
    model_dir: Path,
    results_dir: Path,
    seeds: list[int],
    upper_codebook_mask: int,
    periodic_prompt: int,
    attack_ms: float,
    temperature: float,
    sampling_steps: int,
    mask_profile: str,
    fine_resample_period: int,
    coarse_lora: Path | None,
    allow_unsafe_vram: bool,
) -> Path:
    assets = validate_model_assets(
        model_dir,
        required=("codec.pth", "coarse.pth", "c2f.pth"),
    )
    gpu = require_safe_gpu(allow_unsafe_vram=allow_unsafe_vram)
    if not seeds:
        raise ValueError("Нужен хотя бы один seed")
    if len(seeds) > 1 and len(set(seeds)) != len(seeds):
        raise ValueError("Seed-ы вариаций должны быть уникальны")
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature должна быть в диапазоне 0.1..2.0")
    if not 1 <= sampling_steps <= 24:
        raise ValueError("sampling_steps должны быть в диапазоне 1..24")
    if not 0.0 <= attack_ms <= 500.0:
        raise ValueError("attack_ms должны быть в диапазоне 0..500")
    if mask_profile not in {"conservative", "tiered-mid", "tiered-event"}:
        raise ValueError("Неизвестный mask_profile")
    if not 2 <= fine_resample_period <= 16:
        raise ValueError("fine_resample_period должен быть в диапазоне 2..16")

    try:
        import audiotools as at
        from vampnet.interface import Interface
    except ImportError as error:
        raise RuntimeError(
            "VampNet зависимости не найдены. Запускайте через "
            "artifacts\\vampnet_env\\Scripts\\python.exe"
        ) from error

    reference = load_reference_mono(reference_path)
    prepared, original_loudness = prepare_reference_on_gpu(reference)

    print("[+] Загрузка codec + coarse + coarse-to-fine на GPU...", flush=True)
    load_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    interface = Interface(
        coarse_ckpt=str(model_dir / "coarse.pth"),
        coarse2fine_ckpt=str(model_dir / "c2f.pth"),
        codec_ckpt=str(model_dir / "codec.pth"),
        wavebeat_ckpt=None,
        device="cuda",
        compile=False,
    )
    coarse_lora_report = None
    if coarse_lora is not None:
        coarse_lora_report = load_coarse_lora(interface.coarse, coarse_lora)
        print(
            f"[+] Task-aligned coarse LoRA загружена: {coarse_lora.name}",
            flush=True,
        )
    interface.eval().requires_grad_(False)
    print(f"[+] Модели загружены за {time.perf_counter() - load_started:.1f} с", flush=True)

    with torch.inference_mode():
        encoded = interface.codec.encode(prepared, SAMPLE_RATE)
        codes = encoded["codes"]
        codec_audio = interface.codec.decode(encoded["z"], encoded["length"])["audio"]
    codec_roundtrip = restore_reference_loudness(
        codec_audio,
        frames=reference.size,
        original_loudness=original_loudness,
    )

    attack_tokens = int(np.ceil(attack_ms * SAMPLE_RATE / 1000.0 / interface.codec.hop_length))
    results_dir.mkdir(parents=True, exist_ok=True)
    sf.write(results_dir / "reference_mono_44100.wav", reference, SAMPLE_RATE, subtype="PCM_24")
    sf.write(results_dir / "codec_roundtrip.wav", codec_roundtrip, SAMPLE_RATE, subtype="PCM_24")

    rows: list[dict[str, object]] = []
    generated_waveforms: list[np.ndarray] = []
    token_archives: dict[str, np.ndarray] = {"reference_codes": codes.detach().cpu().numpy()}
    for index, seed in enumerate(seeds, start=1):
        at.util.seed(seed)
        offset = seed % periodic_prompt if periodic_prompt else 0
        if mask_profile in {"tiered-mid", "tiered-event"}:
            mask_np = build_tiered_reference_mask(
                tuple(codes.shape),
                coarse_start=1 if mask_profile == "tiered-event" else 2,
                coarse_stop=4,
                coarse_anchor_period=periodic_prompt,
                coarse_anchor_offset=offset,
                fine_start=4,
                fine_resample_period=fine_resample_period,
                fine_resample_offset=seed % fine_resample_period,
                attack_tokens=attack_tokens,
            )
        else:
            mask_np = build_reference_mask(
                tuple(codes.shape),
                upper_codebook_mask=upper_codebook_mask,
                periodic_prompt=periodic_prompt,
                periodic_offset=offset,
                attack_tokens=attack_tokens,
            )
        mask = torch.from_numpy(mask_np).to(device=codes.device, dtype=torch.long)
        print(
            f"[+] variation {index}/{len(seeds)}, seed={seed}: "
            f"masked={mask_np.mean():.3f}, attack anchors={attack_tokens} tokens...",
            flush=True,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output_codes = interface.vamp(
                codes,
                mask,
                batch_size=1,
                feedback_steps=1,
                temperature=temperature,
                typical_filtering=False,
                _sampling_steps=sampling_steps,
                seed=seed,
            )
            output_signal = interface.decode(output_codes)
        torch.cuda.synchronize()
        waveform = restore_reference_loudness(
            output_signal.audio_data,
            frames=reference.size,
            original_loudness=original_loudness,
        )
        passed, failures = technical_audio_gate(reference, waveform)
        metrics = comparison_metrics(reference, waveform)
        output_path = results_dir / f"variation_{index:02d}_seed_{seed}.wav"
        sf.write(output_path, waveform, SAMPLE_RATE, subtype="PCM_24")

        output_codes_np = output_codes.detach().cpu().numpy()
        changed = output_codes_np != token_archives["reference_codes"]
        changed_per_codebook = changed.mean(axis=(0, 2)).tolist()
        token_archives[f"variation_{index:02d}_codes"] = output_codes_np
        generated_waveforms.append(waveform)
        rows.append(
            {
                "file": output_path.name,
                "seed": seed,
                "seconds": float(time.perf_counter() - started),
                "technical_gate_passed": passed,
                "technical_gate_failures": failures,
                "masked_token_fraction": float(mask_np.mean()),
                "changed_token_fraction": float(changed.mean()),
                "changed_fraction_per_codebook": changed_per_codebook,
                "reference_metrics_diagnostic_only": metrics,
                "codec_relative_metrics_diagnostic_only": codec_relative_metrics(
                    codec_roundtrip,
                    waveform,
                ),
            }
        )
        print(
            f"    done in {rows[-1]['seconds']:.1f} с; "
            f"tokens changed={rows[-1]['changed_token_fraction']:.3f}; "
            f"envelope corr={metrics['envelope_correlation']:.4f}; "
            f"gate={'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    pairwise: list[dict[str, object]] = []
    for left_index in range(len(generated_waveforms)):
        for right_index in range(left_index + 1, len(generated_waveforms)):
            pairwise.append(
                {
                    "left": rows[left_index]["file"],
                    "right": rows[right_index]["file"],
                    "diagnostic_metrics": comparison_metrics(
                        generated_waveforms[left_index],
                        generated_waveforms[right_index],
                    ),
                }
            )

    np.savez_compressed(results_dir / "token_diagnostics.npz", **token_archives)
    write_envelope_overview(
        results_dir / "envelope_overview.png",
        reference=reference,
        variations=generated_waveforms,
    )
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / (1024**2))
    metadata = {
        "stage": "prompt_free_masked_acoustic_token_variations",
        "source": str(reference_path.resolve()),
        "source_commits": {
            "vampnet": VAMPNET_SOURCE_COMMIT,
            "lac": LAC_SOURCE_COMMIT,
        },
        "model_assets": assets,
        "gpu": gpu,
        "peak_vram_mib": peak_vram_mib,
        "reference": serializable_description(reference, SAMPLE_RATE),
        "codec_roundtrip_metrics_diagnostic_only": comparison_metrics(
            reference, codec_roundtrip
        ),
        "configuration": {
            "prompt": None,
            "coarse_lora": coarse_lora_report,
            "mask_profile": mask_profile,
            "seeds": seeds,
            "upper_codebook_mask": upper_codebook_mask,
            "periodic_prompt": periodic_prompt,
            "attack_ms": attack_ms,
            "attack_tokens": attack_tokens,
            "temperature": temperature,
            "sampling_steps": sampling_steps,
            "fine_resample_period": fine_resample_period,
            "codec_input_normalization": "official mono -24 integrated LUFS on GPU",
        },
        "variations": rows,
        "pairwise_variation_metrics_diagnostic_only": pairwise,
        "requires_human_listening": True,
        "known_limitations": [
            "mono prototype",
            "pretrained weights were not trained specifically on the project's SFX set",
            "objective metrics do not prove perceptual identity, diversity, or naturalness",
        ],
    }
    metadata_path = results_dir / "generation_report.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    demo_path = write_supervisor_demo(results_dir, metadata)
    print(f"[+] Peak VRAM: {peak_vram_mib:.0f} MiB", flush=True)
    print(f"[+] Результаты: {results_dir.resolve()}", flush=True)
    print(f"[+] Demo: {demo_path.resolve()}", flush=True)
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt-free вариации SFX через masked acoustic tokens"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--codec-gate", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--allow-unsafe-vram", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--upper-codebook-mask", type=int, default=3)
    parser.add_argument("--periodic-prompt", type=int, default=7)
    parser.add_argument("--attack-ms", type=float, default=80.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--sampling-steps", type=int, default=12)
    parser.add_argument(
        "--mask-profile",
        choices=("conservative", "tiered-mid", "tiered-event"),
        default="conservative",
    )
    parser.add_argument("--fine-resample-period", type=int, default=4)
    parser.add_argument(
        "--coarse-lora",
        type=Path,
        help="Optional task-aligned coarse LoRA checkpoint",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.preflight_only:
            required = ("codec.pth",)
            validate_model_assets(arguments.model_dir, required=required)
            require_safe_gpu(allow_unsafe_vram=arguments.allow_unsafe_vram)
            print("[+] VampNet codec preflight: OK; модель не загружалась")
            return 0
        if arguments.reference is None or arguments.results_dir is None:
            raise ValueError("Для запуска нужны --reference и --results-dir")
        if arguments.codec_gate:
            run_codec_gate(
                reference_path=arguments.reference,
                model_dir=arguments.model_dir,
                results_dir=arguments.results_dir,
                allow_unsafe_vram=arguments.allow_unsafe_vram,
            )
        else:
            run_generation(
                reference_path=arguments.reference,
                model_dir=arguments.model_dir,
                results_dir=arguments.results_dir,
                seeds=arguments.seeds,
                upper_codebook_mask=arguments.upper_codebook_mask,
                periodic_prompt=arguments.periodic_prompt,
                attack_ms=arguments.attack_ms,
                temperature=arguments.temperature,
                sampling_steps=arguments.sampling_steps,
                mask_profile=arguments.mask_profile,
                fine_resample_period=arguments.fine_resample_period,
                coarse_lora=arguments.coarse_lora,
                allow_unsafe_vram=arguments.allow_unsafe_vram,
            )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] VampNet prototype blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
