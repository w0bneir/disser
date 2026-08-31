"""Causal test of VampNet fine-codebook reconciliation on one fixed event body."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from run_vampnet_reference_variations import (
    prepare_reference_on_gpu,
    require_safe_gpu,
    restore_reference_loudness,
)
from sfx_metrics import compare_to_reference
from vampnet_reference_variations import (
    SAMPLE_RATE,
    build_fine_reconciliation_mask,
    comparison_metrics,
    load_reference_mono,
    technical_audio_gate,
    validate_model_assets,
)


RECIPES = (
    {
        "name": "coarse_original_fine",
        "label": "Изменённые CB1–3 + исходные fine CB4–13",
        "period": None,
        "steps": 0,
    },
    {
        "name": "current_sparse_p4_s2",
        "label": "Текущая схема: 1/4 fine-token, 2 шага",
        "period": 4,
        "steps": 2,
    },
    {
        "name": "sparse_p8_s6",
        "label": "Редкое согласование: 1/8 fine-token, 6 шагов",
        "period": 8,
        "steps": 6,
    },
    {
        "name": "sparse_p4_s6",
        "label": "Среднее согласование: 1/4 fine-token, 6 шагов",
        "period": 4,
        "steps": 6,
    },
    {
        "name": "sparse_p2_s6",
        "label": "Плотное согласование: 1/2 fine-token, 6 шагов",
        "period": 2,
        "steps": 6,
    },
    {
        "name": "full_fine_s6",
        "label": "Полное согласование fine CB4–13, 6 шагов",
        "period": 1,
        "steps": 6,
    },
)


def _write_page(results_dir: Path, rows: list[dict[str, object]]) -> Path:
    cards = []
    for row in rows:
        cards.append(
            "<section><h2>"
            + html.escape(str(row["label"]))
            + "</h2><audio controls preload=\"metadata\" src=\""
            + html.escape(str(row["file"]))
            + "\"></audio></section>"
        )
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VampNet fine reconciliation</title>
<style>body{{font:17px/1.45 system-ui;max-width:920px;margin:35px auto;padding:0 20px;background:#f3f4f6}}
section{{background:white;padding:16px 20px;margin:14px 0;border-radius:12px}}audio{{width:100%}}
.note{{border-left:5px solid #4263eb;padding:12px 16px;background:white}}</style></head><body>
<h1>Причинный тест согласования fine-codebook</h1>
<p class="note">Во всех шести файлах тело события CB0–3 абсолютно одинаковое. Меняется только
способ согласования верхних CB4–13. Поэтому различия в металлическом налёте вызваны fine-этапом,
а не новой генерацией выстрела.</p>
{''.join(cards)}
<h2>Что сообщить</h2><ol><li>Какой первый вариант перестал звучать металлически?</li>
<li>В каком варианте полезное отличие от codec-контроля ещё слышно?</li>
<li>Не появился ли новый артефакт при полном согласовании?</li></ol></body></html>"""
    path = results_dir / "fine_reconciliation.html"
    path.write_text(page, encoding="utf-8")
    return path


def _torch_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    return compare_to_reference(
        torch.from_numpy(np.asarray(reference, dtype=np.float32)),
        torch.from_numpy(np.asarray(candidate, dtype=np.float32)),
        SAMPLE_RATE,
    )


def run_experiment(
    *,
    token_diagnostics: Path,
    reference_path: Path,
    model_dir: Path,
    results_dir: Path,
    seed: int,
    attack_ms: float,
) -> Path:
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(f"Каталог результата не пуст: {results_dir}")
    assets = validate_model_assets(
        model_dir,
        required=("codec.pth", "coarse.pth", "c2f.pth"),
    )
    gpu = require_safe_gpu()
    archive = np.load(token_diagnostics)
    if "reference_codes" not in archive or "variation_01_codes" not in archive:
        raise ValueError("В token diagnostics нет reference_codes/variation_01_codes")
    reference_codes = archive["reference_codes"]
    variation_codes = archive["variation_01_codes"]
    if reference_codes.shape != variation_codes.shape or reference_codes.shape[1] != 14:
        raise ValueError("Ожидались одинаковые 14-codebook token tensors")

    # Fix the event body once. Every recipe below gets these exact CB0-3.
    event_codes = reference_codes.copy()
    event_codes[:, :4, :] = variation_codes[:, :4, :]

    try:
        import audiotools as at
        from vampnet.interface import Interface
    except ImportError as error:
        raise RuntimeError(
            "Запускайте через artifacts\\vampnet_env\\Scripts\\python.exe"
        ) from error

    reference = load_reference_mono(reference_path)
    _, original_loudness = prepare_reference_on_gpu(reference)
    print("[+] Загрузка codec + VampNet для изолированного fine-теста...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    interface = Interface(
        coarse_ckpt=str(model_dir / "coarse.pth"),
        coarse2fine_ckpt=str(model_dir / "c2f.pth"),
        codec_ckpt=str(model_dir / "codec.pth"),
        wavebeat_ckpt=None,
        device="cuda",
        compile=False,
    )
    interface.eval().requires_grad_(False)
    attack_tokens = int(
        np.ceil(attack_ms * SAMPLE_RATE / 1000.0 / interface.codec.hop_length)
    )
    device = torch.device("cuda")
    reference_codes_tensor = torch.from_numpy(reference_codes).to(device=device, dtype=torch.long)
    event_codes_tensor = torch.from_numpy(event_codes).to(device=device, dtype=torch.long)

    def decode(codes: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            signal = interface.decode(codes).audio_data
        return restore_reference_loudness(
            signal,
            frames=reference.size,
            original_loudness=original_loudness,
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    sf.write(results_dir / "reference_mono_44100.wav", reference, SAMPLE_RATE, subtype="PCM_24")
    codec_reference = decode(reference_codes_tensor)
    sf.write(results_dir / "00_codec_reference.wav", codec_reference, SAMPLE_RATE, subtype="PCM_24")

    rows: list[dict[str, object]] = [
        {
            "name": "codec_reference",
            "label": "Codec reference — контроль, не вариация",
            "file": "00_codec_reference.wav",
            "fine_mask_fraction": 0.0,
            "fine_sampling_steps": 0,
            "metrics_vs_codec": _torch_metrics(codec_reference, codec_reference),
        }
    ]
    token_outputs: dict[str, np.ndarray] = {
        "reference_codes": reference_codes,
        "fixed_event_codes": event_codes,
    }
    for index, recipe in enumerate(RECIPES, start=1):
        started = time.perf_counter()
        period = recipe["period"]
        if period is None:
            output_codes = event_codes_tensor.clone()
            mask_fraction = 0.0
        else:
            mask_np = build_fine_reconciliation_mask(
                tuple(event_codes.shape),
                fine_start=4,
                resample_period=int(period),
                resample_offset=seed % int(period),
                attack_tokens=attack_tokens,
            )
            mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.long)
            at.util.seed(seed)
            with torch.inference_mode():
                output_codes = interface.coarse_to_fine(
                    event_codes_tensor,
                    mask=mask,
                    typical_filtering=True,
                    _sampling_steps=int(recipe["steps"]),
                    temperature=0.9,
                    seed=seed,
                )
            mask_fraction = float(mask_np[:, 4:, :].mean())
        waveform = decode(output_codes)
        passed, failures = technical_audio_gate(reference, waveform)
        filename = f"{index:02d}_{recipe['name']}.wav"
        sf.write(results_dir / filename, waveform, SAMPLE_RATE, subtype="PCM_24")
        output_codes_np = output_codes.detach().cpu().numpy()
        token_outputs[str(recipe["name"])] = output_codes_np
        rows.append(
            {
                "name": recipe["name"],
                "label": recipe["label"],
                "file": filename,
                "seconds": float(time.perf_counter() - started),
                "fine_mask_fraction": mask_fraction,
                "fine_sampling_steps": int(recipe["steps"]),
                "technical_gate_passed": passed,
                "technical_gate_failures": failures,
                "changed_fraction_per_codebook": (
                    output_codes_np != reference_codes
                ).mean(axis=(0, 2)).tolist(),
                "metrics_vs_raw_reference_diagnostic_only": comparison_metrics(
                    reference, waveform
                ),
                "metrics_vs_codec": _torch_metrics(codec_reference, waveform),
            }
        )
        print(
            f"    {recipe['name']}: {rows[-1]['seconds']:.2f} с; "
            f"fine mask={mask_fraction:.3f}; "
            f"env={rows[-1]['metrics_vs_codec']['envelope_pearson']:.4f}",
            flush=True,
        )

    np.savez_compressed(results_dir / "fine_reconciliation_tokens.npz", **token_outputs)
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / (1024**2))
    report = {
        "stage": "vampnet_fixed_event_fine_reconciliation",
        "hypothesis": (
            "metallic colour is caused by disagreement between changed CB1-3 event tokens "
            "and stale or under-sampled CB4-13 fine tokens"
        ),
        "token_diagnostics": str(token_diagnostics.resolve()),
        "source": str(reference_path.resolve()),
        "seed": seed,
        "attack_ms": attack_ms,
        "attack_tokens": attack_tokens,
        "model_assets": assets,
        "gpu": gpu,
        "peak_vram_mib": peak_vram_mib,
        "items": rows,
        "requires_human_listening": True,
    }
    report_path = results_dir / "fine_reconciliation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    page = _write_page(results_dir, rows)
    print(f"[+] Fine reconciliation: {page.resolve()}", flush=True)
    print(f"[+] Peak VRAM: {peak_vram_mib:.0f} MiB", flush=True)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VampNet fixed-event fine reconciliation")
    parser.add_argument("--token-diagnostics", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/vampnet_models"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--attack-ms", type=float, default=80.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_experiment(
            token_diagnostics=args.token_diagnostics,
            reference_path=args.reference,
            model_dir=args.model_dir,
            results_dir=args.results_dir,
            seed=args.seed,
            attack_ms=args.attack_ms,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] Fine reconciliation blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
