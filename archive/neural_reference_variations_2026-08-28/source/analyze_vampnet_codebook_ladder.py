"""Декодировать причинную лестницу RVQ-codebook без повторной генерации."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from run_vampnet_reference_variations import (
    require_safe_gpu,
    restore_reference_loudness,
)
from vampnet_reference_variations import (
    SAMPLE_RATE,
    build_codebook_hybrids,
    comparison_metrics,
    load_reference_mono,
    serializable_description,
    technical_audio_gate,
    validate_model_assets,
)


LABELS = {
    "codec_reference": "Codec reference — ни один token не изменён",
    "cb1_only": "Только codebook 1",
    "cb1_2": "Codebook 1–2",
    "cb1_3": "Codebook 1–3, исходный fine detail",
    "cb2_3_only": "Только codebook 2–3",
    "fine_4_13_only": "Только разреженные fine codebook 4–13",
    "full_variation": "Полный tiered-event candidate",
}


def write_ladder_page(results_dir: Path, rows: list[dict[str, object]]) -> Path:
    cards = []
    for row in rows:
        filename = html.escape(str(row["file"]))
        label = html.escape(str(row["label"]))
        cards.append(
            f'<section><h2>{label}</h2><audio controls preload="metadata" '
            f'src="{filename}"></audio></section>'
        )
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VampNet codebook ladder</title>
<style>body{{font:17px/1.45 system-ui;max-width:900px;margin:35px auto;padding:0 20px;background:#f3f4f6}}
section{{background:white;padding:16px 20px;margin:14px 0;border-radius:12px}}audio{{width:100%}}</style>
</head><body><h1>Причинная лестница VampNet codebook</h1>
<p>Все файлы декодированы из одного и того же tiered-event результата. Меняется только набор
подставленных codebook-ов. Слушать сверху вниз и отмечать первый файл, где появляется полезное
отличие, и первый файл, где появляется металлический артефакт.</p>{''.join(cards)}</body></html>"""
    path = results_dir / "codebook_ladder.html"
    path.write_text(page, encoding="utf-8")
    return path


def run_ladder(
    *,
    token_diagnostics: Path,
    reference_path: Path,
    model_dir: Path,
    results_dir: Path,
) -> Path:
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(f"Каталог результата не пуст: {results_dir}")
    assets = validate_model_assets(model_dir, required=("codec.pth", "coarse.pth"))
    gpu = require_safe_gpu()
    archive = np.load(token_diagnostics)
    if "reference_codes" not in archive or "variation_01_codes" not in archive:
        raise ValueError("В token diagnostics нет reference_codes/variation_01_codes")
    hybrids = build_codebook_hybrids(
        archive["reference_codes"],
        archive["variation_01_codes"],
    )

    from lac.model.lac import LAC
    from vampnet.modules.transformer import VampNet

    print("[+] Загрузка codec + coarse decoder; генерация не выполняется...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    codec = LAC.load(model_dir / "codec.pth", map_location="cpu").eval().requires_grad_(False).to("cuda")
    coarse = VampNet.load(model_dir / "coarse.pth", map_location="cpu", strict=False)
    coarse.eval().requires_grad_(False).to("cuda")

    reference = load_reference_mono(reference_path)
    from run_vampnet_reference_variations import prepare_reference_on_gpu

    _, original_loudness = prepare_reference_on_gpu(reference)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    decoded: dict[str, np.ndarray] = {}
    for index, (name, codes_np) in enumerate(hybrids.items()):
        codes = torch.from_numpy(codes_np).to(device="cuda", dtype=torch.long)
        with torch.inference_mode():
            signal = coarse.decode(codes, codec).audio_data
        waveform = restore_reference_loudness(
            signal,
            frames=reference.size,
            original_loudness=original_loudness,
        )
        passed, failures = technical_audio_gate(reference, waveform)
        filename = f"{index:02d}_{name}.wav"
        sf.write(results_dir / filename, waveform, SAMPLE_RATE, subtype="PCM_24")
        decoded[name] = waveform
        rows.append(
            {
                "name": name,
                "label": LABELS[name],
                "file": filename,
                "technical_gate_passed": passed,
                "technical_gate_failures": failures,
            }
        )

    codec_reference = decoded["codec_reference"]
    for row in rows:
        row["metrics_vs_codec_reference_diagnostic_only"] = comparison_metrics(
            codec_reference,
            decoded[str(row["name"])],
        )
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / (1024**2))
    report = {
        "stage": "vampnet_codebook_causal_ladder",
        "token_diagnostics": str(token_diagnostics.resolve()),
        "source": str(reference_path.resolve()),
        "model_assets": assets,
        "gpu": gpu,
        "peak_vram_mib": peak_vram_mib,
        "reference": serializable_description(reference, SAMPLE_RATE),
        "items": rows,
        "requires_human_listening": True,
    }
    report_path = results_dir / "codebook_ladder.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    page = write_ladder_page(results_dir, rows)
    print(f"[+] Codebook ladder: {page.resolve()}", flush=True)
    print(f"[+] Peak VRAM: {peak_vram_mib:.0f} MiB", flush=True)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VampNet codebook causal ladder")
    parser.add_argument("--token-diagnostics", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/vampnet_models"))
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_ladder(
            token_diagnostics=args.token_diagnostics,
            reference_path=args.reference,
            model_dir=args.model_dir,
            results_dir=args.results_dir,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] Codebook ladder blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
