"""Build a deterministic blind A/B page for base versus SFX-adapted VampNet."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

from sfx_metrics import compare_to_reference, load_mono_audio


def build_comparison(
    *,
    reference_path: Path,
    base_result: Path,
    adapted_result: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Каталог A/B не пуст: {output_dir}")
    base_files = sorted(base_result.glob(f"variation_*_seed_{seed}.wav"))
    adapted_files = sorted(adapted_result.glob(f"variation_*_seed_{seed}.wav"))
    if len(base_files) != 1 or len(adapted_files) != 1:
        raise ValueError(f"Blind A/B не нашёл ровно одну variation для seed={seed}")
    codec_path = base_result / "codec_roundtrip.wav"
    if not reference_path.is_file() or not codec_path.is_file():
        raise FileNotFoundError("Не найден reference или codec control")

    # Deterministic blinding: labels do not expose model identity in the HTML.
    assignments = {
        "candidate_x.wav": base_files[0],
        "candidate_y.wav": adapted_files[0],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_path, output_dir / "reference.wav")
    shutil.copy2(codec_path, output_dir / "codec_control.wav")
    for destination, source in assignments.items():
        shutil.copy2(source, output_dir / destination)

    codec, sample_rate = load_mono_audio(output_dir / "codec_control.wav")
    metrics = {}
    for destination in assignments:
        candidate, candidate_rate = load_mono_audio(output_dir / destination)
        if candidate_rate != sample_rate:
            raise ValueError("A/B files имеют разную sample rate")
        metrics[destination] = compare_to_reference(codec, candidate, sample_rate)
    candidate_x, _ = load_mono_audio(output_dir / "candidate_x.wav")
    candidate_y, _ = load_mono_audio(output_dir / "candidate_y.wav")
    pairwise = compare_to_reference(candidate_x, candidate_y, sample_rate)

    key = {
        "stage": "vampnet_sfx_lora_blind_ab",
        "reference": str(reference_path.resolve()),
        "assignments": {
            destination: str(source.resolve()) for destination, source in assignments.items()
        },
        "metrics_vs_codec_control_diagnostic_only": metrics,
        "candidate_x_vs_y_diagnostic_only": pairwise,
        "requires_human_listening": True,
    }
    (output_dir / "comparison_key.json").write_text(
        json.dumps(key, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    page = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Blind VampNet SFX LoRA A/B</title>
<style>body{font:17px/1.45 system-ui;max-width:900px;margin:35px auto;padding:0 20px;background:#f3f4f6}
section{background:white;padding:16px 20px;margin:14px 0;border-radius:12px}audio{width:100%}
.note{border-left:5px solid #4263eb;padding:12px 16px;background:white}</style></head><body>
<h1>Слепое A/B: базовая и SFX-адаптированная модель</h1>
<p class="note">X и Y созданы с одинаковыми reference, seed, маской, температурой и числом шагов.
Единственная переменная — 50 шагов task-aligned SFX-LoRA. Не открывайте comparison_key.json до оценки.</p>
<section><h2>Reference</h2><audio controls src="reference.wav"></audio></section>
<section><h2>Codec control</h2><audio controls src="codec_control.wav"></audio></section>
<section><h2>Candidate X</h2><audio controls src="candidate_x.wav"></audio></section>
<section><h2>Candidate Y</h2><audio controls src="candidate_y.wav"></audio></section>
<h2>Три вопроса</h2><ol><li>У X или Y меньше металлического налёта?</li>
<li>У X или Y полезнее отличие от reference?</li><li>Оба ли остаются тем же одиночным выстрелом?</li></ol>
</body></html>"""
    page_path = output_dir / "blind_ab.html"
    page_path.write_text(page, encoding="utf-8")
    print(f"[+] Blind A/B: {page_path.resolve()}", flush=True)
    return page_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind base-vs-LoRA VampNet A/B")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--adapted-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        build_comparison(
            reference_path=args.reference,
            base_result=args.base_result,
            adapted_result=args.adapted_result,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, sf.LibsndfileError) as error:
        print(f"[!] Blind A/B blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
