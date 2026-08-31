"""Generate the six pre-registered causal candidates for structured SFX Gate 1."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import soundfile as sf

from structured_resynthesis import (
    analyze_event_regions,
    diagnostic_metrics,
    generate_causal_variations,
    load_mono,
    technical_gate,
)


LABELS = {
    "A1_attack_mild": "A1 — атака, мягкая межполосная микровариация",
    "A2_attack_medium": "A2 — атака, средняя межполосная микровариация",
    "B1_body_mild": "B1 — тело, мягкий перенос спектральной микротекстуры",
    "B2_body_medium": "B2 — тело, средний перенос спектральной микротекстуры",
    "T1_tail_mild": "T1 — хвост, мягкая фазовая декорреляция",
    "T2_tail_medium": "T2 — хвост, средняя фазовая декорреляция",
}


def _write_listening_page(
    results_dir: Path,
    *,
    regions: dict[str, float | int],
    rows: list[dict[str, object]],
) -> Path:
    cards = []
    for row in rows:
        cards.append(
            f"<section><h2>{html.escape(str(row['label']))}</h2>"
            f"<audio controls preload=\"metadata\" src=\"{html.escape(str(row['file']))}\"></audio>"
            "<p>Идентичность __/5 · естественность __/5 · полезное отличие __/5 · "
            "дополнительное событие: да/нет</p></section>"
        )
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Structured SFX Gate 1</title>
<style>body{{font:17px/1.45 system-ui;max-width:940px;margin:35px auto;padding:0 20px;background:#f3f4f6}}
section{{background:white;padding:16px 20px;margin:14px 0;border-radius:12px}}audio{{width:100%}}
.note{{border-left:5px solid #4263eb;padding:12px 16px;background:white}}</style></head><body>
<h1>Gate 1 — причинные вариации одного выстрела</h1>
<p class="note">Каждая пара меняет только один компонент. Комбинированных обработок нет.
Сначала несколько раз прослушайте reference, затем сравнивайте внутри A, B и T.</p>
<section><h2>Reference</h2><audio controls preload="metadata" src="reference.wav"></audio></section>
<p>Автоматические границы: атака до {float(regions['attack_end_seconds'])*1000:.1f} мс;
тело до {float(regions['body_end_seconds'])*1000:.1f} мс; далее хвост.</p>
{''.join(cards)}
<h2>Stop/go</h2><p>Механизм проходит дальше, только если хотя бы один его вариант получает:
идентичность ≥4/5, естественность ≥4/5, полезное отличие ≥3/5 и не создаёт
дополнительного события. До этой оценки параметры не меняются.</p></body></html>"""
    path = results_dir / "gate1.html"
    path.write_text(page, encoding="utf-8")
    return path


def run_gate1(*, reference_path: Path, results_dir: Path) -> Path:
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(f"Каталог результата не пуст: {results_dir}")
    reference, sample_rate = load_mono(reference_path)
    regions = analyze_event_regions(reference, sample_rate)
    started = time.perf_counter()
    variations = generate_causal_variations(reference, sample_rate, regions)
    results_dir.mkdir(parents=True, exist_ok=True)
    sf.write(results_dir / "reference.wav", reference, sample_rate, subtype="PCM_24")

    rows: list[dict[str, object]] = []
    for name, audio in variations.items():
        passed, failures = technical_gate(reference, audio)
        metrics = diagnostic_metrics(reference, audio, sample_rate, regions)
        structure_gate = bool(
            passed
            and metrics["envelope_correlation"] >= 0.98
            and metrics["candidate_strong_peak_count"]
            == metrics["reference_strong_peak_count"]
        )
        filename = f"{name}.wav"
        sf.write(results_dir / filename, audio, sample_rate, subtype="PCM_24")
        rows.append(
            {
                "name": name,
                "label": LABELS[name],
                "file": filename,
                "technical_gate_passed": passed,
                "technical_gate_failures": failures,
                "structure_prescreen_passed": structure_gate,
                "diagnostic_metrics_not_a_perceptual_verdict": metrics,
            }
        )
        print(
            f"    {name}: env={metrics['envelope_correlation']:.4f}; "
            f"wave={metrics['waveform_correlation']:.4f}; "
            f"residual={metrics['copy_residual_db']:.1f} dB; "
            f"prescreen={'PASS' if structure_gate else 'REVIEW'}",
            flush=True,
        )

    report = {
        "stage": "structured_resynthesis_causal_gate1",
        "method_version": "structured_resynthesis_v2_gate1_preregistered",
        "source": str(reference_path.resolve()),
        "sample_rate": sample_rate,
        "frames": int(reference.size),
        "regions": regions.to_dict(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "candidates": rows,
        "decision_rule": {
            "identity_min": 4,
            "naturalness_min": 4,
            "useful_difference_min": 3,
            "extra_events_allowed": False,
            "no_parameter_change_before_listening": True,
        },
        "requires_human_listening": True,
    }
    report_path = results_dir / "gate1_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    page = _write_listening_page(results_dir, regions=report["regions"], rows=rows)
    print(f"[+] Gate 1: {page.resolve()}", flush=True)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured single-reference SFX Gate 1")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_gate1(reference_path=args.reference, results_dir=args.results_dir)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] Gate 1 blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
