"""Build the first blind draft of natural-statistics-calibrated SFX synthesis."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from perceptual_variation_synthesis import (
    METHOD_VERSION,
    common_peak_safe,
    fit_natural_variation_profile,
    profile_distance,
    synthesize_variation,
    waveform_correlation,
)
from sfx_pool_optimizer import analyze_directory, assemble_sequence, rms_match_sequences


PROTOCOL = "perceptual_variation_draft_v0"
PEAK_LIMIT = 10.0 ** (-1.0 / 20.0)
STRENGTHS = {"low": 0.55, "mid": 0.75, "high": 0.95}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64))) + 1e-12))


def _direct_fields() -> str:
    return """
      <label>A и B — одинаковые или разные?<select data-field="same_or_different"><option value="">—</option><option value="same">одинаковые</option><option value="different">разные</option></select></label>
      <label>Остаётся ли это тем же событием?<select data-field="same_event"><option value="">—</option><option value="yes">да</option><option value="uncertain">не уверен</option><option value="no">нет</option></select></label>
      <label>Если различие слышно — полезно ли оно?<select data-field="useful_difference"><option value="">—</option><option value="not_applicable">не различаю</option><option value="none">нет</option><option value="slight">слегка</option><option value="clear">явно</option></select></label>
      <label>Какой вариант звучит естественнее?<select data-field="more_natural"><option value="">—</option><option value="A">A</option><option value="same">одинаково</option><option value="B">B</option></select></label>
      <label>Где слышнее артефакты?<select data-field="artifacts"><option value="">—</option><option value="A">A</option><option value="neither">нигде</option><option value="B">B</option><option value="both">в обоих</option></select></label>
      <label>Уверенность 1–5<select data-field="confidence"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label>
    """


def _loop_fields() -> str:
    return """
      <label>Где меньше «пулемётный эффект»?<select data-field="less_repetitive"><option value="">—</option><option value="A">A</option><option value="same">одинаково</option><option value="B">B</option></select></label>
      <label>Где выше естественность?<select data-field="more_natural"><option value="">—</option><option value="A">A</option><option value="same">одинаково</option><option value="B">B</option></select></label>
      <label>Где слышнее артефакты?<select data-field="artifacts"><option value="">—</option><option value="A">A</option><option value="neither">нигде</option><option value="B">B</option><option value="both">в обоих</option></select></label>
      <label>Какую серию выбрать для игры?<select data-field="preferred"><option value="">—</option><option value="A">A</option><option value="same">без предпочтения</option><option value="B">B</option></select></label>
      <label>Уверенность 1–5<select data-field="confidence"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label>
    """


def _experiment_html(direct_pairs: dict[str, dict[str, str]], loop_pair: dict[str, str]) -> str:
    cards = []
    for pair_id, sides in direct_pairs.items():
        cards.append(
            f"""<section class="card" data-kind="direct" data-id="{pair_id}" data-a="{sides['A']}" data-b="{sides['B']}">
            <h2>{pair_id}: одиночное событие</h2><p>Быстро переключай A/B. Не пытайся определить, какой файл исходный.</p>
            <div class="players"><div><h3>A</h3><audio controls preload="metadata" src="{sides['A']}.wav"></audio></div><div><h3>B</h3><audio controls preload="metadata" src="{sides['B']}.wav"></audio></div></div>
            <div class="questions">{_direct_fields()}</div></section>"""
        )
    cards.append(
        f"""<section class="card" data-kind="loop" data-id="L01" data-a="{loop_pair['A']}" data-b="{loop_pair['B']}">
        <h2>L01: игровая последовательность</h2><p>Прослушай обе серии целиком и сравни утомляющую повторяемость.</p>
        <div class="players"><div><h3>A</h3><audio controls preload="metadata" src="{loop_pair['A']}.wav"></audio></div><div><h3>B</h3><audio controls preload="metadata" src="{loop_pair['B']}.wav"></audio></div></div>
        <div class="questions">{_loop_fields()}</div></section>"""
    )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Черновик синтеза вариаций SFX</title>
<style>:root{{--bg:#0b1016;--panel:#161e28;--line:#304052;--text:#eff5fa;--muted:#a9bacb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}main{{max-width:980px;margin:auto;padding:30px 18px 70px}}.warning{{background:#261f12;border-left:4px solid #f1b954;padding:15px;border-radius:6px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;margin:18px 0}}.players{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}audio{{width:100%}}.questions{{display:grid;gap:12px;margin-top:18px}}label{{display:grid;grid-template-columns:1fr 190px;gap:12px;align-items:center}}select,textarea{{background:#0e151d;color:var(--text);border:1px solid #43566b;border-radius:6px;padding:8px}}textarea{{grid-column:1/-1;width:100%}}button{{background:#1489c1;color:white;border:0;border-radius:8px;padding:12px 18px;font-weight:700;cursor:pointer}}#status{{color:#a9e3a4;margin-left:12px}}@media(max-width:700px){{.players{{grid-template-columns:1fr}}label{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Черновик перцептивно ограниченного синтеза</h1><p>Пять слепых сравнений одиночных событий и одна игровая последовательность. Среди пар есть точная копия и натуральный контроль.</p>
<p class="warning">Не открывай private-каталог до сохранения ответов. Порядок карточек и стороны A/B меняются при каждой загрузке — после начала не обновляй страницу.</p>
<div id="cards">{''.join(cards)}</div><button id="save">Скачать ответы JSON</button><span id="status"></span>
<script>const sessionId=(globalThis.crypto&&typeof globalThis.crypto.randomUUID==='function')?globalThis.crypto.randomUUID():`local-${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;const cards=Array.from(document.querySelectorAll('.card'));cards.forEach(card=>{{if(Math.random()<.5){{const a=card.dataset.a;card.dataset.a=card.dataset.b;card.dataset.b=a;}}const players=card.querySelectorAll('audio');players[0].src=`${{card.dataset.a}}.wav`;players[1].src=`${{card.dataset.b}}.wav`;}});for(let i=cards.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[cards[i],cards[j]]=[cards[j],cards[i]];}}const container=document.getElementById('cards');cards.forEach(c=>container.appendChild(c));document.getElementById('save').addEventListener('click',()=>{{const answers={{protocol:'{PROTOCOL}',session_id:sessionId,saved_at:new Date().toISOString(),direct:{{}},loops:{{}}}};let missing=0;cards.forEach(card=>{{const row={{blind_A:card.dataset.a,blind_B:card.dataset.b}};card.querySelectorAll('[data-field]').forEach(input=>{{row[input.dataset.field]=input.value;if(input.tagName==='SELECT'&&!input.value)missing++;}});answers[card.dataset.kind==='direct'?'direct':'loops'][card.dataset.id]=row;}});if(missing&&!confirm(`Не заполнено полей: ${{missing}}. Всё равно сохранить?`))return;const blob=new Blob([JSON.stringify(answers,null,2)],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='perceptual_variation_ratings.json';link.click();URL.revokeObjectURL(link.href);document.getElementById('status').textContent='Ответы сохранены';}});</script></main></body></html>"""


def verify_package(directory: Path) -> dict[str, object]:
    root = Path(directory)
    experiment = root / "experiment"
    private = root / "private_do_not_open_before_scoring"
    failures: list[str] = []
    required = [root / "run_manifest.json", experiment / "blind_test.html", experiment / "manifest_public.json", private / "blind_key.json"]
    for path in required:
        if not path.is_file():
            failures.append(f"missing: {path}")
    if failures:
        return {"passed": False, "failures": failures}
    public = json.loads((experiment / "manifest_public.json").read_text(encoding="utf-8"))
    key = json.loads((private / "blind_key.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    expected = set(public.get("blind_ids", []))
    actual = {path.stem for path in experiment.glob("*.wav")}
    if public.get("protocol") != PROTOCOL or len(expected) != 12 or actual != expected:
        failures.append("public protocol or twelve-file inventory mismatch")
    if set(key.get("blind_mapping", {})) != expected:
        failures.append("private/public blind IDs differ")
    html_text = (experiment / "blind_test.html").read_text(encoding="utf-8")
    for forbidden in ("exact_copy", "synthetic_low", "synthetic_mid", "synthetic_high", "natural_ceiling", "repeat_reference", "synthetic_cycle", "SHOT "):
        if forbidden in html_text:
            failures.append(f"blind leakage: {forbidden}")
    hashes = manifest.get("experiment", {}).get("blind_audio_hashes", {})
    direct_ids = {code for pair in public.get("direct_pairs", {}).values() for code in pair.values()}
    loop_ids = set(public.get("loop_pair", {}).values())
    direct_shapes: set[tuple[int, int]] = set()
    loop_shapes: set[tuple[int, int]] = set()
    sample_rates: set[int] = set()
    for code in expected:
        path = experiment / f"{code}.wav"
        if not path.is_file():
            continue
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
        sample_rates.add(int(sample_rate))
        (direct_shapes if code in direct_ids else loop_shapes).add(tuple(audio.shape))
        if code not in direct_ids and code not in loop_ids:
            failures.append(f"unassigned blind file: {code}")
        if info.subtype != "PCM_16":
            failures.append(f"not PCM_16: {code}")
        if float(np.max(np.abs(audio))) > PEAK_LIMIT + 1e-5:
            failures.append(f"peak above -1 dBFS: {code}")
        if _sha256(path) != hashes.get(code):
            failures.append(f"hash mismatch: {code}")
    if len(sample_rates) != 1 or len(direct_shapes) != 1 or len(loop_shapes) != 1:
        failures.append("inconsistent stimulus format")
    same_codes = [code for code, value in key.get("blind_mapping", {}).items() if str(value).startswith("exact_copy_")]
    if len(same_codes) != 2 or hashes.get(same_codes[0]) != hashes.get(same_codes[1]):
        failures.append("exact-copy control is not sample-identical")
    return {
        "passed": not failures,
        "failures": failures,
        "blind_files": len(actual),
        "sample_rate": next(iter(sample_rates)) if len(sample_rates) == 1 else None,
        "direct_shape": list(next(iter(direct_shapes))) if len(direct_shapes) == 1 else None,
        "loop_shape": list(next(iter(loop_shapes))) if len(loop_shapes) == 1 else None,
    }


def build_package(
    input_dir: Path,
    results_dir: Path,
    *,
    group: str = "1",
    events: int = 8,
    interval_ms: float = 1200.0,
    seed: int = 31_082_026,
) -> dict[str, object]:
    target = Path(results_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Results directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        experiment = staging / "experiment"
        private = staging / "private_do_not_open_before_scoring"
        experiment.mkdir()
        private.mkdir()
        clips = [clip for clip in analyze_directory(Path(input_dir)) if clip.group == str(group)]
        if len(clips) < 4:
            raise ValueError("Draft requires at least four natural takes in one group")
        sample_rates = {clip.sample_rate for clip in clips}
        prepared_shapes = {clip.prepared.shape for clip in clips}
        if len(sample_rates) != 1 or len(prepared_shapes) != 1:
            raise ValueError("Prepared natural takes must share sample rate and shape")
        sample_rate = clips[0].sample_rate
        bank = [clip.prepared for clip in clips]
        names = [clip.metrics.name for clip in clips]
        profile = fit_natural_variation_profile(bank, sample_rate, names=names)
        reference_index = profile.reference_index
        reference = bank[reference_index]
        donor_candidates = [index for index in range(len(bank)) if index != reference_index]
        donor_index = max(donor_candidates, key=lambda index: profile_distance(reference, bank[index], profile))
        donor = bank[donor_index]
        synthesized: dict[str, np.ndarray] = {}
        candidate_metrics: dict[str, object] = {}
        for label, strength in STRENGTHS.items():
            audio, transform = synthesize_variation(reference, donor, sample_rate, strength=strength)
            synthesized[label] = audio
            candidate_metrics[label] = {
                "strength": strength,
                "profile_distance_from_reference": profile_distance(reference, audio, profile),
                "waveform_correlation": waveform_correlation(reference, audio),
                "rms_delta_db": 20.0 * np.log10(_rms(audio) / max(_rms(reference), 1e-12)),
                "transform": transform.json_dict(),
            }
        scaled = common_peak_safe([reference, donor, synthesized["low"], synthesized["mid"], synthesized["high"]])
        reference_safe, donor_safe, low_safe, mid_safe, high_safe = scaled
        assets = {
            "reference": reference_safe,
            "natural_donor": donor_safe,
            "synthetic_low": low_safe,
            "synthetic_mid": mid_safe,
            "synthetic_high": high_safe,
        }
        direct_methods = {
            "exact_copy": ("reference", "reference"),
            "synthetic_low": ("reference", "synthetic_low"),
            "synthetic_mid": ("reference", "synthetic_mid"),
            "synthetic_high": ("reference", "synthetic_high"),
            "natural_ceiling": ("reference", "natural_donor"),
        }
        rng = np.random.default_rng(seed)
        pair_ids = [f"D{index:02d}" for index in range(1, 6)]
        comparison_names = list(direct_methods)
        rng.shuffle(comparison_names)
        direct_assignment = dict(zip(pair_ids, comparison_names))
        blind_codes = [f"X{index:02d}" for index in range(1, 13)]
        rng.shuffle(blind_codes)
        code_cursor = 0
        blind_audio: dict[str, np.ndarray] = {}
        blind_mapping: dict[str, str] = {}
        direct_pairs: dict[str, dict[str, str]] = {}
        for pair_id, comparison in direct_assignment.items():
            methods = direct_methods[comparison]
            codes = blind_codes[code_cursor : code_cursor + 2]
            code_cursor += 2
            direct_pairs[pair_id] = {"A": codes[0], "B": codes[1]}
            for side, code, method in zip(("A", "B"), codes, methods):
                blind_audio[code] = assets[method]
                blind_mapping[code] = f"{comparison}_{side}:{method}"
        event_bank = {0: low_safe, 1: mid_safe, 2: high_safe}
        repeat = assemble_sequence({0: reference_safe}, [0] * events, sample_rate=sample_rate, interval_ms=interval_ms)
        base_schedule = [0, 1, 2, 1, 0, 2, 0, 1]
        cycle_schedule = [base_schedule[index % len(base_schedule)] for index in range(events)]
        cycle = assemble_sequence(event_bank, cycle_schedule, sample_rate=sample_rate, interval_ms=interval_ms)
        matched_loops = rms_match_sequences(
            {"repeat": repeat, "cycle": cycle},
            anchor="repeat",
        )
        repeat, cycle = matched_loops["repeat"], matched_loops["cycle"]
        loop_codes = blind_codes[code_cursor : code_cursor + 2]
        loop_pair = {"A": loop_codes[0], "B": loop_codes[1]}
        blind_audio[loop_codes[0]] = repeat
        blind_audio[loop_codes[1]] = cycle
        blind_mapping[loop_codes[0]] = "repeat_reference"
        blind_mapping[loop_codes[1]] = "synthetic_cycle"
        for code, audio in blind_audio.items():
            sf.write(experiment / f"{code}.wav", audio, sample_rate, subtype="PCM_16")
        public = {
            "protocol": PROTOCOL,
            "blind_ids": sorted(blind_audio),
            "direct_pairs": direct_pairs,
            "loop_pair": loop_pair,
            "sample_rate": sample_rate,
            "channels": int(reference.shape[1]),
            "events": events,
            "interval_ms": interval_ms,
        }
        _json_dump(experiment / "manifest_public.json", public)
        (experiment / "blind_test.html").write_text(_experiment_html(direct_pairs, loop_pair), encoding="utf-8")
        private_key = {
            "warning": "Do not open before ratings are saved.",
            "protocol": PROTOCOL,
            "blind_mapping": blind_mapping,
            "direct_assignment": direct_assignment,
            "reference_name": names[reference_index],
            "natural_donor_name": names[donor_index],
            "loop_truth": {"repeat": loop_codes[0], "synthetic_cycle": loop_codes[1], "schedule": cycle_schedule},
        }
        _json_dump(private / "blind_key.json", private_key)
        source_files = [{"path": str(clip.path.resolve()), "sha256": _sha256(clip.path), "name": clip.metrics.name} for clip in clips]
        manifest = {
            "protocol": PROTOCOL,
            "method_version": METHOD_VERSION,
            "settings": {"group": str(group), "events": events, "interval_ms": interval_ms, "seed": seed, "strengths": STRENGTHS},
            "profile": profile.json_dict(),
            "selected": {"reference": names[reference_index], "natural_donor": names[donor_index]},
            "candidate_metrics": candidate_metrics,
            "source_files": source_files,
            "implementation_sha256": {
                "synthesis": _sha256(Path(__file__).with_name("perceptual_variation_synthesis.py")),
                "runner": _sha256(Path(__file__)),
            },
            "experiment": {
                "blind_audio_hashes": {code: _sha256(experiment / f"{code}.wav") for code in sorted(blind_audio)},
                "html_sha256": _sha256(experiment / "blind_test.html"),
                "public_manifest_sha256": _sha256(experiment / "manifest_public.json"),
            },
        }
        _json_dump(staging / "run_manifest.json", manifest)
        verification = verify_package(staging)
        _json_dump(staging / "verification_report.json", verification)
        if not verification["passed"]:
            raise RuntimeError("Package verification failed: " + "; ".join(verification["failures"]))
        staging.replace(target)
        return {"target": str(target), "profile": profile.json_dict(), "selected": manifest["selected"], "candidate_metrics": candidate_metrics, "verification": verification}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("references/group_1"))
    parser.add_argument("--group", default="1")
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--interval-ms", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=31_082_026)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_package(args.input_dir, args.results_dir, group=args.group, events=args.events, interval_ms=args.interval_ms, seed=args.seed)
    print(f"[+] Reference: {result['selected']['reference']}")
    print(f"[+] Natural donor: {result['selected']['natural_donor']}")
    for label, metrics in result["candidate_metrics"].items():
        print(f"[+] {label}: distance={metrics['profile_distance_from_reference']:.3f}; correlation={metrics['waveform_correlation']:.6f}")
    print(f"[+] Package verified: {result['verification']['passed']}")
    print(f"[+] Open: {Path(result['target']) / 'experiment' / 'blind_test.html'}")


if __name__ == "__main__":
    main()
