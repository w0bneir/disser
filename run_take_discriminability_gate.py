"""Build a blind CPU-only gate for natural-take discriminability."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import html
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from sfx_pool_optimizer import (
    analyze_directory,
    assemble_sequence,
    build_groupwise_distance_matrices,
    discover_wav_files,
    group_index_map,
    medoid_index,
    names_for_schedule,
    normalize_prepared_bank,
    rms_match_sequences,
    shuffle_schedule,
    technical_audio_gate,
)


PROTOCOL = "take_discriminability_gate_v1"
PEAK_LIMIT = 10.0 ** (-1.0 / 20.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def select_discriminability_pairs(
    indices: list[int],
    distance_matrix: np.ndarray,
    *,
    medoid: int,
) -> dict[str, tuple[int, int]]:
    """Select a same control plus near, median and far distinct pairs."""

    if len(indices) < 3 or medoid not in indices:
        raise ValueError("Gate requires at least three takes and a medoid inside the group")
    distinct = sorted(
        (
            (float(distance_matrix[left, right]), int(left), int(right))
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    near = distinct[0]
    far = distinct[-1]
    target = float(np.median([item[0] for item in distinct]))
    middle_candidates = [item for item in distinct if item[1:] not in {near[1:], far[1:]}]
    middle = min(
        middle_candidates or distinct,
        key=lambda item: (abs(item[0] - target), item[1], item[2]),
    )
    return {
        "same_control": (int(medoid), int(medoid)),
        "near_pair": (near[1], near[2]),
        "median_pair": (middle[1], middle[2]),
        "far_pair": (far[1], far[2]),
    }


def _common_peak_scale(bank: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    peak = max(float(np.max(np.abs(audio))) for audio in bank.values())
    scale = min(1.0, PEAK_LIMIT / max(peak, 1e-12))
    return {index: (np.asarray(audio, dtype=np.float64) * scale).astype(np.float32) for index, audio in bank.items()}


def _alternate(pair: tuple[int, int], events: int) -> list[int]:
    return [pair[position % 2] for position in range(events)]


def _direct_fields() -> str:
    return """
      <label>Это одинаковый или другой дубль?
        <select data-field="same_or_different"><option value="">—</option><option value="same">одинаковый</option><option value="different">другой</option></select></label>
      <label>Если отличается — полезность различия
        <select data-field="useful_difference"><option value="">—</option><option value="not_applicable">не различаю</option><option value="none">неполезное</option><option value="slight">слегка полезное</option><option value="clear">явно полезное</option></select></label>
      <label>Остаётся ли это тем же событием и контекстом?
        <select data-field="same_event"><option value="">—</option><option value="yes">да</option><option value="uncertain">не уверен</option><option value="no">нет</option></select></label>
      <label>Уверенность <span>1 — почти наугад, 5 — уверен</span>
        <select data-field="confidence"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label>
    """


def _loop_fields() -> str:
    return """
      <label>Где меньше механическая повторяемость?
        <select data-field="less_repetitive"><option value="">—</option><option value="A">A</option><option value="same">различий нет</option><option value="B">B</option></select></label>
      <label>Где заметнее полезная вариативность между выстрелами?
        <select data-field="clearer_variation"><option value="">—</option><option value="A">A</option><option value="same">различий нет</option><option value="B">B</option></select></label>
      <label>Где выше естественность?
        <select data-field="more_natural"><option value="">—</option><option value="A">A</option><option value="same">различий нет</option><option value="B">B</option></select></label>
      <label>Какую серию выбрать для игры?
        <select data-field="preferred"><option value="">—</option><option value="A">A</option><option value="same">без предпочтения</option><option value="B">B</option></select></label>
      <label>Уверенность <span>1 — почти наугад, 5 — уверен</span>
        <select data-field="confidence"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label>
    """


def _gate_html(
    direct_pairs: dict[str, dict[str, str]],
    loop_pairs: dict[str, dict[str, str]],
    *,
    events: int,
    interval_ms: float,
) -> str:
    cards = []
    for pair_id, sides in direct_pairs.items():
        cards.append(
            f"""
            <section class="card" data-kind="direct" data-id="{pair_id}" data-a="{sides['A']}" data-b="{sides['B']}">
              <h2>{pair_id}: одиночные дубли</h2>
              <p>Прослушай A и B несколько раз с быстрым переключением.</p>
              <div class="players"><div><h3>A</h3><audio controls preload="metadata" src="{sides['A']}.wav"></audio></div><div><h3>B</h3><audio controls preload="metadata" src="{sides['B']}.wav"></audio></div></div>
              <div class="questions">{_direct_fields()}</div>
            </section>
            """
        )
    for pair_id, sides in loop_pairs.items():
        cards.append(
            f"""
            <section class="card" data-kind="loop" data-id="{pair_id}" data-a="{sides['A']}" data-b="{sides['B']}">
              <h2>{pair_id}: короткие циклы</h2>
              <p>Сравни серию A и B целиком, затем повтори наиболее спорный участок.</p>
              <div class="players"><div><h3>A</h3><audio controls preload="metadata" src="{sides['A']}.wav"></audio></div><div><h3>B</h3><audio controls preload="metadata" src="{sides['B']}.wav"></audio></div></div>
              <div class="questions">{_loop_fields()}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Gate различимости натуральных дублей</title>
<style>
:root{{--bg:#0b1016;--panel:#161e28;--line:#304052;--text:#eff5fa;--muted:#a9bacb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}main{{max-width:980px;margin:auto;padding:30px 18px 70px}}.warning{{background:#261f12;border-left:4px solid #f1b954;padding:15px;border-radius:6px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;margin:18px 0}}.players{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}audio{{width:100%}}.questions{{display:grid;gap:12px;margin-top:18px}}label{{display:grid;grid-template-columns:1fr 190px;gap:12px;align-items:center}}label span{{display:block;color:var(--muted);font-size:.8rem}}select,textarea{{background:#0e151d;color:var(--text);border:1px solid #43566b;border-radius:6px;padding:8px}}textarea{{grid-column:1/-1;width:100%}}button{{background:#1489c1;color:white;border:0;border-radius:8px;padding:12px 18px;font-weight:700;cursor:pointer}}#status{{color:#a9e3a4;margin-left:12px}}@media(max-width:700px){{.players{{grid-template-columns:1fr}}label{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Gate различимости натуральных дублей</h1>
<p>Сначала четыре пары одиночных выстрелов, затем три пары коротких серий по {events} событий с интервалом {interval_ms:g} мс. Среди одиночных пар может быть идентичный контроль.</p>
<p class="warning">Не открывай корень пакета и private-каталог до сохранения JSON. Имена, порядок карточек и стороны A/B слепые и меняются при загрузке.</p>
<div id="cards">{''.join(cards)}</div>
<button id="save">Скачать ответы JSON</button><span id="status"></span>
<script>
const sessionId=(globalThis.crypto&&typeof globalThis.crypto.randomUUID==='function')?globalThis.crypto.randomUUID():`local-${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
const cards=Array.from(document.querySelectorAll('.card'));
cards.forEach(card=>{{if(Math.random()<0.5){{const a=card.dataset.a;card.dataset.a=card.dataset.b;card.dataset.b=a;}}const players=card.querySelectorAll('audio');players[0].src=`${{card.dataset.a}}.wav`;players[1].src=`${{card.dataset.b}}.wav`;}});
for(let i=cards.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[cards[i],cards[j]]=[cards[j],cards[i]];}}
const container=document.getElementById('cards');cards.forEach(card=>container.appendChild(card));
document.getElementById('save').addEventListener('click',()=>{{const answers={{protocol:'{PROTOCOL}',session_id:sessionId,saved_at:new Date().toISOString(),direct:{{}},loops:{{}}}};let missing=0;cards.forEach(card=>{{const row={{blind_A:card.dataset.a,blind_B:card.dataset.b}};card.querySelectorAll('[data-field]').forEach(input=>{{row[input.dataset.field]=input.value;if(input.tagName==='SELECT'&&!input.value)missing++;}});answers[card.dataset.kind==='direct'?'direct':'loops'][card.dataset.id]=row;}});if(missing&&!confirm(`Не заполнено полей: ${{missing}}. Всё равно сохранить?`))return;const blob=new Blob([JSON.stringify(answers,null,2)],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='take_discriminability_ratings.json';link.click();URL.revokeObjectURL(link.href);document.getElementById('status').textContent='Ответы сохранены';}});
</script></main></body></html>"""


def verify_gate(directory: Path) -> dict[str, object]:
    root = directory.resolve()
    failures: list[str] = []
    experiment = root / "experiment"
    private = root / "private_do_not_open_before_scoring"
    required = [
        root / "run_manifest.json",
        experiment / "discriminability_gate.html",
        experiment / "manifest_public.json",
        private / "blind_key.json",
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"missing: {path}")
    if failures:
        return {"passed": False, "failures": failures}
    public = json.loads((experiment / "manifest_public.json").read_text(encoding="utf-8"))
    key = json.loads((private / "blind_key.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if public.get("protocol") != PROTOCOL:
        failures.append("unexpected public protocol")
    expected_ids = set(public.get("blind_ids", []))
    actual_wavs = {path.stem for path in experiment.glob("*.wav")}
    if len(expected_ids) != 12 or actual_wavs != expected_ids:
        failures.append("expected exactly twelve registered blind WAV files")
    if set(key.get("blind_mapping", {})) != expected_ids:
        failures.append("private/public blind IDs differ")
    public_text = (experiment / "discriminability_gate.html").read_text(encoding="utf-8")
    for forbidden in ("SHOT ", "same_control", "near_pair", "median_pair", "far_pair", "repeat_raw", "alternate_far_raw", "shuffle_raw", "shuffle_clip_matched"):
        if forbidden in public_text:
            failures.append(f"blind leakage: {forbidden}")
    hashes = manifest.get("experiment", {}).get("blind_audio_hashes", {})
    if set(hashes) != expected_ids:
        failures.append("blind audio hash inventory differs")
    direct_ids = {
        blind_id
        for sides in public.get("direct_pairs", {}).values()
        for blind_id in sides.values()
    }
    loop_ids = {
        blind_id
        for sides in public.get("loop_pairs", {}).values()
        for blind_id in sides.values()
    }
    direct_shapes: set[tuple[int, int]] = set()
    loop_shapes: set[tuple[int, int]] = set()
    sample_rates: set[int] = set()
    for blind_id in expected_ids:
        path = experiment / f"{blind_id}.wav"
        if not path.is_file():
            continue
        info = sf.info(path)
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        sample_rates.add(int(sample_rate))
        (direct_shapes if blind_id in direct_ids else loop_shapes).add(tuple(audio.shape))
        if info.subtype != "PCM_16":
            failures.append(f"not PCM_16: {path.name}")
        if float(np.max(np.abs(audio))) > PEAK_LIMIT + 1e-5:
            failures.append(f"peak above -1 dBFS: {path.name}")
        if _sha256(path) != hashes.get(blind_id):
            failures.append(f"SHA-256 mismatch: {path.name}")
    if len(sample_rates) != 1 or len(direct_shapes) != 1 or len(loop_shapes) != 1:
        failures.append("inconsistent sample rate or stimulus shapes")
    mapping = key.get("blind_mapping", {})
    same_codes = [blind_id for blind_id, value in mapping.items() if str(value).startswith("direct_same_control_")]
    if len(same_codes) != 2 or hashes.get(same_codes[0]) != hashes.get(same_codes[1]):
        failures.append("same-control pair is not sample-identical")
    for name, audio in key.get("loop_asset_codes", {}).items():
        if audio not in expected_ids or name not in {"repeat_raw", "alternate_far_raw", "shuffle_raw", "shuffle_clip_matched"}:
            failures.append("invalid private loop mapping")
    return {
        "passed": not failures,
        "failures": failures,
        "blind_files": len(actual_wavs),
        "sample_rate": next(iter(sample_rates)) if len(sample_rates) == 1 else None,
        "direct_shape": list(next(iter(direct_shapes))) if len(direct_shapes) == 1 else None,
        "loop_shape": list(next(iter(loop_shapes))) if len(loop_shapes) == 1 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("references/group_1"))
    parser.add_argument("--group", default="1")
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--interval-ms", type=float, default=1200.0)
    parser.add_argument("--clip-seconds", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=31_082_026)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.events < 6:
        raise ValueError("Gate requires at least six loop events")
    if not 300.0 <= args.interval_ms <= 2_000.0:
        raise ValueError("Interval must be between 300 and 2000 ms")
    final = args.results_dir.resolve()
    if final.exists():
        raise FileExistsError(f"Results directory already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [path.resolve() for path in discover_wav_files(args.input_dir)]
    source_hashes = {str(path): _sha256(path) for path in source_paths}
    implementation_paths = {
        "runner": Path(__file__).resolve(),
        "optimizer": Path(__file__).with_name("sfx_pool_optimizer.py").resolve(),
    }
    implementation_hashes = {name: _sha256(path) for name, path in implementation_paths.items()}
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    published = {"value": False}

    def cleanup() -> None:
        resolved = staging.resolve()
        if (
            not published["value"]
            and resolved.is_dir()
            and resolved.parent == final.parent
            and resolved.name.startswith(f".{final.name}.staging-")
        ):
            shutil.rmtree(resolved)

    atexit.register(cleanup)
    clips = analyze_directory(args.input_dir, clip_duration_s=args.clip_seconds)
    groups = group_index_map(clips)
    if args.group not in groups or len(groups[args.group]) < 3:
        raise ValueError(f"Group {args.group!r} must contain at least three takes")
    indices = groups[args.group]
    sample_rates = {clips[index].sample_rate for index in indices}
    channels = {clips[index].metrics.channels for index in indices}
    if len(sample_rates) != 1 or len(channels) != 1:
        raise ValueError("Selected group has incompatible sample rates or channel counts")
    sample_rate = next(iter(sample_rates))
    distance_matrix, _ = build_groupwise_distance_matrices(clips)
    medoid = medoid_index(indices, distance_matrix)
    direct_truth = select_discriminability_pairs(indices, distance_matrix, medoid=medoid)
    raw_bank = _common_peak_scale({index: clips[index].prepared for index in indices})
    matched_bank = normalize_prepared_bank(clips, indices)
    far_pair = direct_truth["far_pair"]
    schedules = {
        "repeat_raw": [medoid] * args.events,
        "alternate_far_raw": _alternate(far_pair, args.events),
        "shuffle_raw": shuffle_schedule(indices, count=args.events, seed=args.seed + 1),
    }
    rendered = {
        "repeat_raw": assemble_sequence(raw_bank, schedules["repeat_raw"], sample_rate, interval_ms=args.interval_ms),
        "alternate_far_raw": assemble_sequence(raw_bank, schedules["alternate_far_raw"], sample_rate, interval_ms=args.interval_ms),
        "shuffle_raw": assemble_sequence(raw_bank, schedules["shuffle_raw"], sample_rate, interval_ms=args.interval_ms),
        "shuffle_clip_matched": assemble_sequence(matched_bank, schedules["shuffle_raw"], sample_rate, interval_ms=args.interval_ms),
    }
    rendered = rms_match_sequences(rendered, anchor="repeat_raw")
    for name, audio in rendered.items():
        passed, failures = technical_audio_gate(audio)
        if not passed:
            raise RuntimeError(f"Loop {name} failed technical gate: {failures}")

    experiment = staging / "experiment"
    private = staging / "private_do_not_open_before_scoring"
    experiment.mkdir(parents=True)
    private.mkdir()
    assets: list[tuple[str, np.ndarray]] = []
    for pair_name, (left, right) in direct_truth.items():
        assets.append((f"direct_{pair_name}_A", raw_bank[left]))
        assets.append((f"direct_{pair_name}_B", raw_bank[right]))
    for name, audio in rendered.items():
        assets.append((f"loop_{name}", audio))
    codes = [f"X{position:02d}" for position in range(1, len(assets) + 1)]
    np.random.default_rng(args.seed + 20).shuffle(codes)
    asset_to_code = {asset_name: code for code, (asset_name, _) in zip(codes, assets)}
    blind_mapping = {code: asset_name for asset_name, code in asset_to_code.items()}
    for asset_name, audio in assets:
        sf.write(experiment / f"{asset_to_code[asset_name]}.wav", audio, sample_rate, subtype="PCM_16")
    pair_rng = np.random.default_rng(args.seed + 30)
    direct_order = list(direct_truth)
    pair_rng.shuffle(direct_order)
    direct_pairs: dict[str, dict[str, str]] = {}
    direct_comparisons: dict[str, str] = {}
    for position, pair_name in enumerate(direct_order, start=1):
        pair_id = f"D{position:02d}"
        direct_pairs[f"D{position:02d}"] = {
            "A": asset_to_code[f"direct_{pair_name}_A"],
            "B": asset_to_code[f"direct_{pair_name}_B"],
        }
        direct_comparisons[pair_id] = pair_name
    loop_design = [
        ("repeat_raw", "alternate_far_raw"),
        ("repeat_raw", "shuffle_raw"),
        ("shuffle_raw", "shuffle_clip_matched"),
    ]
    pair_rng.shuffle(loop_design)
    loop_comparisons = {
        f"L{position:02d}": comparison
        for position, comparison in enumerate(loop_design, start=1)
    }
    loop_pairs = {
        pair_id: {
            "A": asset_to_code[f"loop_{left}"],
            "B": asset_to_code[f"loop_{right}"],
        }
        for pair_id, (left, right) in loop_comparisons.items()
    }
    (experiment / "discriminability_gate.html").write_text(
        _gate_html(direct_pairs, loop_pairs, events=args.events, interval_ms=args.interval_ms),
        encoding="utf-8",
    )
    public_manifest = {
        "protocol": PROTOCOL,
        "blind_ids": sorted(blind_mapping),
        "direct_pairs": direct_pairs,
        "loop_pairs": loop_pairs,
        "events": args.events,
        "interval_ms": args.interval_ms,
        "sample_rate": sample_rate,
        "channels": next(iter(channels)),
    }
    _json_dump(experiment / "manifest_public.json", public_manifest)
    _json_dump(
        private / "blind_key.json",
        {
            "warning": "Do not open before ratings are saved.",
            "group": args.group,
            "blind_mapping": blind_mapping,
            "direct_truth": {
                pair_name: {
                    "sources": [clips[left].metrics.name, clips[right].metrics.name],
                    "distance": float(distance_matrix[left, right]),
                }
                for pair_name, (left, right) in direct_truth.items()
            },
            "direct_comparisons": direct_comparisons,
            "loop_asset_codes": {
                name: asset_to_code[f"loop_{name}"] for name in rendered
            },
            "loop_comparisons": loop_comparisons,
            "schedules": {
                name: names_for_schedule(schedule, clips) for name, schedule in schedules.items()
            },
            "processing": {
                "direct": "onset alignment/crop plus one common peak scale; natural level differences preserved",
                "raw_loops": "same raw bank, whole-sequence RMS comparison and common peak safety only",
                "matched_loop": "bounded clip-level early-RMS matching, then the same whole-sequence RMS/peak treatment",
            },
        },
    )
    run_manifest = {
        "protocol": PROTOCOL,
        "settings": {
            "group": args.group,
            "events": args.events,
            "interval_ms": args.interval_ms,
            "clip_seconds": args.clip_seconds,
            "seed": args.seed,
        },
        "files": [
            {"path": str(clip.path.resolve()), "sha256": source_hashes[str(clip.path.resolve())], "group": clip.group}
            for clip in clips
        ],
        "implementation_sha256": implementation_hashes,
        "experiment": {
            "blind_audio_hashes": {
                blind_id: _sha256(experiment / f"{blind_id}.wav") for blind_id in blind_mapping
            },
            "html_sha256": _sha256(experiment / "discriminability_gate.html"),
            "public_manifest_sha256": _sha256(experiment / "manifest_public.json"),
        },
    }
    _json_dump(staging / "run_manifest.json", run_manifest)
    for path in source_paths:
        if _sha256(path) != source_hashes[str(path)]:
            raise RuntimeError(f"Source changed during build: {path.name}")
    for name, path in implementation_paths.items():
        if _sha256(path) != implementation_hashes[name]:
            raise RuntimeError(f"Implementation changed during build: {path.name}")
    verification = verify_gate(staging)
    _json_dump(staging / "verification_report.json", verification)
    if not verification["passed"]:
        raise RuntimeError(f"Gate package verification failed: {verification['failures']}")
    if final.exists():
        raise FileExistsError(f"Target appeared during build: {final}")
    staging.replace(final)
    published["value"] = True
    atexit.unregister(cleanup)
    print(f"[+] Gate package: {final}")
    print(f"[+] Open only: {final / 'experiment' / 'discriminability_gate.html'}")


if __name__ == "__main__":
    main()
