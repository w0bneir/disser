"""Build a blind v1 draft of transient-locked stochastic SFX variations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from microstructure_synthesis import (
    METHOD_VERSION,
    calibrate_microstructure_strength,
    common_peak_safe,
    fit_microstructure_profile,
    leading_attack_error,
    microstructure_distance,
    synthesize_microstructure,
)
from perceptual_variation_synthesis import synthesize_variation, waveform_correlation
from sfx_pool_optimizer import analyze_directory, assemble_sequence, group_index_map, technical_audio_gate


PROTOCOL = "microstructure_factorial_gate_v1"
TARGET_LABELS = ("q25", "median", "q75")
TARGET_SEEDS = {"q25": 17, "median": 17, "q75": 17}
LOOP_SEEDS = (17, 42, 2026, 73, 314, 2718, 1618, 808)


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


def _candidate_metrics(reference: np.ndarray, candidate: np.ndarray, profile) -> dict[str, float]:
    return {
        "microstructure_distance": microstructure_distance(reference, candidate, profile),
        "waveform_correlation": waveform_correlation(reference, candidate),
        "leading_attack_max_abs_error": leading_attack_error(
            reference, candidate, profile.sample_rate
        ),
        "rms_delta_db": 20.0 * np.log10(_rms(candidate) / max(_rms(reference), 1e-12)),
        "peak_dbfs": 20.0 * np.log10(max(float(np.max(np.abs(candidate))), 1e-12)),
    }


def _experiment_html(direct_pairs: dict[str, dict[str, str]], loop_pair: dict[str, str]) -> str:
    cards: list[str] = []
    for pair_id, sides in direct_pairs.items():
        cards.append(
            f"""<section class="card" data-kind="direct" data-id="{pair_id}">
            <h2>{pair_id}: одиночный выстрел</h2>
            <p>Сравни A/B. Не пытайся угадать, где исходник.</p>
            <div class="players"><div><b>A</b><audio controls preload="metadata" src="{sides['A']}.wav"></audio></div>
            <div><b>B</b><audio controls preload="metadata" src="{sides['B']}.wav"></audio></div></div>
            <label>Различие слышно?<select data-field="different"><option value="">—</option><option value="no">нет</option><option value="slight">слегка</option><option value="clear">явно</option></select></label>
            <label>Это всё ещё тот же выстрел?<select data-field="same_event"><option value="">—</option><option value="yes">да</option><option value="uncertain">не уверен</option><option value="no">нет</option></select></label>
            <label>Различие полезно как новый дубль?<select data-field="useful"><option value="">—</option><option value="none">нет</option><option value="slight">слегка</option><option value="clear">да</option></select></label>
            <label>Где слышнее артефакты?<select data-field="artifacts"><option value="">—</option><option value="A">A</option><option value="neither">нигде</option><option value="B">B</option><option value="both">в обоих</option></select></label>
            <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label></section>"""
        )
    cards.append(
        f"""<section class="card" data-kind="loop" data-id="L01"><h2>L01: серия выстрелов</h2>
        <p>Сравни повтор одной записи с серией новых микроструктурных вариантов.</p>
        <div class="players"><div><b>A</b><audio controls preload="metadata" src="{loop_pair['A']}.wav"></audio></div>
        <div><b>B</b><audio controls preload="metadata" src="{loop_pair['B']}.wav"></audio></div></div>
        <label>Где меньше «пулемётный эффект»?<select data-field="less_repetitive"><option value="">—</option><option value="A">A</option><option value="same">одинаково</option><option value="B">B</option></select></label>
        <label>Какую серию выбрать для игры?<select data-field="preferred"><option value="">—</option><option value="A">A</option><option value="same">без предпочтения</option><option value="B">B</option></select></label>
        <label>Где слышнее артефакты?<select data-field="artifacts"><option value="">—</option><option value="A">A</option><option value="neither">нигде</option><option value="B">B</option><option value="both">в обоих</option></select></label>
        <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label></section>"""
    )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Слепой тест микроструктуры v1</title><style>
:root{{--bg:#10151d;--panel:#19212c;--line:#344252;--text:#eef4fa;--muted:#aebdca;--accent:#55c2ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,sans-serif}}main{{max-width:960px;margin:auto;padding:28px 18px 80px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}}.players{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}}audio{{display:block;width:100%;margin-top:8px}}
label{{display:block;margin:11px 0}}select,textarea{{display:block;width:100%;margin-top:5px;padding:9px;background:#0e141c;color:var(--text);border:1px solid var(--line);border-radius:7px}}button{{padding:11px 16px;border:0;border-radius:8px;background:var(--accent);color:#071018;font-weight:700}}p{{color:var(--muted)}}
@media(max-width:650px){{.players{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Слепой factorial gate v1</h1><p>Семь одиночных сравнений и одна серия. Сначала сохрани ответы, затем можно раскрывать ключ.</p>
{''.join(cards)}<button id="save">Сохранить ответы JSON</button><span id="status"></span></main><script>
document.getElementById('save').onclick=()=>{{const ratings={{protocol:'{PROTOCOL}',ratings:[]}};document.querySelectorAll('.card').forEach(card=>{{const row={{id:card.dataset.id,kind:card.dataset.kind}};card.querySelectorAll('[data-field]').forEach(el=>row[el.dataset.field]=el.value);ratings.ratings.push(row)}});const blob=new Blob([JSON.stringify(ratings,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='microstructure_factorial_ratings.json';a.click();URL.revokeObjectURL(a.href);document.getElementById('status').textContent=' Ответы сохранены.'}};
</script></body></html>"""


def _analysis_html(profile_dict: dict[str, object], candidates: dict[str, object]) -> str:
    corridor = profile_dict["natural_microstructure_distance"]
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{values['microstructure_distance']:.3f}</td>"
        f"<td>{values['waveform_correlation']:.4f}</td><td>{values['rms_delta_db']:+.2f}</td>"
        f"<td>{values['leading_attack_max_abs_error']:.2e}</td></tr>"
        for name, values in candidates.items()
    )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Microstructure v1 analysis</title>
<style>body{{max-width:1000px;margin:30px auto;font:16px system-ui;background:#111720;color:#eaf2f8}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #405064;padding:8px;text-align:left}}code{{color:#71d1ff}}</style></head><body>
<h1>Калибровка микроструктуры v1</h1><p>Профиль построен по {profile_dict['files']} WAV, но статистика различий считается только внутри групп.</p>
<p>Натуральный коридор: Q25=<code>{corridor['q25']:.3f}</code>, медиана=<code>{corridor['median']:.3f}</code>, Q75=<code>{corridor['q75']:.3f}</code>; пар: {profile_dict['within_group_pair_count']}.</p>
<table><thead><tr><th>Кандидат</th><th>Micro-distance</th><th>Корреляция waveform</th><th>ΔRMS dB</th><th>Ошибка защищённой атаки</th></tr></thead><tbody>{rows}</tbody></table>
<p>Расстояние — исследовательский proxy, а не доказательство качества. Решение принимается только по слепому тесту.</p></body></html>"""


def build_package(
    input_dir: Path,
    results_dir: Path,
    *,
    experiment_group: str = "1",
    reference_name: str = "SHOT 1.4.wav",
    seed: int = 31_082_026,
    events: int = 8,
    interval_ms: float = 1200.0,
) -> dict[str, object]:
    input_dir, target = Path(input_dir), Path(results_dir)
    if target.exists():
        raise FileExistsError(f"Results directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_", dir=target.parent))
    try:
        analysis_dir = staging / "analysis"
        experiment_dir = staging / "experiment"
        private_dir = staging / "private"
        analysis_dir.mkdir()
        experiment_dir.mkdir()
        private_dir.mkdir()
        clips = analyze_directory(input_dir)
        if len(clips) < 6:
            raise ValueError("Need at least six natural takes")
        if len({clip.sample_rate for clip in clips}) != 1 or len({clip.prepared.shape for clip in clips}) != 1:
            raise ValueError("Prepared takes must share sample rate and shape")
        sample_rate = clips[0].sample_rate
        profile = fit_microstructure_profile(
            [clip.prepared for clip in clips],
            sample_rate,
            groups=[clip.group for clip in clips],
            names=[clip.metrics.name for clip in clips],
        )
        group_indices = group_index_map(clips).get(str(experiment_group), [])
        if len(group_indices) < 3:
            raise ValueError(f"Experiment group {experiment_group!r} has fewer than three takes")
        matches = [index for index in group_indices if clips[index].metrics.name.casefold() == reference_name.casefold()]
        if len(matches) != 1:
            available = ", ".join(clips[index].metrics.name for index in group_indices)
            raise ValueError(f"Reference {reference_name!r} not found uniquely; available: {available}")
        reference_index = matches[0]
        reference = clips[reference_index].prepared
        donor_index = max(
            (index for index in group_indices if index != reference_index),
            key=lambda index: microstructure_distance(reference, clips[index].prepared, profile),
        )
        natural_donor = clips[donor_index].prepared
        targets = {
            "q25": profile.corridor_low,
            "median": profile.corridor_median,
            "q75": profile.corridor_high,
        }
        micro_candidates: dict[str, np.ndarray] = {}
        calibration: dict[str, dict[str, float | int]] = {}
        for label in TARGET_LABELS:
            audio, strength, distance = calibrate_microstructure_strength(
                reference,
                profile,
                seed=TARGET_SEEDS[label],
                target_distance=targets[label],
            )
            micro_candidates[label] = audio
            calibration[label] = {
                "seed": TARGET_SEEDS[label],
                "target_distance": targets[label],
                "selected_strength": strength,
                "achieved_distance": distance,
            }
        macro, _ = synthesize_variation(reference, natural_donor, sample_rate, strength=0.95)
        macro_plus_micro = synthesize_microstructure(
            macro,
            sample_rate,
            seed=TARGET_SEEDS["median"],
            strength=float(calibration["median"]["selected_strength"]),
        )
        assets = {
            "reference": reference,
            "macro_only": macro,
            "micro_q25": micro_candidates["q25"],
            "micro_median": micro_candidates["median"],
            "micro_q75": micro_candidates["q75"],
            "macro_plus_micro": macro_plus_micro,
            "natural_ceiling": natural_donor,
        }
        safe_values = common_peak_safe(list(assets.values()))
        assets = dict(zip(assets, safe_values))
        reference_safe = assets["reference"]
        metrics = {
            name: _candidate_metrics(reference_safe, audio, profile)
            for name, audio in assets.items()
            if name != "reference"
        }
        comparisons = {
            "exact_copy": ("reference", "reference"),
            "macro_only": ("reference", "macro_only"),
            "micro_q25": ("reference", "micro_q25"),
            "micro_median": ("reference", "micro_median"),
            "micro_q75": ("reference", "micro_q75"),
            "macro_plus_micro": ("reference", "macro_plus_micro"),
            "natural_ceiling": ("reference", "natural_ceiling"),
        }
        rng = np.random.default_rng(seed)
        comparison_names = list(comparisons)
        rng.shuffle(comparison_names)
        codes = [f"X{index:02d}" for index in range(1, 17)]
        rng.shuffle(codes)
        blind_audio: dict[str, np.ndarray] = {}
        blind_mapping: dict[str, str] = {}
        direct_pairs: dict[str, dict[str, str]] = {}
        cursor = 0
        direct_truth: dict[str, str] = {}
        for number, comparison in enumerate(comparison_names, start=1):
            pair_id = f"D{number:02d}"
            left_name, right_name = comparisons[comparison]
            if rng.random() < 0.5:
                left_name, right_name = right_name, left_name
            left_code, right_code = codes[cursor : cursor + 2]
            cursor += 2
            direct_pairs[pair_id] = {"A": left_code, "B": right_code}
            direct_truth[pair_id] = comparison
            for code, asset_name in ((left_code, left_name), (right_code, right_name)):
                blind_audio[code] = assets[asset_name]
                blind_mapping[code] = asset_name
        loop_variants = []
        for loop_seed in LOOP_SEEDS[:events]:
            variant, _, _ = calibrate_microstructure_strength(
                reference,
                profile,
                seed=loop_seed,
                target_distance=profile.corridor_median,
                iterations=7,
            )
            loop_variants.append(variant)
        loop_variants = common_peak_safe(loop_variants)
        repeat = assemble_sequence({0: reference_safe}, [0] * events, sample_rate=sample_rate, interval_ms=interval_ms)
        cycle = assemble_sequence(
            {index: audio for index, audio in enumerate(loop_variants)},
            list(range(events)),
            sample_rate=sample_rate,
            interval_ms=interval_ms,
        )
        repeat, cycle = common_peak_safe([repeat, cycle])
        loop_names = [("repeat_reference", repeat), ("microstructure_cycle", cycle)]
        if rng.random() < 0.5:
            loop_names.reverse()
        loop_codes = codes[cursor : cursor + 2]
        loop_pair = {"A": loop_codes[0], "B": loop_codes[1]}
        for code, (asset_name, audio) in zip(loop_codes, loop_names):
            blind_audio[code] = audio
            blind_mapping[code] = asset_name
        failures: list[str] = []
        for code, audio in blind_audio.items():
            passed, reasons = technical_audio_gate(audio)
            if not passed:
                failures.extend(f"{code}: {reason}" for reason in reasons)
            sf.write(experiment_dir / f"{code}.wav", audio, sample_rate, subtype="PCM_16")
        if failures:
            raise RuntimeError("Technical audio gate failed: " + "; ".join(failures))
        profile_dict = profile.json_dict(include_arrays=True)
        _json_dump(analysis_dir / "microstructure_profile.json", profile_dict)
        with (analysis_dir / "within_group_distances.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("group", "distance"))
            writer.writerows(zip(profile.pairwise_groups, profile.pairwise_distances))
        (analysis_dir / "analysis.html").write_text(
            _analysis_html(profile.json_dict(), metrics), encoding="utf-8"
        )
        public = {
            "protocol": PROTOCOL,
            "direct_pairs": direct_pairs,
            "loop_pair": loop_pair,
            "sample_rate": sample_rate,
            "channels": int(reference.shape[1]),
            "events": events,
            "interval_ms": interval_ms,
        }
        _json_dump(experiment_dir / "manifest_public.json", public)
        (experiment_dir / "blind_test.html").write_text(
            _experiment_html(direct_pairs, loop_pair), encoding="utf-8"
        )
        private = {
            "warning": "Do not open before ratings are saved.",
            "protocol": PROTOCOL,
            "blind_mapping": blind_mapping,
            "direct_truth": direct_truth,
            "loop_truth": {side: blind_mapping[code] for side, code in loop_pair.items()},
            "reference": clips[reference_index].metrics.name,
            "natural_donor": clips[donor_index].metrics.name,
        }
        _json_dump(private_dir / "blind_key.json", private)
        manifest = {
            "protocol": PROTOCOL,
            "method_version": METHOD_VERSION,
            "settings": {
                "input_dir": str(input_dir),
                "experiment_group": experiment_group,
                "reference_name": reference_name,
                "seed": seed,
                "events": events,
                "interval_ms": interval_ms,
                "attack_protection": "sample-exact through 55 ms from prepared clip start",
                "cross_group_policy": "groups never form natural-pair distances",
            },
            "profile_summary": profile.json_dict(),
            "calibration": calibration,
            "candidate_metrics": metrics,
            "selected": {
                "reference": clips[reference_index].metrics.name,
                "natural_donor": clips[donor_index].metrics.name,
            },
            "source_files": [
                {
                    "path": str(clip.path.resolve()),
                    "group": clip.group,
                    "sha256": _sha256(clip.path),
                }
                for clip in clips
            ],
            "implementation_sha256": {
                "synthesis": _sha256(Path(__file__).with_name("microstructure_synthesis.py")),
                "runner": _sha256(Path(__file__)),
            },
            "output_hashes": {
                path.name: _sha256(path) for path in sorted(experiment_dir.glob("*.wav"))
            },
        }
        _json_dump(staging / "run_manifest.json", manifest)
        verification = {
            "passed": True,
            "audio_files": len(blind_audio),
            "source_files": len(clips),
            "within_group_pairs": int(profile.pairwise_distances.size),
            "failures": [],
        }
        _json_dump(staging / "verification_report.json", verification)
        staging.replace(target)
        return {
            "target": str(target),
            "profile": profile.json_dict(),
            "calibration": calibration,
            "candidate_metrics": metrics,
            "selected": manifest["selected"],
            "verification": verification,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("references/group_1"))
    parser.add_argument("--experiment-group", default="1")
    parser.add_argument("--reference-name", default="SHOT 1.4.wav")
    parser.add_argument("--seed", type=int, default=31_082_026)
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--interval-ms", type=float, default=1200.0)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_package(
        args.input_dir,
        args.results_dir,
        experiment_group=args.experiment_group,
        reference_name=args.reference_name,
        seed=args.seed,
        events=args.events,
        interval_ms=args.interval_ms,
    )
    corridor = result["profile"]["natural_microstructure_distance"]
    print(f"[+] Profile: 26 files; natural corridor {corridor['q25']:.3f} / {corridor['median']:.3f} / {corridor['q75']:.3f}")
    print(f"[+] Reference: {result['selected']['reference']}")
    print(f"[+] Natural ceiling: {result['selected']['natural_donor']}")
    for label, values in result["calibration"].items():
        print(f"[+] micro {label}: strength={values['selected_strength']:.4f}; distance={values['achieved_distance']:.3f}")
    print(f"[+] Package verified: {result['verification']['passed']}")
    print(f"[+] Open: {Path(result['target']) / 'experiment' / 'blind_test.html'}")


if __name__ == "__main__":
    main()
