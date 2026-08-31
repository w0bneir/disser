"""Build a preregistered blind sequence-level anti-repetition listening test."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from anti_repetition import (
    VNFParameters,
    adaptive_schedule,
    assemble_sequence,
    energy_match,
    load_mono,
    loudness_match_sequences,
    no_repeat_schedule,
    pitch_gain_variant,
    rms,
    spectral_profile,
    technical_gate,
    velvet_spectral_variant,
)


METHOD_LABELS = {
    "repeat": "точное повторение",
    "pitch_gain": "стандартная pitch/gain-рандомизация",
    "legacy_micro": "прежние причинные микровариации",
    "adaptive_vnf": "sequence-aware velvet-noise spectral variations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("references/shot_sound.wav"))
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=Path("results/2026-08-28_structured_resynthesis_shot_gate1_01"),
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--interval-ms", type=float, default=450.0)
    parser.add_argument("--seed", type=int, default=28_082_026)
    return parser.parse_args()


def _load_legacy(directory: Path, reference: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    names = [
        "A1_attack_mild.wav",
        "A2_attack_medium.wav",
        "B1_body_mild.wav",
        "B2_body_medium.wav",
        "T1_tail_mild.wav",
        "T2_tail_medium.wav",
    ]
    result = []
    for name in names:
        audio, rate = load_mono(directory / name)
        if rate != sample_rate or audio.size != reference.size:
            raise ValueError(f"Несовместимый legacy candidate: {name}")
        result.append(energy_match(reference, audio))
    return result


def _build_pitch_gain(reference: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    semitones = rng.uniform(-0.65, 0.65, size=count)
    gains = rng.uniform(-0.8, 0.8, size=count)
    return [
        pitch_gain_variant(reference, semitones=float(pitch), gain_db=float(gain))
        for pitch, gain in zip(semitones, gains)
    ]


def _build_vnf_bank(reference: np.ndarray, sample_rate: int, seed: int) -> tuple[list[np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    candidates: list[np.ndarray] = []
    parameters: list[dict] = []
    for index in range(48):
        item = VNFParameters(
            length_ms=float(rng.uniform(1.4, 3.2)),
            pulses=int(rng.integers(6, 13)),
            decay_db=float(rng.uniform(10.0, 22.0)),
            wet_mix=float(rng.uniform(0.055, 0.13)),
            highpass_hz=float(rng.uniform(450.0, 1800.0)),
            seed=seed + 10_000 + index,
        )
        candidate = velvet_spectral_variant(reference, sample_rate, item)
        passed, failures = technical_gate(reference, candidate)
        if not passed:
            raise RuntimeError(f"VNF candidate {index} не прошёл gate: {failures}")
        candidates.append(candidate)
        parameters.append(item.__dict__)
    return candidates, parameters


def _html_page(reference_name: str, blind_ids: list[str], events: int, interval_ms: float) -> str:
    cards = []
    for blind_id in blind_ids:
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(blind_id)}</h2>
              <audio controls preload="metadata" src="{html.escape(blind_id)}.wav"></audio>
              <p>Механическая повторяемость (1 — не слышна, 5 — очень сильная): ____ /5</p>
              <p>Идентичность события: ____ /5</p>
              <p>Естественность: ____ /5</p>
              <p>Артефакты: нет / лёгкие / сильные</p>
              <p>Пригодность для игры: ____ /5</p>
              <p>Комментарий: __________________________________________</p>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Sequence Gate 0</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:920px;margin:32px auto;padding:0 18px;background:#11151b;color:#eaf0f6}}
.card{{background:#1b222c;border:1px solid #344252;border-radius:12px;padding:18px;margin:18px 0}}
audio{{width:100%}} code{{color:#9bd4ff}} .warning{{color:#ffd38a}}
</style></head><body>
<h1>Слепой sequence-level Gate 0</h1>
<p>Четыре серии содержат по {events} событий с интервалом {interval_ms:.0f} мс. Громкость серий выровнена.</p>
<p class="warning">Не открывайте <code>manifest.json</code> до завершения оценивания: там находится слепой ключ.</p>
<h2>Референс</h2><audio controls preload="metadata" src="{html.escape(reference_name)}"></audio>
<p>Слушайте серии в разном порядке. Оценивайте всю последовательность, а не различимость соседней пары.</p>
{''.join(cards)}
<p>После заполнения передайте оценки без попытки угадать методы.</p>
</body></html>"""


def main() -> None:
    args = parse_args()
    if args.events != 20:
        raise ValueError("Gate 0 preregistered ровно для 20 событий")
    if abs(args.interval_ms - 450.0) > 1e-9:
        raise ValueError("Gate 0 preregistered для интервала 450 мс")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise FileExistsError(f"Каталог результатов не пуст: {args.results_dir}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    reference, sample_rate = load_mono(args.reference)
    legacy = _load_legacy(args.legacy_dir, reference, sample_rate)
    pitch_gain = _build_pitch_gain(reference, args.events, args.seed + 1)
    vnf_bank, vnf_parameters = _build_vnf_bank(reference, sample_rate, args.seed + 2)
    legacy_indices = no_repeat_schedule(len(legacy), args.events, args.seed + 3)
    vnf_indices = adaptive_schedule(
        vnf_bank,
        reference,
        sample_rate,
        count=args.events,
        history=4,
    )

    methods = {
        "repeat": assemble_sequence(
            [reference] * args.events,
            sample_rate,
            interval_ms=args.interval_ms,
        ),
        "pitch_gain": assemble_sequence(
            pitch_gain,
            sample_rate,
            interval_ms=args.interval_ms,
        ),
        "legacy_micro": assemble_sequence(
            [legacy[index] for index in legacy_indices],
            sample_rate,
            interval_ms=args.interval_ms,
        ),
        "adaptive_vnf": assemble_sequence(
            [vnf_bank[index] for index in vnf_indices],
            sample_rate,
            interval_ms=args.interval_ms,
        ),
    }
    methods = loudness_match_sequences(methods, anchor_name="repeat")
    lengths = {values.size for values in methods.values()}
    if len(lengths) != 1:
        raise RuntimeError("Последовательности имеют разную длину")
    if any(not np.isfinite(values).all() for values in methods.values()):
        raise RuntimeError("NaN/Inf в итоговой последовательности")
    if max(float(np.max(np.abs(values))) for values in methods.values()) > 1.0:
        raise RuntimeError("Peak safety нарушен")

    method_order = list(methods)
    np.random.default_rng(args.seed + 4).shuffle(method_order)
    blind_mapping = {f"X{index + 1}": method for index, method in enumerate(method_order)}
    sf.write(args.results_dir / "reference.wav", reference, sample_rate, subtype="PCM_24")
    for blind_id, method in blind_mapping.items():
        sf.write(args.results_dir / f"{blind_id}.wav", methods[method], sample_rate, subtype="PCM_24")

    reference_profile = spectral_profile(reference, sample_rate)
    report = {
        "stage": "sequence_level_gate0_preregistered",
        "reference": str(args.reference.resolve()),
        "sample_rate": sample_rate,
        "events": args.events,
        "interval_ms": args.interval_ms,
        "seed": args.seed,
        "blind_mapping_do_not_open_before_rating": blind_mapping,
        "method_labels": METHOD_LABELS,
        "legacy_schedule": legacy_indices,
        "adaptive_vnf_schedule": vnf_indices,
        "selected_vnf_parameters": [vnf_parameters[index] for index in vnf_indices],
        "diagnostics_not_a_perceptual_verdict": {
            method: {
                "sequence_rms": rms(values),
                "sequence_peak": float(np.max(np.abs(values))),
                "mean_hit_profile_distance_from_reference": float(
                    np.mean(
                        [
                            np.linalg.norm(spectral_profile(hit, sample_rate) - reference_profile)
                            for hit in (
                                [reference] * args.events
                                if method == "repeat"
                                else pitch_gain
                                if method == "pitch_gain"
                                else [legacy[index] for index in legacy_indices]
                                if method == "legacy_micro"
                                else [vnf_bank[index] for index in vnf_indices]
                            )
                        ]
                    )
                ),
            }
            for method, values in methods.items()
        },
        "decision_rule": {
            "mechanical_repetition_improvement_vs_repeat_min_points": 2,
            "identity_min": 4,
            "naturalness_min": 4,
            "strong_artifacts_allowed": False,
            "game_usefulness_min": 3,
            "no_post_listening_parameter_sweep_in_gate0": True,
        },
    }
    (args.results_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.results_dir / "sequence_gate0.html").write_text(
        _html_page("reference.wav", list(blind_mapping), args.events, args.interval_ms),
        encoding="utf-8",
    )
    print(f"[+] Слепой Gate 0 создан: {(args.results_dir / 'sequence_gate0.html').resolve()}")
    print("[+] До выставления оценок не открывайте manifest.json.")


if __name__ == "__main__":
    main()
