"""Validate and decode ratings from the natural-take discriminability gate."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


DIRECT_VALUES = {
    "same_or_different": {"same", "different"},
    "useful_difference": {"not_applicable", "none", "slight", "clear"},
    "same_event": {"yes", "uncertain", "no"},
}
LOOP_FIELDS = ("less_repetitive", "clearer_variation", "more_natural", "preferred")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_confidence(answer: dict, pair_id: str) -> int:
    value = str(answer.get("confidence", ""))
    if value not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Invalid or missing confidence for {pair_id}")
    return int(value)


def _blind_ids_for_roles(blind_mapping: dict[str, str], roles: set[str]) -> set[str]:
    result = {blind_id for blind_id, role in blind_mapping.items() if role in roles}
    if len(result) != len(roles):
        raise ValueError(f"Blind key is incomplete for roles: {sorted(roles)}")
    return result


def _validate_answer_ids(answer: dict, expected: set[str], pair_id: str) -> tuple[str, str]:
    actual_a = answer.get("blind_A")
    actual_b = answer.get("blind_B")
    if (
        not isinstance(actual_a, str)
        or not isinstance(actual_b, str)
        or actual_a == actual_b
        or {actual_a, actual_b} != expected
    ):
        raise ValueError(f"Blind IDs for {pair_id} do not match the private key")
    return actual_a, actual_b


def _find_loop(report_items: list[dict], first: str, second: str) -> dict:
    expected = {first, second}
    for item in report_items:
        if set(item["methods"]) == expected:
            return item
    raise ValueError(f"Missing loop comparison: {first} vs {second}")


def decode_gate(key: dict, document: dict) -> dict[str, object]:
    """Strictly validate one completed questionnaire and decode randomized A/B sides."""
    if document.get("protocol") != "take_discriminability_gate_v1":
        raise ValueError("Unknown ratings protocol")
    blind_mapping = key.get("blind_mapping")
    direct_comparisons = key.get("direct_comparisons")
    direct_truth = key.get("direct_truth")
    loop_comparisons = key.get("loop_comparisons")
    loop_asset_codes = key.get("loop_asset_codes")
    if not all(
        isinstance(value, dict)
        for value in (blind_mapping, direct_comparisons, direct_truth, loop_comparisons, loop_asset_codes)
    ):
        raise ValueError("Malformed private key")

    direct_answers = document.get("direct")
    loop_answers = document.get("loops")
    if not isinstance(direct_answers, dict) or not isinstance(loop_answers, dict):
        raise ValueError("Ratings must contain direct and loops objects")

    decoded_direct: list[dict] = []
    for pair_id, pair_type in sorted(direct_comparisons.items()):
        answer = direct_answers.get(pair_id)
        if not isinstance(answer, dict):
            raise ValueError(f"Missing direct answer: {pair_id}")
        expected = _blind_ids_for_roles(
            blind_mapping,
            {f"direct_{pair_type}_A", f"direct_{pair_type}_B"},
        )
        _validate_answer_ids(answer, expected, pair_id)
        for field, allowed in DIRECT_VALUES.items():
            if answer.get(field) not in allowed:
                raise ValueError(f"Invalid or missing {pair_id}.{field}")
        confidence = _require_confidence(answer, pair_id)
        truth = direct_truth.get(pair_type)
        if not isinstance(truth, dict):
            raise ValueError(f"Missing direct truth for {pair_type}")
        decoded_direct.append(
            {
                "pair_id": pair_id,
                "pair_type": pair_type,
                "sources": truth.get("sources", []),
                "acoustic_distance": truth.get("distance"),
                "reported": answer["same_or_different"],
                "difference_detected": answer["same_or_different"] == "different",
                "useful_difference": answer["useful_difference"],
                "same_event": answer["same_event"],
                "confidence": confidence,
                "comment": str(answer.get("comment", "")),
            }
        )

    decoded_loops: list[dict] = []
    for pair_id, methods in sorted(loop_comparisons.items()):
        if not isinstance(methods, list) or len(methods) != 2 or methods[0] == methods[1]:
            raise ValueError(f"Malformed loop comparison: {pair_id}")
        answer = loop_answers.get(pair_id)
        if not isinstance(answer, dict):
            raise ValueError(f"Missing loop answer: {pair_id}")
        try:
            expected = {loop_asset_codes[methods[0]], loop_asset_codes[methods[1]]}
        except KeyError as exc:
            raise ValueError(f"Missing loop asset code for {pair_id}") from exc
        actual_a, actual_b = _validate_answer_ids(answer, expected, pair_id)
        method_by_blind = {loop_asset_codes[method]: method for method in methods}
        decoded = {}
        for field in LOOP_FIELDS:
            choice = answer.get(field)
            if choice == "same":
                decoded[field] = "tie"
            elif choice == "A":
                decoded[field] = method_by_blind[actual_a]
            elif choice == "B":
                decoded[field] = method_by_blind[actual_b]
            else:
                raise ValueError(f"Invalid or missing {pair_id}.{field}")
        decoded_loops.append(
            {
                "pair_id": pair_id,
                "methods": methods,
                "decoded_choices": decoded,
                "confidence": _require_confidence(answer, pair_id),
                "comment": str(answer.get("comment", "")),
            }
        )

    direct_by_type = {item["pair_type"]: item for item in decoded_direct}
    control_passed = direct_by_type["same_control"]["reported"] == "same"
    raw_shuffle = _find_loop(decoded_loops, "repeat_raw", "shuffle_raw")
    alternate = _find_loop(decoded_loops, "repeat_raw", "alternate_far_raw")
    matching = _find_loop(decoded_loops, "shuffle_raw", "shuffle_clip_matched")
    raw_choices = raw_shuffle["decoded_choices"]
    alternate_choices = alternate["decoded_choices"]
    matching_choices = matching["decoded_choices"]
    raw_shuffle_wins = (
        raw_choices["less_repetitive"] == "shuffle_raw"
        and raw_choices["clearer_variation"] == "shuffle_raw"
    )
    alternate_wins = (
        alternate_choices["less_repetitive"] == "alternate_far_raw"
        and alternate_choices["clearer_variation"] == "alternate_far_raw"
    )
    matching_reduces_variation = matching_choices["clearer_variation"] == "shuffle_raw"
    detected = [item["pair_type"] for item in decoded_direct if item["difference_detected"]]
    metric_nonmonotonic = (
        direct_by_type["near_pair"]["difference_detected"]
        and not direct_by_type["median_pair"]["difference_detected"]
    )

    return {
        "protocol": "take_discriminability_analysis_v1",
        "session_id": document.get("session_id"),
        "saved_at": document.get("saved_at"),
        "direct": decoded_direct,
        "loops": decoded_loops,
        "decision": {
            "control_passed": control_passed,
            "detected_direct_pair_types": detected,
            "all_detected_pairs_preserved_event_identity": all(
                item["same_event"] == "yes"
                for item in decoded_direct
                if item["difference_detected"] and item["pair_type"] != "same_control"
            ),
            "raw_shuffle_beats_repeat": raw_shuffle_wins,
            "alternate_far_beats_repeat": alternate_wins,
            "clip_matching_reduces_clear_variation": matching_reduces_variation,
            "acoustic_distance_ranking_is_not_perceptually_monotonic": metric_nonmonotonic,
            "gate_passed": control_passed and raw_shuffle_wins and alternate_wins,
            "recommended_next_step": (
                "Build the natural-pool scheduler pilot without clip-level RMS matching."
                if control_passed and raw_shuffle_wins and alternate_wins
                else "Do not optimize the scheduler until the failed sensitivity condition is resolved."
            ),
        },
        "scope_warning": (
            "One expert-listener session is an exploratory engineering gate, not statistical validation. "
            "Confirm the result on more groups and listeners before making inferential claims."
        ),
    }


def _choice_label(value: str) -> str:
    return {
        "tie": "нет различий",
        "repeat_raw": "повтор одного дубля",
        "shuffle_raw": "случайная последовательность",
        "alternate_far_raw": "чередование далёкой пары",
        "shuffle_clip_matched": "случайная, поклипово выровненная",
    }.get(value, value)


def _html_report(report: dict[str, object]) -> str:
    direct_rows = []
    for item in report["direct"]:
        direct_rows.append(
            "<tr>"
            f"<td>{html.escape(item['pair_id'])}</td>"
            f"<td><code>{html.escape(item['pair_type'])}</code></td>"
            f"<td>{item['acoustic_distance']:.3f}</td>"
            f"<td>{'разные' if item['difference_detected'] else 'одинаковые'}</td>"
            f"<td>{html.escape(item['useful_difference'])}</td>"
            f"<td>{html.escape(item['same_event'])}</td>"
            f"<td>{item['confidence']}/5</td></tr>"
        )
    loop_rows = []
    for item in report["loops"]:
        choices = item["decoded_choices"]
        loop_rows.append(
            "<tr>"
            f"<td>{html.escape(item['pair_id'])}</td>"
            f"<td><code>{html.escape(' vs '.join(item['methods']))}</code></td>"
            f"<td>{html.escape(_choice_label(choices['less_repetitive']))}</td>"
            f"<td>{html.escape(_choice_label(choices['clearer_variation']))}</td>"
            f"<td>{html.escape(_choice_label(choices['more_natural']))}</td>"
            f"<td>{html.escape(_choice_label(choices['preferred']))}</td>"
            f"<td>{item['confidence']}/5</td></tr>"
        )
    decision = report["decision"]
    flags = [
        ("Контроль одинаковой пары пройден", decision["control_passed"]),
        ("Raw Shuffle слышимо лучше Repeat", decision["raw_shuffle_beats_repeat"]),
        ("Чередование далёкой пары слышимо лучше Repeat", decision["alternate_far_beats_repeat"]),
        ("Поклиповое RMS matching ослабляет вариативность", decision["clip_matching_reduces_clear_variation"]),
        ("Gate пройден", decision["gate_passed"]),
    ]
    flag_html = "".join(
        f"<li class=\"{'pass' if value else 'fail'}\">{'ДА' if value else 'НЕТ'} — {html.escape(label)}</li>"
        for label, value in flags
    )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Gate различимости — раскрытие</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;background:#0d1218;color:#edf4fa}}h1,h2{{color:#f5fbff}}.hero,.note{{background:#182431;border-left:4px solid #55c9ff;padding:16px;margin:18px 0}}.pass{{color:#8df0b2}}.fail{{color:#ffab9f}}table{{border-collapse:collapse;width:100%;margin:16px 0 28px}}th,td{{border:1px solid #344557;padding:9px;text-align:left}}th{{background:#182431}}code{{color:#b7e7ff}}</style></head><body>
<h1>Раскрытие gate различимости натуральных дублей</h1>
<div class="hero"><b>Итог: {'gate пройден' if decision['gate_passed'] else 'gate не пройден'}.</b><ul>{flag_html}</ul></div>
<h2>Одиночные дубли</h2><table><thead><tr><th>ID</th><th>Тип</th><th>Дистанция</th><th>Ответ</th><th>Полезность</th><th>То же событие</th><th>Уверенность</th></tr></thead><tbody>{''.join(direct_rows)}</tbody></table>
<h2>Короткие циклы</h2><table><thead><tr><th>ID</th><th>Сравнение</th><th>Меньше повтор</th><th>Яснее вариация</th><th>Естественнее</th><th>Предпочтение</th><th>Уверенность</th></tr></thead><tbody>{''.join(loop_rows)}</tbody></table>
<p class="note">Следующий шаг: исключить поклиповое RMS-выравнивание и проверить natural-pool scheduler на нескольких группах и слушателях. {html.escape(report['scope_warning'])}</p>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    key_path = args.gate_dir / "private_do_not_open_before_scoring" / "blind_key.json"
    key = _load(key_path)
    document = _load(args.ratings)
    report = decode_gate(key, document)
    report["input_integrity"] = {
        "ratings_sha256": _sha256(args.ratings),
        "blind_key_sha256": _sha256(key_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ratings_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ratings_analysis.html").write_text(
        _html_report(report), encoding="utf-8"
    )
    print(f"[+] Gate passed: {report['decision']['gate_passed']}")
    print(f"[+] Report: {(args.output_dir / 'ratings_analysis.html').resolve()}")


if __name__ == "__main__":
    main()
