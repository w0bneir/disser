"""Validate and decode one perceptual-variation draft questionnaire."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


DIRECT_FIELDS = {
    "same_or_different": {"same", "different"},
    "same_event": {"yes", "uncertain", "no"},
    "useful_difference": {"not_applicable", "none", "slight", "clear"},
    "more_natural": {"A", "same", "B"},
    "artifacts": {"A", "neither", "B", "both"},
}
LOOP_FIELDS = {
    "less_repetitive": {"A", "same", "B"},
    "more_natural": {"A", "same", "B"},
    "artifacts": {"A", "neither", "B", "both"},
    "preferred": {"A", "same", "B"},
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confidence(answer: dict, pair_id: str) -> int:
    value = str(answer.get("confidence", ""))
    if value not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Invalid or missing confidence for {pair_id}")
    return int(value)


def _validate_ids(answer: dict, expected: set[str], pair_id: str) -> tuple[str, str]:
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


def _decode_side(choice: str, actual_a: str, actual_b: str, method_by_blind: dict[str, str]) -> str:
    if choice == "same":
        return "tie"
    if choice == "A":
        return method_by_blind[actual_a]
    if choice == "B":
        return method_by_blind[actual_b]
    raise ValueError(f"Cannot decode side choice: {choice}")


def _decode_artifacts(choice: str, actual_a: str, actual_b: str, method_by_blind: dict[str, str]) -> str:
    if choice in {"neither", "both"}:
        return choice
    return _decode_side(choice, actual_a, actual_b, method_by_blind)


def decode_ratings(key: dict, document: dict, manifest: dict | None = None) -> dict[str, object]:
    if document.get("protocol") != "perceptual_variation_draft_v0":
        raise ValueError("Unknown ratings protocol")
    if key.get("protocol") != document.get("protocol"):
        raise ValueError("Ratings protocol does not match private key")
    mapping = key.get("blind_mapping")
    assignments = key.get("direct_assignment")
    loop_truth = key.get("loop_truth")
    if not isinstance(mapping, dict) or not isinstance(assignments, dict) or not isinstance(loop_truth, dict):
        raise ValueError("Malformed private key")
    direct_answers = document.get("direct")
    loop_answers = document.get("loops")
    if not isinstance(direct_answers, dict) or not isinstance(loop_answers, dict):
        raise ValueError("Ratings must contain direct and loops objects")

    metric_by_comparison: dict[str, dict] = {}
    if isinstance(manifest, dict):
        for label, metrics in manifest.get("candidate_metrics", {}).items():
            metric_by_comparison[f"synthetic_{label}"] = metrics

    decoded_direct = []
    for pair_id, comparison in sorted(assignments.items()):
        answer = direct_answers.get(pair_id)
        if not isinstance(answer, dict):
            raise ValueError(f"Missing answer: {pair_id}")
        expected = {code for code, value in mapping.items() if str(value).startswith(f"{comparison}_")}
        if len(expected) != 2:
            raise ValueError(f"Private key does not define two sides for {comparison}")
        actual_a, actual_b = _validate_ids(answer, expected, pair_id)
        for field, allowed in DIRECT_FIELDS.items():
            if answer.get(field) not in allowed:
                raise ValueError(f"Invalid or missing {pair_id}.{field}")
        method_by_blind = {code: str(mapping[code]).split(":", 1)[1] for code in expected}
        natural_winner = _decode_side(answer["more_natural"], actual_a, actual_b, method_by_blind)
        artifact_location = _decode_artifacts(answer["artifacts"], actual_a, actual_b, method_by_blind)
        decoded_direct.append(
            {
                "pair_id": pair_id,
                "comparison": comparison,
                "reported": answer["same_or_different"],
                "difference_detected": answer["same_or_different"] == "different",
                "same_event": answer["same_event"],
                "useful_difference": answer["useful_difference"],
                "more_natural": natural_winner,
                "artifacts": artifact_location,
                "confidence": _confidence(answer, pair_id),
                "comment": str(answer.get("comment", "")),
                "objective_metrics": metric_by_comparison.get(comparison),
            }
        )

    loop_answer = loop_answers.get("L01")
    if not isinstance(loop_answer, dict):
        raise ValueError("Missing loop answer: L01")
    repeat_code = loop_truth.get("repeat")
    cycle_code = loop_truth.get("synthetic_cycle")
    actual_a, actual_b = _validate_ids(loop_answer, {repeat_code, cycle_code}, "L01")
    for field, allowed in LOOP_FIELDS.items():
        if loop_answer.get(field) not in allowed:
            raise ValueError(f"Invalid or missing L01.{field}")
    loop_method = {repeat_code: "repeat_reference", cycle_code: "synthetic_cycle"}
    decoded_loop = {
        "less_repetitive": _decode_side(loop_answer["less_repetitive"], actual_a, actual_b, loop_method),
        "more_natural": _decode_side(loop_answer["more_natural"], actual_a, actual_b, loop_method),
        "artifacts": _decode_artifacts(loop_answer["artifacts"], actual_a, actual_b, loop_method),
        "preferred": _decode_side(loop_answer["preferred"], actual_a, actual_b, loop_method),
        "confidence": _confidence(loop_answer, "L01"),
        "comment": str(loop_answer.get("comment", "")),
    }

    by_comparison = {item["comparison"]: item for item in decoded_direct}
    control_passed = not by_comparison["exact_copy"]["difference_detected"]
    successful_doses = []
    for comparison in ("synthetic_low", "synthetic_mid", "synthetic_high"):
        item = by_comparison[comparison]
        synthetic_method = comparison
        naturalness_ok = item["more_natural"] in {"tie", synthetic_method}
        artifacts_ok = item["artifacts"] not in {synthetic_method, "both"}
        if (
            item["difference_detected"]
            and item["same_event"] == "yes"
            and item["useful_difference"] in {"slight", "clear"}
            and naturalness_ok
            and artifacts_ok
        ):
            successful_doses.append(comparison)
    detected_doses = [
        comparison
        for comparison in ("synthetic_low", "synthetic_mid", "synthetic_high")
        if by_comparison[comparison]["difference_detected"]
    ]
    no_synthetic_artifacts = all(
        by_comparison[comparison]["artifacts"] not in {comparison, "both"}
        for comparison in ("synthetic_low", "synthetic_mid", "synthetic_high")
    )
    cycle_reduces_repetition = decoded_loop["less_repetitive"] == "synthetic_cycle"
    return {
        "protocol": "perceptual_variation_analysis_v0",
        "session_id": document.get("session_id"),
        "saved_at": document.get("saved_at"),
        "reference_name": key.get("reference_name"),
        "natural_donor_name": key.get("natural_donor_name"),
        "direct": decoded_direct,
        "loop": decoded_loop,
        "decision": {
            "control_passed": control_passed,
            "detected_synthetic_doses": detected_doses,
            "successful_synthetic_doses": successful_doses,
            "all_synthetic_doses_preserved_event_identity": all(
                by_comparison[name]["same_event"] == "yes"
                for name in ("synthetic_low", "synthetic_mid", "synthetic_high")
            ),
            "no_reported_synthetic_artifacts": no_synthetic_artifacts,
            "synthetic_cycle_reduces_repetition": cycle_reduces_repetition,
            "gate_passed": control_passed and bool(successful_doses),
            "diagnosis": (
                "At least one bounded dose is useful; continue cross-group validation."
                if successful_doses
                else "Macro envelope/EQ/width transfer is insufficient; add transient-safe microstructure variation."
            ),
        },
        "scope_warning": "One expert-listener session is a diagnostic gate, not confirmatory evidence.",
    }


def _label(value: str) -> str:
    return {
        "tie": "одинаково",
        "neither": "артефактов нет",
        "both": "в обоих",
        "reference": "референс",
        "natural_donor": "натуральный дубль",
        "synthetic_low": "синтез low",
        "synthetic_mid": "синтез mid",
        "synthetic_high": "синтез high",
        "repeat_reference": "повтор референса",
        "synthetic_cycle": "цикл синтеза",
    }.get(value, value)


def _html_report(report: dict[str, object]) -> str:
    rows = []
    for item in report["direct"]:
        metrics = item.get("objective_metrics") or {}
        distance = metrics.get("profile_distance_from_reference")
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['pair_id'])}</td><td><code>{html.escape(item['comparison'])}</code></td>"
            f"<td>{'да' if item['difference_detected'] else 'нет'}</td><td>{html.escape(item['same_event'])}</td>"
            f"<td>{html.escape(item['useful_difference'])}</td><td>{html.escape(_label(item['more_natural']))}</td>"
            f"<td>{html.escape(_label(item['artifacts']))}</td><td>{'—' if distance is None else f'{distance:.3f}'}</td>"
            f"<td>{item['confidence']}/5</td></tr>"
        )
    decision = report["decision"]
    flags = [
        ("Контроль пройден", decision["control_passed"]),
        ("Событие сохранено во всех дозах", decision["all_synthetic_doses_preserved_event_identity"]),
        ("Артефактов синтеза не отмечено", decision["no_reported_synthetic_artifacts"]),
        ("Синтетический цикл уменьшил повтор", decision["synthetic_cycle_reduces_repetition"]),
        ("Главный gate пройден", decision["gate_passed"]),
    ]
    flag_html = "".join(
        f"<li class=\"{'pass' if value else 'fail'}\">{'ДА' if value else 'НЕТ'} — {html.escape(label)}</li>"
        for label, value in flags
    )
    loop = report["loop"]
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Результат perceptual variation v0</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;background:#0d1218;color:#edf4fa}}.hero,.note{{background:#182431;border-left:4px solid #55c9ff;padding:16px;margin:18px 0}}.pass{{color:#8df0b2}}.fail{{color:#ffab9f}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #344557;padding:9px;text-align:left}}th{{background:#182431}}code{{color:#b7e7ff}}</style></head><body>
<h1>Перцептивно ограниченный синтез v0 — раскрытие</h1><div class="hero"><b>Итог: {'gate пройден' if decision['gate_passed'] else 'gate не пройден'}.</b><ul>{flag_html}</ul></div>
<table><thead><tr><th>ID</th><th>Сравнение</th><th>Различие</th><th>То же событие</th><th>Полезность</th><th>Естественнее</th><th>Артефакты</th><th>Дистанция</th><th>Уверенность</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Цикл</h2><p>Меньше повтор: <b>{html.escape(_label(loop['less_repetitive']))}</b>; естественнее: <b>{html.escape(_label(loop['more_natural']))}</b>; артефакты: <b>{html.escape(_label(loop['artifacts']))}</b>; предпочтение: <b>{html.escape(_label(loop['preferred']))}</b>.</p>
<p class="note">Диагноз: macro-only перенос огибающей, EQ и stereo width достигает слышимости без артефактов, но не создаёт полезного нового дубля. v1 должен добавить статистически ограниченную микроструктурную вариативность тела/хвоста при сохранённой атаке. {html.escape(report['scope_warning'])}</p>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    key_path = args.draft_dir / "private_do_not_open_before_scoring" / "blind_key.json"
    manifest_path = args.draft_dir / "run_manifest.json"
    report = decode_ratings(_load(key_path), _load(args.ratings), _load(manifest_path))
    report["input_integrity"] = {
        "ratings_sha256": _sha256(args.ratings),
        "blind_key_sha256": _sha256(key_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ratings_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "ratings_analysis.html").write_text(_html_report(report), encoding="utf-8")
    print(f"[+] Gate passed: {report['decision']['gate_passed']}")
    print(f"[+] Detected doses: {', '.join(report['decision']['detected_synthetic_doses']) or 'none'}")
    print(f"[+] Successful doses: {', '.join(report['decision']['successful_synthetic_doses']) or 'none'}")
    print(f"[+] Report: {(args.output_dir / 'ratings_analysis.html').resolve()}")


if __name__ == "__main__":
    main()
