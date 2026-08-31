"""Decode and aggregate saved pairwise ratings after blind scoring is complete."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path

from scipy.stats import binomtest


CRITERIA = {
    "less_repetitive": "Меньше механическая повторяемость",
    "more_natural": "Выше естественность",
    "more_consistent": "Устойчивее одно событие/контекст",
    "preferred": "Предпочтение для игры",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _document_fingerprint(document: dict) -> str:
    """Ignore download time while detecting an accidentally imported duplicate."""
    payload = {
        "protocol": document.get("protocol"),
        "session_id": document.get("session_id"),
        "pairs": document.get("pairs"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_ratings(key: dict, rating_documents: list[dict]) -> dict[str, object]:
    pair_key = key["pairwise_key"]
    aggregates: dict[tuple[str, str], Counter[str]] = {}
    confidence_values: dict[str, list[int]] = {}
    warnings = []
    valid_documents = 0
    partial_documents = 0
    invalid_documents = 0
    duplicate_documents = 0
    seen_session_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for document_index, document in enumerate(rating_documents, start=1):
        if not isinstance(document, dict):
            invalid_documents += 1
            warnings.append(f"Документ {document_index}: JSON root не является объектом")
            continue
        session_id = document.get("session_id")
        normalized_session_id = str(session_id).strip() if session_id is not None else ""
        fingerprint = _document_fingerprint(document)
        if document.get("protocol") != "natural_pool_pairwise_v1":
            invalid_documents += 1
            warnings.append(f"Документ {document_index}: неизвестный protocol")
            continue
        answers = document.get("pairs", {})
        if not isinstance(answers, dict):
            invalid_documents += 1
            warnings.append(f"Документ {document_index}: pairs не является объектом")
            continue
        missing_errors: list[str] = []
        invalid_errors: list[str] = []
        decoded_votes: list[tuple[str, str, str]] = []
        decoded_confidences: list[tuple[str, int]] = []
        for pair_id, private in pair_key.items():
            answer = answers.get(pair_id)
            if not isinstance(answer, dict):
                missing_errors.append(f"отсутствует {pair_id}")
                continue
            actual_a = answer.get("blind_A")
            actual_b = answer.get("blind_B")
            expected_ids = {private["A_blind_id"], private["B_blind_id"]}
            if (
                not isinstance(actual_a, str)
                or not isinstance(actual_b, str)
                or actual_a == actual_b
                or {actual_a, actual_b} != expected_ids
            ):
                invalid_errors.append(f"blind IDs {pair_id} не совпадают с ключом")
                continue
            method_by_blind = {
                private["A_blind_id"]: private["A_method"],
                private["B_blind_id"]: private["B_method"],
            }
            hypothesis = private["hypothesis"]
            for field in CRITERIA:
                choice = answer.get(field, "")
                if choice == "A":
                    decoded_votes.append((hypothesis, field, method_by_blind[actual_a]))
                elif choice == "B":
                    decoded_votes.append((hypothesis, field, method_by_blind[actual_b]))
                elif choice == "same":
                    decoded_votes.append((hypothesis, field, "tie"))
                elif choice in {"", None}:
                    missing_errors.append(f"не заполнено {pair_id}.{field}")
                else:
                    invalid_errors.append(f"недопустимое значение {pair_id}.{field}")
            confidence = str(answer.get("confidence", ""))
            if confidence not in {"1", "2", "3", "4", "5"}:
                if confidence:
                    invalid_errors.append(f"недопустимая уверенность {pair_id}")
                else:
                    missing_errors.append(f"не заполнена уверенность {pair_id}")
            else:
                decoded_confidences.append((hypothesis, int(confidence)))

        if invalid_errors:
            invalid_documents += 1
            warnings.append(f"Документ {document_index}: " + "; ".join(invalid_errors))
            continue
        if missing_errors:
            partial_documents += 1
            warnings.append(f"Документ {document_index}: " + "; ".join(missing_errors))
            continue
        duplicate = (
            bool(normalized_session_id) and normalized_session_id in seen_session_ids
        ) or fingerprint in seen_fingerprints
        if duplicate:
            duplicate_documents += 1
            warnings.append(f"Документ {document_index}: дубликат сессии/ответов, пропущен")
            continue
        if normalized_session_id:
            seen_session_ids.add(normalized_session_id)
        seen_fingerprints.add(fingerprint)
        valid_documents += 1
        for hypothesis, field, decoded_choice in decoded_votes:
            aggregates.setdefault((hypothesis, field), Counter())[decoded_choice] += 1
        for hypothesis, confidence in decoded_confidences:
            confidence_values.setdefault(hypothesis, []).append(confidence)

    comparisons = []
    for (hypothesis, field), counts in sorted(aggregates.items()):
        private = next(value for value in pair_key.values() if value["hypothesis"] == hypothesis)
        methods = [private["A_method"], private["B_method"]]
        first_wins = counts[methods[0]]
        second_wins = counts[methods[1]]
        decisive = first_wins + second_wins
        p_value = float(binomtest(first_wins, decisive, 0.5).pvalue) if decisive else None
        hypothesis_confidence = confidence_values.get(hypothesis, [])
        comparisons.append(
            {
                "hypothesis": hypothesis,
                "criterion": field,
                "criterion_label": CRITERIA[field],
                "method_1": methods[0],
                "method_1_wins": first_wins,
                "method_2": methods[1],
                "method_2_wins": second_wins,
                "ties": counts["tie"],
                "missing": counts["missing"],
                "decisive_answers": decisive,
                "two_sided_exact_binomial_p": p_value,
                "mean_confidence": (
                    float(sum(hypothesis_confidence) / len(hypothesis_confidence))
                    if hypothesis_confidence
                    else None
                ),
            }
        )
    return {
        "protocol": "natural_pool_pairwise_analysis_v1",
        "documents_submitted": len(rating_documents),
        "valid_documents": valid_documents,
        "partial_documents": partial_documents,
        "invalid_documents": invalid_documents,
        "duplicate_documents": duplicate_documents,
        "warnings": warnings,
        "interpretation_warning": (
            "Агрегируются валидные документы ответов, а не уникальные люди: без внешнего participant ID один человек может создать несколько сессий. "
            "Это exploratory-пилот, а не подтверждающая статистика; exact p-values не скорректированы за множественные сравнения."
        ),
        "comparisons": comparisons,
    }


def _html_report(report: dict[str, object]) -> str:
    rows = []
    for item in report["comparisons"]:
        p_value = "—" if item["two_sided_exact_binomial_p"] is None else f"{item['two_sided_exact_binomial_p']:.4f}"
        confidence = "—" if item["mean_confidence"] is None else f"{item['mean_confidence']:.2f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['hypothesis'])}</td>"
            f"<td>{html.escape(item['criterion_label'])}</td>"
            f"<td><code>{html.escape(item['method_1'])}</code>: {item['method_1_wins']}</td>"
            f"<td><code>{html.escape(item['method_2'])}</code>: {item['method_2_wins']}</td>"
            f"<td>{item['ties']}</td><td>{confidence}</td><td>{p_value}</td></tr>"
        )
    warnings = "".join(f"<li>{html.escape(value)}</li>" for value in report["warnings"])
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Результаты SFX Pool Pilot</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 18px;background:#0d1218;color:#edf4fa}}.note{{background:#182431;border-left:4px solid #55c9ff;padding:15px}}table{{border-collapse:collapse;width:100%;margin-top:24px}}th,td{{border:1px solid #344557;padding:9px;text-align:left}}th{{background:#182431}}code{{color:#b7e7ff}}</style></head><body>
<h1>Раскрытие слепого пилота</h1>
<p>Загружено: <b>{report['documents_submitted']}</b> · валидных: <b>{report['valid_documents']}</b> ·
неполных: <b>{report['partial_documents']}</b> · невалидных: <b>{report['invalid_documents']}</b> ·
дубликатов: <b>{report['duplicate_documents']}</b>.</p>
<p class="note">{html.escape(report['interpretation_warning'])}</p>
<ul>{warnings}</ul><table><thead><tr><th>Сравнение</th><th>Критерий</th><th>Метод 1</th><th>Метод 2</th><th>Ничья</th><th>Ср. уверенность</th><th>Exact p</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Каталог результатов не пуст: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_path = args.pilot_dir / "private_do_not_open_before_scoring" / "blind_key.json"
    key = _load(key_path)
    documents = [_load(path) for path in args.ratings]
    report = decode_ratings(key, documents)
    (args.output_dir / "ratings_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ratings_analysis.html").write_text(
        _html_report(report), encoding="utf-8"
    )
    print(f"[+] Валидных документов: {report['valid_documents']}")
    print(f"[+] Отчёт: {(args.output_dir / 'ratings_analysis.html').resolve()}")


if __name__ == "__main__":
    main()
