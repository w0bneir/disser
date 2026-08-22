"""Раскрыть ключ и агрегировать полностью заполненный blind listening-пилот."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


INDIVIDUAL_RATINGS = [
    "same_event_identity_1_5",
    "naturalness_1_5",
    "artifact_severity_1_5",
    "sound_design_usefulness_1_5",
]
PACKAGE_RATINGS = [
    "same_event_consistency_1_5",
    "naturalness_1_5",
    "useful_diversity_1_5",
    "artifact_severity_1_5",
    "listening_fatigue_1_5",
    "sound_design_usefulness_1_5",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Нечего сохранять")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_scores(
    rows: list[dict[str, str]],
    *,
    id_column: str,
    rating_columns: list[str],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Таблица оценок пуста")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        item_id = row.get(id_column, "").strip()
        listener_id = row.get("listener_id", "").strip()
        if not item_id or not listener_id:
            raise ValueError(f"Не заполнены listener_id/{id_column}")
        if item_id in seen_ids:
            raise ValueError(f"Повторный идентификатор: {item_id}")
        seen_ids.add(item_id)
        converted: dict[str, Any] = dict(row)
        for column in rating_columns:
            raw_value = row.get(column, "").strip()
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(f"Не заполнена оценка {column} для {item_id}") from error
            if not 1 <= value <= 5:
                raise ValueError(f"Оценка {column} для {item_id} вне диапазона [1, 5]")
            converted[column] = value
        validated.append(converted)
    return validated


def _decode(
    score_rows: list[dict[str, Any]],
    key_rows: list[dict[str, str]],
    *,
    id_column: str,
) -> list[dict[str, Any]]:
    key_by_id = {row[id_column]: row for row in key_rows}
    score_ids = {str(row[id_column]) for row in score_rows}
    if score_ids != set(key_by_id):
        raise ValueError("Набор ID в оценках не совпадает с закрытым ключом")
    return [{**row, **key_by_id[str(row[id_column])]} for row in score_rows]


def _summary(
    decoded_rows: list[dict[str, Any]],
    rating_columns: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decoded_rows:
        grouped[str(row["method"])].append(row)
    summaries: list[dict[str, Any]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        summary: dict[str, Any] = {"method": method, "count": len(rows)}
        for column in rating_columns:
            values = [float(row[column]) for row in rows]
            summary[f"mean_{column}"] = mean(values)
            summary[f"median_{column}"] = median(values)
        summaries.append(summary)
    return summaries


def analyze_listening_package(
    listening_dir: Path,
    *,
    output_dir: Path | None = None,
) -> None:
    public_dir = listening_dir / "public"
    private_dir = listening_dir / "private_do_not_open_before_scoring"
    resolved_output = output_dir or listening_dir / "analysis"
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise FileExistsError(f"Analysis-каталог уже непустой: {resolved_output}")

    individual_scores = _validate_scores(
        _read_csv(public_dir / "individual_scores.csv"),
        id_column="stimulus_id",
        rating_columns=INDIVIDUAL_RATINGS,
    )
    package_scores = _validate_scores(
        _read_csv(public_dir / "package_scores.csv"),
        id_column="package_id",
        rating_columns=PACKAGE_RATINGS,
    )
    individual_decoded = _decode(
        individual_scores,
        _read_csv(private_dir / "individual_answer_key.csv"),
        id_column="stimulus_id",
    )
    package_decoded = _decode(
        package_scores,
        _read_csv(private_dir / "package_answer_key.csv"),
        id_column="package_id",
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    _write_csv(resolved_output / "individual_decoded.csv", individual_decoded)
    _write_csv(
        resolved_output / "individual_method_summary.csv",
        _summary(individual_decoded, INDIVIDUAL_RATINGS),
    )
    _write_csv(resolved_output / "package_decoded.csv", package_decoded)
    _write_csv(
        resolved_output / "package_method_summary.csv",
        _summary(package_decoded, PACKAGE_RATINGS),
    )
    context = json.loads((private_dir / "context.json").read_text(encoding="utf-8"))
    context["analysis_status"] = "complete"
    context["listener_ids"] = sorted(
        {str(row["listener_id"]) for row in individual_decoded + package_decoded}
    )
    (resolved_output / "analysis_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[+] Blind key раскрыт после полной оценки: {resolved_output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listening-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze_listening_package(
        arguments.listening_dir,
        output_dir=arguments.output_dir,
    )

