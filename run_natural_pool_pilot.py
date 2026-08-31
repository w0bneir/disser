"""Analyze natural SFX takes and build a blind anti-repetition pilot."""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import html
from itertools import combinations
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote

import numpy as np
import scipy
import soundfile as sf

from sfx_pool_optimizer import (
    AnalyzedClip,
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_SCHEDULER_HISTORY,
    DEFAULT_SCHEDULER_TEMPERATURE,
    GroupRecommendation,
    METHOD_VERSION,
    SCHEDULER_SCORE_WEIGHTS,
    SELECTION_OBJECTIVE_WEIGHTS,
    analyze_directory,
    assemble_sequence,
    build_distance_matrices,
    build_groupwise_distance_matrices,
    choose_experiment_group,
    discover_wav_files,
    group_index_map,
    rms_match_sequences,
    names_for_schedule,
    normalize_prepared_bank,
    perceptual_schedule,
    random_schedule,
    recommend_groups,
    schedule_diagnostics,
    shuffle_schedule,
    technical_audio_gate,
)


METHOD_LABELS = {
    "repeat_one": "Один натуральный дубль, повторяемый на протяжении серии",
    "random_full": "Случайный выбор с возвращением из полного пула",
    "shuffle_full": "Shuffle без непосредственного повтора, полный пул",
    "perceptual_full": "Content-aware scheduler v1, полный пул",
    "shuffle_optimized": "Shuffle, оптимизированный малый пул",
    "perceptual_optimized": "Content-aware scheduler v1, репрезентативная тройка",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("references/group_1"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-group",
        default="auto",
        help="Номер группы либо auto: выбрать первую по имени среди самых крупных групп.",
    )
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--events", type=int, default=15)
    parser.add_argument("--interval-ms", type=float, default=800.0)
    parser.add_argument("--clip-seconds", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=28_082_026)
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_context(project_directory: Path) -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _runtime_context() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
        "libsndfile": getattr(sf, "__libsndfile_version__", None),
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value).strip("._")
    return cleaned or "take"


def _take_number(name: str) -> int | None:
    match = re.search(r"[._ -](\d+)$", Path(name).stem)
    return int(match.group(1)) if match else None


def _data_warnings(clips: list[AnalyzedClip]) -> list[str]:
    warnings = []
    reaper_bwf = 0
    for clip in clips:
        with clip.path.open("rb") as stream:
            header = stream.read(4096)
        if b"bext" in header and b"REAPER" in header.upper():
            reaper_bwf += 1
    if reaper_bwf:
        warnings.append(
            f"{reaper_bwf} из {len(clips)} WAV имеют BWF/bext с упоминанием REAPER: "
            "это натуральный материал после DAW-экспорта, а не подтверждённые recorder masters."
        )

    groups = group_index_map(clips)
    for left_group, right_group in combinations(groups, 2):
        if len(groups[left_group]) != len(groups[right_group]):
            continue
        left = {
            _take_number(clips[index].metrics.name): clips[index].metrics.duration_s
            for index in groups[left_group]
        }
        right = {
            _take_number(clips[index].metrics.name): clips[index].metrics.duration_s
            for index in groups[right_group]
        }
        shared = sorted(number for number in set(left) & set(right) if number is not None)
        if len(shared) < 3 or len(shared) != len(groups[left_group]):
            continue
        left_values = np.asarray([left[number] for number in shared], dtype=np.float64)
        right_values = np.asarray([right[number] for number in shared], dtype=np.float64)
        correlation = float(np.corrcoef(left_values, right_values)[0, 1])
        deltas = np.abs(left_values - right_values)
        median_delta = float(np.median(deltas))
        close_matches = int(np.count_nonzero(deltas <= 0.10))
        paired_timing = (
            len(shared) >= 5 and correlation >= 0.90 and close_matches >= 2
        ) or (
            correlation >= 0.75 and median_delta <= 0.20
        )
        if np.isfinite(correlation) and paired_timing:
            warnings.append(
                f"Группы {left_group}/{right_group}: длительности одноимённых дублей "
                f"согласованы (r={correlation:.2f}, median Δ={median_delta * 1000.0:.0f} мс). "
                "Возможны параллельные микрофонные stems одних физических выстрелов."
            )
    return warnings


def _write_metrics_csv(path: Path, clips: list[AnalyzedClip]) -> None:
    rows = [clip.metrics.json_dict() for clip in clips]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_distance_csv(path: Path, clips: list[AnalyzedClip], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", *[clip.metrics.name for clip in clips]])
        for clip, row in zip(clips, matrix):
            writer.writerow([clip.metrics.name, *[f"{float(value):.8f}" for value in row]])


def _format_optional(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def _analysis_html(
    clips: list[AnalyzedClip],
    distance_matrix: np.ndarray,
    recommendations: list[GroupRecommendation],
    selected_group: str | None,
    data_warnings: list[str],
) -> str:
    recommendation_cards = []
    for item in recommendations:
        selected = ", ".join(html.escape(clips[index].metrics.name) for index in item.selected_indices)
        mark = " <span class='badge'>пилот</span>" if item.group == selected_group else ""
        recommendation_cards.append(
            f"""
            <section class="group-card">
              <h3>Группа {html.escape(item.group)}{mark}</h3>
              <p><b>{item.count}</b> натуральных дублей · медианное внутригрупповое различие
              <b>{item.median_distance:.3f}</b> · максимальное <b>{item.maximum_distance:.3f}</b></p>
              <p>Центральный дубль: <code>{html.escape(clips[item.medoid_index].metrics.name)}</code></p>
              <p>Рекомендуемый пул: <code>{selected}</code></p>
              <p class="muted">Ошибка среднего покрытия: {item.coverage_mean:.3f}; худшего: {item.coverage_max:.3f}.</p>
            </section>
            """
        )

    metrics_rows = []
    selected_names_by_group = {
        item.group: {clips[index].metrics.name for index in item.selected_indices}
        for item in recommendations
    }
    for clip in clips:
        metric = clip.metrics
        selected = "✓" if metric.name in selected_names_by_group.get(metric.group, set()) else ""
        metrics_rows.append(
            "<tr>"
            f"<td>{html.escape(metric.group)}</td><td>{html.escape(metric.name)}</td><td>{selected}</td>"
            f"<td>{metric.duration_s:.2f}</td><td>{metric.onset_s * 1000.0:.1f}</td>"
            f"<td>{metric.early_rms_dbfs:.1f}</td><td>{metric.peak_dbfs:.1f}</td>"
            f"<td>{metric.spectral_centroid_hz:.0f}</td>"
            f"<td>{_format_optional(metric.decay_20_db_s)}</td>"
            f"<td>{_format_optional(metric.decay_40_db_s)}</td>"
            f"<td>{metric.stereo_correlation:.2f}</td><td>{metric.side_to_mid_db:.1f}</td>"
            f"<td>{metric.near_full_scale_samples}</td></tr>"
        )

    maximum = max(float(np.max(distance_matrix)), 1e-9)
    heat_rows = []
    for row_index, clip in enumerate(clips):
        cells = [f"<th title='{html.escape(clip.metrics.name)}'>{row_index + 1}</th>"]
        for column_index, value in enumerate(distance_matrix[row_index]):
            fraction = min(1.0, float(value) / maximum)
            hue = 205.0 - 170.0 * fraction
            lightness = 18.0 + 36.0 * fraction
            title = (
                f"{clip.metrics.name} ↔ {clips[column_index].metrics.name}: {float(value):.4f}"
            )
            cells.append(
                f"<td title='{html.escape(title)}' style='background:hsl({hue:.0f} 68% {lightness:.0f}%)'>"
                f"{float(value):.2f}</td>"
            )
        heat_rows.append(f"<tr>{''.join(cells)}</tr>")
    heat_header = "".join(f"<th>{index + 1}</th>" for index in range(len(clips)))

    experiment_link = (
        "<a class='button' href='../experiment/pairwise_test.html'>Открыть основной парный тест</a> "
        "<a class='button' href='../experiment/blind_test.html'>Открыть поклиповую оценку серий</a>"
        if selected_group is not None
        else ""
    )
    warnings_html = "".join(f"<li>{html.escape(value)}</li>" for value in data_warnings)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>SFX Pool Optimizer — анализ</title>
<style>
:root{{--bg:#0c1118;--panel:#151d27;--line:#2c3a49;--text:#eef4fa;--muted:#9db0c3;--accent:#52c7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:34px 20px 64px}}h1{{font-size:2.1rem;margin-bottom:8px}}h2{{margin-top:38px}}
.lead{{font-size:1.08rem;color:#c9d6e2;max-width:900px;line-height:1.55}}.notice{{padding:16px;border-left:4px solid var(--accent);background:#111b26;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}}.group-card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
.group-card h3{{margin-top:0}}.badge{{font-size:.72rem;background:#1d6282;padding:3px 7px;border-radius:99px}}.muted{{color:var(--muted)}}code{{color:#b6e5ff}}
.button{{display:inline-block;background:#1385bb;color:white;text-decoration:none;font-weight:700;padding:12px 18px;border-radius:9px;margin:10px 0}}
.scroll{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;min-width:100%;font-size:.84rem}}th,td{{padding:7px 9px;border:1px solid #293746;white-space:nowrap;text-align:right}}th{{background:#182331;position:sticky;top:0}}td:nth-child(1),td:nth-child(2){{text-align:left}}
.heat td{{padding:5px;min-width:38px;text-align:center;font-size:.7rem;color:white;text-shadow:0 1px 2px #000}}.heat th{{position:static;text-align:center}}
</style></head><body><main>
<h1>SFX Pool Optimizer: натуральные дубли СКС</h1>
<p class="lead">CPU-анализ {len(clips)} исходных записей. Система не генерирует и не
морфирует звук: она оценивает атаку, тембр, динамику хвоста, стереокартину и
уровень, выбирает репрезентативный малый пул и строит историю воспроизведения.</p>
<div class="notice"><b>Важно:</b> численные расстояния — инженерная модель, а не
доказательство слышимого различия. Их валидирует слепое прослушивание. Группы
анализируются отдельно, чтобы смена помещения или микрофонной позиции не
выдавалась за полезную вариативность.</div>
<h2>Ограничения исходного материала</h2><ul>{warnings_html}</ul>
{experiment_link}
<h2>Рекомендации по группам</h2><div class="grid">{''.join(recommendation_cards)}</div>
<h2>Поклиповый контроль</h2>
<div class="scroll"><table><thead><tr><th>Группа</th><th>Файл</th><th>В пуле</th><th>с</th><th>Onset, мс</th><th>Early RMS</th><th>Peak</th><th>Centroid</th><th>t−20, с</th><th>t−40, с</th><th>L/R corr</th><th>S/M, dB</th><th>Near FS</th></tr></thead>
<tbody>{''.join(metrics_rows)}</tbody></table></div>
<h2>Глобальная диагностическая матрица</h2><p class="muted">Она нужна только для визуального сравнения контекстов. Выбор пула и scheduler используют отдельно нормализованную внутригрупповую шкалу, поэтому числа в карточках выше не обязаны совпадать с этой шкалой. Наведите курсор на ячейку для имён файлов.</p>
<div class="scroll"><table class="heat"><thead><tr><th>#</th>{heat_header}</tr></thead><tbody>{''.join(heat_rows)}</tbody></table></div>
</main></body></html>"""


def _blind_html(blind_ids: list[str], events: int, interval_ms: float) -> str:
    cards = []
    for blind_id in blind_ids:
        escaped = html.escape(blind_id)
        cards.append(
            f"""
            <section class="card" data-id="{escaped}">
              <h2>{escaped}</h2>
              <audio controls preload="metadata" src="{quote(blind_id)}.wav"></audio>
              <div class="questions">
                <label>Механическая повторяемость <span>1 — не слышна, 5 — очень сильная</span>
                  <select data-field="mechanical_repetition"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
                <label>Полезная вариативность <span>1 — отсутствует, 5 — отчётливо полезна</span>
                  <select data-field="useful_variation"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
                <label>Постоянство одного события <span>1 — разные события/контексты, 5 — один устойчивый выстрел</span>
                  <select data-field="event_consistency"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
                <label>Естественность <span>1 — искусственно, 5 — полностью естественно</span>
                  <select data-field="naturalness"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
                <label>Пригодность для игры <span>1 — непригодно, 5 — готово к использованию</span>
                  <select data-field="game_usefulness"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
                <label>Слышна смена помещения/микрофонной позиции?
                  <select data-field="context_switch"><option value="">—</option><option value="no">нет</option><option value="slight">слегка</option><option value="yes">да</option></select></label>
                <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Слепой пилот натурального SFX-пула</title>
<style>
:root{{--bg:#0b1016;--panel:#161e28;--line:#304052;--text:#eff5fa;--muted:#a9bacb;--accent:#52c7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}main{{max-width:900px;margin:auto;padding:30px 18px 70px}}
.warning{{background:#261f12;border-left:4px solid #f1b954;padding:15px;border-radius:6px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;margin:18px 0}}audio{{width:100%}}
.questions{{display:grid;gap:12px;margin-top:16px}}label{{display:grid;grid-template-columns:1fr 110px;gap:10px;align-items:center}}label span{{display:block;color:var(--muted);font-size:.8rem}}select,textarea{{background:#0e151d;color:var(--text);border:1px solid #43566b;border-radius:6px;padding:8px}}textarea{{grid-column:1/-1;width:100%}}
button{{background:#1489c1;color:white;border:0;border-radius:8px;padding:12px 18px;font-weight:700;cursor:pointer}}#status{{color:#a9e3a4;margin-left:12px}}
@media(max-width:650px){{label{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Слепой пилот натурального пула</h1>
<p>Шесть последовательностей содержат по {events} выстрелов с интервалом {interval_ms:.0f} мс.
Все события — реальные записи; синтеза, pitch-shift, EQ и морфинга нет. Различаются только состав пула и порядок выбора.</p>
<p class="warning">До сохранения оценок работайте <b>только с этой listening-папкой</b>. Не открывайте соседние <code>analysis</code>, <code>optimized_pool</code>, корневой manifest и <code>private_do_not_open_before_scoring</code>. Слушайте в наушниках при неизменной комфортной громкости.</p>
<p><a href="pairwise_test.html">Перейти к основному парному сравнению</a></p>
{''.join(cards)}
<button id="save">Скачать заполненные оценки</button><span id="status"></span>
<script>
const sessionId=(globalThis.crypto&&typeof globalThis.crypto.randomUUID==='function')
  ? globalThis.crypto.randomUUID()
  : `local-${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
const blindCards=Array.from(document.querySelectorAll('.card'));
for(let i=blindCards.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[blindCards[i],blindCards[j]]=[blindCards[j],blindCards[i]];}}
const saveButton=document.getElementById('save');
blindCards.forEach(card=>saveButton.parentNode.insertBefore(card,saveButton));
document.getElementById('save').addEventListener('click',()=>{{
  const ratings={{protocol:'natural_pool_pilot_v1',session_id:sessionId,saved_at:new Date().toISOString(),conditions:{{}}}};
  let missing=0;
  document.querySelectorAll('.card').forEach(card=>{{
    const row={{}};
    card.querySelectorAll('[data-field]').forEach(input=>{{row[input.dataset.field]=input.value;if(input.tagName==='SELECT'&&!input.value)missing++;}});
    ratings.conditions[card.dataset.id]=row;
  }});
  if(missing&&!confirm(`Не заполнено полей: ${{missing}}. Всё равно сохранить?`))return;
  const blob=new Blob([JSON.stringify(ratings,null,2)],{{type:'application/json'}});
  const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='natural_pool_ratings.json';link.click();URL.revokeObjectURL(link.href);
  document.getElementById('status').textContent='Оценки сохранены';
}});
</script></main></body></html>"""


def _pairwise_html(pairs: dict[str, dict[str, str]], events: int, interval_ms: float) -> str:
    cards = []
    for pair_id, sides in pairs.items():
        escaped_pair = html.escape(pair_id)
        cards.append(
            f"""
            <section class="card" data-id="{escaped_pair}" data-a="{html.escape(sides['A'])}" data-b="{html.escape(sides['B'])}">
              <h2>{escaped_pair}</h2>
              <div class="players">
                <div><h3>A</h3><audio controls preload="metadata" src="{quote(sides['A'])}.wav"></audio></div>
                <div><h3>B</h3><audio controls preload="metadata" src="{quote(sides['B'])}.wav"></audio></div>
              </div>
              <div class="questions">
                <label>Где меньше слышна механическая повторяемость?
                  <select data-field="less_repetitive"><option value="">—</option><option value="A">A</option><option value="same">различий нет</option><option value="B">B</option></select></label>
                <label>Где выше естественность?
                  <select data-field="more_natural"><option value="">—</option><option value="A">A</option><option value="same">различий нет</option><option value="B">B</option></select></label>
                <label>Где устойчивее ощущение одного события и контекста?
                  <select data-field="more_consistent"><option value="">—</option><option value="A">A</option><option value="same">различий нет</option><option value="B">B</option></select></label>
                <label>Какую серию вы выбрали бы для игры?
                  <select data-field="preferred"><option value="">—</option><option value="A">A</option><option value="same">без предпочтения</option><option value="B">B</option></select></label>
                <label>Уверенность <span>1 — почти наугад, 5 — уверенно</span>
                  <select data-field="confidence"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
                <label>Комментарий<textarea data-field="comment" rows="2"></textarea></label>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Парный тест SFX Pool Optimizer</title>
<style>
:root{{--bg:#0b1016;--panel:#161e28;--line:#304052;--text:#eff5fa;--muted:#a9bacb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}main{{max-width:980px;margin:auto;padding:30px 18px 70px}}
.warning{{background:#261f12;border-left:4px solid #f1b954;padding:15px;border-radius:6px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;margin:18px 0}}.players{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}audio{{width:100%}}
.questions{{display:grid;gap:12px;margin-top:18px}}label{{display:grid;grid-template-columns:1fr 180px;gap:12px;align-items:center}}label span{{display:block;color:var(--muted);font-size:.8rem}}select,textarea{{background:#0e151d;color:var(--text);border:1px solid #43566b;border-radius:6px;padding:8px}}textarea{{grid-column:1/-1;width:100%}}button{{background:#1489c1;color:white;border:0;border-radius:8px;padding:12px 18px;font-weight:700;cursor:pointer}}#status{{color:#a9e3a4;margin-left:12px}}
@media(max-width:700px){{.players{{grid-template-columns:1fr}}label{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Основной слепой парный тест</h1>
<p>Каждая серия содержит {events} натуральных выстрелов с интервалом {interval_ms:.0f} мс.
Сначала прослушайте A, затем B, при необходимости повторите один раз. Сравнивайте серию целиком.</p>
<p class="warning">До сохранения ответов работайте <b>только с этой listening-папкой</b>. Не открывайте соседние <code>analysis</code>, <code>optimized_pool</code>, корневой manifest и private-каталог. Положение A/B и порядок пар перемешиваются на этом компьютере.</p>
{''.join(cards)}
<button id="save">Скачать ответы парного теста</button><span id="status"></span>
<p><a href="blind_test.html">Дополнительная абсолютная оценка всех серий</a></p>
<script>
const sessionId=(globalThis.crypto&&typeof globalThis.crypto.randomUUID==='function')
  ? globalThis.crypto.randomUUID()
  : `local-${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
const pairCards=Array.from(document.querySelectorAll('.card'));
pairCards.forEach(card=>{{
  if(Math.random()<0.5){{const previousA=card.dataset.a;card.dataset.a=card.dataset.b;card.dataset.b=previousA;}}
  const players=card.querySelectorAll('audio');
  players[0].src=`${{card.dataset.a}}.wav`;players[1].src=`${{card.dataset.b}}.wav`;
}});
for(let i=pairCards.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[pairCards[i],pairCards[j]]=[pairCards[j],pairCards[i]];}}
const saveButton=document.getElementById('save');
pairCards.forEach(card=>saveButton.parentNode.insertBefore(card,saveButton));
document.getElementById('save').addEventListener('click',()=>{{
  const ratings={{protocol:'natural_pool_pairwise_v1',session_id:sessionId,saved_at:new Date().toISOString(),pairs:{{}}}};let missing=0;
  document.querySelectorAll('.card').forEach(card=>{{const row={{blind_A:card.dataset.a,blind_B:card.dataset.b}};card.querySelectorAll('[data-field]').forEach(input=>{{row[input.dataset.field]=input.value;if(input.tagName==='SELECT'&&!input.value)missing++;}});ratings.pairs[card.dataset.id]=row;}});
  if(missing&&!confirm(`Не заполнено полей: ${{missing}}. Всё равно сохранить?`))return;
  const blob=new Blob([JSON.stringify(ratings,null,2)],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='natural_pool_pairwise_ratings.json';link.click();URL.revokeObjectURL(link.href);document.getElementById('status').textContent='Ответы сохранены';
}});
</script></main></body></html>"""


def _choose_requested_group(
    recommendations: list[GroupRecommendation], requested: str
) -> GroupRecommendation:
    if requested.casefold() == "auto":
        return choose_experiment_group(recommendations)
    matches = [item for item in recommendations if item.group == requested]
    if not matches:
        available = ", ".join(item.group for item in recommendations)
        raise ValueError(f"Группа {requested!r} не найдена. Доступны: {available}")
    if matches[0].count < 3:
        raise ValueError("Для пилота нужны минимум три натуральных дубля")
    return matches[0]


def _build_schedules(
    full_pool: list[int],
    optimized_pool: list[int],
    medoid: int,
    distance_matrix: np.ndarray,
    *,
    events: int,
    seed: int,
) -> dict[str, list[int]]:
    return {
        "repeat_one": [medoid] * events,
        "random_full": random_schedule(full_pool, count=events, seed=seed + 11),
        "shuffle_full": shuffle_schedule(full_pool, count=events, seed=seed + 12),
        "perceptual_full": perceptual_schedule(
            full_pool, distance_matrix, count=events, seed=seed + 13
        ),
        "shuffle_optimized": shuffle_schedule(
            optimized_pool, count=events, seed=seed + 14
        ),
        "perceptual_optimized": perceptual_schedule(
            optimized_pool, distance_matrix, count=events, seed=seed + 15
        ),
    }


def _scheduler_seed_audit(
    pool: list[int],
    distance_matrix: np.ndarray,
    *,
    events: int,
    seed: int,
    trials: int = 500,
) -> dict[str, object]:
    metrics = (
        "mean_adjacent_distance",
        "mean_history_distance",
        "aba_patterns",
        "transition_entropy",
    )
    deltas = {metric: [] for metric in metrics}
    for trial in range(trials):
        shuffle = shuffle_schedule(pool, count=events, seed=seed + 2 * trial)
        content_aware = perceptual_schedule(
            pool, distance_matrix, count=events, seed=seed + 2 * trial + 1
        )
        shuffle_diagnostics = schedule_diagnostics(shuffle, distance_matrix)
        content_diagnostics = schedule_diagnostics(content_aware, distance_matrix)
        for metric in metrics:
            deltas[metric].append(
                float(content_diagnostics[metric]) - float(shuffle_diagnostics[metric])
            )
    higher_is_better = {
        "mean_adjacent_distance": True,
        "mean_history_distance": True,
        "aba_patterns": False,
        "transition_entropy": True,
    }
    return {
        "trials": trials,
        "warning": "Objective schedule diagnostics only; not a perceptual verdict.",
        "content_aware_minus_shuffle": {
            metric: {
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                "content_aware_win_fraction": float(
                    np.mean(np.asarray(values) > 0.0)
                    if higher_is_better[metric]
                    else np.mean(np.asarray(values) < 0.0)
                ),
            }
            for metric, values in deltas.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.events < 10:
        raise ValueError("Для sequence-level пилота нужно минимум 10 событий")
    if not 300.0 <= args.interval_ms <= 2_000.0:
        raise ValueError("Интервал должен быть от 300 до 2000 мс")
    if not 1.0 <= args.clip_seconds <= 6.0:
        raise ValueError("Длительность события должна быть от 1 до 6 секунд")
    if args.pool_size < 2:
        raise ValueError("Малый пул должен содержать минимум два дубля")
    source_paths_snapshot = [path.resolve() for path in discover_wav_files(args.input_dir)]
    source_hashes_snapshot = {
        str(path): _sha256(path)
        for path in source_paths_snapshot
    }
    implementation_paths = {
        "runner": Path(__file__).resolve(),
        "optimizer": Path(__file__).with_name("sfx_pool_optimizer.py").resolve(),
        "verifier": Path(__file__).with_name("verify_natural_pool_package.py").resolve(),
        "ratings_analyzer": Path(__file__).with_name("analyze_natural_pool_ratings.py").resolve(),
    }
    implementation_hashes_snapshot = {
        name: _sha256(path)
        for name, path in implementation_paths.items()
    }
    final_results_dir = args.results_dir.expanduser().resolve()
    if final_results_dir.exists():
        raise FileExistsError(f"Каталог результатов уже существует: {final_results_dir}")
    final_results_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_results_dir.name}.staging-",
            dir=final_results_dir.parent,
        )
    )
    staging_state = {"published": False}

    def cleanup_staging() -> None:
        resolved = staging_dir.resolve()
        expected_prefix = f".{final_results_dir.name}.staging-"
        safe_target = (
            resolved.parent == final_results_dir.parent
            and resolved.name.startswith(expected_prefix)
        )
        if not staging_state["published"] and safe_target and resolved.is_dir():
            shutil.rmtree(resolved)
            print(f"[!] Неполный staging-каталог удалён: {resolved}")

    atexit.register(cleanup_staging)
    args.results_dir = staging_dir

    print(f"[+] Анализ натуральных дублей: {args.input_dir.resolve()}")
    clips = analyze_directory(args.input_dir, clip_duration_s=args.clip_seconds)
    analyzed_paths = {clip.path.resolve() for clip in clips}
    if analyzed_paths != set(source_paths_snapshot):
        raise RuntimeError("Состав входных WAV изменился во время начала анализа")
    global_distance_matrix, global_component_matrices = build_distance_matrices(clips)
    within_distance_matrix, within_component_matrices = build_groupwise_distance_matrices(clips)
    recommendations = recommend_groups(clips, within_distance_matrix, pool_size=args.pool_size)
    if not recommendations:
        raise RuntimeError("Не найдено ни одной группы минимум из двух файлов")

    selected_recommendation = None
    if not args.analysis_only:
        selected_recommendation = _choose_requested_group(recommendations, args.experiment_group)
    selected_group = selected_recommendation.group if selected_recommendation else None
    data_warnings = _data_warnings(clips)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = args.results_dir / "analysis"
    analysis_dir.mkdir()

    _write_metrics_csv(analysis_dir / "clip_metrics.csv", clips)
    _write_distance_csv(analysis_dir / "distance_global_context.csv", clips, global_distance_matrix)
    _write_distance_csv(analysis_dir / "distance_within_group.csv", clips, within_distance_matrix)
    for name, matrix in global_component_matrices.items():
        _write_distance_csv(analysis_dir / f"distance_global_{name}.csv", clips, matrix)
    for name, matrix in within_component_matrices.items():
        _write_distance_csv(analysis_dir / f"distance_within_group_{name}.csv", clips, matrix)
    _json_dump(
        analysis_dir / "recommendations.json",
        {
            "distance_model": {
                "components": ["attack", "timbre", "envelope", "spatial", "level"],
                "weights": DEFAULT_COMPONENT_WEIGHTS,
                "selection_and_scheduler_normalization": "fitted independently inside each filename group",
                "global_context_matrix": "diagnostic only; never used for pool selection or scheduling",
                "warning": "Objective diagnostics require perceptual validation.",
            },
            "groups": [item.json_dict(clips) for item in recommendations],
            "auto_selected_experiment_group": selected_group,
        },
    )
    (analysis_dir / "report.html").write_text(
        _analysis_html(clips, global_distance_matrix, recommendations, selected_group, data_warnings),
        encoding="utf-8",
    )

    run_manifest: dict[str, object] = {
        "protocol": METHOD_VERSION,
        "input_directory": str(args.input_dir.resolve()),
        "files": [
            {
                "path": str(clip.path.resolve()),
                "sha256": source_hashes_snapshot[str(clip.path.resolve())],
                "group": clip.group,
            }
            for clip in clips
        ],
        "settings": {
            "pool_size": args.pool_size,
            "events": args.events,
            "interval_ms": args.interval_ms,
            "clip_seconds": args.clip_seconds,
            "seed": args.seed,
            "analysis_only": args.analysis_only,
            "distance_component_weights": DEFAULT_COMPONENT_WEIGHTS,
            "selection_objective_weights": SELECTION_OBJECTIVE_WEIGHTS,
            "scheduler_history": DEFAULT_SCHEDULER_HISTORY,
            "scheduler_temperature": DEFAULT_SCHEDULER_TEMPERATURE,
            "scheduler_score_weights": SCHEDULER_SCORE_WEIGHTS,
            "normalization_scope": "within filename group for selection and scheduling",
            "sequence_rms_matching": "each complete sequence matched to repeat_one before one common peak scale",
        },
        "implementation_sha256": implementation_hashes_snapshot,
        "runtime": _runtime_context(),
        "git": _git_context(Path(__file__).resolve().parent),
        "data_warnings": data_warnings,
    }

    if selected_recommendation is not None:
        groups = group_index_map(clips)
        full_pool = groups[selected_recommendation.group]
        optimized_pool = list(selected_recommendation.selected_indices)
        if not 2 <= args.pool_size < len(full_pool):
            raise ValueError(
                "Для listening-пилота малый пул должен содержать минимум два "
                "дубля и быть строго меньше полного пула выбранной группы"
            )
        if len(optimized_pool) != args.pool_size:
            raise RuntimeError("Размер оптимизированного пула не совпадает с заданным K")
        sample_rate = clips[full_pool[0]].sample_rate
        bank = normalize_prepared_bank(clips, full_pool)
        schedules = _build_schedules(
            full_pool,
            optimized_pool,
            selected_recommendation.medoid_index,
            within_distance_matrix,
            events=args.events,
            seed=args.seed,
        )
        rendered = {
            method: assemble_sequence(
                bank,
                schedule,
                sample_rate,
                interval_ms=args.interval_ms,
            )
            for method, schedule in schedules.items()
        }
        rendered = rms_match_sequences(rendered, anchor="repeat_one")
        lengths = {audio.shape for audio in rendered.values()}
        if len(lengths) != 1:
            raise RuntimeError("Blind stimuli имеют разные формы")
        for method, audio in rendered.items():
            passed, failures = technical_audio_gate(audio)
            if not passed:
                raise RuntimeError(f"Stimulus {method} не прошёл technical gate: {failures}")

        experiment_dir = args.results_dir / "experiment"
        private_dir = args.results_dir / "private_do_not_open_before_scoring"
        pool_dir = args.results_dir / "optimized_pool" / f"group_{_safe_name(selected_recommendation.group)}"
        experiment_dir.mkdir()
        private_dir.mkdir()
        pool_dir.mkdir(parents=True)

        method_order = list(rendered)
        np.random.default_rng(args.seed + 101).shuffle(method_order)
        blind_mapping = {f"P{position + 1:02d}": method for position, method in enumerate(method_order)}
        for blind_id, method in blind_mapping.items():
            sf.write(experiment_dir / f"{blind_id}.wav", rendered[method], sample_rate, subtype="PCM_16")
        (experiment_dir / "blind_test.html").write_text(
            _blind_html(list(blind_mapping), args.events, args.interval_ms),
            encoding="utf-8",
        )
        method_to_blind = {method: blind_id for blind_id, method in blind_mapping.items()}
        hypothesis_pairs = [
            ("perceptual_full", "shuffle_full", "H1_scheduler_vs_shuffle"),
            ("perceptual_optimized", "shuffle_full", "H2_small_pool_exploratory"),
            ("perceptual_full", "random_full", "H1b_scheduler_vs_random"),
            ("repeat_one", "shuffle_full", "sanity_repeat_vs_natural_pool"),
        ]
        pair_rng = np.random.default_rng(args.seed + 202)
        public_pairs: dict[str, dict[str, str]] = {}
        private_pairs: dict[str, dict[str, str]] = {}
        for position, (left_method, right_method, hypothesis) in enumerate(hypothesis_pairs, start=1):
            methods = [left_method, right_method]
            if bool(pair_rng.integers(0, 2)):
                methods.reverse()
            pair_id = f"Q{position:02d}"
            public_pairs[pair_id] = {
                "A": method_to_blind[methods[0]],
                "B": method_to_blind[methods[1]],
            }
            private_pairs[pair_id] = {
                "hypothesis": hypothesis,
                "A_method": methods[0],
                "B_method": methods[1],
                "A_blind_id": method_to_blind[methods[0]],
                "B_blind_id": method_to_blind[methods[1]],
            }
        (experiment_dir / "pairwise_test.html").write_text(
            _pairwise_html(public_pairs, args.events, args.interval_ms),
            encoding="utf-8",
        )
        _json_dump(
            experiment_dir / "pairwise_manifest_public.json",
            {"protocol": "natural_pool_pairwise_v1", "pairs": public_pairs},
        )

        exported_pool = []
        for position, index in enumerate(optimized_pool, start=1):
            filename = f"take_{position:02d}_{_safe_name(clips[index].metrics.name)}"
            destination = pool_dir / filename
            sf.write(destination, bank[index], sample_rate, subtype="PCM_24")
            exported_pool.append(
                {
                    "source": clips[index].metrics.name,
                    "output": destination.relative_to(args.results_dir).as_posix(),
                    "sha256": _sha256(destination),
                }
            )

        diagnostics = {
            method: schedule_diagnostics(schedule, within_distance_matrix)
            for method, schedule in schedules.items()
        }
        seed_robustness = {
            "full_pool": _scheduler_seed_audit(
                full_pool,
                within_distance_matrix,
                events=args.events,
                seed=args.seed + 20_000,
            ),
            "representative_pool": _scheduler_seed_audit(
                optimized_pool,
                within_distance_matrix,
                events=args.events,
                seed=args.seed + 40_000,
            ),
        }
        public_manifest = {
            "protocol": "natural_pool_blind_pilot_v1",
            "blind_ids": list(blind_mapping),
            "events": args.events,
            "interval_ms": args.interval_ms,
            "sample_rate": sample_rate,
            "channels": int(next(iter(rendered.values())).shape[1]),
            "audio_processing": "onset alignment, bounded within-group gain matching, whole-sequence RMS matching, then one common peak-safety scale; no synthesis, pitch shift, EQ, or morphing",
        }
        _json_dump(experiment_dir / "manifest_public.json", public_manifest)
        _json_dump(
            private_dir / "blind_key.json",
            {
                "warning": "Do not open before ratings are saved.",
                "group": selected_recommendation.group,
                "blind_mapping": blind_mapping,
                "pairwise_key": private_pairs,
                "method_labels": METHOD_LABELS,
                "optimized_pool": [clips[index].metrics.name for index in optimized_pool],
                "schedules": {
                    method: names_for_schedule(schedule, clips)
                    for method, schedule in schedules.items()
                },
                "objective_diagnostics_not_a_listening_verdict": diagnostics,
                "objective_seed_robustness_not_a_listening_verdict": seed_robustness,
            },
        )
        _json_dump(
            pool_dir / "pool_manifest.json",
            {
                "group": selected_recommendation.group,
                "selection_method": "medoid-constrained corpus coverage optimization",
                "files": exported_pool,
            },
        )
        run_manifest["experiment"] = {
            "group": selected_recommendation.group,
            "full_pool_count": len(full_pool),
            "optimized_pool_count": len(optimized_pool),
            "asset_reduction_fraction": 1.0 - len(optimized_pool) / len(full_pool),
            "blind_stimulus_hashes": {
                blind_id: _sha256(experiment_dir / f"{blind_id}.wav")
                for blind_id in blind_mapping
            },
            "public_artifact_hashes": {
                path.name: _sha256(path)
                for path in (
                    experiment_dir / "blind_test.html",
                    experiment_dir / "pairwise_test.html",
                    experiment_dir / "manifest_public.json",
                    experiment_dir / "pairwise_manifest_public.json",
                )
            },
        }

    current_source_paths = [path.resolve() for path in discover_wav_files(args.input_dir)]
    if current_source_paths != source_paths_snapshot:
        raise RuntimeError("Состав входных WAV изменился во время сборки")
    for path in source_paths_snapshot:
        if _sha256(path) != source_hashes_snapshot[str(path)]:
            raise RuntimeError(f"Входной WAV изменился во время сборки: {path.name}")
    for name, path in implementation_paths.items():
        if _sha256(path) != implementation_hashes_snapshot[name]:
            raise RuntimeError(f"Исходный код изменился во время сборки: {path.name}")

    _json_dump(args.results_dir / "run_manifest.json", run_manifest)
    if args.analysis_only:
        verification_report = {
            "passed": (analysis_dir / "report.html").is_file(),
            "scope": "analysis-only package",
        }
    else:
        from verify_natural_pool_package import verify

        verification_report = verify(args.results_dir, require_external=True)
    _json_dump(args.results_dir / "verification_report.json", verification_report)
    if not verification_report["passed"]:
        print(json.dumps(verification_report, ensure_ascii=False, indent=2))
        raise RuntimeError(
            "Staging-пакет не прошёл самопроверку; "
            "целевой каталог не создан."
        )
    if final_results_dir.exists():
        raise FileExistsError(
            f"Целевой каталог появился во время сборки: {final_results_dir}"
        )
    args.results_dir.replace(final_results_dir)
    staging_state["published"] = True
    atexit.unregister(cleanup_staging)
    print(f"[+] Пакет прошёл самопроверку и опубликован атомарно.")
    print(f"[+] Аналитический отчёт: {(final_results_dir / 'analysis' / 'report.html').resolve()}")
    if selected_recommendation is not None:
        print(f"[+] Для пилота выбрана группа {selected_recommendation.group}.")
        print(f"[+] Основной слепой тест: {(final_results_dir / 'experiment' / 'pairwise_test.html').resolve()}")
        print("[+] До сохранения оценок открывайте только experiment; не открывайте analysis, optimized_pool и private-каталог.")


if __name__ == "__main__":
    main()
