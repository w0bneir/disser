# Reference-guided SFX Variations

Исследовательский прототип локальной генерации новых дублей звукового эффекта
по одному аудиореференсу. Метод должен сохранять узнаваемое событие, его
макродинамику и ритм, но создавать слышимые естественные вариации вместо точных
копий.

Тема магистерской диссертации:

> Разработка и оценка метода референс-управляемой генерации вариаций звуковых
> эффектов с сохранением перцептивной идентичности для интерактивных медиа и
> игровых аудиосистем.

Полные формулировки исследовательского вопроса, гипотезы и критериев успеха
зафиксированы в [docs/RESEARCH_SCOPE.md](docs/RESEARCH_SCOPE.md).

## Текущий метод

Ветка Stable Audio Open / Reference SDEdit завершена как абляция. На
`wood_creak` она дала предварительно положительный слуховой результат, но на
валидном `shot_sound` не нашла рабочей области между точной копией и дрейфом
идентичности. Аддитивный residual и spectral magnitude/phase post-processing
тоже остановлены по заранее заданному правилу.

Следующий основной кандидат — **Stable Audio 3 Small-SFX** с нативным
audio-to-audio conditioning и новым semantic-acoustic autoencoder. Он будет
установлен в отдельное окружение после фиксации текущей контрольной точки.
Резервный single-example маршрут — SpecSinGAN. Обоснование выбора и безопасные
gate описаны в [docs/NEXT_MODEL_EVALUATION.md](docs/NEXT_MODEL_EVALUATION.md).

## Текущая контрольная точка

На `wood_creak` и трёх seed Reference SDEdit улучшил Pearson относительно
paired text-only во всех случаях: `0.1257 → 0.6602`, `-0.1176 → 0.5180` и
`0.0348 → 0.7535`. Медианный MSE снизился примерно на `59%`, с `0.1286` до
`0.0525`. Результаты не являются линейными копиями и сохраняют двухчастную
структуру с поздним акцентом около 2.6–2.8 с.

Перенос на `shot_sound` показал, что высокий Envelope Pearson не гарантирует
перцептивную идентичность. Global SDEdit получил identity `2/5`, а
reference-preserving post-processing либо оставался копией, либо снижал
identity до `3/5`. Поэтому прежняя реализация не считается основным методом.

Зафиксированный DSP v1 является сильным контролем: на трёх wood-creak
вариациях медианный Envelope Pearson равен `0.8273`, у Reference SDEdit —
`0.6602`. Поэтому дальнейшая проверка оценивает не только совпадение огибающей,
но и естественность, идентичность события и межвариантное разнообразие.

Решения по экспериментам собраны в [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md),
а заранее определённый протокол — в
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Структура проекта

```text
configs/
  reference_variations.json       основная same-family серия
  dsp_baseline.json               зафиксированный DSP v1
  text_probe.json                 диагностическая text-only генерация
  stress_tests/                   дополнительные стресс-тесты
docs/
  RESEARCH_SCOPE.md               тема, вопрос, гипотеза и критерии
  EXPERIMENT_PROTOCOL.md          сравниваемые методы и метрики
  METRICS.md                      точные определения CPU-метрик v1
  RESULTS_INDEX.md                индекс значимых экспериментов
references/                       исходные WAV-референсы
requirements/                     зависимости для разных поколений GPU
tools/                            preflight, мониторинг и служебные сценарии
results/                          локальные артефакты запусков, не входят в Git
archive/audioldm/                 исторические эксперименты AudioLDM
run_stable_audio_experiments.py   основной безопасный runner
run_dsp_baseline.py               CPU pitch/time/EQ-контроль
evaluate_reference_variations.py единая CPU-оценка четырёх методов
prepare_listening_test.py         сборка анонимного listening-пакета
analyze_listening_test.py         раскрытие ключа после полной оценки
dsp_baseline.py                   воспроизводимые DSP-преобразования
sfx_metrics.py                    структура, спектр и non-copy метрики
stable_audio_guidance.py          denoising и reference-latent методы
stable_audio_probe.py             диагностический text-only probe
test_*.py                         CPU-модульные тесты
```

## Окружение RTX 5070

Рабочая конфигурация:

- Windows, Python 3.11, conda `sfx_gen_5070`;
- PyTorch 2.7.1 с CUDA 12.8;
- NVIDIA RTX 5070, 12227 MiB VRAM;
- Stable Audio Open 1.0 с model CPU offload.

Для чистого окружения сначала устанавливается сборка PyTorch под GPU, затем
общие зависимости:

```bat
conda activate sfx_gen_5070
cd /d C:\Users\godmi\Projects\disser
python -m pip install -r requirements\windows-cu128-blackwell.txt
python -m pip install -r requirements\base.txt
python verify_setup.py
```

Корневой `requirements.txt` содержит только общие зависимости и намеренно не
выбирает CUDA-сборку автоматически. Для старых GPU существует отдельный файл
`requirements\windows-cu121-legacy.txt`.

## Проверки без генерации

После изменения кода:

```bat
python -B -m unittest discover -v
python -m pip check
```

Перед каждым GPU-запуском:

```bat
git pull --ff-only
powershell -ExecutionPolicy Bypass -File tools\preflight_gpu.ps1
python run_stable_audio_experiments.py --preflight-only --config configs\reference_variations.json --max-new-pairs 1
```

Предохранитель требует не менее 12000 MiB общей и 10000 MiB свободной VRAM.
Параметр `--allow-unsafe-vram` не используется в основном исследовании.

## DSP-контроль и единая оценка

DSP v1 полностью выполняется на CPU и создаёт три pitch/time/EQ-вариации для
каждого референса:

```bat
python run_dsp_baseline.py --resume
```

Когда paired text-only и Reference SDEdit готовы в одном каталоге, четыре
метода оцениваются одной командой:

```bat
python evaluate_reference_variations.py --case-id wood_creak --reference references\wood_creak.wav --generation-results-dir results\2026-08-19_wood_reference_sde_50step_01 --output-dir results\2026-08-19_wood_reference_sde_50step_01\evaluation_3seed_v1 --seeds 17 42 2026
```

Создаются `file_metrics.csv`, сводки по методам и разнообразию, а также общий
график RMS-огибающих. Все сигналы анализируются при 44,1 кГц.

## Безопасный запуск Reference SDEdit

Любой новый маршрут сначала проверяется четырёхшаговым smoke-test одной пары:

```bat
python run_stable_audio_experiments.py --config configs\reference_variations.json --smoke-test --case-id wood_creak --seed 42 --reference-sde-strength 0.30 --export-latent-diagnostics --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\YYYY-MM-DD_wood_reference_sde_smoke_01
```

Только после успешного smoke-test запускается одна 50-шаговая пара:

```bat
python run_stable_audio_experiments.py --config configs\reference_variations.json --num-inference-steps 50 --case-id wood_creak --seed 42 --reference-sde-strength 0.30 --export-latent-diagnostics --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\YYYY-MM-DD_wood_reference_sde_50step_01
```

Для продолжения прерванной серии используется тот же каталог и `--resume`.
Одновременный запуск нескольких генераций запрещён. Внешний монитор VRAM
рекомендуется оставлять включённым на весь GPU-прогон.

## Что считается результатом исследования

Один удачный WAV не подтверждает метод. Для каждой серии сравниваются:

- неизменённый Repeat;
- умеренные pitch/time/EQ-модификации;
- text-only Stable Audio;
- Reference SDEdit.

Текущий `sfx_metrics_v1` оценивает огибающую, атаки, спектр, внутрипакетное
разнообразие и отсутствие тривиального копирования. Аудиоэмбеддинги,
VRAM/время и слепые оценки слушателей добавляются в итоговый протокол отдельно.
Веб-интерфейс создаётся после того, как метод даст воспроизводимый результат на
нескольких референсах и seed.

Текущий пилотный listening-пакет собирается по протоколу
`docs/LISTENING_PROTOCOL.md`. Ключ рандомизации хранится отдельно и не
раскрывается до заполнения индивидуальной и пакетной таблиц.

## Данные и воспроизводимость

`results/`, кэши моделей и крупные артефакты не коммитятся. Значимые численные
выводы вручную заносятся в `docs/RESULTS_INDEX.md`, а каждый каталог эксперимента
должен содержать параметры, метрики и контекст запуска. Исходные WAV в
`references/` не изменяются во время серии.
