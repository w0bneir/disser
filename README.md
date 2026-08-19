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

Основная экспериментальная ветка использует **Reference SDEdit**:

1. WAV-референс кодируется VAE модели Stable Audio Open 1.0.
2. Его latent-представление умеренно зашумляется с заданным seed.
3. Модель выполняет условный денойзинг с текстовым описанием того же семейства
   SFX.
4. Результат сравнивается с text-only генерацией из того же шума и с честным
   DSP-baseline.

Параметр `--reference-sde-strength` управляет компромиссом между сохранением
референса и разнообразием. Первая зафиксированная серия использует strength
`0.30`: при расписании из 50 шагов выполняются 15 эффективных SDEdit-шагов.

Прежние latent-RMS, probe, final decoder и decoder-denoising режимы сохранены
для абляционного раздела. Они снижали внутренний loss, но не обеспечили
достаточного переноса длинной временной структуры.

## Текущая контрольная точка

На `wood_creak`, seed 17 и same-family prompt Reference SDEdit улучшил Envelope
Pearson с `0.1257` до `0.6602`, снизил MSE с `0.1286` до `0.0436` и перенёс
слышимый поздний акцент около 2.65 с. Waveform результата не является простой
копией референса. Это обнадёживающий калибровочный результат, но ещё не
подтверждение гипотезы: обязательны повторения на seed 42, 2026, новых
референсах и слепая слуховая оценка.

Зафиксированный DSP v1 является сильным контролем: на трёх wood-creak
вариациях медианный Envelope Pearson равен `0.8273`, тогда как текущий
Reference SDEdit seed 17 даёт `0.6602`. Поэтому дальнейшая проверка оценивает
не только совпадение огибающей, но и естественность, идентичность события и
межвариантное разнообразие.

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

## Данные и воспроизводимость

`results/`, кэши моделей и крупные артефакты не коммитятся. Значимые численные
выводы вручную заносятся в `docs/RESULTS_INDEX.md`, а каждый каталог эксперимента
должен содержать параметры, метрики и контекст запуска. Исходные WAV в
`references/` не изменяются во время серии.
