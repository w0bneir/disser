# SFX Direct Latent Guidance — черновой демонстратор

Текущая рабочая основа проекта — **Stable Audio Open 1.0**. На этом этапе
мы проверяем, что текстовая модель сама создаёт узнаваемые SFX, а затем
добавим управление динамикой по RMS-огибающей пользовательского WAV.

AudioLDM-эксперименты сохранены как архив: модель дала некачественный
звуковой результат для этой задачи, поэтому новые запуски на ней не ведутся.

## Структура

```text
analyzer.py                 извлечение RMS-огибающей референса
audio_io.py                 безопасное сохранение WAV без клиппинга
stable_audio_probe.py       текущий baseline Stable Audio без guidance
stable_audio_probe.json     prompt, seed и параметры baseline
stable_audio_guidance.py    ручной denoising и Direct Latent Guidance
run_stable_audio_experiments.py  безопасный эксперимент baseline/guided
stable_audio_experiments.json    конфигурация эксперимента с референсами
references/                 пользовательские WAV-референсы
results/stable_audio_probe/ актуальные контрольные результаты
archive/audioldm/           старый код AudioLDM, сохранён для истории
results/archive_audioldm/   прежние WAV, графики и метрики AudioLDM
```

## Подготовка

Откройте **Miniconda Prompt**, активируйте среду и перейдите в проект:

```bat
conda activate sfx_gen
cd /d D:\YandexDisk\disser
python verify_setup.py
```

Для разделённой схемы «код локально, GPU удалённо» и настройки RTX 5070
используйте [инструкцию удалённого запуска](docs/REMOTE_EXECUTION.md).

Проверка библиотек без запуска модели:

```bat
python -m pip check
python -c "from diffusers import StableAudioPipeline; print('Stable Audio: OK')"
```

## Безопасный запуск baseline

Запускайте только **один** новый вариант за раз. Уже созданный вариант
`metal_impact / seed_17` будет пропущен благодаря `--resume`.

```bat
python stable_audio_probe.py --resume --max-new-runs 1 --cooldown-seconds 15
```

Результат появляется в `results\stable_audio_probe\<id>\seed_<seed>\`:

- `audio.wav` — сгенерированный звук;
- `metadata.json` — prompt, seed, время и пиковая VRAM.

Если Windows начинает заметно тормозить, сразу нажмите `Ctrl+C`. Не запускайте
несколько вариантов подряд и не используйте пока команду без `--max-new-runs 1`.

## Следующий этап

Guidance реализован, но ещё не запускался на GPU: референс задаёт форму
RMS-огибающей, а prompt — акустический материал. Перед первым запуском пройдите
модульные проверки (они не загружают модель):

```bat
python -B -m unittest -v test_stable_audio_guidance.py
```

Ручной guidance теперь заблокирован на GPU с менее чем **12 ГБ VRAM**: Stable
Audio Open с 8 ГБ (включая GTX 1070 и RTX 3070 Laptop) способен подвесить
драйвер Windows. В CFG ветви выполняются последовательно, без batch=2, что
снижает пиковую VRAM ценой примерно двукратного времени denoising.

Первый GPU smoke-test запускайте только на GPU с 12+ ГБ VRAM, одной парой
baseline/guided. Он использует четыре шага denoising; VAE декодирует звук только
после каждого полного цикла:

```bat
python run_stable_audio_experiments.py --smoke-test --max-new-pairs 1 --cooldown-seconds 20
```

Если система остаётся отзывчивой и в `results\stable_audio_guidance\` созданы
два WAV, следующий шаг — один полный 50-шаговый pair:

Перед ним выполните промежуточные одиночные прогоны на 10 и 20 шагах. Они
требуют явного `--max-new-pairs 1`, поэтому не смогут случайно запустить серию
GPU-пар:

```bat
python run_stable_audio_experiments.py --num-inference-steps 10 --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\stable_audio_10step
python run_stable_audio_experiments.py --num-inference-steps 20 --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\stable_audio_20step
```

Только после проверки этих двух результатов можно запускать один полный
50-шаговый pair:

```bat
python run_stable_audio_experiments.py --results-dir results\stable_audio_trial_gamma20 --max-new-pairs 1 --cooldown-seconds 20
```

Полный пробный прогон использует `gamma=20`, ограничение градиента `0.1` и
дополнительный предел коррекции 3% от нормы активного латента на шаг. Он также
сохраняет `guidance_trace.csv` и `guidance_diagnostics.png` для проверки того,
что latent-loss действительно уменьшается.

Референс и сгенерированный WAV анализируются на одной частоте модели (44,1 кГц).
Это принципиально для корректной RMS-метрики: окно в 2048 отсчётов должно иметь
одинаковую физическую длительность у обеих сравниваемых записей.

Для одиночной проверки силы guidance можно переопределить `gamma`, не изменяя
JSON-конфигурацию. Диапазон ограничен пользовательским интервалом `(0, 50]`, а
команда обязательно требует предохранитель `--max-new-pairs 1`:

```bat
python run_stable_audio_experiments.py --num-inference-steps 20 --gamma 50 --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\stable_audio_20step_gamma50
```

Чтобы `gamma` не ослабевал с ростом длительности SFX, mean-loss gradient
нормируется относительно 0,5 секунды latent-времени. Для файлов короче этого
порога поведение не меняется; для более длинных коррекция усиливается
пропорционально числу активных latent-позиций и всё равно ограничивается
`max_relative_step`.

Один конкретный case и seed можно безопасно выбрать без изменения JSON. Это
нужно для изолированного smoke-test после алгоритмических изменений:

```bat
python run_stable_audio_experiments.py --smoke-test --case-id wood_creak --seed 17 --gamma 50 --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\wood_duration_scaled_smoke
```

Для построения waveform-aware envelope probe предусмотрен диагностический
экспорт. Флаг сохраняет `latent_diagnostics.npz` с active latents и выровненными
target/latent/waveform-огибающими. NPZ содержит только числовые массивы и
открывается с `allow_pickle=False`; экспорт всегда ограничен одной GPU-парой:

```bat
python run_stable_audio_experiments.py --smoke-test --case-id wood_creak --seed 17 --gamma 50 --export-latent-diagnostics --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\wood_latent_diagnostics_smoke
```

Следующий этап не подключается к generation автоматически: сначала по
нескольким независимым NPZ обучается малый waveform-aware probe. Он использует
знаковую ridge-проекцию latent-каналов: это учитывает, что VAE декодирует каналы
с разными знаками, и остаётся полностью differentiable по входному latent.
Ridge alpha выбирается leave-one-pair-out проверкой только внутри train-набора.
Разделение
train/validation выполняется по целым baseline/guided-парам, поэтому два почти
одинаковых результата одного seed не могут попасть по разные стороны проверки.
По умолчанию обучение CPU требует минимум шесть независимых 50-шаговых пар
(текущий набор: два case × три seed). Smoke-результаты в обучающий набор не
включаются, потому что распределение финальных latents после 4 и 50 шагов
различается:

```bat
python train_envelope_probe.py results\probe_dataset --output models\envelope_probe.safetensors
```

Вместе с весами сохраняется JSON-отчёт, где качество probe на отложенных парах
сопоставлено с текущей latent-RMS огибающей. Подключать probe к guidance можно
только после улучшения validation Pearson без ухудшения validation MSE.

На наборе `2026-08-18_probe_dataset_50step_01` версия
`signed_latent_ridge_v2` выбрала `alpha=1.0` внутренним leave-one-pair-out CV.
На полностью отложенном seed 2026 для обоих case средний Pearson вырос с
`0.6166` до `0.8238`, а MSE снизился с `0.0951` до `0.0360`. Контрольный барьер
пройден; checkpoint сохранён как `models/envelope_probe.safetensors`.

Probe подключается только явно через `--envelope-probe` и пока остаётся
ограниченным одной парой. Старый latent-RMS без этого флага работает без
изменений. Первый GPU-запуск после интеграции должен быть только 4-шаговым
smoke-test с новым каталогом результатов:

```bat
python run_stable_audio_experiments.py --smoke-test --case-id wood_creak --seed 17 --gamma 50 --envelope-probe models\envelope_probe.safetensors --probe-guidance-mode final --final-guidance-steps 10 --export-latent-diagnostics --max-new-pairs 1 --cooldown-seconds 20 --results-dir results\wood_probe_guidance_smoke
```

Режим `denoising` оставлен для абляционного сравнения, но не рекомендуется:
35 локальных ограничений по 3% накопились в итоговое отклонение latent на 12.5%
и позволили оптимизации обмануть surrogate. Режим `final` сначала завершает
обычный denoising, затем оптимизирует только final latent. Каждая временная
позиция всё время остаётся внутри 3% от исходного anchor, поэтому локальные
шаги не могут накопиться в неограниченный drift.
