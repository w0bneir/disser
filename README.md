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

```bat
python run_stable_audio_experiments.py --results-dir results\stable_audio_trial_gamma20 --max-new-pairs 1 --cooldown-seconds 20
```

Полный пробный прогон использует `gamma=20`, ограничение градиента `0.1` и
дополнительный предел коррекции 3% от нормы активного латента на шаг. Он также
сохраняет `guidance_trace.csv` и `guidance_diagnostics.png` для проверки того,
что latent-loss действительно уменьшается.
