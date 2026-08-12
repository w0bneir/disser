# Черновой демонстратор Direct Latent Guidance для SFX

Этот проект показывает управление **временной динамикой** предобученной AudioLDM без обучения и дообучения ее весов. Пользовательский WAV задает целевую RMS-огибающую `E_target`, а английский prompt задает акустическую текстуру SFX. Во время обратной диффузии корректируется только латентный вектор.

## Что будет получено

Для двух референсов, трех фиксированных seed (`17`, `42`, `2026`) и двух режимов создаются 12 WAV-файлов:

- `baseline.wav` — обычная генерация AudioLDM;
- `guided.wav` — генерация с Direct Latent Guidance.

Для каждой пары сохраняется график трех RMS-огибающих. В `results/metrics.csv` находятся MSE, корреляция Пирсона, время и пиковая VRAM, а `summary_metrics.png` показывает средние метрики по seed.

## Подготовка

Работайте из среды Miniconda `sfx_gen`. В обычном системном терминале `conda` может не быть в `PATH` — откройте Miniconda Prompt либо активируйте среду полным путем.

```powershell
conda activate sfx_gen
cd D:\YandexDisk\disser
python verify_setup.py
```

Если в среде отсутствуют библиотеки, установите зависимости из `requirements.txt`. Версии PyTorch в этом файле рассчитаны на CUDA 12.1 и GTX 1070.

```powershell
pip install -r requirements.txt
```

## Референсы и конфиг

Поместите два собственных WAV-файла в `references/`:

- `metal_impact.wav` — короткий удар;
- `wood_creak.wav` — более протяженный звук.

Пути, prompt и параметры находятся в `experiments.json`. Prompt описывает материал и характер звука, но не его длительность и динамику. В конфиге зафиксированы: 30 шагов диффузии, CFG `7.5`, `gamma=20.0`, ограничение нормы градиента `0.1` и три seed.

## Запуск

### 1. Новая базовая модель: Stable Audio Open

AudioLDM 1 оставлен в проекте как зафиксированный неудачный baseline, но больше не используется для новых экспериментов. Новая контрольная модель — `stabilityai/stable-audio-open-1.0`: она рассчитана на текстовую генерацию stereo-аудио и принимает требуемую длительность как условие модели.

`stable_audio_probe.py` вызывает только официальный `StableAudioPipeline`, без ручного цикла и без guidance. На GTX 1070 он загружает модель в FP16 и выгружает неактивные компоненты в CPU, поэтому запускайте по одному образцу и прослушивайте `audio.wav`:

```powershell
python stable_audio_probe.py --max-new-runs 1 --cooldown-seconds 8
python stable_audio_probe.py --resume --max-new-runs 1 --cooldown-seconds 8
```

Повторите вторую команду, пока в `results\stable_audio_probe\` не появятся шесть вариантов. Если хотя бы один prompt даёт узнаваемый SFX в нескольких seed, Stable Audio Open становится новой основой для guidance. Если модели нет в локальном кеше, добавьте `--allow-download`.

### 2. Guidance-эксперимент

Сначала выполните короткую проверку формы тензоров и связки VAE → вокодер. Она использует один пример, один seed и один шаг денойзинга:

```powershell
python run_experiments.py --smoke-test --results-dir results\smoke
```

Затем запустите полный эксперимент:

```powershell
python run_experiments.py --config experiments.json --results-dir results
```

На GTX 1070, которая одновременно выводит изображение Windows, безопаснее выполнять по одной новой паре `baseline/guided`. После каждой пары скрипт освобождает CUDA-кеш, сохраняет `metrics.csv` и делает паузу. Если запуск был прерван, продолжайте его без перегенерации готовых WAV:

```powershell
python run_experiments.py --config experiments.json --results-dir results --resume --cooldown-seconds 5 --max-new-pairs 1
```

Повторяйте последнюю команду, пока не будут сформированы все 12 WAV. При обычной стабильной работе ограничения `--max-new-pairs` не нужны.

Структура полного результата:

```text
results/
  metal_impact/seed_17/{baseline.wav, guided.wav, envelope_comparison.png}
  metal_impact/seed_42/...
  metal_impact/seed_2026/...
  wood_creak/seed_17/...
  wood_creak/seed_42/...
  wood_creak/seed_2026/...
  metrics.csv
  summary_metrics.png
```

## Интерпретация

`mse` — ошибка между нормированными RMS-огибающими, поэтому меньшее значение лучше. `pearson_correlation` измеряет сходство формы кривых, поэтому большее значение лучше. Для демонстрации гипотезы guided-вариант должен показать меньшую среднюю MSE хотя бы на одном референсе; целевой результат — улучшение на обоих.

Direct Latent Guidance оценивает временную энергию из `z-hat-0` как RMS по канальному и частотному измерениям латента. VAE не вызывается внутри цикла денойзинга: она декодирует только финальный латент, после чего применяется штатный метод AudioLDM `mel_spectrogram_to_waveform`. Это снижает нагрузку на VRAM и исключает ошибку пяти измерений на входе вокодера.

## Проверки без модели

```powershell
python -m unittest -v test_guided_pipeline.py
```

Они проверяют нормализацию и форму latent-огибающей, интерполяцию, метрики, клиппинг градиента, короткий WAV и отклонение некорректной 5D-мели до запуска вокодера.
