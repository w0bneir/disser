# Перцептивно ограниченный синтез вариаций SFX — research prototype

Локальный CPU-прототип для создания вариаций одного SFX в диапазоне изменений,
измеренном по натуральным дублям того же события. Цель — получить слышимо новые,
но естественные дубли без свободной нейрогенерации, потери атаки и дополнительных
импульсных артефактов. Natural-pool optimizer сохранён как baseline и как источник
калибровочной статистики.

## Текущий исследовательский вопрос

Можно ли оценить допустимую структуру и величину вариативности по группам
натуральных дублей, а затем использовать эту статистику для создания новых
вариаций одного референса, которые слышимо отличаются, сохраняют идентичность
события и не получают заметных артефактов?

Основные проверяемые гипотезы:

1. сегментно-зависимые изменения огибающей, спектрального баланса и stereo width,
   ограниченные статистикой натуральных дублей, сохраняют идентичность события;
2. существует диапазон силы преобразования между неслышимой копией и потерей
   естественности — перцептивный коридор;
3. такой метод даёт больше полезных вариаций, чем фиксированные pitch/time/EQ
   baseline при сопоставимой идентичности события.

Для будущего подтверждающего теста ещё нужно заранее зафиксировать первичный показатель,
практически значимую границу, правило вывода по доверительному интервалу и единицу анализа.

## Что уже работает

- stereo-preserving загрузка WAV и автоматическое определение атаки;
- отдельный анализ атаки, тембра, хвоста, пространства и уровня;
- составная матрица различий с устойчивым масштабированием;
- выбор репрезентативного поднабора заданного размера по покрытию исходного
  корпуса с обязательным центральным дублем;
- три стратегии: Random, Shuffle/no-repeat и content-aware scheduler v1;
- равное использование файлов в Shuffle и content-aware режиме при совместимой длине
  серии;
- штрафы немедленных повторов, повторяющихся переходов и триграмм;
- экспорт выровненного 24-bit пула;
- слепые 16-bit browser-compatible последовательности, отдельный закрытый ключ,
  абсолютная и основная парная формы прослушивания;
- воспроизводимый manifest с SHA-256 исходных WAV, слепых стимулов и
  экспортированного пула.
- профиль натуральной вариативности по 45 динамическим, спектральным и
  пространственным признакам;
- перенос bounded natural delta на один референс с защищённой атакой;
- дозы `low/mid/high`, натуральный ceiling-контроль и слепой sequence gate.

Объективные показатели используются только как диагностика. Решение о слышимой
пользе принимает слепое прослушивание.

## Безопасный запуск

Активный маршрут работает только на CPU и не загружает генеративные модели или
GPU. В окружении `sfx_gen_5070`:

```powershell
python -m unittest -v `
  test_sfx_pool_optimizer.py `
  test_analyze_natural_pool_ratings.py `
  test_natural_pool_pipeline.py `
  test_take_discriminability_gate.py `
  test_analyze_take_discriminability_ratings.py `
  test_perceptual_variation_synthesis.py `
  test_perceptual_variation_draft.py `
  test_analyze_perceptual_variation_ratings.py

python run_perceptual_variation_draft.py `
  --input-dir references\group_1 `
  --group 1 `
  --events 8 `
  --interval-ms 1200 `
  --results-dir results\YYYY-MM-DD_perceptual_variation_draft_v0_01

python analyze_perceptual_variation_ratings.py `
  --draft-dir results\YYYY-MM-DD_perceptual_variation_draft_v0_01 `
  --ratings C:\path\to\perceptual_variation_ratings.json `
  --output-dir results\YYYY-MM-DD_perceptual_variation_analysis_v0_01

python run_natural_pool_pilot.py `
  --input-dir references\group_1 `
  --experiment-group 1 `
  --events 15 `
  --interval-ms 800 `
  --pool-size 3 `
  --results-dir results\YYYY-MM-DD_natural_pool_pilot_01
```

Если sequence-пилот не отличает даже Repeat-one от Shuffle, сначала запускается
отдельный gate различимости исходных дублей:

```powershell
python run_take_discriminability_gate.py `
  --input-dir references\group_1 `
  --group 1 `
  --events 8 `
  --interval-ms 1200 `
  --results-dir results\YYYY-MM-DD_take_discriminability_gate_01
```

Раскрытие заполненной анкеты:

```powershell
python analyze_take_discriminability_ratings.py `
  --gate-dir results\YYYY-MM-DD_take_discriminability_gate_01 `
  --ratings C:\path\to\take_discriminability_ratings.json `
  --output-dir results\YYYY-MM-DD_take_discriminability_analysis_01
```

`requirements.txt` задаёт совместимые диапазоны для разработки. Точные версии
Python-пакетов, использованные проверенным пилотом `v1_10`, зафиксированы в
`requirements-natural-pool-lock.txt`; Python, ОС и версия libsndfile также
записываются в `run_manifest.json` каждого пакета.

Сценарий отказывается писать в уже существующий каталог результатов. Пакет сначала собирается
в staging-каталоге, проходит самопроверку и только потом атомарно публикуется. Оригинальные WAV не
изменяются.

После выполнения сначала пройдите слепой тест и только потом смотрите аналитику:

1. `experiment/pairwise_test.html` — основной слепой тест;
2. `experiment/blind_test.html` — необязательная дополнительная оценка всех серий;
3. `analysis/report.html` — технический отчёт и выбранный поднабор, только после сохранения оценок.

Каталог `private_do_not_open_before_scoring/` разрешается открывать только после
скачивания ответов из HTML.

Внешнему слушателю передаётся **только** папка `experiment/`. Корень пакета,
`analysis/`, `optimized_pool/` и закрытый ключ остаются у исследователя до
получения ответов.

Проверка готового пакета:

```powershell
python verify_natural_pool_package.py `
  results\YYYY-MM-DD_natural_pool_pilot_01 `
  --require-external
```

Флаг `--require-external` предназначен для строгой локальной проверки: он также
сверяет исходные WAV и текущую реализацию. Без флага проверка остаётся
переносимой и подтверждает внутреннюю целостность уже собранного пакета, даже
если исходники отсутствуют на другом компьютере.

После сбора ответов и раскрытия ключа:

```powershell
python analyze_natural_pool_ratings.py `
  --pilot-dir results\YYYY-MM-DD_natural_pool_pilot_01 `
  --ratings path\to\listener_*.json `
  --output-dir results\YYYY-MM-DD_natural_pool_ratings_01
```

## Материал пилота

В `references/group_1/` находятся 26 stereo PCM24/44.1 kHz файлов СКС,
разделённых на шесть групп `5 + 5 + 5 + 5 + 3 + 3`. По сообщению владельца это
натуральные записи без намеренной обработки, однако WAV экспортированы из
REAPER, а точное происхождение микрофонных слоёв пока не документировано.

В текущем пилоте группы не объединяются: акустический контекст и стереокартина
между ними различаются существенно сильнее, чем дубли внутри группы. Сходство
длин и стереоподписей допускает, но не доказывает, что пары групп 3/4 и 5/6 могут
быть параллельными микрофонными перспективами одних физических выстрелов. До
уточнения метаданных каждая группа анализируется отдельно.

## Структура активного метода

- `perceptual_variation_synthesis.py` — оценка natural variation profile и
  bounded перенос огибающей, спектра и stereo width на один референс;
- `run_perceptual_variation_draft.py` — атомарная сборка первого слепого
  synthesis gate;
- `test_perceptual_variation_synthesis.py` и
  `test_perceptual_variation_draft.py` — unit и end-to-end проверки нового
  метода;
- `analyze_perceptual_variation_ratings.py` и
  `test_analyze_perceptual_variation_ratings.py` — строгое раскрытие synthesis
  gate и проверка его критерия успеха;
- `sfx_pool_optimizer.py` — анализ, расстояния, выбор пула, scheduler и rendering;
- `run_natural_pool_pilot.py` — воспроизводимый анализ и слепой пилот;
- `test_sfx_pool_optimizer.py` — автоматические проверки активного метода;
- `test_analyze_natural_pool_ratings.py` — проверки валидации и раскрытия анкет;
- `test_natural_pool_pipeline.py` — интеграционные проверки сборки, staging,
  verifier и обработки ответов;
- `run_take_discriminability_gate.py` — direct-take и short-loop gate перед
  дальнейшей настройкой scheduler;
- `test_take_discriminability_gate.py` — проверки выбора near/median/far пар и
  сборки слепого gate-пакета;
- `analyze_take_discriminability_ratings.py` — строгая расшифровка заполненного
  gate с учётом случайной перестановки A/B;
- `test_analyze_take_discriminability_ratings.py` — проверки расшифровки,
  повреждённых blind ID и неполных ответов;
- `verify_natural_pool_package.py` — проверка слепоты, WAV и SHA-256 пакета;
- `analyze_natural_pool_ratings.py` — раскрытие и агрегирование оценок после теста;
- `requirements-natural-pool-lock.txt` — точные CPU-зависимости проверенного
  пилота `v1_10`;
- `docs/NATURAL_POOL_METHOD.md` — научная формализация;
- `docs/NATURAL_POOL_PILOT_PROTOCOL.md` — зафиксированный протокол пилота.
- `docs/TAKE_DISCRIMINABILITY_GATE.md` — диагностический протокол после
  непрохождения sequence sanity-check.
- `docs/SUPERVISOR_DEMO_BRIEF.md` — одностраничная памятка для показа научному руководителю.
- `docs/PROJECT_STATUS_2026-08-31.md` — точка продолжения, проверки и следующий шаг.

Предыдущие reference-conditioned нейросетевые, токенные, DSP и однореференсные
sequence-level маршруты сохранены как отрицательный исследовательский результат
в `archive/`. Они не являются зависимостью активного прототипа.
