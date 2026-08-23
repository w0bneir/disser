# Prompt-free VampNet draft

## Зачем введён новый маршрут

Итоговая система не должна требовать удачного текстового промпта: пользователь
загружает один SFX и получает новые дубли того же события. Text-to-audio и
Reference SDEdit оставлены как исследовательские baseline. Текущий основной
черновик использует masked acoustic token modeling из VampNet: референс сначала
кодируется в дискретные акустические токены, затем пересэмплируется только их
часть.

Это проверяет более прямую гипотезу, чем text-guided denoising:

> низкие codebook-и могут удерживать идентичность и временной каркас события,
> пока высокие codebook-и создают отличия тембра и микродеталей.

Текстовый промпт в прототипе отсутствует.

## Зафиксированный draft v1

- tokenizer: официальный LAC checkpoint VampNet, 44,1 кГц, 14 codebook-ов;
- codebook 0–2 всегда копируются из референса;
- начиная с codebook 3 токены пересэмплируются;
- один временной якорь сохраняется через каждые 7 token-frame;
- первые 80 мс атаки сохраняются во всех codebook-ах;
- temperature 0.9, 12 coarse sampling steps, один c2f проход;
- выход v1: mono, 44,1 кГц, PCM24, исходная длительность и loudness;
- seeds первой серии: 17, 42, 2026.

Параметры выбраны до прослушивания полного пакета и не подгонялись под отдельный
результат.

## Первый технический результат на shot_sound

Демонстрационный каталог: `results/2026-08-23_vampnet_shot_token_draft_02`.

- длительность каждого WAV: 1,7142857 с;
- время одной вариации после загрузки: 0,33–0,52 с;
- пик VRAM: 3682 MiB;
- codebook 0–2 не изменились ни в одном seed;
- суммарно изменилось 52,3–52,7% акустических токенов;
- корреляция RMS-огибающей с reference: 0,9967–0,9972;
- попарная waveform-корреляция вариаций: 0,656–0,713.

Числа подтверждают только технические свойства: одинаковую структуру и
отсутствие точного копирования. Они **не подтверждают** перцептивную
идентичность, естественность и практическую полезность. Эти три критерия должны
пройти слуховой gate.

Повторный запуск с теми же параметрами дал побитово идентичные WAV. SHA256:

- seed 17: `30261c6bedc91024d4f2c29e6ee56a2a2001c5068f701d0be7853baf1cadeac5`;
- seed 42: `7e21f018a399ffae0853e265dbc97d73bbd0a5bc54fb814045da6dd47f9ee324`;
- seed 2026: `1c6d08e8512a5b19547ecb8b8e154d573e878f694e1811cb355f020218c2654d`.

Для демонстрации научному руководителю открыть локальный `demo.html` внутри
каталога результата и последовательно прослушать reference, codec round-trip и
три вариации.

## Безопасный запуск

Основное conda-окружение не изменяется. Изолированный runtime находится в
игнорируемом `artifacts/vampnet_env`, исходники и веса — в `artifacts/`.

Проверка codec без генерации:

```bat
artifacts\vampnet_env\Scripts\python.exe run_vampnet_reference_variations.py --codec-gate --reference references\shot_sound.wav --results-dir results\YYYY-MM-DD_vampnet_codec_gate
```

Одна короткая генеративная проверка:

```bat
artifacts\vampnet_env\Scripts\python.exe run_vampnet_reference_variations.py --generate --reference references\shot_sound.wav --seeds 17 --upper-codebook-mask 3 --periodic-prompt 7 --attack-ms 80 --temperature 0.9 --sampling-steps 2 --results-dir results\YYYY-MM-DD_vampnet_smoke
```

Зафиксированный пакет запускается только после успешного smoke:

```bat
artifacts\vampnet_env\Scripts\python.exe run_vampnet_reference_variations.py --generate --reference references\shot_sound.wav --seeds 17 42 2026 --upper-codebook-mask 3 --periodic-prompt 7 --attack-ms 80 --temperature 0.9 --sampling-steps 12 --results-dir results\YYYY-MM-DD_vampnet_draft
```

Предохранитель требует минимум 12000 MiB общей и 10000 MiB свободной VRAM.

## Воспроизводимость и лицензия

- VampNet source commit: `72e2675790091fe28ecfd8391303a46b25a703db`;
- LAC source commit: `7761206878d1fba79aad314a38f975e9589af0a4`;
- веса проверяются по точному размеру до загрузки;
- `codec.pth`: 600996465 bytes;
- `coarse.pth`: 1332182321 bytes;
- `c2f.pth`: 1101898865 bytes.

SHA256 также проверяется до загрузки:

- `codec.pth`: `3db3fa43ab5d160439ddb81fc540b5573ad5ae962230de3fc5b47d218845b855`;
- `coarse.pth`: `78e4ad4f8398e8ec3651bc5e5c6ea2995e1080b6226be186723ccf4320c9756c`;
- `c2f.pth`: `b10ea2d45459d34edb773cbacd71f40f7baa1f4e75ac8bcd93b022ac69f8fa63`.

README VampNet указывает для pretrained weights лицензию CC BY-NC-SA 4.0.
Поэтому текущий маршрут подходит для некоммерческого диссертационного
прототипа, но ограничение должно быть пересмотрено до коммерческого продукта.

## Gate дальнейшей работы

Сначала оценивается `draft_01` без изменения параметров. Продолжение имеет
смысл, если вариации воспринимаются как тот же выстрел, естественны и слышимо
различаются. При копиях меняется глубина mask; при артефактах mask делается
консервативнее или выполняется SFX fine-tuning. Параметрический sweep до
слухового заключения запрещён.

### Listening gate draft v1

Результат прослушивания `shot_token_draft_02`:

- идентичность события сохранена;
- полезного различия между дублями нет;
- слышна небольшая металлическая окраска;
- дополнительных выстрелов нет.

Вердикт: v1 технически корректен, но задачу machine-gun effect пока не решает.
Следующая причинная проверка — профиль `tiered-mid`: изменение переносится из
fine detail в codebook 2–3, а в codebook 4–13 пересэмплируется только один из
четырёх token-frame. Атака и codebook 0–1 остаются неизменными. До успешного
smoke пакет seed-ов не запускается.

Технический smoke `2026-08-23_vampnet_shot_tiered_mid_smoke_01` прошёл:
codebook 0–1 не изменились, в 2–3 изменилось 63,6–65,7% токенов, в 4–13 —
19,2–22,2%; Envelope Pearson 0,9950, пик VRAM 3679 MiB. Полный пакет
заблокирован до слуховой оценки этого единственного файла.
