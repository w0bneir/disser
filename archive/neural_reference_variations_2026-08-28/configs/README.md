# Конфигурации

- `reference_variations.json` — основной same-family эксперимент диссертации;
- `text_probe.json` — диагностическая text-only генерация без референса;
- `dsp_baseline.json` — зафиксированный pitch/time/EQ-контроль `DSP v1`;
- `stress_tests/wood_to_metal.json` — дополнительный тест замены материала,
  не являющийся основной задачей.

Результаты и временные конфигурации внутри `results/` не считаются
каноническими. После научно значимого опыта его параметры переносятся сюда.

Честный paired text-only baseline основной серии создаётся самим
`run_stable_audio_experiments.py`: он использует ту же длительность, seed и
исходный Gaussian noise, что и Reference SDEdit. Поля `gamma`,
`gradient_clip_norm` и `guidance_*` пока остаются в общей конфигурационной схеме
для воспроизводимости старых абляций и игнорируются в режиме Reference SDEdit.
