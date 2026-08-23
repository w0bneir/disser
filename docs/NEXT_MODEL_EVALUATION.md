# Выбор следующей генеративной модели

## Причина смены модели

Stable Audio Open 1.0 является прежде всего text-to-audio моделью. В
`shot_sound` глобальный latent SDEdit разрушал перцептивную идентичность, а
reference-preserving post-processing не создал одновременно узнаваемую и
слышимую полезную вариацию. Дальнейшая настройка тех же коэффициентов нарушила
бы правило остановки.

## Основной кандидат: Stable Audio 3 Small-SFX

Официальный Stable Audio 3 предоставляет отдельный SFX-checkpoint и нативные
режимы audio-to-audio и inpainting. Small-SFX содержит 433 млн параметров,
работает с 44.1 kHz stereo, использует SAME semantic-acoustic autoencoder и по
официальной таблице требует около 1.69 GB VRAM для пяти секунд. Это соответствует
локальному и безопасному сценарию RTX 5070.

Источники:

- <https://github.com/Stability-AI/stable-audio-3>
- <https://huggingface.co/stabilityai/stable-audio-3-small-sfx>
- <https://arxiv.org/abs/2605.17991>

Checkpoint gated: перед загрузкой пользователь должен принять Stability AI
Community License и условия Gemma на Hugging Face. Модель устанавливается в
отдельное окружение и не изменяет воспроизводимое окружение `sfx_gen_5070`.

## Резерв: SpecSinGAN

SpecSinGAN создан именно для генерации новых дублей по одному one-shot SFX и
оценивался на footsteps, gunshots и character jumps. В опубликованном listening
test участвовали 30 человек; сравнивались правдоподобие и вариативность.
Недостатки для инженерного прототипа: авторская реализация публично не
предоставлена, обучение требует адаптации ConSinGAN и ручных гиперпараметров,
а лучшие результаты статьи использовали разложенные sound layers.

Источники:

- <https://arxiv.org/abs/2110.07311>
- <https://www.adrianbarahonarios.com/specsingan/>

## Безопасная последовательность

1. Зафиксировать текущую отрицательную контрольную точку в Git.
2. Создать изолированное окружение Stable Audio 3 и проверить импорт без модели.
3. Принять gated-лицензию и загрузить только Small-SFX.
4. Выполнить CPU/preflight-проверку checkpoint и GPU/CPU backend без генерации.
5. Проверить SAME encode/decode round-trip `shot_sound`; это нижняя граница
   идентичности новой архитектуры.
6. Выполнить один минимальный audio-to-audio smoke candidate с внешним
   мониторингом VRAM.
7. Только после слухового успеха заморозить noise level и создать разные seed.
8. Если нативный audio-to-audio не проходит, не подбирать post-processing, а
   перейти к реализации single-example SpecSinGAN.
