# RealSense dataset tools

Дополнение к существующим `tools/realsense_preview.py` и `tools/realsense_capture.py`.

## Назначение

Эти скрипты закрывают следующий шаг после первичного захвата:

- сбор сцен с объектными метаданными;
- ручная разметка Cornell grasp rectangles;
- проверка собранного датасета;
- object-wise split для fine-tune.

## Скрипты

- `capture_session.py` — захват aligned RGB + depth с сохранением `meta.csv`.
- `annotate_cornell.py` — ручная разметка `cpos/cneg`.
- `verify_cornell_dataset.py` — сводка по датасету.
- `build_objectwise_split.py` — разбиение по объектам.

## Пример

```bash
python tools/realsense_dataset/capture_session.py --out captures/my_dataset --subdir 01 --object-name mug
python tools/realsense_dataset/annotate_cornell.py --root captures/my_dataset --subdir 01
python tools/realsense_dataset/verify_cornell_dataset.py --root captures/my_dataset
python tools/realsense_dataset/build_objectwise_split.py --root captures/my_dataset --test-frac 0.2 --seed 0
```
