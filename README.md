# HRNet-W18 для grasp-сегментации на Jacquard V2

Пайплайн обучения HRNet-W18 (через `timm`) на датасете
[Jacquard V2](https://github.com/lqh12345/Jacquard_V2) для задачи семантической
сегментации области захвата (grasp segmentation). Поддерживается RGB / D / RGB-D
вход, три формулировки маски (binary / angle / multitask) и обучение
с AMP + gradient accumulation, заточенное под **RTX 3060 6 ГБ**.

## Что внутри

```
configs/default.yaml        # все гиперпараметры в одном месте, можно переопределять с CLI
grasp_seg/
  data/
    grasp_rect.py           # парсинг *_grasps.txt и растеризация маски (compact-polygon, 1/3 длины)
    transforms.py           # синхронные RGB-D + grasp-list аугментации
    splits.py               # object-wise train/val/test
    jacquard_v2.py          # PyTorch Dataset
  models/hrnet.py           # HRNet-W18 (+ Small-v2) + 4-канальный первый conv
  losses/                   # BCE+Dice / CE+Dice / multitask
  engine/                   # Trainer (AMP + accum) и валидация (mIoU / Dice / F1)
tools/
  prepare_split.py          # построение object-wise сплитов
  train.py                  # запуск обучения
  eval.py                   # инференс/метрики на сплите
scripts/smoke_test.py       # smoke-тест на 8 синтетических сценах
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# Если nvidia-драйвер CUDA 11.x — поставьте torch с подходящим колесом:
# https://pytorch.org/get-started/locally/
```

Smoke-тест без датасета (≈30 с):

```bash
python scripts/smoke_test.py
```

Должен пройти без ошибок и напечатать `ALL MODES OK`.

## Подготовка датасета

1. Скачать 4 архива Jacquard V2 (`JacquardV2_Dataset_0..3`) с
   [OneDrive ссылки в репозитории lqh12345/Jacquard_V2](https://github.com/lqh12345/Jacquard_V2).
2. Распаковать так, чтобы получилась структура:

   ```
   /data/JacquardV2_Dataset/
       JacquardV2_Dataset_0/<object_id>/<idx>_<object_id>_RGB.png
       JacquardV2_Dataset_0/<object_id>/<idx>_<object_id>_perfect_depth.tiff
       JacquardV2_Dataset_0/<object_id>/<idx>_<object_id>_stereo_depth.tiff
       JacquardV2_Dataset_0/<object_id>/<idx>_<object_id>_mask.png
       JacquardV2_Dataset_0/<object_id>/<idx>_<object_id>_grasps.txt
       JacquardV2_Dataset_1/...
       ...
   ```

3. Сгенерировать object-wise сплит:

   ```bash
   python tools/prepare_split.py \
       --root /data/JacquardV2_Dataset \
       --out splits/jacquard_v2.json \
       --val-frac 0.1 --test-frac 0.1 --seed 0
   ```

## Обучение (RTX 3060 6 ГБ)

```bash
python tools/train.py \
    --config configs/default.yaml \
    dataset.root=/data/JacquardV2_Dataset \
    dataset.splits_path=splits/jacquard_v2.json \
    trainer.save_dir=outputs/hrnet_w18_rgbd_angle
```

По умолчанию: HRNet-W18, RGB-D, 384×384, **batch_size=2**, **accum_steps=4**
(эффективный batch 8), AMP, SGD (lr=1e-2, momentum=0.9, wd=5e-4, poly LR),
80 эпох, mask_mode=`angle` (19 классов: фон + 18 угловых бинов по 10°).

### Если падает по памяти (OOM)

В порядке простоты:

1. `dataset.image_size=320` — экономит ~40% VRAM.
2. `model.backbone=hrnet_w18_small_v2` — лёгкая версия HRNet.
3. `trainer.batch_size=1 trainer.accum_steps=8` — тот же эффективный batch.
4. Закройте Chrome / VS Code helpers / OBS — на Windows они едят VRAM.

### Переключение режима маски

```bash
# бинарная Q-маска (graspable / no)
python tools/train.py --config configs/default.yaml dataset.mask_mode=binary

# multi-task GG-CNN-like: pos + cos2θ + sin2θ + width
python tools/train.py --config configs/default.yaml dataset.mask_mode=multitask
```

### Только RGB или только Depth

```bash
python tools/train.py --config configs/default.yaml dataset.input_mode=rgb
python tools/train.py --config configs/default.yaml dataset.input_mode=depth
```

## Метрики

- **binary**: mIoU, Dice, precision/recall на классе «graspable».
- **angle**: общий и foreground-only mIoU/Dice по всем угловым бинам.
- **multitask**: mIoU/Dice по `pos` + MSE по `cos`/`sin` на положительных пикселях.

`Trainer` сохраняет:
- `outputs/.../best.pth` (по `miou_fg`),
- `outputs/.../last.pth` (после каждой эпохи),
- `outputs/.../resolved_config.yaml` — фактический конфиг запуска.

## Маска grasp-захвата — почему compact-polygon, а не полный прямоугольник

Каждая строка `*_grasps.txt` это `x;y;θ;w;h` — параметризация **одной позы
гриппера**, а не «области, где можно хватать». Контакт гриппера происходит
в центральной полоске вдоль оси раскрытия (длина `w`). Поэтому растеризовать
весь прямоугольник в маску — некорректно: его края соответствуют пластинам
гриппера, а не graspable-пикселям.

Здесь используется тот же приём, что в официальном тулбоксе и в GG-CNN:
длина прямоугольника сжимается в `1/3` (`length_scale=0.3333`), и
рисуется только это compact-ядро. Получается маска центров захвата,
которая корректно интерпретируется как «здесь гриппер физически смыкается».

См. `grasp_seg/data/grasp_rect.py:rasterize_grasp_mask`.

## Перенос на Colab

Если 6 ГБ оказывается мало даже на small-v2, перенос почти автоматический:

1. `pip install -r requirements.txt`.
2. Залить датасет на Google Drive (или использовать Drive с уже
   распакованным `/JacquardV2_Dataset`) и смонтировать его.
3. В Colab Free (T4, 16 ГБ) можно поднять `batch_size=4..8` и
   `image_size=480`. Меняется только конфиг.

## Известные ограничения

- Тулбокс Jacquard V2 не публикует «эталонную» grasp-mask — постановка
  именно с маской вводится здесь и совместима с подходом GG-CNN/GR-ConvNet.
- Аугментации ничего не делают со stereo-depth NaN’ами; стерео
  включается с вероятностью `augmentation.use_stereo_depth_p` (0.3 по
  умолчанию). Если NaN’ы мешают — выставьте `0.0` и тренируйте только на
  `perfect_depth`.
- `eval.py` оценивает в пиксельном пространстве. Для grasp-rectangle IoU-
  метрики (Jacquard standard) нужно дополнительно декодировать предсказания
  обратно в прямоугольники — это можно добавить позже.
