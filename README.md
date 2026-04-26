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

# 1) Сначала ставим CUDA-сборку torch вручную, иначе с PyPI приедет CPU-only.
# Узнайте версию CUDA у вашего драйвера: `nvidia-smi` → строка `CUDA Version: ...`.
# Подберите соответствующий индекс на https://pytorch.org/get-started/locally/
# Примеры:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

# 2) Остальные зависимости — обычным образом
pip install -r requirements.txt

# 3) Проверка GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Должно напечатать что-то вроде `2.x.y+cu126 True NVIDIA GeForce RTX 3060`.
Если `torch.cuda.is_available()` возвращает `False` — снесите torch (`pip uninstall -y torch torchvision`) и поставьте сборку под другую версию CUDA.

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
- `outputs/.../epoch_NNN.pth` для каждой эпохи (если `trainer.save_every_epoch=true`, по умолчанию включено — нужно для построения графиков обучения по чекпоинтам),
- `outputs/.../metrics.csv` — построчная история train/val метрик и lr по эпохам (готова к загрузке в pandas/matplotlib),
- `outputs/.../resolved_config.yaml` — фактический конфиг запуска.

### Возобновление обучения

```bash
python tools/train.py --config configs/default.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    trainer.save_dir=outputs/hrnet_w18_rgbd_angle \
    --resume outputs/hrnet_w18_rgbd_angle/last.pth
```

`--resume` загружает model + optimizer + scheduler + scaler + epoch counter, обучение продолжится со следующей эпохи. Для дообучения «с чистым оптимизатором» добавьте `--resume-model-only` — тогда восстановятся только веса модели.

## Облачные варианты

Готовые рецепты для трёх популярных платформ:

- [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) — Google Colab (free T4 / Pro). Скачивает датасет с публичной ссылки Я.Диска прямо в `/content/`, чекпоинты пишутся в Google Drive (переживают сбросы сессии).
- [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) — Kaggle Notebooks (бесплатно, 30 ч/нед на T4 ×2 / P100). Датасет один раз заливается как Kaggle Dataset, потом доступен read-only из любого ноутбука.
- [`docs/yandex_datasphere.md`](docs/yandex_datasphere.md) — Yandex DataSphere (платно, рубли). T4 / V100 / A100, нативный доступ к Я.Диску и Object Storage.

Все три варианта используют `--resume` для продолжения обучения после разрыва сессии.

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
