# Визуализация результатов

В репозитории есть отдельный модуль `grasp_seg/viz/`, отвечающий за все
визуализации, которые могут понадобиться в отчёте, презентации и на защите
проекта. Все надписи на русском, формат — PNG, размер DPI настраивается.

## Что покрывает модуль

| Раздел | Файл | Что показывает |
| ---- | ---- | ---- |
| Датасет | `grasp_seg/viz/dataset_viz.py` | Исходное RGB + GT-захваты, resize до 384×384, маски (binary / angle / multitask), compact-polygon vs full rect, шаги аугментации (флипы, повороты, масштаб, jitter цвета, шум RGB, jitter и dropout глубины), Cornell сцена с GT-прямоугольниками |
| Кривые обучения | `grasp_seg/viz/metrics_viz.py` | Полная панель по `metrics.csv` (loss / IoU / Dice / lr / multitask cos+sin+width+ang_mae / GPU память+util / тайминг), а также сравнение нескольких моделей |
| Эволюция эпох | `grasp_seg/viz/epoch_evolution.py` | Сетка сцена×эпоха для одной модели на сохранённых чекпоинтах `epoch_001/005/010/015/…/last.pth` |
| Лучшая эпоха | `grasp_seg/viz/eval_viz.py` | Для каждой модели: вход / GT / предсказание / GT-rect vs decoded-rect / карта ошибок; bar-chart per-bin IoU |
| Сравнение моделей | `grasp_seg/viz/compare_viz.py` | Side-by-side всех загруженных моделей на нескольких сценах Jacquard/Cornell |
| Дополнительно | `grasp_seg/viz/extra_viz.py` | IoU vs угол, depth contribution heatmap (RGB-D − RGB), failure case catalog с эвристическими причинами |

## Декодер masks → grasp rectangles

Модуль `grasp_seg/viz/decoder.py` реализует декодирование выходов модели
обратно в ориентированные прямоугольники захвата:

- **angle-режим**: сглаживание уверенности `1 − p_bg`, peak-finding с
  `scipy.ndimage.maximum_filter`, NMS в пространстве (центр, угол) с
  порогами `nms_dist_px` и `nms_angle_deg`.
- **multitask-режим**: то же на канале `pos` (sigmoid), угол
  `½·atan2(sin2θ, cos2θ)`, ширина `sigmoid(width)·max_width_px`.

Поддерживается стандартная для Jacquard / Cornell оценка качества
захвата: `IoU > 0.25 ∧ |Δθ| < 30°` (`decoder.jacquard_match`,
`decoder.topk_correct_rate`).

## Как запускать

### Вариант 1 — Jupyter-ноутбук
Файл `notebooks/visualize.ipynb` поэтапно проходит все секции с
комментариями. Параметры путей задаются в первой ячейке. Поддерживает
Colab, локальный VS Code/JupyterLab и Kaggle.

### Вариант 2 — CLI
```bash
python tools/visualize.py \
    --jacquard-root /path/to/Jacquard_V2 \
    --splits-path splits/jacquard_v2.json \
    --cornell-root /path/to/cornell \
    --run multitask_rgbd=/runs/multitask_rgbd \
    --run multitask_rgb=/runs/multitask_rgb \
    --run angle=/runs/angle \
    --out outputs/viz/report \
    --dpi 140
```
По умолчанию строятся все секции (`dataset / training / epoch_evolution /
best_epoch / compare / extras`); для подмножества используйте
`--sections training compare`.

## Где лучше запускать

1. **Google Colab Pro / A100 (рекомендация).** Быстрый GPU; чекпоинты на
   подмонтированном Drive; распакованный Jacquard уже там; легко
   поделиться ноутбуком и сохранить итоговые PNG обратно на Drive.
2. **Локальный VS Code (RTX 3060).** Работает без проблем, но потребует
   скачать релевантные `epoch_*.pth` + `metrics.csv` + `resolved_config.yaml`
   для каждой из 3 моделей; Jacquard объёмен, можно ограничиться нужным
   подмножеством сцен из `splits/jacquard_v2.json`.
3. **Kaggle.** Возможно, но не оптимально (T4 медленнее A100; неудобный
   доступ к pre-trained ckpt).

## Зависимости

- `matplotlib`, `pandas`, `scipy`, `seaborn` — добавлены в
  `requirements.txt` (всё лёгкое, ставится за < 1 минуты).
- `tensorboard` / `wandb` **не требуются** для статичных PNG — модуль
  работает целиком на matplotlib.
- `torch`, `timm`, `opencv-python`, `tifffile`, `imagecodecs` уже были
  обязательны для тренировки и переиспользуются.

## Качественная vs количественная оценка на Cornell

Cornell Grasp Dataset не содержит пиксельных GT-масок — только
прямоугольники (`pcdNNNNcpos.txt`). Поэтому для cross-domain оценки мы:

- **Качественно**: рисуем GT-прямоугольники + декодированные top-K
  предсказания на тех же сценах.
- **Количественно**: считаем стандартный grasp-rectangle accuracy
  (`IoU > 0.25 ∧ |Δθ| < 30°` через `decoder.jacquard_match`).
- **Не считаем** пиксельный mIoU, поскольку он потребовал бы
  растеризовать Cornell-rectangles (а это дополнительное допущение).
