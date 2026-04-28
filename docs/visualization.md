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
| Что видит модель | `grasp_seg/viz/heatmap_viz.py` | Per-head декомпозиция (`pos`, `cos2θ`, `sin2θ`, `width` или `fg_conf` + раскраска углов) и Grad-CAM по последнему общему слою HRNet |

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

## Раскладка Cornell

Лоадер `grasp_seg/data/cornell.py` рекурсивно ищет файлы `pcdNNNNr.png` и
поддерживает обе стандартные раскладки:

- **Плоская** — все `pcdNNNN*` в одной директории.
- **Оригинальная** — 10 подпапок `01/` … `10/` (плюс `backgrounds/`,
  которая игнорируется).

В ноутбуке `notebooks/visualize.ipynb` для `ENV="local"` дефолтный путь —
`<repo>/datasets/Cornel_grasp_dataset/`.

### Cornell + Depth (cross-domain RGB-D)

Если рядом с `pcdNNNNr.png` лежит `pcdNNNNd.tiff` (поставляется с
большинством редистрибуций Cornell, в т.ч. на Kaggle), лоадер
автоматически:

1. Читает TIFF через `tifffile.imread` (общий хелпер из
   `grasp_seg/data/jacquard_v2.py:_load_depth`).
2. Если разрешение depth не совпадает с RGB — приводит nearest-resize'ом.
3. Нормирует тем же 1/99-перцентильным robust-нормированием, что и в
   `_normalise_depth` Jacquard'а — это держит распределение глубин
   схожим с тем, на котором обучались RGB-D модели, и заметно улучшает
   cross-domain предсказания.

Результат кладётся в `CornellSample.depth` (нормированный) и
`CornellSample.depth_raw` (исходный TIFF). `figure_compare_models_cornell`
автоматически использует `scene.depth`, если он есть, и падает обратно
к нулевой карте, если `pcdNNNNd.tiff` отсутствует. Чтобы пропустить
загрузку depth целиком, передайте `load_depth=False` в `load_scene` /
`iter_scenes`.

Замечание для отчёта: депт-сенсор Cornell (Kinect v1, активная
структурированная подсветка, ~640×480) распределён иначе, чем
рендеренный depth Jacquard V2, поэтому даже с реальным дептом cross-
domain метрики смещены — это ожидаемо.

## Качественная vs количественная оценка на Cornell

Cornell Grasp Dataset не содержит пиксельных GT-масок — только
прямоугольники (`pcdNNNNcpos.txt`). Поэтому для cross-domain оценки мы:

- **Качественно**: рисуем GT-прямоугольники + декодированные top-K
  предсказания на тех же сценах (`figure_compare_models_cornell`,
  ноутбук, секция 5.2).
- **Количественно**: считаем стандартный grasp-rectangle accuracy
  (`IoU > 0.25 ∧ |Δθ| < 30°` через `decoder.jacquard_match`).
  Реализация: `grasp_seg/viz/cornell_eval.py`:
  - `evaluate_cornell(runner, scenes)` — per-scene records (`top1_ok`,
    `topk_any_ok`, `top1_iou`, `top1_angle_err_deg`).
  - `summarize_cornell(records)` — агрегаты `top1_acc`, `topk_any_acc`,
    `mean_top1_iou`, `mean_top1_angle_err_deg`.
  Ноутбук: секция **5.3** строит таблицу + bar-plot top-1 accuracy для
  всех загруженных моделей на N Cornell-сценах (N задаётся
  `NUM_CORNELL_EVAL`).
- **Failure cases**: `extra_viz.figure_cornell_failures` сортирует
  Cornell-сцены по top-1 IoU и показывает первые `NUM_CORNELL_FAILURES`
  худших — GT (зелёные) + top-3 предсказаний (красные). Ноутбук:
  секция **5.4**.
- **Не считаем** пиксельный mIoU, поскольку он потребовал бы
  растеризовать Cornell-rectangles (а это дополнительное допущение).

### Coordinate frame для Cornell-метрик

Cornell — 480×640, модели — `image_size × image_size` (обычно 384).
`evaluate_cornell` приводит каждую сцену в model-frame через
`compare_viz._scene_to_model_space`: пад до квадрата `max(H, W)` (с
чёрными полосами на короткой стороне) → uniform-resize до
`image_size`. Аспект сохраняется, углы граспов остаются неизменными,
центры/длины/ширины масштабируются равномерно. Это тот же
координатный frame, в котором работает `figure_compare_models_cornell`.

## Что видит модель — heatmap-карты

Модуль `grasp_seg/viz/heatmap_viz.py` отвечает на два смежных вопроса:

### Per-head декомпозиция (`figure_per_head_heatmap`)

Что **выдаёт** модель для одной сцены. Раскладка зависит от `mask_mode`:

- **multitask** — 2×3: RGB | depth | `pos` (графическая «вероятность
  захвата», sigmoid) | `cos2θ` | `sin2θ` | `width` (sigmoid). У каждой
  карты свой colorbar; диапазоны фиксированы (`pos`, `width` ∈ [0, 1];
  `cos2θ`, `sin2θ` ∈ [−1, 1] на палитре `RdBu`).
- **angle** — 1×4: RGB | depth | `fg_conf` (1 − p_bg) | argmax-bin,
  раскрашенный палитрой `palette.angle_cmap`.
- **binary** — 1×3: RGB | depth | `pos`.

Источник сцены может быть:
- путём к `*_grasps.txt` (Jacquard);
- объектом `CornellSample` (в этом случае автоматически применяется
  `_scene_to_model_space`, чтобы аспект 4:3 сохранялся);
- кортежем `(rgb, depth)`.

### Grad-CAM (`figure_grad_cam`, `compute_grad_cam`)

«Куда модель смотрит» — карта градиентов, классическая Grad-CAM по
последнему общему фичемап-стеку HRNet (`HRNetSeg.fuse` /
`HRNetMultiTask._seg.fuse`). Реализация:

1. Регистрируются прямой и обратный хуки на `fuse`.
2. Для прохода с `requires_grad=True` параметрам выставляется
   `requires_grad=False`, а сам входной тензор требует grad — этого
   достаточно, чтобы автоград построил граф через всю сеть.
3. Целевой скаляр зависит от `mask_mode`:
   - multitask: среднее по `out["pos"]` (логит до sigmoid);
   - binary:   среднее по логиту;
   - angle:    среднее по логитам каналов 1..K (foreground bins).
4. Веса каналов = пространственное среднее градиентов; CAM =
   ReLU(Σ_k w_k · A_k); upsample bilinear до `image_size`; нормировка
   до `[0, 1]`.

Ноутбук: секция **8** (`8.1` — per-head, `8.2` — Grad-CAM).
Сохраняемые файлы: `outputs/viz/.../heatmaps/*.png`.
