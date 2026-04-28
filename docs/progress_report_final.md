# Итоговый отчёт по проекту HRNet-W18 Jacquard V2 Grasp Segmentation

**Дата составления:** 27 апреля 2026 г.
**Текущий статус:** все три модели обучены до сходимости (RGB multitask, RGB-D multitask, RGB-D angle); пайплайн визуализации, кросс-доменная оценка на Cornell и интерпретируемость (heatmap + Grad-CAM) собраны.
**Репозиторий:** https://github.com/chelovekrazumnii2-png/HRNet_Grasp_Semantic_Segmentation

> Этот отчёт обобщает [progress_report.md](progress_report.md) и [progress_report_v2.md](progress_report_v2.md) и расширяет их фактическими финальными метриками трёх моделей и работой, выполненной поверх обучения (визуализация, кросс-домен Cornell, интерпретируемость).

---

## 1. Краткое резюме

Построен и доведён до production-ready состояния модульный пайплайн обучения **HRNet-W18** для пиксельной сегментации grasp-областей на **Jacquard V2** + кросс-доменная валидация на **Cornell Grasp Dataset** + интерпретируемость предсказаний.

**Что сделано:**

1. **Каркас + обучение** (см. v1/v2): три mask-режима (`binary`/`angle`/`multitask`), три вариантa loss, object-wise split 80/10/10, AMP fp16, per-epoch checkpoints + `--resume`, рецепты для Colab Pro A100 / Kaggle T4 / RTX 3060.

2. **Три обученных модели:**
   - `hrnet_w18_rgb_multitask` — 3 канала, multitask GG-CNN-стиль (`pos`/`cos2θ`/`sin2θ`/`width`).
   - `hrnet_w18_rgbd_multitask` — 4 канала (RGB + perfect depth), тот же multitask.
   - `hrnet_w18_rgbd_angle` — 4 канала, классификация на 19 углов (фон + 18 × 10°).

3. **Пайплайн визуализации** (`grasp_seg/viz/`) — 11 модулей: датасет / метрики обучения / эволюция эпох / per-class IoU / side-by-side сравнение моделей / IoU vs угол / depth contribution / failure-cases / Cornell-eval / heatmap+Grad-CAM. Все в одном ноутбуке `notebooks/visualize.ipynb`.

4. **Кросс-доменная оценка на Cornell:**
   - Loader Cornell (RGB+depth, обе раскладки: вложенная `01..10/` и плоская).
   - `_scene_to_model_space` — pad-to-square + uniform resize (480×640 → 384×384), сохраняющий аспект 4:3 и углы grasp'ов.
   - Качественная: `figure_compare_models_cornell` (side-by-side), `figure_cornell_failures` (худшие N).
   - Количественная: `evaluate_cornell` + `summarize_cornell` — top-1 grasp accuracy по стандартному критерию (IoU > 0.25 ∧ |Δθ| < 30°).

5. **Интерпретируемость:**
   - `figure_per_head_heatmap` — раскладка всех голов модели (multitask: pos/cos2θ/sin2θ/width; angle: fg_conf + раскрашенный argmax).
   - `figure_grad_cam` — Grad-CAM по последнему общему слою HRNet (`fuse`).

**Главный сюрприз:** перевод с `mask_mode=angle` (19 классов) на `mask_mode=multitask` (регрессионные головы GG-CNN-стиля) дал **+0.34 mIoU_fg** при том же бюджете эпох — это существенно выше всех «оптимистичных» прогнозов прошлого отчёта.

---

## 2. Финальные метрики обучения

Три модели обучены на одном и том же object-wise split (seed=0; train ≈ 41 000 grasp-сцен, val ≈ 5 200, test ≈ 5 200, image_size=384). Оптимизация: SGD lr=0.01, momentum=0.9, weight_decay=5e-4, poly schedule (power=0.9), warmup 1 эпоха, grad clip 1.0, AMP fp16.

**Длительность прогона:** 30 эпох (multitask), 31 эпоха (angle). При SGD+poly schedule метрики уже уверенно вышли на плато на 25-й эпохе, поэтому продлевать прогоны нецелесообразно.

### 2.1. Сводная таблица (best-эпоха)

| Модель | Эпоха | val_mIoU_fg | val_Dice_fg | val_ang_mae_deg | s/step |
|---|---|---|---|---|---|
| `hrnet_w18_rgb_multitask`  | 27 | **0.6417** | **0.7817** | 15.86° | 0.31 |
| `hrnet_w18_rgbd_multitask` | 26 | **0.6788** | **0.8086** | **14.97°** | 0.91 |
| `hrnet_w18_rgbd_angle`     | 25 | 0.3282 | 0.4937 | — (нет регрессионной головы) | — |

> `val_ang_mae_deg` — средняя абсолютная ошибка предсказанного угла grasp'а от GT в градусах, считается только в multitask-моделях из голов `cos2θ`/`sin2θ`.

### 2.2. Финальная эпоха (для сравнения с best)

| Модель | val_mIoU_fg | val_Dice_fg | val_precision_fg | val_recall_fg |
|---|---|---|---|---|
| `rgb_multitask`  (epoch 29) | 0.6384 | 0.7793 | 0.702 | 0.875 |
| `rgbd_multitask` (epoch 29) | 0.6774 | 0.8077 | 0.744 | 0.883 |
| `rgbd_angle`     (epoch 30) | 0.3242 | 0.4887 | 0.539 | 0.455 |

### 2.3. Депт даёт чистый бонус

| Метрика | RGB-only (multitask) | RGB-D (multitask) | Δ |
|---|---|---|---|
| val_mIoU_fg | 0.642 | 0.679 | **+3.7 pp** |
| val_Dice_fg | 0.782 | 0.809 | **+2.7 pp** |
| val_ang_mae_deg | 15.86° | 14.97° | **−0.9°** |
| val_recall_fg | 0.860 | 0.883 | **+2.3 pp** |
| s/step | 0.31 | 0.91 | +0.60 (×3 медленнее) |

Депт стабильно улучшает качество, но триплирует время шага. Для реального робота — однозначно RGB-D; для real-time inference на edge-GPU — RGB-only терпимо (потеря 4 п.п. mIoU_fg).

### 2.4. Multitask vs angle

| Метрика | `rgbd_angle` (best) | `rgbd_multitask` (best) | Δ |
|---|---|---|---|
| val_mIoU_fg | 0.328 | 0.679 | **+34.6 pp** |
| val_Dice_fg | 0.494 | 0.809 | **+31.5 pp** |

Это решительная победа multitask-формулировки. Прошлый отчёт (v2) рекомендовал `multitask` как «Вариант B» с ожидаемым приростом +5–10% — фактический прирост в **3–7 раз больше**. Гипотезы, почему так:

- **Регрессионный сигнал плотнее.** В `angle`-варианте 95% пикселей — фон, и большая часть градиента идёт в bg-класс. В `multitask` маска занятости (`pos`) — отдельная голова, а `cos`/`sin`/`width` обучаются только на foreground-пикселях. Сигнал концентрированнее.
- **`pos` — sigmoid (не softmax 19-way).** Угол выводится из `arctan2(sin2θ, cos2θ)` непрерывно — не страдает от дискретизации с шагом 10°.
- **Loss-смесь сбалансирована.** `MultiTaskGraspLoss` взвешивает BCE + Dice + cos + sin + width так, что ни один компонент не доминирует.

### 2.5. Кривые (последние 5 эпох RGB-D multitask)

```
epoch | train_loss | val_mIoU_fg | val_Dice_fg | val_ang_mae_deg
  25  |  0.6207    |   0.6740    |   0.8051    |   15.06°
  26  |  0.6166    |   0.6788    |   0.8086    |   14.97°  ← best
  27  |  0.6129    |   0.6708    |   0.8030    |   14.98°
  28  |  0.6107    |   0.6751    |   0.8061    |   14.88°
  29  |  0.6089    |   0.6774    |   0.8077    |   14.84°
```

Метрики уверенно стабилизировались. Дальнейшее обучение даст не более +1–2 п.п.

### 2.6. Сравнение с прогнозом v2

| Метрика | План v2 (epoch 80, angle) | Факт (epoch 30, multitask) |
|---|---|---|
| Реалистичный прогноз | 0.42–0.46 | — |
| Optimistic прогноз | 0.48–0.50 | — |
| Целевой уровень | ≥ 0.45 | — |
| **Факт RGB-D** | — | **0.679** |
| **Факт RGB** | — | **0.642** |

Все целевые планки v2 закрыты с большим запасом. Минимальный приёмочный уровень (0.40) перевыполнен в 1.7 раза.

---

## 3. Хронология работы

### 3.1. Базовый пайплайн (детали — в [progress_report.md](progress_report.md), [progress_report_v2.md](progress_report_v2.md))

| Этап | Описание | Статус |
|---|---|---|
| 1. Каркас `grasp_seg/` | data + models + losses + engine + utils + 3 mask-режима | ✅ |
| 2. Локальная RTX 3060 | batch=8, ~42 ч/прогон → переход на облако | ✅ |
| 3. Per-epoch checkpoints + resume | Безопасные длинные прогоны | ✅ |
| 4. Облачные рецепты | Colab Pro A100 + Kaggle T4 | ✅ |
| 5. RGB-only конфиг | `_patch_first_conv` для 3-канального input | ✅ |
| 6. Pre-unpacked Kaggle | Экономия 45 мин/сессия | ✅ |
| 7. Multi-GPU попытка | Откат: CPU bottleneck на T4×2 | ✅ (как negative result) |
| 8. Финальный батчинг A100 | batch=48 — sweet spot | ✅ |

### 3.2. Что добавилось после v2 (текущая сессия)

| PR | Цель | Файлы |
|---|---|---|
| **#1** | Базовый visualization-пайплайн (8 модулей) | `grasp_seg/viz/{palette,draw,decoder,inference,dataset_viz,metrics_viz,epoch_evolution,eval_viz}.py`, `notebooks/visualize.ipynb` |
| **#2** | Windows-friendly defaults + раскладка Cornell с подпапками `01..10/` | `grasp_seg/data/cornell.py`, `notebooks/visualize.ipynb` |
| **#3** | Документация для локального Windows-запуска | `local_setup_windows.md`, `README.md`, `requirements.txt` |
| **#4** | Загрузка `pcdNNNNd.tiff` для Cornell в RGB-D моделях | `grasp_seg/data/cornell.py`, `compare_viz`, `dataset_viz` |
| **#5** | `_scene_to_model_space` — pad+resize Cornell под model-frame, фикс broadcast-ошибки | `grasp_seg/viz/compare_viz.py` |
| **#6** | Количественная оценка на Cornell (top-1 acc) + failure-каталог | `grasp_seg/viz/cornell_eval.py`, `extra_viz.py`, ноутбук секции 5.3, 5.4 |
| **#7** | Регистрация `cornell_eval` в `__init__.py` + guard на пустой failure-grid | `grasp_seg/viz/__init__.py`, `extra_viz.py` |
| **#8** | `heatmap_viz`: per-head декомпозиция + Grad-CAM | `grasp_seg/viz/heatmap_viz.py`, ноутбук секция 8 |

Все PR смёржены, CI зелёный.

---

## 4. Кросс-доменная оценка на Cornell

### 4.1. Что такое Cornell

Cornell Grasp Dataset (Jiang 2011) — реальные RGB-D снимки 280 разных бытовых предметов, всего 885 сцен. Раскладка файлов:

```
pcdNNNNr.png      ← RGB
pcdNNNNd.tiff     ← depth (Kinect v1, 640×480, мм)
pcdNNNN.txt       ← point cloud (не используется)
pcdNNNNcpos.txt   ← положительные граспы (4 угла на rect)
pcdNNNNcneg.txt   ← негативные
```

Loader (`grasp_seg/data/cornell.py`) поддерживает:
- **Вложенную раскладку** (оригинал Jiang 2011): `01/pcd0100*`, `02/pcd0200*`, …, `10/...`, `backgrounds/` игнорируется.
- **Плоскую раскладку**: всё в одной директории.
- **Robust normalization** depth — 1/99 percentile clip, аналогично Jacquard'у, чтобы домены были сопоставимы.

### 4.2. Coordinate frame: pad-to-square + uniform resize

Cornell — 480×640 (4:3), модели — 384×384. Два очевидных варианта (просто `cv2.resize` без pad'а или crop под квадрат) портят геометрию: углы grasp'а становятся неверными после неравномерного масштабирования. Поэтому реализован `_scene_to_model_space`:

1. Пад до квадрата `max(H, W)=640` чёрными полосами (центрирование на короткой стороне).
2. Uniform-resize до 384×384.
3. Сдвиг центров grasp'ов на `(pad_top, pad_left)` + умножение `length`/`width` на единый `scale=384/640`.
4. Углы grasp'ов **не меняются** (равномерный resize).

Этот же frame используется во всех Cornell-визуализациях: `figure_compare_models_cornell`, `evaluate_cornell`, `figure_cornell_failures`.

### 4.3. Качественная оценка: `figure_compare_models_cornell`

Side-by-side всех загруженных моделей на N Cornell-сценах. На каждой сцене: GT-rectangles (зелёные) + предсказанная heatmap-overlay + декодированные top-3 grasp-rectangles (синие/красные). В заголовке — индикатор `top-1 ✓/×` (true если top-1 предсказание удовлетворяет критерию IoU > 0.25 ∧ |Δθ| < 30° против любого GT).

### 4.4. Количественная оценка: `evaluate_cornell` + `summarize_cornell`

Стандартный Jacquard/Cornell критерий: предсказанный grasp считается корректным, если **существует** хотя бы один GT-rect с

```
IoU(pred, gt) > 0.25   AND   |angle(pred) − angle(gt)| < 30°
```

Возвращаемые метрики:

- `top1_acc` — доля сцен, где top-1 декодированный grasp правильный (главная метрика).
- `top_k_any_acc` — есть ли хотя бы один правильный grasp среди top-K (по умолчанию K=5).
- `mean_top1_iou`, `mean_top1_angle_err_deg` — средние характеристики top-1 предсказаний.

В ноутбуке (секция 5.3) собирается DataFrame с per-model метриками + bar-plot top-1 acc; секция 5.4 — каталог N худших Cornell-сцен (`figure_cornell_failures`) с подписями `(IoU=…, Δθ=…°)`. Конфигурация — `NUM_CORNELL_EVAL`/`NUM_CORNELL_FAILURES` в первой ячейке.

### 4.5. Что мы не считаем

- **Пиксельный mIoU на Cornell.** Cornell даёт rectangle-граспы, а не pixel-mask, и растеризация требует дополнительных допущений (толщина, перекрытие). Стандарт Cornell — top-1 accuracy по rect-критерию, что мы и используем.
- **Fine-tune на Cornell-train.** Основная цель кросс-домена — **проверить sim2real generalization** обученной на Jacquard модели без дообучения. Fine-tune оставлен как отдельный потенциальный шаг (см. раздел 7).

---

## 5. Визуализация: что собрано

Всё — единым `notebooks/visualize.ipynb`, секции 1–8.

| Секция | Файл viz | Содержание |
|---|---|---|
| 1.1–1.5 | `dataset_viz` | Сырая сцена + GT-rects, resize-pipeline, mask-modes (binary/angle/multitask), augmentation steps |
| 1.6 | `dataset_viz.figure_cornell_raw` | Cornell сцена: RGB+pos-grasps + нормированный depth |
| 2 | `metrics_viz` | Кривые обучения по `metrics.csv` (loss/IoU/Dice/lr/multitask cos+sin+width+ang_mae/GPU память+util/тайминг) + сравнение нескольких моделей в одной фигуре |
| 3 | `epoch_evolution` | Сетка `сцена × эпоха` для одной модели на сохранённых checkpoint'ах |
| 4 | `eval_viz` | Best-epoch: вход / GT / предсказание / GT-rect vs decoded / карта ошибок + bar-chart per-bin IoU |
| 5.1 | `compare_viz.figure_compare_models_jacquard` | Side-by-side всех моделей на test-сценах Jacquard |
| 5.2 | `compare_viz.figure_compare_models_cornell` | То же на Cornell |
| 5.3 | `cornell_eval.evaluate_cornell` + DataFrame | Таблица top-1 acc / top-K acc / mean IoU / mean angle error + bar-plot top-1 acc |
| 5.4 | `extra_viz.figure_cornell_failures` | N худших Cornell-сцен (GT зелёные + top-3 предсказания) |
| 6 | `extra_viz` | IoU vs угол grasp'а; depth contribution heatmap (RGB-D − RGB); failure-каталог Jacquard с эвристическими причинами |
| 7 | (custom) | Сводная таблица top-1 grasp accuracy по моделям |
| **8.1** | `heatmap_viz.figure_per_head_heatmap` | Раскладка всех голов модели для одной сцены |
| **8.2** | `heatmap_viz.figure_grad_cam` | Grad-CAM по последнему общему слою HRNet |

Конфиг ноутбука (первая ячейка): `ENV` (`local`/`colab`/`kaggle`), `JACQUARD_ROOT`, `CORNELL_ROOT`, `RUNS` (mapping name → run-dir), `NUM_*` knobs. Подгружаются те модели из `RUNS`, у которых существует папка с `resolved_config.yaml` и `best.pth` — недостающие пропускаются с warning'ом.

### 5.1. Heatmap-визуализация (PR #8)

**Per-head декомпозиция** (`figure_per_head_heatmap`) — что модель **выдаёт**:

- **multitask** (2×3): `RGB | depth | pos | cos2θ | sin2θ | width`. У каждой панели свой colorbar; `pos`/`width` ∈ [0, 1] (sigmoid), `cos2θ`/`sin2θ` ∈ [−1, 1] (палитра RdBu).
- **angle** (1×4): `RGB | depth | fg_conf (1−p_bg) | argmax-bin (раскрашенный палитрой angle_cmap)`.
- **binary** (1×3): `RGB | depth | pos`.

**Grad-CAM** (`figure_grad_cam` + `compute_grad_cam`) — куда модель **смотрит**:

- Хуки `forward_hook` + `register_full_backward_hook` на последний общий conv-stack: `HRNetSeg.fuse` (для binary/angle) или `HRNetMultiTask._seg.fuse` (для multitask).
- Параметры → `requires_grad=False`, входной тензор → `requires_grad=True` — этого достаточно, чтобы автоград построил граф через всю сеть, не накапливая parameter-grad'ы.
- Целевой скаляр зависит от `mask_mode`:
  - multitask: `out["pos"].mean()` (логит до sigmoid);
  - binary: `logits.mean()`;
  - angle: `out[:, 1:].mean()` (среднее по foreground-bin'ам, исключая bg).
- CAM = ReLU(Σ_k w_k · A_k), bilinear upsample до `image_size`, нормировка до [0, 1].
- Сетка `figure_grad_cam`: каждая сцена — пара панелей `[RGB][RGB+CAM]`, ряды переносятся по `n_cols`.

Источник сцены — единый интерфейс: путь к Jacquard `*_grasps.txt`, объект `CornellSample`, либо кортеж `(rgb, depth)`. Cornell-сцены автоматически проходят через `_scene_to_model_space`.

---

## 6. Архитектурные решения, обоснованные финалом

| # | Решение | Подтверждено финалом? |
|---|---|---|
| 1 | HRNet-W18 backbone | ✅ — train в плато на 25-й эпохе при 30 на A100; запас на w32 был не нужен |
| 2 | `mask_mode = multitask` | ✅✅ — +0.34 mIoU_fg против `angle`; **главный win проекта** |
| 3 | length_scale = 1/3 | ✅ — Dice сходится без проблем |
| 4 | Object-wise split | ✅ — нужно для честной оценки sim2real |
| 5 | SGD + poly LR | ✅ — стандартная сходимость, без сюрпризов |
| 6 | LR = 0.01 | ✅ — gradient stable на batch=48, AMP fp16 |
| 7 | CE+Dice (для angle) / BCE+Dice+cos+sin+width (для multitask) | ✅ — все компоненты multitask-loss убывают синхронно |
| 8 | RGB-D depth-channel = mean(RGB-весов) | ✅ — нет «шока» на старте, depth начинает работать сразу |
| 9 | AMP fp16 | ✅ — без NaN'ов, ×2 throughput |
| 10 | image_size = 384 | ✅ — sweet spot |
| 11 | Per-epoch checkpoints | ✅ — позволили выбрать best=26 (а не 29) |
| 12 | Colab Pro A100 | ✅ — 30 эпох × 13.5 мин ≈ 6.5 ч на multitask |
| 13 | batch=48 | ✅ — gradient stable, VRAM 20/40 GB |
| 14 | `_scene_to_model_space` для Cornell | ✅ — без него получается сквош 4:3 → 1:1, углы grasp'ов искажаются |

---

## 7. Что осталось (опциональные следующие шаги)

Все пункты ниже — **не блокеры**: основная задача (обучить + оценить + визуализировать) закрыта.

### 7.1. Численные результаты Cornell

В текущем ноутбуке секция 5.3 строит таблицу top-1 acc по моделям, но конкретные числа зависят от `NUM_CORNELL_EVAL` (по умолчанию 200). После прогона у пользователя — добавить итоговые числа в этот отчёт + bar-plot:

```
Model                       | top-1 acc | top-5 any | mean IoU | mean Δθ
---------------------------- | --------- | --------- | -------- | -------
hrnet_w18_rgb_multitask      |   <TBD>   |   <TBD>   |  <TBD>   |  <TBD>
hrnet_w18_rgbd_multitask     |   <TBD>   |   <TBD>   |  <TBD>   |  <TBD>
hrnet_w18_rgbd_angle         |   <TBD>   |   <TBD>   |  <TBD>   |  <TBD>
```

### 7.2. Fine-tune на Cornell-train

Все 885 Cornell-сцен можно поделить object-wise (~280 объектов) — например, 80/20 train/test. После ~5–10 эпох fine-tune на Cornell-train цифры из 7.1 могут вырасти в 1.5–2 раза. Это уже не cross-domain generalization, но полезно для production-сценария «робот работает на тех предметах, что я ему показал».

### 7.3. Inference на собственных RGB-D снимках

Пользователь располагает RTX 3060 + (потенциально) реальной RGB-D камерой. `tools/infer.py` нужен только для real-снимков:
- Вход: RGB+depth (`.png`+`.tiff` или раскадровка с камеры).
- Подготовка: pad-to-square + uniform-resize (тот же `_scene_to_model_space`).
- Forward → top-K grasp rectangles + JSON.
- Опционально — overlay для отчёта.

Скрипт мелкий (~80 строк), но требует от пользователя: модель камеры + 10–20 пробных снимков.

### 7.4. Оптимизации, которые не понадобились

В прошлом отчёте (v2 раздел 8) рассматривались как «варианты при недостатке качества». С финальным mIoU_fg = 0.679 они не нужны:

- ❌ `length_scale = 0.5` — текущие 0.33 дают чистые маски.
- ❌ `hrnet_w32` backbone — w18 не упёрся в потолок ёмкости.
- ❌ Big batch (64) + larger LR — текущие 48 + 0.01 стабильны.

---

## 8. Ключевые ссылки

- **Репо:** https://github.com/chelovekrazumnii2-png/HRNet_Grasp_Semantic_Segmentation
- **Базовая статья HRNet:** https://arxiv.org/abs/1908.07919
- **GG-CNN (multitask formulation):** https://arxiv.org/abs/1804.05172
- **Jacquard V2:** https://jacquard.liris.cnrs.fr/
- **Cornell Grasp Dataset:** http://pr.cs.cornell.edu/grasping/rect_data/data.php
- **Pretrained backbone:** https://huggingface.co/timm/hrnet_w18.ms_aug_in1k
- **Прошлые отчёты:** [progress_report.md](progress_report.md), [progress_report_v2.md](progress_report_v2.md)
- **Документация визуализации:** [visualization.md](visualization.md)
- **Локальный setup (Windows):** [../local_setup_windows.md](../local_setup_windows.md) (если опубликован), либо `local_setup_windows.md` в корне репо
- **Файл-источник метрик:** `train_results/<run>/metrics.csv` (по одному на каждый run)

### 8.1. PR-история текущей стадии

| # | Заголовок | Состояние |
|---|---|---|
| 1 | viz: comprehensive visualization pipeline for the report | merged |
| 2 | viz: Windows local-setup defaults + nested Cornell layout | merged |
| 3 | docs: Windows local setup guide + RTX 3060 visualization defaults | merged |
| 4 | cornell: load pcdNNNNd.tiff depth for cross-domain RGB-D inference | merged |
| 5 | viz: pad+resize Cornell scenes to model space (fix broadcast error) | merged |
| 6 | viz: Cornell quantitative top-1 accuracy + failure-case catalog | merged |
| 7 | viz: register cornell_eval in package + guard empty failure grid | merged |
| 8 | viz: per-head heatmaps + Grad-CAM (что видит модель) | merged |

---

## 9. Итог

Ключевые цифры, которые имеют смысл показать в защите/докладе:

- **Best val_mIoU_fg = 0.6788** (RGB-D multitask, epoch 26 из 30) — в **1.5×** выше целевой планки (0.45) и в **1.7×** выше минимально приемлемой (0.40), которые были обозначены в прошлом отчёте.
- **Депт даёт +3.7 п.п. mIoU_fg** (RGB-only → RGB-D) при цене ×3 по времени шага.
- **Multitask GG-CNN-формулировка даёт +34 п.п. mIoU_fg** против classical 19-class классификации углов — это самое крупное изменение качества за всю историю проекта.
- **Cross-domain пайплайн** на Cornell готов и работает (loader + frame-перевод + side-by-side + top-1 acc + failure-catalog + heatmap/Grad-CAM); цифры зависят от выбранного `NUM_CORNELL_EVAL` и подтягиваются прямо в ноутбуке.

Все технические компоненты — модульные, переиспользуемые, документированные. Код и ноутбук готовы для воспроизведения как на локальной RTX 3060 (после описанных в `local_setup_windows.md` шагов), так и на Colab/Kaggle.
