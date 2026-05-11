# Training metrics — что значит каждое число в логах

Этот файл объясняет каждое поле, которое выводит наш `Trainer` по
ходу обучения. Пример строки лога (из реального RGB-D multitask-рана):

```
2026-04-28 00:09:50,681 | INFO | epoch 28 step 24920 lr 0.000493 |
  loss=0.6104 pos_bce=0.0462 pos_dice=0.1792 cos=0.1934 sin=0.1907
  width=0.0089 data_time_s=0.0203 compute_time_s=0.9376 step_time_s=0.9579
  dataload_fraction=0.0072 gpu_mem_alloc_gb=0.4132 gpu_util_pct=62.8248

[epoch 28] train: loss=0.6107 pos_bce=0.0462 pos_dice=0.1792 cos=0.1934
  sin=0.1909 width=0.0089 data_time_s=0.0198 compute_time_s=0.9267
  step_time_s=0.9465 dataload_fraction=0.0071 gpu_mem_alloc_gb=0.4132
  gpu_util_pct=63.6453 gpu_mem_peak_gb=17.6400

[epoch 28] timing: step=946.5ms (data=19.8ms, compute=926.7ms,
  dataload_frac=0.01) -> GPU-bound (good)

[epoch 28] gpu: util=64% mem_alloc=0.41GB mem_peak=17.64GB

[epoch 28] val: miou=0.8341 miou_fg=0.6751 dice=0.9013 dice_fg=0.8061
  precision_fg=0.7379 recall_fg=0.8881 miou_fg_ang=0.2099 dice_fg_ang=0.3464
  cos_mse=0.1944 sin_mse=0.1753 ang_mae_deg=14.8812
```

Всё разделено на четыре блока: **training step line** (раз в
`log_interval` шагов), **train epoch summary**, **timing/gpu summary** и
**val metrics**.

---

## 1. Step-line — `epoch X step Y lr ... | <metrics>`

Печатается раз в `log_interval` оптимизаторных шагов (по умолчанию
20). Это бегущее среднее всех потерь и таймингов, накопленное с начала
текущей эпохи.

| Поле | Что значит | Хороший знак |
|---|---|---|
| `epoch` | Номер текущей эпохи (0-indexed). | — |
| `step` | Глобальный номер оптимизаторного шага с начала рана. | — |
| `lr` | Текущий learning rate (после warm-up + poly/cosine decay). | Падает плавно от `lr_init` до 0 к концу. Резкие скачки = баг scheduler'а. |
| `loss` | Полный взвешенный loss, минимизируемый оптимизатором. | Монотонное падение в первые ~10 эпох, потом плато. |

Дальше идут **компоненты лосса** — конкретные слагаемые, которые суммируются (с весами из `loss:` блока конфига) в общий `loss`. Состав зависит от `mask_mode`.

### Multitask mode (`mask_mode=multitask`, GG-CNN-style)

| Компонент | Что считает | Норма к концу обучения |
|---|---|---|
| `pos_bce` | BCE-with-logits по `pos`-голове (вероятность graspability в каждом пикселе). | ~0.04–0.06 (низко — позитивы редкие). |
| `pos_dice` | Soft Dice по `pos`-голове (комплементарно к BCE — штрафует false negatives). | ~0.15–0.25. |
| `cos` | MSE между предсказанным `cos2θ` и таргетом, **только на positive-пикселях**. | ~0.15–0.25. |
| `sin` | MSE между предсказанным `sin2θ` и таргетом, **только на positive-пикселях**. | ~0.15–0.25. |
| `width` | MSE между предсказанной шириной grasp'а (нормированной к 150 px → [0,1]) и таргетом. | ~0.005–0.015. |

`pos_bce + pos_dice` бьются за **геометрию маски графаемых регионов**;
`cos + sin` — за **ориентацию** (через unit-circle parametrization,
избегая wrap-around 0°↔180°); `width` — за **раскрытие захвата**.

### Angle mode (`mask_mode=angle`, 19-class classification)

| Компонент | Что считает | Норма |
|---|---|---|
| `ce` | Cross-entropy по 19 классам (background + 18 angle bins). | Падает с ~3.0 до ~0.5. |
| `dice` | Multi-class soft Dice. | Падает с ~0.95 до ~0.6. |

### Binary mode (`mask_mode=binary`)

| Компонент | Что считает |
|---|---|
| `bce` | BCE-with-logits на бинарной маске «есть grasp / нет grasp». |
| `dice` | Soft Dice на той же бинарной маске. |

### Тайминги (общие для всех режимов)

| Поле | Что значит | Что делать, если плохо |
|---|---|---|
| `data_time_s` | Сколько секунд main-thread ждал, пока worker'ы DataLoader'а подготовят следующий батч. | Если > 30% от `step_time_s` — увеличь `num_workers` и/или `prefetch_factor`. |
| `compute_time_s` | Forward + backward + optimizer step (на GPU). | Это «полезная» работа; должна доминировать. |
| `step_time_s` | Полное время одного шага (data + compute). | Делает ~`step_time_s × steps/epoch` секунд / эпоха. |
| `dataload_fraction` | `data_time_s / step_time_s`. Доля времени, потерянная в ожидании данных. | < 0.10 → GPU-bound (хорошо); 0.10–0.30 → balanced; > 0.30 → DATALOADER-BOUND. |
| `gpu_mem_alloc_gb` | Среднее живое VRAM в момент шага (`torch.cuda.memory_allocated`). | Должно быть < `gpu_mem_peak_gb`. |
| `gpu_util_pct` | nvidia-smi GPU utilization %, усреднённый по шагам (через pynvml). | 60–95% — норма. < 50% → GPU простаивает (см. dataload_fraction). |

---

## 2. `[epoch N] train: ...` — итоговая строка эпохи

Точно те же ключи, что и в step-line, но усреднённые **за всю эпоху**, а
не за `log_interval` шагов. Плюс одно дополнительное поле:

| Поле | Что значит |
|---|---|
| `gpu_mem_peak_gb` | Пиковое VRAM за эпоху (`torch.cuda.max_memory_allocated`). Не усредняется — это абсолютный максимум. Если приближается к лимиту GPU → риск OOM. |

---

## 3. `[epoch N] timing: step=...ms` и `[epoch N] gpu: ...`

Человекочитаемый verdict по тем же тренировочным метрикам:

```
[epoch 28] timing: step=946.5ms (data=19.8ms, compute=926.7ms,
                   dataload_frac=0.01) -> GPU-bound (good)
[epoch 28] gpu: util=64% mem_alloc=0.41GB mem_peak=17.64GB
```

Verdict-таблица:

| `dataload_fraction` | Verdict |
|---|---|
| < 0.10 | **GPU-bound (good)** — добавлять worker'ов бессмысленно. |
| 0.10–0.30 | **balanced** — DataLoader не блокирует, но не доминирует. |
| > 0.30 | **DATALOADER-BOUND** — увеличь `num_workers` / `prefetch_factor` / `persistent_workers`. |

---

## 4. `[epoch N] val: ...` — метрики на валидации

Считаются раз в `eval_every` эпох (по умолчанию каждую). Состав **зависит от `mask_mode`**.

### Multitask mode (binary fg + per-pixel orientation)

| Метрика | Что значит | Хорошо |
|---|---|---|
| `miou` | Mean IoU по двум классам (background + foreground). | > 0.80. |
| `miou_fg` | IoU только класса foreground (graspable region). **Это `target_metric` — по нему выбирается `best.pth`.** | > 0.65 — отличный результат на Jacquard V2. |
| `dice` | Mean Dice по двум классам (более «поощрительная» метрика, чем mIoU). | > 0.90. |
| `dice_fg` | Dice класса foreground. | > 0.80. |
| `precision_fg` | Точность foreground: из всех предсказанных positive — какая доля правильная. | > 0.70. |
| `recall_fg` | Полнота foreground: из всех настоящих positive — какую долю модель нашла. | > 0.85. |
| `miou_fg_ang` | mIoU foreground'а **per angle bin** (модель предсказывает `cos/sin` → argbin → 19-класс маска → IoU). Сопоставимо с `miou_fg` из angle mode. | > 0.20 на multitask (меньше, чем у angle mode, потому что angle здесь регрессионный, а не classification). |
| `dice_fg_ang` | То же, но Dice. | > 0.30. |
| `cos_mse` | Per-pixel MSE между предсказанным `cos2θ` и таргетом, на positive-пикселях. | < 0.20. |
| `sin_mse` | Per-pixel MSE между предсказанным `sin2θ` и таргетом, на positive-пикселях. | < 0.20. |
| `ang_mae_deg` | **Mean absolute error** угла grasp'а в **градусах**, на positive-пикселях. Восстановлен из (cos2θ, sin2θ) → atan2 / 2 → unwrap к [0°, 180°). | < 15° — практически пригодно для робота; < 25° — приемлемо. |

### Angle mode (19-class)

| Метрика | Что значит |
|---|---|
| `miou` | Mean IoU по 19 классам. |
| `miou_fg` | Mean IoU по 18 angle-bin'ам (без background). **target_metric.** |
| `dice` / `dice_fg` | То же, но Dice. |
| `pixel_acc` | Доля пикселей, чей класс предсказан верно. |
| `top1_acc_grasp` | Top-1 grasp accuracy по Jacquard match-criteria (IoU>0.25 и Δθ<30° с любым GT-грипером). |

### Binary mode

| Метрика | Что значит |
|---|---|
| `miou` / `miou_fg` / `dice` / `dice_fg` | Те же, что выше, но на 2 класса (есть/нет grasp). |
| `precision_fg` / `recall_fg` | Точность/полнота foreground. |

---

## 5. Где лежат файлы метрик

После завершения каждой эпохи `Trainer` дописывает строку в:

| Файл (под `save_dir`) | Содержание | Гранулярность |
|---|---|---|
| `metrics.csv` | Все train_* и val_* метрики этой эпохи + lr + global_step. | 1 строка / эпоху |
| `iter_log.jsonl` (если включено `iter_log_path`) | Loss-компоненты + lr на каждом optimizer-шаге. | 1 строка / шаг |

Чекпоинты:
- `best.pth` — обновляется каждую эпоху, если `val_<target_metric>` (по умолчанию `miou_fg`) превысил предыдущий максимум.
- `last.pth` — обновляется каждую эпоху, всегда последняя.
- `epoch_NNN.pth` — снэпшоты раз в `save_every_n_epochs` эпох (или каждую, если `save_every_epoch=true`).

---

## 6. Что не выводится напрямую, но можно посчитать

- **Throughput (samples/sec)** = `batch_size × accum_steps / step_time_s`.
  Например, `batch=2 accum=8 step=4s` → 4 sample/s → 14.4k samples/час.
- **Эффективный размер батча** = `batch_size × accum_steps`. Должен быть ≥ 8 для стабильного BatchNorm; типично 16–32 для сегментации.
- **ETA до конца обучения** = `(epochs - epoch) × step_time_s × steps_per_epoch`.
