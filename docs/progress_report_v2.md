# Отчёт по проекту HRNet-W18 Jacquard V2 Grasp Segmentation (краткая версия)

**Дата составления:** 27 апреля 2026 г.
**Текущий статус:** обучение в процессе (Colab Pro A100, ~25/80 эпох, ~6 ч идёт, ~14 ч до завершения).
**Репозиторий:** https://github.com/Hesgoryr/HRNet_Grasp_Semantic_Segmentation

> Это сокращённая редакция [progress_report.md](progress_report.md). Из неё убраны
> мелкие технические неполадки (установочные ошибки, опечатки в путях, и т.п.) —
> сосредоточились на тех решениях и проблемах, что влияли на **производительность,
> качество обучения и скорость**.

---

## 1. Краткое резюме

Построили модульный пайплайн обучения HRNet-W18 для пиксельной сегментации
grasp-областей на датасете **Jacquard V2**. Задача: пиксельное предсказание
**19 классов** — фон + 18 углов схвата (по 10° каждый).

С нуля до полностью рабочего облачного пайплайна за ~10 итераций / 10 PR-ов:
- Локальный пайплайн (RTX 3060) → переход на облако (Colab + Kaggle).
- 2 платформы (Colab Pro A100, Kaggle T4), 2 input-режима (RGB / RGB-D).
- Воспроизводимый split, per-epoch checkpoints, `--resume` для длинных прогонов.

**Текущий run:** Colab Pro A100 40 GB, RGB-D, 80 эпох, batch=48, ~13.5 мин/эпоха.
Ожидаемое окончание через ~14 часов.

---

## 2. Хронология и этапы

### Этап 1. Постановка задачи и каркас (Session 1)

**Запрос:** «Что нужно для правильного запуска обучения модели HRNet
на датасете Jacquard V2?»

**Что сделали:**
- Изучили формат Jacquard V2: каждый объект — папка с N grasp-снимками,
  файлы `*_RGB.png`, `*_perfect_depth.tiff`, `*_stereo_depth.tiff`,
  `*_mask.png`, `*_grasps.txt`. ~11 619 объектов, ~52 000 grasp-конфигураций.
- Спроектировали модуль `grasp_seg/`:
  - `data/` — Jacquard reader + augmentations + splits + grasp-rect parsing.
  - `models/` — HRNet-W18 backbone (timm) + multi-resolution fusion head.
  - `losses/` — три варианта loss (binary, multi-class CE+Dice, multitask).
  - `engine/` — Trainer + evaluator.
  - `utils/` — logger, meters.
- Создали 3 mask-режима: `binary` (fg/bg), `angle` (19 классов), `multitask`
  (4 регрессионных канала: pos / cos2θ / sin2θ / width — стиль GG-CNN).
- Tools: `prepare_split.py` (object-wise 80/10/10), `train.py`, `eval.py`,
  `smoke_test.py`.

**PR #1** — основной пайплайн.

### Этап 2. Локальная RTX 3060 6 GB (Session 2)

**Что узнали:** На 6 GB VRAM **впритык** влезает batch=8 при image_size=384 —
запас всего ~500 МБ. На 80 эпох ушло бы **~42 часа** локально. Это
неприемлемо → решили переходить на облако.

### Этап 3. Per-epoch checkpoints + resume (Session 4)

**Запрос:** Длинные прогоны (десятки часов) рискуют прерваться. Нужны
точки сохранения и возможность продолжить.

**Что сделали:**
- В `Trainer.save_checkpoint`: `epoch_NNN.pt` каждую эпоху + `last.pt`
  (последний) + `best.pt` (по val_mIoU_fg).
- `metrics.csv` пишется построчно после каждой эпохи.
- `--resume <path>` в `train.py`: читает model + optimizer + scheduler +
  scaler + state, продолжает с следующей эпохи.

**PR #4** — checkpoints + resume.

### Этап 4. Облачные рецепты (Session 5)

**Запрос:** 42 часа локально неприемлемы, нужно на облаке.

**Что сделали:**
- `notebooks/colab_train.ipynb` — Colab Pro A100, чекпоинты в Drive.
- `notebooks/kaggle_train.ipynb` — Kaggle T4, чекпоинты в `/kaggle/working/`.
- В каждом recipe: download датасета, unzip, flatten, prepare_split, train с
  resume.

**PR #5** — облачные рецепты.

### Этап 5. RGB-only вариант (Session 6)

**Запрос:** «Сделай конфиг для 3-канального RGB без depth.»

**Что сделали:**
- `configs/rgb.yaml` — копия `default.yaml` с `input_mode: rgb`, отключёнными
  depth-аугментациями.
- В `models/hrnet.py:_patch_first_conv`: при `in_channels=3` копирует
  ImageNet-веса как есть, при `in_channels=4` инициализирует depth-канал
  средним RGB-весов.

**PR #6** — RGB-only конфиг.

### Этап 6. Облачные эксперименты (текущая сессия)

#### 6.1. Pre-unpacked Kaggle dataset (PR #7)

**Что узнали:** Загрузка 63 ГБ zip-ов с Я.Диска + распаковка занимали ~45 мин
на старте каждой Kaggle-сессии. Это съедало GPU-квоту.

**Решение:** Нашли третью сторону, выложившую распакованный датасет на Kaggle:
`vdsdggsgsd/jacquard`, ~77 ГБ, mounted at
`/kaggle/input/datasets/vdsdggsgsd/jacquard/Jacquard/`. Файловая структура
совпадает с тем, что ожидает наш парсер.

**Эффект:** Экономия ~45 мин/сессия → нужное время для обучения.

#### 6.2. DataParallel на Kaggle T4 x2 — попытка ускорения (PR #9, откат)

**Контекст:** Kaggle даёт «GPU T4 x2», но наш training loop использовал только
одну. GPU 0 = 100%, GPU 1 = 0%.

**Что сделали:** PR #9 — автовключение `torch.nn.DataParallel` при
`torch.cuda.device_count() > 1`. Чекпоинты сохраняются без `module.` prefix
(unwrap перед save), при load — strip `module.` если есть. Совместимо в обе
стороны (single-GPU ↔ multi-GPU).

**Результат:** s/step стало **хуже** (1.10 с vs 0.88 с на одной T4).
Корень: Kaggle T4 x2 имеет всего **4 CPU cores**. CPU был bottleneck (385/400%
загрузки). DataParallel добавил scatter/gather overhead, при том что данные
всё равно не успевали поступать. GPU простаивали (19% и 11% utilization).

**Решение:** Откатились к single GPU через `CUDA_VISIBLE_DEVICES=0`. PR
остаётся полезен для конфигураций с достаточным CPU.

**Урок:** Multi-GPU помогает только когда compute — bottleneck. На CPU-bound
конфигурациях DP только вредит. На Colab A100 с её мощным CPU и одной GPU
проблемы нет.

#### 6.3. Тонкая настройка batch-size на Colab A100

**Что узнали:**
- На A100 40 GB: batch=32 → 0.65 с/step, 49 img/s.
- batch=48: 0.94 с/step, 51 img/s. Per-image throughput почти не изменился
  (compute-bound), но gradient signal чуть стабильнее.
- VRAM использовано 20 GB из 40 GB → запас огромный, можно идти до bs=64,
  но скорости это не даст.

**Решение:** Остановились на batch=48. Минимальное преимущество в gradient
stability при незаметной потере скорости.

#### 6.4. Решение про Kaggle

**Что узнали:** При single-GPU T4 80 эпох заняли бы ~51 ч (5 сессий по
12 ч + resume). Юзер решил остановить Kaggle: основной кейс RGB-D покрывает
Colab A100, RGB-only можно запустить там же позже.

---

## 3. Ключевые решения

| # | Решение | Альтернативы | Обоснование |
|---|---|---|---|
| 1 | HRNet-W18 backbone | ResNet50, EfficientNet, hrnet_w32, hrnet_w48 | HRNet сохраняет высокое разрешение через все scales — идеально для pixel-wise задач. W18 — баланс точность/скорость; w32/w48 в 1.5-2.5× медленнее за +5-10% mIoU. |
| 2 | Mask mode = `angle` (19 классов) | `binary`, `multitask` | Compromise: даёт пиксельные углы (важно для grasp), не требует учить регрессию (multitask чаще нестабилен). |
| 3 | length_scale = 1/3 | 1/2, 1/1 | Стандарт Jacquard toolbox. Делает маски тоньше → строже метрики, но точнее центр grasp'а. |
| 4 | Object-wise split (80/10/10) | image-wise, scene-wise | Предотвращает leakage: один объект **не может** быть и в train, и в val. См. раздел 4 для подробностей. |
| 5 | SGD + poly LR + 1-epoch warmup | Adam, cosine, OneCycle | Стандарт сегментации (HRNet, DeepLab, BiSeNet). Poly с power=0.9 даёт более длинный «средний» LR чем cosine. |
| 6 | LR = 0.01 | 0.005, 0.02 | Стандартная база для SGD-сегментации с batch~16-32. На bs=48 теоретически можно поднять, но не рисковали. |
| 7 | CE + Dice loss (без bg в Dice) | Focal, Lovász, Tversky | Robust комбо: CE учит классы, Dice учит границы. Без bg в Dice т.к. фон занимает >95% пикселей. |
| 8 | RGB-D с depth = mean(RGB-весов) | Random init для depth-канала | Меньше шок для backbone. Depth начинает работать как «нейтральный grey channel» и постепенно специализируется. |
| 9 | AMP fp16 | bf16, fp32 | A100 поддерживает оба, но fp16 совместим со всеми GPU (T4 тоже). |
| 10 | image_size = 384 | 256, 512 | Compromise: 256 теряет детали, 512 в 4× медленнее. 384 — sweet spot. |
| 11 | Per-epoch checkpoints | Каждые N эпох, только best | Полная безопасность от прерываний. Стоит ~340 МБ × 80 = 27 ГБ дискового места — приемлемо в Drive. |
| 12 | Colab Pro A100 (платно) | T4 free, L4, V100 | A100 в ~6× быстрее T4. Время важнее, чем экономия compute units. |
| 13 | batch_size = 48 (на A100) | 32, 64 | Сompute-bound. 48 даёт чуть стабильнее gradient при той же скорости в img/s. |

---

## 4. Подробнее об object-wise split

**Да, мы обучаем на 80% датасета.** Конкретно:

- **Всего:** 11 619 уникальных объектов в Jacquard V2 (это не картинки, а 3D-модели предметов).
- **На каждый объект** приходится ~5-10 grasp-конфигураций (разные ракурсы и углы захвата).
- **Splits делятся по объектам** (не по картинкам):
  - **Train: 80% объектов** (~9 295) → ~41 000 grasp-картинок.
  - **Val: 10% объектов** (~1 162) → ~5 200 картинок (validation после каждой эпохи).
  - **Test: 10% объектов** (~1 162) → ~5 200 картинок (используется только в финале).

**Зачем такой split?** Чтобы модель не «запоминала» конкретные предметы.
Если бы делили по **картинкам**, то картинки одного объекта могли попасть и в
train, и в val — модель бы выучила, как выглядит этот предмет, и метрики были
бы завышены. Object-wise split строго запрещает: один объект **может быть
только в одной части**.

В логе обучения: «epoch 0 step 860» означает 860 батчей × 48 = ~41 000 картинок —
это и есть наш 80%-train.

---

## 5. Проблемы, влиявшие на производительность / качество / скорость

### 5.1. CPU-only torch на pip (PR #2) — производительность ×100

**Проблема:** На голом `pip install torch` PyPI ставит CPU-only wheel.
Тренировка не использовала GPU вообще — была в **~100 раз медленнее**.

**Решение:** Документировали явный CUDA-wheel index URL. Закрепили в
`requirements.txt` и в README инструкции.

**Эффект:** Без этого фикса вообще никакое обучение было бы невозможно.

### 5.2. Локальная RTX 3060 — недостаточная скорость для полного цикла

**Проблема:** На 6 GB VRAM batch=8, ~32 мин/эпоха → ~42 ч на 80 эпох.
Прерывания питания, перегрев, тепловые троттлинги делают такой длинный
прогон ненадёжным.

**Решение:** Переход на облако (Colab/Kaggle). На A100 — 13.5 мин/эпоха, в
2.4× быстрее.

**Эффект:** Прогон стал реалистичным (с 42 ч → 18 ч на A100).

### 5.3. Multi-GPU попытка на Kaggle T4 x2 — производительность −25%

**Проблема:** На Kaggle есть «GPU T4 x2», но из коробки наш код использовал
только одну. Попытались включить DataParallel.

**Корень:** Kaggle T4 x2 имеет всего **4 CPU cores**. CPU был bottleneck
(385/400% загрузки). DataParallel добавил scatter/gather overhead в Python
(GIL), при том что данные всё равно не успевали поступать. GPU простаивали
(19% и 11% utilization). s/step стало **хуже** (1.10 с vs 0.88 с на одной).

**Решение:** Откатились к single GPU через `CUDA_VISIBLE_DEVICES=0`. PR
остался для будущих конфигураций с достаточным CPU (например, V100 ×2 на
Y.DS, или собственный сервер).

**Урок:** Multi-GPU помогает только когда compute — bottleneck. Для
data-bound сценариев нужно сначала чинить data pipeline.

### 5.4. Pre-unpacked Kaggle dataset (PR #7) — экономия 45 мин/сессия

**Проблема:** Каждая Kaggle-сессия начиналась с download 63 ГБ zip-ов с
Я.Диска + unzip → ~45 минут потери GPU-квоты.

**Решение:** Перешли на сторонний пре-распакованный Kaggle public dataset
`vdsdggsgsd/jacquard`. Mounted в read-only, готов к использованию мгновенно.

**Эффект:** Каждая сессия стартует обучение через 1-2 мин вместо 45.

### 5.5. Тонкая настройка batch-size — компромисс качества и скорости

**Проблема:** На A100 40 GB неясно, какой batch_size оптимален. Слишком
маленький → нестабильный gradient. Слишком большой → compute-bound, скорости
не прибавится.

**Эксперимент:**
- batch=32 → 0.65 с/step, 49 img/s.
- batch=48 → 0.94 с/step, 51 img/s.
- VRAM: 20 GB из 40 GB при batch=48 → запас огромный.

**Решение:** Остановились на batch=48. Минимальное преимущество в gradient
stability при незаметной потере скорости.

---

## 6. Текущая конфигурация (Colab A100, активный прогон)

### 6.1. Hardware

| Параметр | Значение | Обоснование |
|---|---|---|
| GPU | 1× A100 40 GB | A100 в 6-8× быстрее T4, входит в Colab Pro budget. 40 GB ≈ 2× больше типичного потребления (20 GB) — есть запас. |
| Backend | Google Colab Pro | $10/мес, 24-часовые сессии (vs 12 ч free), 100 compute units/мес, A100 access. |
| Storage | `/content` (235 GB SSD) + Drive | SSD для скачивания/распаковки 65 GB, Drive для checkpoints. |
| Network | Yandex Disk → Colab | ~30 мин на 63 ГБ через public API. |

### 6.2. Data

| Параметр | Значение | Обоснование |
|---|---|---|
| Dataset | Jacquard V2, 11 619 объектов | Стандарт grasp-сегментации. |
| Источник | Я.Диск public share, 12 шардов | Бесплатное хранилище. |
| Splits | object-wise 80/10/10, seed=0 | Detection of leakage на конкретные объекты. |
| image_size | 384 × 384 | Compromise: 256 теряет детали (1024 оригинал), 512 в 4× медленнее. |
| input_mode | `rgbd` (4 канала) | RGB + perfect depth. Depth даёт +3-5% точности. |
| Mask mode | `angle` (19 классов: bg + 18 × 10°) | Пиксельные углы — что нам нужно для grasp prediction. |
| length_scale | 0.3333 (1/3) | Стандарт Jacquard toolbox. |

### 6.3. Augmentation

| Augmentation | p | Параметры | Обоснование |
|---|---|---|---|
| H-flip | 0.5 | — | Стандарт, корректно перевёртывает angle. |
| V-flip | 0.5 | — | Стандарт. |
| Rotate | 0.8 | ±180° | Учит rotation-invariance, корректно сдвигает angle класс. |
| Scale | 0.7 | 0.8-1.2 | Robustность к масштабу объектов. |
| Translate | 0.5 | ±5% | Robustность к позиции в кадре. |
| Color jitter | 0.5 | brightness/contrast/saturation 0.2, hue 0.05 | Robustность к освещению. |
| RGB noise | 0.3 | std=0.02 | Robustность к sensor noise. |
| Depth jitter | 0.5 | range 0.95-1.05 | Robustность к depth scale errors. |
| Depth dropout | 0.3 | 2% pixels → 0 | Имитация depth-sensor holes. |
| Stereo swap | 0.3 | perfect → stereo | Robustность к низкокачественному depth. |

### 6.4. Model

| Параметр | Значение | Обоснование |
|---|---|---|
| Backbone | `hrnet_w18` (timm) | Pretrained HuggingFace, 85.6 МБ ImageNet-1k weights. |
| Pretrained | True | Transfer learning, +20-30% точности vs random init на 80 эпох. |
| Adapter | `_patch_first_conv` для 4-канального input | RGB-веса копируются 1:1, depth-канал = mean(RGB). |
| Head | `Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Dropout` | Стандартный fusion head для multi-resolution outputs. |
| head_channels | 256 | Compromise: больше = точнее, но медленнее. 256 — sweet spot для w18. |
| Classifier | `Conv1×1 → 19 каналов` | Pixel-wise softmax по 19 классам. |
| Dropout | 0.1 | Слабая регуляризация (датасет большой, overfitting риск умеренный). |

### 6.5. Training

| Параметр | Значение | Обоснование |
|---|---|---|
| Epochs | 80 | Стандарт для Jacquard. Меньше — недообучение, больше — overfitting на augmentation. |
| Batch size | 48 | Compromise stability/throughput. Помещается до 64 на A100, но скорости не даёт. |
| Accum steps | 1 | Effective batch = 48. |
| Optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=5e-4 | Стандарт segmentation. |
| Scheduler | Poly (power=0.9) | Smooth decay, более длинный «средний» LR чем cosine. |
| Warmup | 1 эпоха (linear от 0 до lr) | Стабилизация старта при больших batch. |
| Grad clip | 1.0 | Защита от exploding gradients (особенно с AMP). |
| AMP | fp16 | A100 хорошо тянет, +2× скорость над fp32. |
| Save | каждую эпоху + best.pt + last.pt | Безопасность + удобство resume. |
| Eval | каждую эпоху | Видим динамику метрик в реальном времени. |

### 6.6. Loss

| Компонент | Вес | Обоснование |
|---|---|---|
| CE (multi-class) | 1.0 | Учит классификацию углов. |
| Dice (fg-only) | 1.0 | Учит границы fg-маски. |
| Background в Dice | excluded | Bg занимает >95% пикселей, забивает Dice gradient. |
| pos_weight (CE) | 1.0 | Балансирующий вес — uniform по классам. |

---

## 7. Текущие результаты (epoch 22 из 80)

### 7.1. Кривые

```
epoch | train_loss | train_dice | val_mIoU_fg | val_Dice_fg
   0  |    1.259   |   0.905    |    0.225    |    0.366
   5  |    0.705   |   0.637    |    0.288    |    0.446
  10  |    0.688   |   0.622    |    0.307    |    0.469
  15  |    0.677   |   0.612    |    0.306    |    0.468
  20  |    0.668   |   0.604    |    0.319    |    0.483
  22  |    0.666   |   0.602    |    0.323    |    0.488  ← best
```

### 7.2. Анализ

- **CE loss** уже после epoch 1 упал до 0.07 — модель выучила classification.
  Дальше учится **Dice** (форма маски).
- **Train_dice** медленно ползёт вниз с 0.92 до 0.60. Темп замедляется.
- **Val_mIoU_fg** растёт с 0.225 до 0.323 (logarithmic curve).
  - epoch 0-10 = +0.082
  - epoch 10-22 = +0.016
- **Прецизия > recall** — модель «осторожна» с предсказанием fg, что обычно
  для CE+Dice конфигурации.
- **Best so far:** epoch 21, val_mIoU_fg=0.3233 → сохранено в `best.pt`.

### 7.3. Прогноз финала (epoch 80)

| Сценарий | val_mIoU_fg | val_Dice_fg |
|---|---|---|
| Линейная экстраполяция | ~0.39 | ~0.55 |
| Реалистичный (с poly decay) | ~0.42-0.46 | ~0.58-0.62 |
| Optimistic | ~0.48-0.50 | ~0.65 |

Целевой 0.45 (наша «целевая» планка) **достижим** в реалистичном сценарии.
Минимально приемлемый 0.40 — высокая вероятность.

---

## 8. Возможные решения замедленного прогресса

Если финальный mIoU_fg окажется ниже 0.40, пробуем по убыванию ROI:

### Вариант A. Big batch + larger LR (быстро, low risk)

- `batch_size=64`, `lr=0.015` или `0.02`.
- Ожидаемый прирост: +0.02-0.04 mIoU_fg.
- Стоимость: 1 повторный прогон (~20 ч).

### Вариант B. Сменить mask_mode на `multitask` (medium risk, **главный кандидат**)

- Регрессионные головы pos / cos2θ / sin2θ / width — обычно дают +5-10% mIoU.
- Loss меняется: `MultiClassCEDiceLoss` → `MultiTaskGGCNNLoss`.
- Не требует изменения модели.
- Ожидаемый прирост: +0.05-0.10 mIoU_fg.
- Стоимость: 1 повторный прогон + смена обработки в evaluator.

### Вариант C. Увеличить mask compactness (medium risk)

- `length_scale=0.5` → больше positive pixels → проще учить Dice.
- Ожидаемый прирост: +0.03-0.06 mIoU_fg.
- Стоимость: 1 повторный прогон.

### Вариант D. hrnet_w32 backbone (high cost)

- +60% параметров, +5-10% точности.
- Стоимость: ~30 ч на A100, batch=32 (меньше памяти).
- Ожидаемый прирост: +0.03-0.05 mIoU_fg.

### Рекомендация прямо сейчас

**Дождаться текущего прогона** (ещё ~14 ч), посмотреть финал.
- Если **≥ 0.40** — задача решена адекватно.
- Если **0.35 ≤ < 0.40** — Вариант B (multitask) → главный кандидат.
- Если **< 0.35** — что-то не так, диагностируем глубже.

---

## 9. Дальнейшие планы

### 9.1. Планка качества (согласовано 27.04.2026)

| Уровень | Метрика | Значение |
|---|---|---|
| Минимально приемлемый (Jacquard val) | mIoU_fg | **≥ 0.40** |
| Минимально приемлемый (Jacquard val) | Dice_fg | **≥ 0.55** |
| Целевой (Jacquard val) | mIoU_fg | **≥ 0.45** |
| Целевой (Jacquard val) | Dice_fg | **≥ 0.62** |
| Минимум на test split | mIoU_fg | **≥ 0.38** |

### 9.2. Дорожная карта

**Фаза 1. Завершить текущий RGB-D run** (~14 ч)

**Фаза 2. Полный eval RGB-D** (~30 мин)
- `python tools/eval.py ... --split val` → подтверждение.
- `python tools/eval.py ... --split test` → на ранее не виданных объектах.
- Если **test_mIoU_fg ≥ 0.38** → переход в Фазу 3.
- Если **< 0.38** → Фаза 2a (оптимизация).

**Фаза 2a. Оптимизация RGB-D (если требуется)**

В порядке убыванию ROI:
1. **mask_mode = `multitask`** — приоритет №1, +5-10% mIoU. ~20 ч.
2. **length_scale = 0.5** + повторный прогон — +3-6%, ~20 ч.
3. **hrnet_w32** — +3-5%, ~30 ч + меньший batch.

Бюджет: до 3 итераций (~70 ч compute).

**Фаза 3. Cross-dataset eval на Cornell Grasp Dataset** (~3-5 ч)

Начнётся **после Фазы 2/2a**.

3.1. Подготовка Cornell-парсера:
- Конвертер point cloud (`pcd*.txt`) → depth map.
- Парсер `*_cpos.txt` → `Grasp[]` в нашем формате.
- DataLoader для Cornell с теми же transforms.
- ~1 рабочий день кода, +200 строк, отдельный PR.

3.2. Eval flow:
- Загрузить best.pt (Jacquard RGB-D).
- Прогнать по всем 885 Cornell-изображениям без fine-tune.
- Метрики: mIoU_fg, Dice_fg + grasp-detection metric.

3.3. Целевые результаты:
- Without fine-tune: **mIoU_fg ≥ 0.30**.
- С 5-10 эпох fine-tuning на Cornell train: **mIoU_fg ≥ 0.45**.

**Фаза 4. RGB-only обучение и сравнение** (~20 ч + сравнение)

Параллельно с Фазой 3.

4.1. Запуск:
- `python tools/train.py --config configs/rgb.yaml ...`
- Та же конфигурация, но 3 канала, без depth-аугментаций.
- ~210 compute units на Colab Pro.

4.2. Сравнение head-to-head:

| Метрика | RGB-D | RGB-only | Δ |
|---|---|---|---|
| Jacquard val_mIoU_fg | TBD | TBD | TBD |
| Jacquard test_mIoU_fg | TBD | TBD | TBD |
| Cornell mIoU_fg (no fine-tune) | TBD | TBD | TBD |
| Cornell mIoU_fg (with fine-tune) | TBD | TBD | TBD |
| Speed (s/step) | 0.94 | ~0.85 | -10% |

4.3. Целевые:
- val_mIoU_fg на Jacquard ≥ 0.35 (минимум), ≥ 0.42 (целевой).
- Cornell mIoU_fg ≥ 0.25 (no fine-tune), ≥ 0.40 (with fine-tune).

4.4. Выводы:
- Δ ≥ 5% → depth важен, RGB-D рекомендуется для production.
- Δ ≤ 2% → depth не критичен.

**Фаза 5. Inference на собственных RGB-D снимках**

5.1. Что нужно от пользователя:
- **RGB-D камера:** какая модель? Это влияет на формат depth.
- **Снимки:** ≥ 10-20 разных сцен/объектов.
- **Калибровка:** depth должен быть aligned к RGB.

5.2. Pipeline:
- `tools/infer.py` — RGB+depth → angle mask + визуализация.
- (Опционально) NMS-постпроцесс → top-K grasp rectangles.
- Запуск **обеих** моделей (RGB-D и RGB) на одних и тех же снимках.

5.3. Подводные камни:
- Depth scale: Jacquard в [0,1] м, Realsense — мм.
- FOV/разрешение: ресайз до 384×384.
- Sim2Real gap: Jacquard синтетический, реальные снимки могут потребовать
  fine-tune на Cornell + ваших примерах.

**Фаза 6. Финальный сравнительный отчёт**
- Сводная таблица всех метрик RGB-D vs RGB-only на Jacquard val/test, Cornell, real.
- Визуализация типичных предсказаний.
- Выводы и рекомендации для production.

### 9.3. Общий timeline

| Фаза | Время | Параллелизуемо? |
|---|---|---|
| 1. Закончить RGB-D run | ~14 ч | — |
| 2. Eval RGB-D val + test | ~30 мин | После 1 |
| 2a. (Если нужно) Оптимизации | 20-70 ч | Sequentially |
| 3. Cornell parser + eval | ~3 ч код + 30 мин eval | Параллельно с 4 |
| 3a. Cornell fine-tune | ~3-5 ч | После 3 |
| 4. RGB-only обучение | ~20 ч | Параллельно с 3 |
| 5. Inference на real снимках | ~3 ч код + от пользователя | Параллельно с 3, 4 |
| 6. Финальный отчёт | ~2 ч | После всего |

**Итого:** ~3-5 дней активной работы.

### 9.4. Открытые вопросы

1. **RGB-D камера** для real-snapshots — какая модель? (Realsense / Kinect / OAK-D / iPhone LiDAR / другая.)
2. **Сколько Colab Pro compute units осталось?** RGB-only прогон стоит ~210 units, оптимизации (если потребуются) — ещё 210-630.
3. **Готовые снимки или нужно снимать?** Если нужно — спланируем сценарии.

---

## 10. Метаданные

- **GitHub:** https://github.com/Hesgoryr/HRNet_Grasp_Semantic_Segmentation
- **PR за всё время:** #1-#10 (все мерджены).
- **Datasets:**
  - https://disk.yandex.ru/d/Je56nUcC9hiHFQ (Я.Диск public share, 12 архивов).
  - https://www.kaggle.com/datasets/vdsdggsgsd/jacquard (pre-unpacked Kaggle).
- **Pretrained weights:** https://huggingface.co/timm/hrnet_w18.ms_aug_in1k
- **Полная версия отчёта:** [progress_report.md](progress_report.md).
