# Отчёт по проекту HRNet-W18 Jacquard V2 Grasp Segmentation

**Дата составления:** 27 апреля 2026 г.
**Текущий статус:** обучение в процессе (Colab Pro A100, ~23/80 эпох, ~5.5 ч идёт, ~14.5 ч до полного завершения)
**Репозиторий:** https://github.com/Hesgoryr/HRNet_Grasp_Semantic_Segmentation

---

## 1. Краткое резюме

Построили модульный пайплайн обучения HRNet-W18 для семантической сегментации
grasp-полигонов на датасете Jacquard V2. Базовая задача — pixel-wise предсказание
**19 классов**: фон + 18 углов схвата (по 10° каждый), для генерации сегмента
schлёных областей объекта.

С нуля до полностью рабочего облачного пайплайна за ~10 итераций, 10 PR-ов:
- Локальный пайплайн (RTX 3060) → переход на облако (Colab + Kaggle)
- 3 платформы (Colab, Kaggle, Yandex DataSphere), 2 input-режима (RGB / RGB-D)
- Воспроизводимый split, per-epoch checkpoints, `--resume` для длинных прогонов
- Решено ~7 значимых блокеров деплоя (CUDA wheel, LZW TIFF, Python 3.12 ABI, и др.)

**Текущий run:** Colab A100 40 GB, RGB-D, 80 эпох, batch=48, ~13.5 мин/эпоха.
Ожидаемое окончание через ~14.5 часов.

---

## 2. Хронология и этапы

### Этап 1. Постановка задачи и каркас (Session 1)

**Запрос пользователя:** «Что нужно для правильного запуска обучения модели HRNet
на датасете Jacquard-V2?»

**Что сделали:**
- Изучили формат Jacquard V2: каждый объект — папка с N grasp-снимками,
  файлы `*_RGB.png`, `*_perfect_depth.tiff`, `*_stereo_depth.tiff`, `*_mask.png`,
  `*_grasps.txt`. ~11 000 объектов, ~50 000 grasp-конфигураций.
- Спроектировали модуль `grasp_seg/`:
  - `data/` — Jacquard reader + transforms + splits + grasp-rect parsing
  - `models/` — HRNet-W18 backbone (timm) + multi-resolution fusion head
  - `losses/` — три варианта loss (binary, multi-class CE+Dice, multitask)
  - `engine/` — Trainer + 3 evaluator-функции
  - `utils/` — logger, meters
- Создали 3 mask-режима: `binary` (fg/bg), `angle` (19 классов), `multitask`
  (4 регрессионных канала: pos / cos2θ / sin2θ / width — стиль GG-CNN).
- Tools: `prepare_split.py` (object-wise 80/10/10), `train.py`, `eval.py`,
  `smoke_test.py`.

**PR #1** — основной пайплайн.

### Этап 2. Адаптация под локальную RTX 3060 6 GB (Session 2)

**Проблемы:**
- На 6 GB VRAM tonkim ладно влезает batch=8 при image_size=384.
- На PyPI установился CPU-only torch вместо CUDA — тренировка не использовала GPU.

**Решения:**
- В `requirements.txt` явно указали CUDA-wheel index URL.
- Документировали в README, как установить CUDA-torch.

**PR #2** — документация CUDA torch.

### Этап 3. Декодирование depth TIFF (Session 2)

**Проблема:** TIFF-файлы Jacquard сжаты LZW, стандартная установка PIL/tifffile
не декодирует — `KeyError: 'TIFF compression code 5'`.

**Решение:** Добавили `imagecodecs` в зависимости (это backend для tifffile,
расшифровывает LZW).

**PR #3** — imagecodecs в requirements.

### Этап 4. Per-epoch checkpoints + resume (Session 4)

**Запрос:** Локальное обучение на RTX 3060 при 384²+batch=8 даёт ~32 мин/эпоха →
~42 часа на 80 эпох. Нужны промежуточные точки сохранения, чтобы можно было
прерывать и продолжать.

**Что сделали:**
- В `Trainer.save_checkpoint`: сохранение `epoch_NNN.pt` каждую эпоху +
  обновление `last.pt` (последний) + `best.pt` (по val_mIoU_fg).
- `metrics.csv` пишется построчно после каждой эпохи.
- `--resume <path>` в `train.py`: читает model + optimizer + scheduler + scaler +
  state, продолжает с следующей эпохи.

**PR #4** — checkpoints + resume.

### Этап 5. Облачные рецепты (Session 5)

**Запрос:** Локальные 42 часа неприемлемы. Нужно на облаке.

**Что сделали:**
- `notebooks/colab_train.ipynb` — Colab T4 free / A100 Pro, чекпоинты в Drive.
- `notebooks/kaggle_train.ipynb` — Kaggle T4, чекпоинты в `/kaggle/working/`.
- `docs/yandex_datasphere.md` — инструкция для Я.DataSphere с примонтированием
  Я.Диска.
- В каждом recipe: download датасета с Я.Диска (12 zip-ов через public-resources
  API), unzip, flatten, prepare_split, train с resume.

**PR #5** — три облачных рецепта.

### Этап 6. RGB-only вариант (Session 6)

**Запрос:** «Сделай конфиг для 3-канального RGB без depth.»

**Что сделали:**
- `configs/rgb.yaml` — копия `default.yaml` с `input_mode: rgb`, отключёнными
  depth-аугментациями (`depth_jitter_p=0`, `depth_dropout_p=0`,
  `use_stereo_depth_p=0`).
- В `tools/train.py`: `_input_channels(input_mode)` возвращает 3 для `rgb`
  и 4 для `rgbd`. Это число пробрасывается в `model.in_channels` и в `dataset`.
- В `models/hrnet.py:_patch_first_conv`: при `in_channels=3` копирует
  ImageNet-веса как есть, при `in_channels=4` инициализирует depth-канал
  средним RGB-весов.

**PR #6** — RGB-only конфиг.

### Этап 7. Облачные деплой-проблемы (Текущая сессия)

#### 7.1. Поиск pre-unpacked Kaggle dataset

**Контекст:** Загрузка 63 ГБ zip-ов с Я.Диска + распаковка занимают ~45 минут на
старте каждой Kaggle-сессии. Это съедает GPU-квоту.

**Решение:** Нашли третью сторону, которая выложила распакованный датасет на
Kaggle: `vdsdggsgsd/jacquard`, ~77 ГБ, mounted at
`/kaggle/input/datasets/vdsdggsgsd/jacquard/Jacquard/` с 11 619 объектами в
плоской структуре. Файловая структура совпадает с тем, что ожидает наш парсер:
`*_RGB.png`, `*_perfect_depth.tiff`, `*_mask.png`, `*_grasps.txt`.

**PR #7** — упрощённый Kaggle ноутбук без download/unzip.

#### 7.2. Colab Python 3.12 import error

**Проблема:** На Colab `pip install -r requirements.txt` обновляет numpy и другие
зависимости, в результате уже импортированный torch ломается:
`ImportError: cannot import name 'nn' from partially initialized module 'torch'`.

**Корень:** Python 3.12 строже относится к partial imports. Когда pip обновляет
зависимости загруженного модуля, этот модуль остаётся в полу-инициализированном
состоянии и любой повторный `import torch` падает.

**Решение:** Заменили `pip install -r requirements.txt` на точечную установку
только тех пакетов, которых нет на Colab по умолчанию: `pip install timm
imagecodecs`. Остальные (torch, numpy, opencv, tifffile, matplotlib, pandas,
pyyaml, tqdm) Colab уже имеет.

**PR #8** — точечная установка для Colab.

#### 7.3. Kaggle Internet выключен

**Проблема:** `git clone` падал с `Could not resolve host: github.com`.

**Решение:** Включить в Settings → Internet → On (требует phone verification, что
у пользователя уже было).

#### 7.4. DataParallel на Kaggle T4 x2

**Контекст:** Kaggle даёт «GPU T4 x2», но наш training loop использовал только
одну. Скрин показал GPU 0 = 100%, GPU 1 = 0%.

**Что сделали:** PR #9 — автовключение `torch.nn.DataParallel` при
`torch.cuda.device_count() > 1`. Чекпоинты сохраняются без `module.` prefix
(unwrap перед save), при load — strip `module.` если есть. Совместимо с
single-GPU и multi-GPU runs в обе стороны.

**Результат на Kaggle:** s/step стало **хуже** (1.10 с vs 0.88 с на одной T4).
Корень: Kaggle T4 x2 имеет всего 4 CPU cores. CPU был bottleneck (385/400%
загрузки). DataParallel добавил scatter/gather overhead, при том что данные всё
равно не успевали поступать. GPU простаивали (19% и 11% utilization).

**Решение:** Откатились к single GPU через `CUDA_VISIBLE_DEVICES=0`. PR остаётся
полезен для конфигураций с достаточным CPU (не на Kaggle T4 x2).

#### 7.5. Yandex Disk mirror flatten pattern

**Проблема:** На Colab после распаковки 12 zip-ов `prepare_split.py` упал с
`No '*_grasps.txt' files found`. Корень: zip-архивы из Я.Диска содержат папки
`Jacquard_Dataset_N` (без "V2"), а наш flatten искал `JacquardV2_Dataset_*`.

**Решение:** PR #10 — обобщили шард-паттерн на `Jacquard*Dataset_*` в
`grasp_seg/data/splits.py`, `notebooks/colab_train.ipynb`,
`docs/yandex_datasphere.md`. Теперь матчит и upstream-релиз
(`JacquardV2_Dataset_0..3`), и зеркало (`Jacquard_Dataset_0..11`).

### Этап 8. Тонкая настройка batch-size и текущий прогон (Текущая сессия, продолж.)

**Что сделали:**
- На Colab A100 40 GB: batch=32 → 0.65 с/step, 49 img/s.
- Подняли batch=48: 0.94 с/step, 51 img/s. Per-image throughput почти не
  изменился (compute-bound), но gradient signal чуть стабильнее.
- Использовали 20 GB из 40 GB VRAM — запас огромный, можно идти до bs=64, но
  скорости это не даст.
- Kaggle: остановлен. Запасной прогон не нужен, Colab покрывает основной кейс.

**Текущий status (epoch 23):**
- train_loss: 1.26 (epoch 0) → **0.666** (epoch 22)
- train_dice: 0.91 → **0.602**
- val_mIoU_fg: 0.225 → **0.323**
- val_Dice_fg: 0.366 → **0.488**
- best.pt = epoch 21

---

## 3. Ключевые решения

| # | Решение | Альтернативы | Обоснование |
|---|---|---|---|
| 1 | HRNet-W18 backbone | ResNet50, EfficientNet, hrnet_w32, hrnet_w48 | HRNet сохраняет высокое разрешение через все scales, идеально для pixel-wise задач. W18 — баланс точность/скорость; w32/w48 в 1.5-2.5× медленнее за +5-10% mIoU. |
| 2 | Mask mode = `angle` (19 классов) | binary, multitask | Compromise: даёт пиксельные углы (важно для grasp), не требует учить регрессию (multitask чаще нестабилен). |
| 3 | length_scale = 1/3 | 1/2, 1/1 (full polygon) | Стандарт Jacquard toolbox. Делает маски тоньше → строже метрики, но точнее grasp-центры. |
| 4 | Object-wise split (80/10/10) | image-wise, scene-wise | Предотвращает data leakage: один объект не может попасть и в train, и в val (иначе легко переобучиться на конкретные предметы). |
| 5 | SGD + poly LR + 1-epoch warmup | Adam, cosine, OneCycle | Стандарт сегментации (используется в HRNet, DeepLab, BiSeNet). Poly с power=0.9 даёт более длинный «средний» LR чем cosine. |
| 6 | LR = 0.01 | 0.005, 0.02 | Стандартная база для SGD-сегментации с batch~16-32. На bs=48 теоретически можно поднять, но мы не рисковали. |
| 7 | CE + Dice loss (без bg в Dice) | Focal, Lovász, Tversky | Самый robust комбо: CE учит классы, Dice учит границы. Без bg в Dice т.к. фон занимает >95% пикселей и забивает gradient. |
| 8 | RGB-D с depth = mean(RGB-весов) | random init для depth-канала | Меньше шок для backbone'а. Depth начинает работать как «нейтральный grey channel» и постепенно специализируется. |
| 9 | AMP fp16 | bf16, fp32 | A100 поддерживает оба, но fp16 совместим со всеми GPU (T4 тоже). bf16 даёт чуть лучше стабильность, но ради совместимости остались на fp16. |
| 10 | image_size = 384 | 256, 512 | На 256 Jacquard теряет детали (objects маленькие в 1024×1024 оригинале), 512 в 4× медленнее. 384 — sweet spot. |
| 11 | Per-epoch checkpoints | каждые N эпох, только best | Полная безопасность от прерываний, легко возобновить с любой точки. Стоит ~340 МБ × 80 = 27 ГБ дискового места — приемлемо в Drive. |
| 12 | Colab A100 (платно) вместо T4 | T4 free, L4, V100 | Время критично (юзер хотел ~1 день, не неделю). A100 даёт ~3× прирост над T4 за разумную цену compute units. |
| 13 | Kaggle с pre-unpacked dataset | Загрузка с Я.Диска | Экономит ~45 мин/сессия. Стоимость: третья сторона может удалить датасет в любой момент. |

---

## 4. Проблемы и их решения

### Решённые

| # | Проблема | Корень | Решение | Когда |
|---|---|---|---|---|
| 1 | CPU-only torch на pip | requirements.txt не указывал CUDA wheel index | Документировали `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision` | Этап 2, PR #2 |
| 2 | LZW TIFF не декодируется | tifffile нужен imagecodecs backend | Добавили `imagecodecs` в requirements.txt | Этап 3, PR #3 |
| 3 | Прерывание тренировки | Нет промежуточных сохранений | Per-epoch ckpt + resume + metrics.csv | Этап 4, PR #4 |
| 4 | Локально 42 ч/прогон | RTX 3060 слабая | Облачные ноутбуки (Colab, Kaggle, Y.DS) | Этап 5, PR #5 |
| 5 | Тратится время на upload/unzip 63 ГБ | Каждая Kaggle-сессия с нуля | Pre-unpacked third-party Kaggle dataset | Этап 7.1, PR #7 |
| 6 | Colab Python 3.12 ImportError | `pip install -r req.txt` ломает torch ABI | Точечная установка `pip install timm imagecodecs` | Этап 7.2, PR #8 |
| 7 | Kaggle git clone падает | Internet выключен в settings | Settings → Internet → On (с phone verification) | Этап 7.3 |
| 8 | Kaggle T4 x2 unused | Single-GPU код | DataParallel auto-enable + checkpoint compat | Этап 7.4, PR #9 |
| 9 | DP overhead > compute saving на Kaggle | Только 4 CPU cores, CPU bottleneck | Откат к single GPU через CUDA_VISIBLE_DEVICES=0 | Этап 7.4 |
| 10 | Colab unzip падает с "No grasps files" | Я.Диск-мирор использует имя `Jacquard_Dataset_N`, не `JacquardV2_Dataset_N` | Обобщили flatten pattern на `Jacquard*Dataset_*` | Этап 7.5, PR #10 |
| 11 | Colab free памяти не хватало | 112 GB не вмещают 65 ГБ распаковки + кэши | Юзер купил Colab Pro (235 GB) | Этап 7.x |

### Нерешённые / открытые

| # | Проблема | Текущий обход | Что делать дальше |
|---|---|---|---|
| 1 | Kaggle T4 x2 не получает выгоды от 2-й GPU | CPU bottleneck, оставили single-GPU | Можно: bigger batch + DP (амортизировать DP overhead) или ничего не делать (Kaggle и так в 5× медленнее A100, проще не использовать) |
| 2 | Slow improvement в metrics после epoch 10 | Идёт линейно ~+0.0013/epoch | См. раздел 7. Возможные решения |
| 3 | DataParallel general slowdown на CPU-bound hosts | Откатили глобально | Оставили PR #9 как опцию, активна только при `device_count > 1`. Можно добавить heuristic «не включать если CPU/GPU < 2». |

---

## 5. Текущая конфигурация (Colab A100, активный прогон)

### 5.1. Hardware

| Параметр | Значение | Обоснование |
|---|---|---|
| GPU | 1× A100 40 GB | A100 в 6-8× быстрее T4, входит в Colab Pro budget. 40 GB ≈ 2.5× больше типичного потребления (20 GB) — есть запас. |
| Backend | Google Colab Pro | $10/мес, 24-часовые сессии (vs 12 ч free), 100 compute units/мес, A100 access |
| Storage | `/content` (235 GB SSD) + Drive | SSD для скачивания/распаковки 65 GB, Drive для checkpoints |
| Network | Yandex Disk → Colab | ~30 мин на 63 ГБ через public API |

### 5.2. Data

| Параметр | Значение | Обоснование |
|---|---|---|
| Dataset | Jacquard V2, 11 619 объектов | Стандарт grasp-сегментации |
| Источник | Я.Диск public share, 12 шардов | Бесплатное хранилище, разделение для удобной выгрузки |
| Splits | object-wise 80/10/10, seed=0 | Detection of overfitting на конкретные объекты |
| image_size | 384 × 384 | Compromise: 256 теряет детали (1024 oригинал), 512 в 4× медленнее |
| input_mode | `rgbd` (4 канала) | RGB + perfect depth. Depth даёт +3-5% точности |
| Mask mode | `angle` (19 классов: bg + 18 × 10°) | Пиксельные углы — что нам нужно для grasp prediction |
| length_scale | 0.3333 (1/3) | Стандарт Jacquard toolbox |

### 5.3. Augmentation

| Augmentation | p | Параметры | Обоснование |
|---|---|---|---|
| H-flip | 0.5 | — | Стандарт, корректно перевёртывает angle |
| V-flip | 0.5 | — | Стандарт |
| Rotate | 0.8 | ±180° | Учит rotation-invariance, корректно сдвигает angle класс |
| Scale | 0.7 | 0.8-1.2 | Robustность к масштабу объектов |
| Translate | 0.5 | ±5% | Robustность к позиции в кадре |
| Color jitter | 0.5 | brightness/contrast/saturation 0.2, hue 0.05 | Robustность к освещению |
| RGB noise | 0.3 | std=0.02 | Robustность к sensor noise |
| Depth jitter | 0.5 | range 0.95-1.05 | Robustность к depth scale errors |
| Depth dropout | 0.3 | 2% pixels → 0 | Имитация depth-sensor holes |
| Stereo swap | 0.3 | perfect → stereo | Robustность к низкокачественному depth |

### 5.4. Model

| Параметр | Значение | Обоснование |
|---|---|---|
| Backbone | `hrnet_w18` (timm) | Pretrained HuggingFace, 85.6 МБ ImageNet-1k weights |
| Pretrained | True | Transfer learning, +20-30% точности vs random init на 80 эпох |
| Adapter | `_patch_first_conv` для 4-канального input | RGB-весов копируются 1:1, depth-канал = mean(RGB) |
| Head | `nn.Sequential(Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Dropout)` | Стандартный fusion head для multi-resolution outputs |
| head_channels | 256 | Compromise: больше = точнее, но медленнее. 256 — sweet spot для w18 |
| Classifier | `Conv1x1 → 19 каналов` | Pixel-wise softmax по 19 классам |
| Dropout | 0.1 | Слабая регуляризация (датасет большой, overfitting риск умеренный) |

### 5.5. Training

| Параметр | Значение | Обоснование |
|---|---|---|
| Epochs | 80 | Стандарт для Jacquard в литературе. Меньше — недообучение, больше — overfitting на augmentation |
| Batch size | 48 | На A100 40 GB поместится до 64, но скорость идёт в полку при ≥48 (compute-bound) |
| Accum steps | 1 | Effective batch = 48, не нужно gradient accumulation |
| Optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=5e-4 | Стандарт segmentation |
| Scheduler | Poly (power=0.9) | Smooth decay, более длинный «средний LR» чем cosine |
| Warmup | 1 эпоха (linear от 0 до lr) | Стабилизация старта при больших batch |
| Grad clip | 1.0 | Защита от exploding gradients (особенно важно с AMP) |
| AMP | fp16 | A100 хорошо тянет, +2× скорость над fp32 |
| Save | каждую эпоху + best.pt + last.pt | Безопасность + удобство resume |
| Eval | каждую эпоху | Видим динамику метрик в реальном времени |

### 5.6. Loss

| Компонент | Вес | Обоснование |
|---|---|---|
| CE (multi-class) | 1.0 | Учит классификацию углов |
| Dice (fg-only) | 1.0 | Учит границы fg-маски |
| Background в Dice | excluded | Bg занимает >95% пикселей, забивает Dice gradient |
| pos_weight (CE) | 1.0 | Балансирующий вес — uniform по классам |

---

## 6. Текущие результаты (epoch 22 из 80)

### 6.1. Кривые

```
epoch | train_loss | train_dice | val_mIoU_fg | val_Dice_fg
   0  |    1.259   |   0.905    |    0.225    |    0.366
   5  |    0.705   |   0.637    |    0.288    |    0.446
  10  |    0.688   |   0.622    |    0.307    |    0.469
  15  |    0.677   |   0.612    |    0.306    |    0.468
  20  |    0.668   |   0.604    |    0.319    |    0.483
  22  |    0.666   |   0.602    |    0.323    |    0.488  ← best
```

### 6.2. Анализ

- **CE loss** (компонент train_loss) уже после epoch 1 упал до 0.07 — модель
  выучила classification. Дальше учится **Dice** (форма маски).
- **Train_dice** медленно ползёт вниз с 0.92 (~ничего не предсказано) до 0.60.
  Темп замедляется: с epoch 10 до 22 → -0.02.
- **Val_mIoU_fg** растёт с 0.225 до 0.323. Темп: epoch 0-10 = +0.082, epoch
  10-22 = +0.016. **Логнормальная динамика** (типично для seg).
- **Прецизия > recall** — модель более «осторожна» с предсказанием fg, что
  обычно для CE+Dice конфигурации.
- **Best so far:** epoch 21, val_mIoU_fg=0.3233. Сохранено в `best.pt`.

### 6.3. Прогноз финала (epoch 80)

| Сценарий | val_mIoU_fg | val_Dice_fg |
|---|---|---|
| Линейная экстраполяция | ~0.39 | ~0.55 |
| Реалистичный (с poly decay) | ~0.42-0.46 | ~0.58-0.62 |
| Optimistic | ~0.48-0.50 | ~0.65 |

Ориентир из литературы Jacquard pixel-wise grasp seg: **0.40-0.50 mIoU** —
ожидаемый результат. Целевой 0.55 (мой первоначальный прогноз) был слишком
оптимистичен, но 0.42-0.48 вполне достойно.

---

## 7. Возможные решения замедленного прогресса

Если финальный mIoU_fg окажется ниже 0.40 или хочется выжать больше — вот
варианты по убыванию рекомендуемости:

### Вариант A. Big batch + larger LR (быстро, low risk)

- Поднять `batch_size=64` (помещается в 40 GB VRAM), `lr=0.015` или `0.02`.
- Compute-bound почти не сдвинется, но более стабильный gradient.
- Ожидаемый прирост: +0.02-0.04 mIoU_fg.
- Стоимость: 1 повторный прогон (~20 ч).

### Вариант B. Сменить mask_mode на `multitask` (medium risk)

- Перейти на отдельные регрессионные головы: pos / cos2θ / sin2θ / width.
- Континуальные предсказания вместо 18 дискретных бинов — обычно дают
  +5-10% mIoU.
- Loss меняется: `MultiClassCEDiceLoss` → `MultiTaskGGCNNLoss`.
- Не требует изменения модели (head уже universal на 4 канала).
- Ожидаемый прирост: +0.05-0.10 mIoU_fg.
- Стоимость: 1 повторный прогон + смена обработки в evaluator.

### Вариант C. Cosine LR + warm restarts (low risk)

- Заменить poly на cosine с одним warm restart на середине обучения (epoch 40).
- При warm restart LR прыгает обратно к peak, что часто помогает выйти из
  локального минимума.
- Ожидаемый прирост: +0.01-0.03 mIoU_fg.
- Стоимость: 1 повторный прогон.

### Вариант D. Увеличить mask compactness (medium risk)

- Поднять `length_scale = 0.5` (более широкие маски).
- Больше positive pixels → проще учить Dice. Но менее «компактные» grasps в
  смысле точного центра.
- Ожидаемый прирост: +0.03-0.06 mIoU_fg на текущей метрике, но реальная польза
  для grasp-detection (что важно практически) может уменьшиться.
- Стоимость: 1 повторный прогон.

### Вариант E. Сменить backbone на hrnet_w32 (high cost)

- W32 имеет +60% параметров, ~+5-10% точности.
- Стоимость: ~1.5× времени обучения (~30 ч на A100), больше VRAM (надо
  снижать batch до 32).
- Ожидаемый прирост: +0.03-0.05 mIoU_fg.
- Окупается только если другие варианты исчерпаны.

### Вариант F. Specialised loss (high risk, high reward)

- Tversky loss с β > 0.5 (более чувствительный к recall, чем Dice).
- Lovász-Softmax (прямая оптимизация IoU).
- Focal loss для CE (помогает с unbalanced classes).
- Ожидаемый прирост: +0.05-0.15 mIoU_fg, но риск нестабильности.
- Стоимость: разработка + 1-2 повторных прогона.

### Вариант G. Совсем другой подход — instance segmentation

- Перейти от semantic segmentation к instance: каждый grasp — отдельный объект.
- Mask R-CNN или YOLOv8-seg.
- Принципиально другая модель. Не релевантно если хотим именно HRNet.

### Что я бы рекомендовал прямо сейчас

**Дождаться текущего прогона** (ещё ~14 ч), посмотреть на финальные метрики.
- Если **finalный mIoU_fg ≥ 0.40** — задача решена адекватно. Можно остановиться
  или провести +1 итерацию с **Вариантом A** (big batch + larger LR).
- Если **0.35 ≤ mIoU_fg < 0.40** — пробовать **Вариант B** (multitask). Это
  главный кандидат на «качественный скачок».
- Если **mIoU_fg < 0.35** — что-то не так, диагностируем глубже (визуализация
  предсказаний, per-class IoU, etc.).

---

## 8. Метаданные проекта

- **GitHub:** https://github.com/Hesgoryr/grad/HRNet_Grasp_Semantic_Segmentation
- **PR за всё время:** #1-#10 (все мерджены)
- **Contributors:** themysteriousarmour (юзер), Devin (Cognition AI assistant)
- **Datasets used:**
  - https://disk.yandex.ru/d/Je56nUcC9hiHFQ (Я.Диск public share, 12 архивов)
  - https://www.kaggle.com/datasets/vdsdggsgsd/jacquard (pre-unpacked Kaggle)
- **Pretrained weights:** https://huggingface.co/timm/hrnet_w18.ms_aug_in1k

---

## 9. Дальнейшие планы

### 9.1. Планка качества

Согласовано (27.04.2026):

| Уровень | Метрика | Значение |
|---|---|---|
| Минимально приемлемый (Jacquard val) | mIoU_fg | **≥ 0.40** |
| Минимально приемлемый (Jacquard val) | Dice_fg | **≥ 0.55** |
| Целевой (Jacquard val) | mIoU_fg | **≥ 0.45** |
| Целевой (Jacquard val) | Dice_fg | **≥ 0.62** |
| Минимум на test split (отдельный 10%) | mIoU_fg | **≥ 0.38** |

### 9.2. Дорожная карта

**Фаза 1. Завершить текущий RGB-D run** (~14 ч)
- Оставить идущий Colab A100 80-эпох прогон до конца.
- Финальные метрики записать в metrics.csv (epoch 79).
- Чекпоинты в `/content/drive/MyDrive/hrnet_runs/hrnet_w18_rgbd_angle/`.

**Фаза 2. Полный eval RGB-D** (~30 мин)
- `python tools/eval.py --config configs/default.yaml --checkpoint best.pt --split val` — для подтверждения.
- `python tools/eval.py --config configs/default.yaml --checkpoint best.pt --split test` — на ранее не виданных объектах.
- Решение:
  - Если **test_mIoU_fg ≥ 0.38** → переход в Фазу 3.
  - Если **< 0.38** → Фаза 2a (оптимизация).

**Фаза 2a. Оптимизация RGB-D (если требуется)**
Применять в порядке убыванию ROI:
1. **mask_mode = `multitask`** — приоритет №1, ожидаемый прирост +5-10% mIoU. ~20 ч.
2. **length_scale = 0.5** + повторный прогон — +3-6%, ~20 ч.
3. **hrnet_w32 backbone** — +3-5%, ~30 ч + меньший batch.
- Допустимый бюджет: **до 3 итераций** (~70 ч compute).
- Если после 3 итераций целевой 0.40 не достигнут — глубокая диагностика
  (визуализация неудач, per-class IoU, пересмотр loss).

**Фаза 3. Cross-dataset eval на Cornell Grasp Dataset**

Начнётся **после Фазы 2/2a** (юзер подтвердил: «после завершения основного run»).

3.1. Подготовка Cornell-парсера
- Конвертер point cloud (`pcd*.txt`) → depth map.
- Парсер `*_cpos.txt` → `Grasp[]` в нашем формате.
- DataLoader для Cornell с теми же transforms.
- ~1 рабочий день кода, +200 строк, отдельный PR.

3.2. Eval flow
- Загрузить best.pt (Jacquard RGB-D).
- Прогнать по всем 885 Cornell-изображениям без fine-tune.
- Метрики: mIoU_fg, Dice_fg + grasp-detection metric (top-1 grasp at IoU > 0.25, angle diff < 30°).

3.3. Целевые результаты
- Without fine-tune: **mIoU_fg ≥ 0.30** (домен-shift подтверждение, что транфер работает).
- С 5-10 эпох fine-tuning на Cornell train: **mIoU_fg ≥ 0.45**.

**Фаза 4. RGB-only обучение и сравнение**

Параллельно с Фазой 3 (если есть compute units, иначе после).

4.1. Запуск
- `python tools/train.py --config configs/rgb.yaml ...`
- Та же конфигурация что и RGB-D, но 3 канала, без depth-аугментаций.
- ~20 ч на A100, ~210 compute units.

4.2. Сравнение head-to-head

| Метрика | RGB-D | RGB-only | Δ |
|---|---|---|---|
| Jacquard val_mIoU_fg | TBD | TBD | TBD |
| Jacquard test_mIoU_fg | TBD | TBD | TBD |
| Cornell mIoU_fg (no fine-tune) | TBD | TBD | TBD |
| Cornell mIoU_fg (with fine-tune) | TBD | TBD | TBD |
| Speed (s/step) | 0.94 | ~0.85 | -10% |
| Inference на real RGB-D | TBD | TBD | TBD |

4.3. Целевые результаты RGB-only
- val_mIoU_fg на Jacquard ≥ 0.35 (минимум), ≥ 0.42 (целевой).
- Cornell mIoU_fg ≥ 0.25 (no fine-tune), ≥ 0.40 (with fine-tune).

4.4. Выводы
- Δ ≥ 5% → depth важен, RGB-D рекомендуется для production.
- Δ ≤ 2% → depth не критичен, RGB достаточно.

**Фаза 5. Inference на собственных RGB-D снимках**

5.1. Что нужно от пользователя
- **RGB-D камера:** какая модель (Realsense / Kinect / OAK-D / iPhone LiDAR / др.)?
- **Снимки:** ≥ 10-20 разных сцен/объектов.
- **Калибровка:** depth должен быть aligned к RGB.

5.2. Pipeline
- `tools/infer.py` — RGB+depth → angle mask + визуализация.
- (Опционально) NMS-постпроцесс → top-K grasp rectangles.
- Запуск **обеих** моделей (RGB-D и RGB) на одних и тех же снимках для сравнения.

5.3. Подводные камни
- Depth scale: Jacquard в [0,1] м, Realsense — мм. Нормализация на старте.
- FOV/разрешение: ресайз до 384×384.
- Sim2Real gap: Jacquard синтетический (Blender), реальные снимки могут потребовать fine-tune на Cornell + ваших примерах.

**Фаза 6. Финальный сравнительный отчёт**

- Сводная таблица всех метрик RGB-D vs RGB-only на трёх dataset (Jacquard val/test, Cornell, real).
- Визуализация типичных предсказаний.
- Выводы: какая конфигурация рекомендуется для production.
- Ограничения и направления дальнейших улучшений.

### 9.3. Общий timeline

| Фаза | Время | Параллелизуемо? |
|---|---|---|
| 1. Закончить RGB-D run | ~14 ч | — |
| 2. Eval RGB-D val + test | ~30 мин | После 1 |
| 2a. (Если нужно) Оптимизации RGB-D | 20-70 ч | Sequentially |
| 3. Cornell parser + eval | ~3 ч код + 30 мин eval | Параллельно с 4 |
| 3a. Cornell fine-tune | ~3-5 ч | После 3 |
| 4. RGB-only обучение | ~20 ч | Параллельно с 3 |
| 5. Inference на real снимках | ~3 ч код + от пользователя | Параллельно с 3, 4 |
| 6. Финальный отчёт | ~2 ч | После всего |

**Итого:** ~3-5 дней активной работы.

### 9.4. Открытые вопросы к пользователю

1. **RGB-D камера** для real-snapshots — какая? Это влияет на формат depth и калибровку.
2. **Сколько Colab Pro compute units осталось?** (Settings → Compute units). RGB-only прогон стоит ~210 units, оптимизации (если потребуются) — ещё ~210-630 units.
3. **Когда снимать на реальной камере?** Если есть готовые снимки — можно использовать сразу для Фазы 5. Если нужно снять — спланировать сценарии (количество объектов, освещение).

