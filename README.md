# HRNet-W18 для grasp-сегментации (Jacquard V2 → Cornell → RealSense D435 → Fairino FR5)

End-to-end пайплайн **pixel-wise grasp detection** на основе **HRNet-W18** (timm) с разделяемым RGB/Depth/RGB-D стволом и тремя сменными формулировками маски (`binary` / `angle` / `multitask`). Базовое обучение — на **[Jacquard V2](https://github.com/lqh12345/Jacquard_V2)**; кросс-доменная валидация — на **[Cornell Grasp Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)**; целевое применение — захват реальных объектов с **Intel RealSense D435** манипулятором **Fairino FR5**.

> **Текущий статус (май 2026):**
> Этапы 1–6 (обучение, кросс-домен, интерпретируемость, локальный 150-эпочный run на RTX 4060, инструменты захвата D435) — выполнены и в `main`.
> Этап 7 (разметка собственного датасета и fine-tune под кухню D435) и Этап 8 (интеграция с FR5) — в проектировании; см. [§ 16. Дорожная карта](#16-дорожная-карта).

---

## Содержание

1. [Этапы проекта (обзор)](#1-этапы-проекта-обзор)
2. [Финальные метрики](#2-финальные-метрики)
3. [Структура репозитория](#3-структура-репозитория)
4. [Выберите путь запуска](#4-выберите-путь-запуска)
5. [Установка](#5-установка)
6. [Подготовка датасета](#6-подготовка-датасета)
7. [Обучение](#7-обучение)
8. [Возобновление и формат чекпоинтов](#8-возобновление-и-формат-чекпоинтов)
9. [Метрики и логи](#9-метрики-и-логи)
10. [Визуализация для отчёта](#10-визуализация-для-отчёта)
11. [Кросс-домен: Cornell](#11-кросс-домен-cornell)
12. [Интерпретируемость: heatmap + Grad-CAM](#12-интерпретируемость-heatmap--grad-cam)
13. [RealSense D435: захват своего датасета](#13-realsense-d435-захват-своего-датасета)
14. [Архитектурные заметки](#14-архитектурные-заметки)
15. [Конфиги — справочник](#15-конфиги--справочник)
16. [Дорожная карта](#16-дорожная-карта)
17. [Заметки для новых участников и ИИ-агентов](#17-заметки-для-новых-участников-и-ии-агентов)
18. [Ссылки](#18-ссылки)

---

## 1. Этапы проекта (обзор)

| # | Этап | Статус | Артефакты |
|---|---|---|---|
| **1** | Каркас и три mask-режима                                | done    | PR #1 — `configs/`, `grasp_seg/{data,models,losses,engine}`, `tools/{prepare_split,train,eval,visualize,smoke_test}` |
| **2** | Обучение трёх моделей на Jacquard V2                    | done    | Colab Pro A100, 30 эпох × 3 (rgb_multitask / rgbd_multitask / rgbd_angle). Лучшая: **rgbd_multitask, val_mIoU_fg = 0.6788**. |
| **3** | Кросс-доменная оценка на Cornell                        | done    | PR #2 / #4 / #5 / #6 / #7 — loader, pad+resize в model-space, top-1/K accuracy по стандарту Jacquard (IoU>0.25 ∧ \|Δθ\|<30°), side-by-side, каталог failure-cases. |
| **4** | Интерпретируемость (что видит модель)                   | done    | PR #8 / #10 — per-head heatmap (`pos`/`cos2θ`/`sin2θ`/`width` или `fg_conf`+argmax) + Grad-CAM по последнему общему слою HRNet. |
| **5** | Финальный отчёт                                         | done    | PR #9 — [docs/progress_report_final.md](docs/progress_report_final.md). |
| **6a**| Инструменты RealSense D435 (захват в Cornell-формат)    | done    | PR #11 — [`tools/realsense_preview.py`](tools/realsense_preview.py), [`tools/realsense_capture.py`](tools/realsense_capture.py), [docs/realsense_setup.md](docs/realsense_setup.md). |
| **6b**| Локальный 150-эпочный RGB-D multitask на RTX 4060       | done    | PR #12 — [`notebooks/local_train_rgbd_multitask.ipynb`](notebooks/local_train_rgbd_multitask.ipynb), [`configs/multitask_local512.yaml`](configs/multitask_local512.yaml) (512×512, batch=2, accum=8, save_every_n_epochs=5, `iter_log.jsonl`), [docs/local_training_windows.md](docs/local_training_windows.md), [docs/training_metrics.md](docs/training_metrics.md). |
| **7** | Свой датасет на D435 + fine-tune (Cornell-style)        | design  | ~100–200 сцен, ~20–30 объектов. UI разметки (SAM-предложения + ручная правка), seed-разметка → промежуточный fine-tune → разметка остальных через предсказания модели (active learning). |
| **8** | Интеграция с манипулятором Fairino FR5                  | planned | Eye-to-hand калибровка (ChArUco), pixel+depth → base-frame, обратная кинематика FR5, тестирование захвата на ~20 объектах. |

История PR-ов: [docs/progress_report_final.md § 8.1](docs/progress_report_final.md#81-pr-история-текущей-стадии).

---

## 2. Финальные метрики

Все три модели обучены на одном **object-wise** split'е (seed=0; train ≈ 41 000 сцен, val ≈ 5 200, test ≈ 5 200, image_size=384). Оптимизация: SGD lr=0.01, momentum=0.9, weight_decay=5e-4, poly schedule (power=0.9), warmup 1 эпоха, grad clip 1.0, AMP fp16.

| Модель                         | Режим маски     | Каналы | Эпоха | val mIoU_fg | val Dice_fg | val ang_MAE | s/step |
|-------------------------------|----------------|-------:|------:|------------:|------------:|------------:|-------:|
| `hrnet_w18_rgb_multitask`     | multitask       |     3 |    27 |  **0.6417** |  **0.7817** |    15.86°   |   0.31 |
| `hrnet_w18_rgbd_multitask`    | multitask       |     4 |    26 |  **0.6788** |  **0.8086** | **14.97°**  |   0.91 |
| `hrnet_w18_rgbd_angle`        | angle (19 cls)  |     4 |    25 |    0.3282   |    0.4937   |       —     |   —    |

**Что отсюда видно:**

- Multitask GG-CNN формулировка даёт **+34 п.п. mIoU_fg** против классической классификации углов в 19 классов.
- Депт даёт **+3.7 п.п. mIoU_fg** (RGB-only → RGB-D) при цене ×3 по времени шага.
- Best run — `rgbd_multitask`, **в 1.5× выше** ранее обозначенного таргета 0.45.

Подробно — [docs/progress_report_final.md § 2](docs/progress_report_final.md#2-финальные-метрики-обучения).

---

## 3. Структура репозитория

```
configs/                                  # YAML-конфиги (все гиперпараметры в одном месте)
  default.yaml                            # 19-class angle, 384px, RTX 3060
  rgb.yaml                                # RGB-only, 19-class angle
  multitask.yaml                          # RGB-D multitask (GG-CNN), 384px
  multitask_rgb.yaml                      # RGB multitask
  multitask_local512.yaml                 # RGB-D multitask, 512px, batch=2 accum=8, 150 эпох
                                          # (для локального run на RTX 4060)

grasp_seg/
  data/
    grasp_rect.py                         # парсинг *_grasps.txt + растеризация маски
                                          #   (binary / angle / multitask, compact-polygon 1/3 length)
    transforms.py                         # синхронные RGB-D + grasp-list аугментации
    splits.py                             # object-wise train/val/test (по object_id, не по сцене!)
    jacquard_v2.py                        # PyTorch Dataset (RGB / Depth / RGB-D)
    cornell.py                            # PyTorch loader для Cornell (плоская и 01..10/-раскладки)

  models/hrnet.py                         # HRNet-W18/Small-v2 (через timm) + 4-канальный stem
                                          # + одна голова на режим (binary / angle / multitask)

  losses/
    seg_losses.py                         # BCE+Dice (binary), CE+Dice (angle)
    multitask_loss.py                     # MultiTaskGraspLoss = BCE+Dice(pos) + MSE(cos/sin/width)

  engine/
    trainer.py                            # Trainer: AMP fp16, grad-accum, per-epoch и
                                          #   per-N-epochs чекпоинты, opt.+sched.+scaler resume,
                                          #   per-step JSONL лог, metrics.csv, target_metric
    evaluator.py                          # evaluate_binary / evaluate_angle / evaluate_multitask

  viz/                                    # 11 модулей визуализации (см. § 10)
    palette.py / draw.py                  # цветовая палитра, отрисовка grasp-rect'ов
    dataset_viz.py                        # секции 1.x ноутбука (датасет, маски, аугментации)
    metrics_viz.py                        # секция 2 (кривые обучения из metrics.csv)
    epoch_evolution.py                    # секция 3 (эволюция сцены × эпоха)
    eval_viz.py                           # секция 4 (best-epoch: GT / pred / decoded / errors)
    compare_viz.py                        # секция 5 (side-by-side моделей)
    extra_viz.py                          # секция 6 (IoU vs angle, depth contribution, failures)
    cornell_eval.py                       # секция 5.3 (количественная Cornell top-1 acc)
    heatmap_viz.py                        # секция 8 (per-head heatmap + Grad-CAM)
    decoder.py                            # masks → oriented grasp rects (peak-find + NMS)
    inference.py                          # обёртка вокруг модели для всех viz-функций

  utils/                                  # logger, meters, AMP-helpers

tools/                                    # CLI-входы
  prepare_split.py                        # построение object-wise сплита
  train.py                                # запуск обучения (используется и из ноутбуков)
  eval.py                                 # оценка на split'е (mIoU/Dice/F1)
  visualize.py                            # CLI-генератор всех PNG для отчёта
  realsense_preview.py                    # ЭТАП 6a: live RGB+depth, save по 's'
  realsense_capture.py                    # ЭТАП 6a: структурированный захват в Cornell-формат

notebooks/                                # «оркестраторы» — каждый ноутбук = одна платформа
  colab_train.ipynb                       # Google Colab (Pro A100 / Free T4) — обучение
  kaggle_train.ipynb                      # Kaggle (T4 ×2 / P100, бесплатно 30 ч/нед)
  local_train_rgbd_multitask.ipynb        # ЭТАП 6b: Windows + RTX 4060, 150 эпох × 512px
  visualize.ipynb                         # отчётные графики для всех платформ

scripts/smoke_test.py                     # 30-секундный smoke-тест на 8 синт. сценах
                                          #   (binary / angle / multitask + resume)

docs/
  progress_report.md                      # отчёт версии 1 (после обучения angle-режима)
  progress_report_v2.md                   # отчёт версии 2 (после переключения на multitask)
  progress_report_final.md                # финальный отчёт (этап 5) — главный источник цифр
  multitask_phase2.md                     # обоснование перехода на multitask + new metrics
  visualization.md                        # описание viz-модуля и notebooks/visualize.ipynb
  local_setup_windows.md                  # ЭТАП 3+: установка под Windows + RTX 3060 (viz)
  local_training_windows.md               # ЭТАП 6b: установка под Windows + RTX 4060 (train)
  training_metrics.md                     # расшифровка ВСЕХ полей train-лога (для пользователя)
  realsense_setup.md                      # ЭТАП 6a: установка pyrealsense2 + использование скриптов
  yandex_datasphere.md                    # как запустить на Yandex DataSphere (рублёвая альтернатива)
```

Не в git (см. `.gitignore`): `datasets/`, `train_results/`, `outputs/`, `*.zip`,
`pyrealsense2-*.whl`, `pcdNNNN*.png/tiff` (захваты с D435).

---

## 4. Выберите путь запуска

> Все четыре пути приходят к одному и тому же чекпоинту `best.pth` + `metrics.csv` + `resolved_config.yaml` — выбор зависит только от того, какое железо доступно.

| Путь | Где | Когда удобно | Стартовая точка |
|---|---|---|---|
| **A. Google Colab** | браузер, T4 (free) или A100 (Pro) | нет GPU локально; хочется попробовать без установки | [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) |
| **B. Kaggle Notebooks** | браузер, T4 ×2 / P100 (free, 30 ч/нед) | нет Pro, хочется бесплатно | [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) |
| **C. Локально, RTX 3060 6 GB** | Windows / Linux | визуализация и небольшие эксперименты | [`docs/local_setup_windows.md`](docs/local_setup_windows.md) |
| **D. Локально, RTX 4060 8 GB** | Windows + VS Code | полноценное обучение 150 эпох × 512px | [`notebooks/local_train_rgbd_multitask.ipynb`](notebooks/local_train_rgbd_multitask.ipynb) + [`docs/local_training_windows.md`](docs/local_training_windows.md) |
| **E. Yandex DataSphere** | браузер, T4/V100/A100, RUB | нужна оплата в рублях / нативный доступ к Я.Диску | [`docs/yandex_datasphere.md`](docs/yandex_datasphere.md) |

Архивы Jacquard V2 (12 zip-файлов) лежат на публичном Я.Диске:
**https://disk.yandex.ru/d/Je56nUcC9hiHFQ** — все ноутбуки умеют тянуть их оттуда автоматически (Colab/Kaggle через `gdown`-стиль, local через REST-API Я.Диска параллельно в 4 потока).

---

## 5. Установка

### 5.1. Универсально (Linux / WSL / macOS-CPU)

```bash
git clone https://github.com/chelovekrazumnii2-png/HRNet_Grasp_Semantic_Segmentation.git
cd HRNet_Grasp_Semantic_Segmentation

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1

pip install --upgrade pip

# 1) torch с правильным CUDA-индексом (узнайте свою CUDA через nvidia-smi).
#    Просто `pip install torch` приедет CPU-only — это самая частая ошибка.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6 (RTX 3060)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4 (RTX 4060)

# 2) Остальное — обычным pip
pip install -r requirements.txt

# 3) Проверка GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# 4) 30-секундный smoke-тест без датасета: все три mask-режима, resume, checkpoints
python scripts/smoke_test.py
```

Ожидаемый вывод последней команды — `ALL MODES OK`.

### 5.2. Windows + RTX 3060 (визуализация / лёгкие эксперименты)

Полная инструкция (драйверы → venv → kernel selection → типовые сбои) — в [docs/local_setup_windows.md](docs/local_setup_windows.md). Краткий рецепт:

```powershell
cd D:\
git clone https://github.com/chelovekrazumnii2-png/HRNet_Grasp_Semantic_Segmentation.git
cd HRNet_Grasp_Semantic_Segmentation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name grasp_viz --display-name "Python 3.10 (HRNet viz)"
python scripts\smoke_test.py
```

### 5.3. Windows + RTX 4060 (полноценное обучение)

Подробно — в [docs/local_training_windows.md](docs/local_training_windows.md). Целевой ноутбук — [notebooks/local_train_rgbd_multitask.ipynb](notebooks/local_train_rgbd_multitask.ipynb), он содержит 9 ячеек: проверка GPU → проверка зависимостей → скачивание датасета с Я.Диска → распаковка → split → smoke-test с замером VRAM и ETA → запуск 150-эпочного обучения → графики из `metrics.csv` и `iter_log.jsonl`.

Python 3.11 рекомендован (3.12 имеет проблемы с wheel'ами `pyrealsense2`).

---

## 6. Подготовка датасета

### 6.1. Jacquard V2

1. **Скачать архивы** (12 zip, ~65 GB всего):
   - С Я.Диска **https://disk.yandex.ru/d/Je56nUcC9hiHFQ** — оригинал, на котором обучались все опубликованные чекпоинты. Все наши ноутбуки умеют тянуть автоматически.
   - Альтернативно: 4 архива `JacquardV2_Dataset_{0..3}.zip` с [OneDrive из репозитория lqh12345/Jacquard_V2](https://github.com/lqh12345/Jacquard_V2).

2. **Распаковать** в плоскую структуру:

   ```
   /data/JacquardV2/
       <object_id>/<idx>_<object_id>_RGB.png
       <object_id>/<idx>_<object_id>_perfect_depth.tiff
       <object_id>/<idx>_<object_id>_stereo_depth.tiff
       <object_id>/<idx>_<object_id>_mask.png
       <object_id>/<idx>_<object_id>_grasps.txt
       ...
   ```

   Если архивы распаковались с промежуточным уровнем `JacquardV2_Dataset_<N>/<object_id>/...` — `notebooks/local_train_rgbd_multitask.ipynb` автоматически «выпрямляет» структуру в ячейке 4.

3. **Object-wise split** (80/10/10, делится по объектам, а не сценам — иначе данные одного объекта окажутся и в train, и в val):

   ```bash
   python tools/prepare_split.py \
       --root /data/JacquardV2 \
       --out splits/jacquard_v2.json \
       --val-frac 0.1 --test-frac 0.1 --seed 0
   ```

   Файл сохраняется как JSON со списками полных путей к `*_grasps.txt`. Один раз сгенерировали — переиспользуем во всех runs, чтобы метрики были сравнимы.

### 6.2. Cornell (опционально, для кросс-доменной оценки)

[Cornell Grasp Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) — 885 сцен с прямоугольной grasp-разметкой. Поддерживаются обе стандартные раскладки:

- **Плоская:** все `pcdNNNN{r.png,d.tiff,cpos.txt,cneg.txt}` в одной папке.
- **Оригинальная:** 10 подпапок `01/`…`10/` (`backgrounds/` игнорируется).

Loader находится в `grasp_seg/data/cornell.py` и автоматически детектит раскладку.

### 6.3. RealSense D435 (этап 6a)

Свой датасет можно собрать сразу в Cornell-формате — потом он подхватывается тем же loader'ом. См. [§ 13](#13-realsense-d435-захват-своего-датасета).

---

## 7. Обучение

### 7.1. Базовый вызов (любой конфиг)

```bash
python tools/train.py \
    --config configs/multitask.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    trainer.save_dir=train_results/hrnet_w18_rgbd_multitask
```

CLI допускает **точечные переопределения** любого ключа конфига через `dot.path=value` после `--config` — удобно не плодить YAML'ы под мелкие эксперименты.

### 7.2. Готовые конфиги

| Конфиг | Режим | Каналы | image_size | batch×accum | Эпох | Заметки |
|---|---|--:|--:|---|--:|---|
| `configs/default.yaml`           | `angle`     | 4 (RGB-D) | 384 | 2×4 (=8)   |  80 | классический 19-class baseline |
| `configs/rgb.yaml`               | `angle`     | 3 (RGB)   | 384 | 2×4        |  80 | RGB-only ablation              |
| `configs/multitask.yaml`         | `multitask` | 4 (RGB-D) | 384 | 2×4        |  80 | GG-CNN-стиль, лучшая модель    |
| `configs/multitask_rgb.yaml`     | `multitask` | 3 (RGB)   | 384 | 2×4        |  80 | RGB multitask ablation         |
| `configs/multitask_local512.yaml`| `multitask` | 4 (RGB-D) | **512** | **2×8** (=16) | **150** | локальный run на RTX 4060, чекпоинт раз в 5 эпох, JSONL-лог |

### 7.3. Переключение режима и каналов на лету

```bash
# бинарная Q-маска (graspable / no)
python tools/train.py --config configs/default.yaml dataset.mask_mode=binary

# multi-task GG-CNN-стиль: pos + cos2θ + sin2θ + width
python tools/train.py --config configs/default.yaml dataset.mask_mode=multitask

# только RGB / только Depth
python tools/train.py --config configs/default.yaml dataset.input_mode=rgb
python tools/train.py --config configs/default.yaml dataset.input_mode=depth
```

### 7.4. Что делать, если OOM

В порядке простоты:

1. `dataset.image_size=320` — экономит ~40% VRAM.
2. `model.backbone=hrnet_w18_small_v2` — лёгкая версия HRNet.
3. `trainer.batch_size=1 trainer.accum_steps=8` — тот же эффективный batch.
4. На Windows: закрыть Chrome, Discord, OBS — они едят VRAM.

### 7.5. 150-эпочный локальный run (этап 6b)

См. целевой ноутбук [notebooks/local_train_rgbd_multitask.ipynb](notebooks/local_train_rgbd_multitask.ipynb) и [docs/local_training_windows.md](docs/local_training_windows.md). Особенности:

- **Чекпоинт раз в 5 эпох** (`trainer.save_every_n_epochs=5`) вместо каждой — экономит диск (30 файлов × 75 МБ vs 150 × 75 МБ).
- **`best.pth` обновляется каждую эпоху** независимо от чекпоинт-политики, по метрике `trainer.target_metric` (по умолчанию `miou_fg`).
- **Per-step JSONL-лог** (`trainer.iter_log_path: iter_log.jsonl`) — одна строка на оптимизаторный шаг, **усреднённая** по всем micro-batch'ам внутри `accum_steps`. Это даёт суб-эпочную кривую loss'а с ~50k+ точек на 150 эпох.
- **Resume через `last.pth`** встроен — kernel-restart или прерывание не убьют прогресс.

---

## 8. Возобновление и формат чекпоинтов

```bash
python tools/train.py --config configs/multitask.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    trainer.save_dir=train_results/hrnet_w18_rgbd_multitask \
    --resume train_results/hrnet_w18_rgbd_multitask/last.pth
```

`--resume` загружает **model + optimizer + scheduler + scaler + epoch counter**, обучение продолжится со следующей эпохи. Для дообучения «с чистым оптимизатором» добавьте `--resume-model-only` — тогда восстановятся только веса модели (например, для fine-tune на новом датасете).

**Что появляется в `save_dir`:**

```
train_results/hrnet_w18_rgbd_multitask/
  best.pth                    # обновляется per-epoch по target_metric (val_miou_fg)
  last.pth                    # обновляется per-epoch (для --resume)
  epoch_004.pth, epoch_009.pth, ...  # каждые save_every_n_epochs (для эволюции по эпохам)
  metrics.csv                 # построчно: epoch, lr, train.*, val.*, timing.*, gpu.*
  iter_log.jsonl              # одна JSON-строка на оптимизаторный шаг (если включён)
  resolved_config.yaml        # фактический конфиг запуска (со всеми CLI-overrides)
  train.log                   # human-readable консоль через logger
```

---

## 9. Метрики и логи

Расшифровка **каждого** поля train-лога (`loss / pos_bce / pos_dice / cos / sin / width / data_time_s / compute_time_s / step_time_s / dataload_fraction / gpu_mem_alloc_gb / gpu_util_pct / gpu_mem_peak_gb / miou / miou_fg / dice / dice_fg / precision_fg / recall_fg / miou_fg_ang / dice_fg_ang / cos_mse / sin_mse / ang_mae_deg`) — в [docs/training_metrics.md](docs/training_metrics.md). Документ объясняет диапазоны, что считается «хорошим знаком», и какие проверки делать на ходу.

Краткое сравнение метрик по режимам:

| Режим маски | best.pth выбирается по | Apples-to-apples сравнение с angle |
|---|---|---|
| `binary`     | `val_miou_fg` (foreground vs background) | — (другая задача)            |
| `angle`      | `val_miou_fg` (19-class)                 | то же самое, эталон          |
| `multitask`  | `val_miou_fg` (binary — pos vs bg)       | **`val_miou_fg_ang`** (cos/sin → 18 bins, тот же ConfusionMeter что в angle) |

Подробное обоснование multitask-метрик — [docs/multitask_phase2.md](docs/multitask_phase2.md).

---

## 10. Визуализация для отчёта

Всё собрано в едином ноутбуке [notebooks/visualize.ipynb](notebooks/visualize.ipynb) (русские подписи, 8 секций). Тот же набор картинок можно сгенерировать пакетно через CLI:

```bash
python tools/visualize.py \
    --jacquard-root /path/to/JacquardV2 \
    --splits-path splits/jacquard_v2.json \
    --cornell-root /path/to/cornell \
    --run multitask_rgb=/runs/multitask_rgb \
    --run multitask_rgbd=/runs/multitask_rgbd \
    --run angle_rgbd=/runs/angle_rgbd \
    --out outputs/viz/report \
    --dpi 140
```

Секции (см. [docs/visualization.md](docs/visualization.md) для деталей):

| Секция | Модуль | Содержание |
|---|---|---|
| 1     | `dataset_viz`     | Сцена + GT-rects, resize-pipeline, мaska-modes (binary/angle/multitask), compact-polygon vs full rect, шаги аугментаций, Cornell сцена |
| 2     | `metrics_viz`     | Кривые обучения из `metrics.csv` (loss/IoU/Dice/lr/cos/sin/width/ang_mae/GPU mem/util/тайминг), сравнение моделей |
| 3     | `epoch_evolution` | Сетка сцена × эпоха для одной модели на сохранённых чекпоинтах |
| 4     | `eval_viz`        | Best-epoch: вход / GT / pred / GT-rect vs decoded / error map + per-bin IoU bar-chart |
| 5.1   | `compare_viz`     | Side-by-side всех моделей на Jacquard test |
| 5.2   | `compare_viz`     | Side-by-side всех моделей на Cornell |
| 5.3   | `cornell_eval`    | Количественная top-1 / top-K accuracy на Cornell (см. § 11) |
| 5.4   | `extra_viz`       | Каталог Cornell-failure |
| 6     | `extra_viz`       | IoU vs угол grasp'а; depth contribution heatmap (RGB-D − RGB); Jacquard failure catalog |
| 8.1   | `heatmap_viz`     | Per-head heatmap (pos / cos2θ / sin2θ / width или fg_conf + argmax bin) |
| 8.2   | `heatmap_viz`     | Grad-CAM по последнему общему слою HRNet (см. § 12) |

Декодер `masks → grasp rectangles` (peak-finding + NMS) — `grasp_seg/viz/decoder.py`; стандарт оценки Jacquard/Cornell — `IoU > 0.25 ∧ |Δθ| < 30°` (`decoder.jacquard_match`).

---

## 11. Кросс-домен: Cornell

Цель — проверить sim2real generalization модели, обученной на Jacquard V2, **без** дообучения на Cornell.

**Что есть:**

1. **Loader** `grasp_seg/data/cornell.py` — обе раскладки (плоская и `01..10/`), RGB+depth, robust 1/99-percentile нормирование depth (то же, что в Jacquard).
2. **Перевод координат в model-frame**: `viz/cornell_eval.py:_scene_to_model_space` — pad-to-square + uniform resize 480×640 → 384×384, сохраняющий аспект 4:3. Углы grasp'ов остаются корректными.
3. **Количественная оценка**: `viz.cornell_eval.evaluate_cornell` + `summarize_cornell` — top-1/top-K accuracy, mean IoU, mean angular error по стандартному критерию Jacquard.
4. **Качественные**: `compare_viz.figure_compare_models_cornell` (side-by-side всех моделей), `extra_viz.figure_cornell_failures` (худшие N сцен с GT + top-3 предсказаниями).

Пиксельный mIoU на Cornell **не считается** — Cornell даёт rectangle-граспы, и его pixel-mask растеризация требует допущений; стандарт Cornell — top-1 acc по rect-критерию.

Fine-tune на Cornell-train оставлен как отдельный шаг (см. § 16).

---

## 12. Интерпретируемость: heatmap + Grad-CAM

Два независимых взгляда, обе из `grasp_seg/viz/heatmap_viz.py`:

**Per-head heatmap** (что модель **выдаёт**):

- `multitask` (2×3): `RGB | depth | pos | cos2θ | sin2θ | width`. У каждой панели свой colorbar; `pos`/`width` ∈ [0, 1] (sigmoid), `cos2θ`/`sin2θ` ∈ [−1, 1] (палитра RdBu).
- `angle` (1×4): `RGB | depth | fg_conf (1−p_bg) | argmax-bin (раскрашенный палитрой angle_cmap)`.
- `binary` (1×3): `RGB | depth | pos`.

**Grad-CAM** (куда модель **смотрит**):

- Хуки `forward_hook` + `register_full_backward_hook` на последний общий conv-stack: `HRNetSeg.fuse` (для binary/angle) или `HRNetMultiTask._seg.fuse` (для multitask).
- Параметры модели → `requires_grad=False`, вход → `requires_grad=True`. Этого достаточно, чтобы автоград построил граф через всю сеть, не накапливая parameter-grad'ы.
- Целевой скаляр: `pos.mean()` (multitask), `logits.mean()` (binary), `out[:, 1:].mean()` (angle, среднее по foreground bin'ам).
- CAM = `ReLU(Σ_k w_k · A_k)`, bilinear upsample до `image_size`, нормировка до [0, 1].

Источник сцены — единый интерфейс: путь к Jacquard `*_grasps.txt`, объект `CornellSample`, либо кортеж `(rgb, depth)`.

---

## 13. RealSense D435: захват своего датасета

**Все детали** (установка `pyrealsense2`, форматы потоков, известные грабли Python 3.12) — в [docs/realsense_setup.md](docs/realsense_setup.md).

### 13.1. Что выдаёт камера

| Поток | Разрешение | Формат |
|---|---|---|
| Color | 1280×720 @ 30 fps | BGR8                                  |
| Depth | 1280×720 @ 30 fps | Z16 (uint16, мм × `depth_scale` ≈ 0.001) |

RGB и IR-стерео физически разнесены (~15 мм baseline), поэтому скрипты делают `rs.align(rs.stream.color)` — depth перепроецируется в координатный frame RGB. Точность глубины — ~2% от расстояния (на столе 0.3–0.8 м это ±5–15 мм, достаточно для grasp-сегментации).

### 13.2. Превью + захват

```powershell
# 1) Превью live RGB+depth-colormap, save кадра по 's', выход по 'q'.
#    Smoke-тест RealSense-стека (отсюда видно, что камера видна и pipeline собирается).
python tools/realsense_preview.py --out captured_previews

# 2) Структурированный захват в Cornell-формат
python tools/realsense_capture.py \
    --out datasets/D435_grasp_v1 \
    --subdir 01 \
    --mode manual                # 's' = save, 'b' = burst (5 кадров), 'q' = quit
```

На выходе — стандартная Cornell-раскладка, которую без изменений подхватывает `grasp_seg.data.cornell`:

```
datasets/D435_grasp_v1/
  intrinsics.json    # fx, fy, cx, cy, baseline (один на всю сессию)
  capture.log        # timestamps всех захватов
  01/
    pcd0000r.png     # RGB
    pcd0000d.tiff    # depth в МЕТРАХ (float32, после депт-scale)
    pcd0001r.png
    pcd0001d.tiff
    ...
```

Grasp-разметка (`pcdNNNNcpos.txt` / `pcdNNNNcneg.txt`) — отдельный этап (§ 16, разметка).

---

## 14. Архитектурные заметки

### 14.1. Почему compact-polygon, а не полный rect

Каждая строка `*_grasps.txt` — это `x;y;θ;w;h` — параметризация **одной позы гриппера**, а не «области, где можно хватать». Контакт гриппера происходит в центральной полоске вдоль оси раскрытия (длина `w`). Поэтому растеризовать **весь** прямоугольник в маску некорректно: его края соответствуют пластинам гриппера, а не graspable-пикселям.

Здесь используется тот же приём, что в официальном тулбоксе Jacquard и в GG-CNN: длина прямоугольника сжимается в `1/3` (`length_scale=0.3333`), и рисуется только это compact-ядро. Получается маска центров захвата, корректно интерпретируемая как «здесь гриппер физически смыкается».

Реализация — `grasp_seg/data/grasp_rect.py:rasterize_grasp_mask`.

### 14.2. Multitask GG-CNN — почему +34 п.п. над angle-классификацией

| Channel | Активация | Loss          | Смысл |
|---|---|---|---|
| `pos`    | sigmoid  | BCE + Dice    | вероятность foreground |
| `cos2t`  | identity | MSE on positives | cos(2θ) — непрерывный угол |
| `sin2t`  | identity | MSE on positives | sin(2θ) |
| `width`  | sigmoid  | MSE on positives | ширина раскрытия / 150 px (clip в [0,1]) |

На инференсе `θ = ½·atan2(sin2θ, cos2θ)`, что даёт **непрерывный** угол и убирает дискретизационные артефакты 18-bin классификации (два grasp'а с углами 19° и 21° имеют одну и ту же цель в multitask, но разные классы в angle).

Подробнее — [docs/multitask_phase2.md](docs/multitask_phase2.md).

### 14.3. Robust depth normalisation

Depth (perfect и stereo) нормируется по **1/99 percentile** конкретного кадра (`_normalise_depth` в `jacquard_v2.py`) — это убирает влияние outlier'ов от поверхности стола и фона, и держит распределение глубин стабильным между Jacquard / Cornell / D435.

### 14.4. Object-wise split

Сплит делается **по `object_id`** (`grasp_seg/data/splits.py`), а не по сценам. Если делить по сценам, то один и тот же физический объект окажется и в train, и в val — и метрики будут сильно завышены, потому что модели нужно лишь запомнить расположение grasp'ов на конкретной геометрии, а не обобщить.

Файл сплита (`splits/jacquard_v2.json`) — простой JSON с тремя списками полных путей к `*_grasps.txt`. Один файл переиспользуется во всех runs.

### 14.5. Trainer: что умеет

`grasp_seg/engine/trainer.py` — единственный train loop проекта. Поддерживает:

- AMP fp16 + `torch.cuda.amp.GradScaler`.
- Gradient accumulation (`accum_steps` — эффективный batch).
- Per-epoch и per-N-epochs checkpoints (`save_every_epoch` + `save_every_n_epochs`).
- `best.pth` обновляется per-epoch по `target_metric` (по умолчанию `miou_fg`, конфигурируется).
- Полный resume (`--resume`): model + optimizer + scheduler + scaler + epoch counter.
- Per-step JSONL-лог (`iter_log_path`), усреднённый по `accum_steps` micro-batch'ам — реальный «эффективный батч-loss», который видел оптимизатор.
- `metrics.csv` (per-epoch): train+val метрики, learning rate, тайминг шага, GPU memory/util.
- Идемпотентный close файлового handle через try/finally (важно на Windows, где незакрытый handle блокирует чтение файла другими процессами).
- Поддержка target_metric `miou_fg` / `miou_fg_ang` / `dice_fg` / любой другой ключ из dict'а evaluator'а.

---

## 15. Конфиги — справочник

Все конфиги — обычный YAML. CLI допускает точечные переопределения через `dot.path=value`. Полная схема (значения по умолчанию):

```yaml
dataset:
  splits_path: splits/jacquard_v2.json     # путь к JSON с object-wise сплитом
  image_size: 384                          # сторона квадратного входа в модель
  input_mode: rgbd                         # rgb | depth | rgbd
  mask_mode: multitask                     # binary | angle | multitask
  num_angle_bins: 18                       # только для mask_mode=angle
  length_scale: 0.3333                     # сжатие длины prямоугольника для маски
  use_stereo_depth: false                  # false → perfect_depth (CAD); true → stereo_depth (с шумом)
  num_workers: 4
  pin_memory: true
  persistent_workers: true                 # экономит ~5 с/эпоха на спавне воркеров
  prefetch_factor: 2

augmentation:                              # включается только на train-split
  enable: true
  hflip_p: 0.5
  vflip_p: 0.5
  rotate_p: 0.8        # ±180°
  rotate_max_deg: 180.0
  scale_p: 0.7         # [0.8, 1.2]
  scale_range: [0.8, 1.2]
  translate_p: 0.5     # ±5%
  translate_max_frac: 0.05
  color_jitter_p: 0.5  # brightness / contrast / saturation (luma-preserving) / hue
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.05
  rgb_noise_p: 0.3
  rgb_noise_std: 0.02
  depth_jitter_p: 0.5
  depth_jitter_range: [0.95, 1.05]
  depth_dropout_p: 0.3
  depth_dropout_frac: 0.02
  use_stereo_depth_p: 0.3                  # ★ key for sim2real — учит модель работать с шумным depth

model:
  backbone: hrnet_w18                      # или hrnet_w18_small_v2
  pretrained: true                         # ImageNet-pretrained веса через timm

loss:                                      # только для mask_mode=multitask
  pos_weight: 1.0
  cos_weight: 1.0
  sin_weight: 1.0
  width_weight: 0.4
  bce_pos_weight: 1.0
  smooth: 1.0

trainer:
  epochs: 80
  batch_size: 2
  accum_steps: 4                           # эффективный batch = batch_size * accum_steps
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4
  warmup_epochs: 1
  poly_power: 0.9
  grad_clip_norm: 1.0
  amp: true
  log_interval: 20                         # каждые N оптимизаторных шагов — INFO-строка
  save_dir: outputs/run
  save_every_epoch: true                   # сохранять epoch_NNN.pth каждую эпоху
  save_every_n_epochs: 0                   # если >0 — переопределяет save_every_epoch (только каждые N)
  target_metric: miou_fg                   # по чему обновлять best.pth
  iter_log_path: ""                        # если задано — пишет JSONL построчно (один шаг = одна строка)
```

---

## 16. Дорожная карта

### 16.1. Этап 7 — Свой датасет на D435 + fine-tune

Цель: **fine-tune модели `rgbd_multitask` под кухню Fairino FR5** (рабочее освещение, типы объектов, угол камеры).

Подходы к разметке (см. [docs/realsense_setup.md § «Annotation roadmap»](docs/realsense_setup.md) — если не публиковался, то в обсуждениях PR-ов):

1. **Ручная Cornell-style** — 30–60 с на rect × 3–8 rect/сцена → 3–5 мин/сцена; 10 ч на 150 сцен.
2. **SAM + principal axis** — клик на объект → бинарная маска → PCA → oriented bbox → авто rect'ы; ~5 с/сцена → 15 мин на 150 сцен. Не учитывает «удобство захвата», но даёт baseline.
3. **Гибрид SAM + ручная правка** — SAM-предложения → проверка/правка мышкой → save; 10–30 с/сцена → ~1 ч на 150 сцен. **Рекомендованный путь.**
4. **Active learning loop** — Variant 1 на seed'е (30 сцен) → промежуточный fine-tune → модель размечает остальные → пользователь правит → финальный fine-tune.

Ожидаемый результат: top-1 acc на D435-валидации 70–85% (vs ~30–50% до fine-tune, на основе опыта Cornell при 885 сценах).

Что нужно сделать в коде:
- `tools/annotate_grasps/` — Flask + HTML5 canvas UI (план — гибрид SAM + ручная правка).
- `configs/finetune_d435.yaml` — производный от `multitask_local512.yaml`, `--resume-model-only` от `rgbd_multitask`/best.pth, lr=1e-4, 10–30 эпох.

### 16.2. Этап 8 — Интеграция с Fairino FR5

Цель: автономный pipeline «кадр → grasp → захват».

1. **Hand-eye calibration** — eye-to-hand (камера над столом, статика) предпочтительнее eye-in-hand (на фланце робота).
2. **ChArUco-калибровка** через OpenCV → матрица `T_camera_base` (преобразование `(px, py, depth)` в base-frame робота).
3. **Pipeline захвата**: D435-кадр → инференс модели → top-1 grasp в pixel-frame → через intrinsics+depth в camera-frame → через `T_camera_base` в base-frame → обратная кинематика FR5 → захват.
4. **Gripper interface**: какой именно end-effector (параллельный двухпальцевый / вакуумный) определит интерпретацию `width`-канала и I/O-протокол.
5. **SDK**: [github.com/FAIR-INNOVATION/fairino-python-sdk](https://github.com/FAIR-INNOVATION/fairino-python-sdk).

---

## 17. Заметки для новых участников и ИИ-агентов

### 17.1. Где что лежит — в одном предложении на файл

- **Хотите начать обучение** → `notebooks/colab_train.ipynb` (бесплатно) или `notebooks/local_train_rgbd_multitask.ipynb` (RTX 4060).
- **Хотите построить графики из готового run'а** → `notebooks/visualize.ipynb`.
- **Хотите изменить гиперпараметр** → `configs/*.yaml`, не код.
- **Хотите изменить аугментации** → секция `augmentation:` в YAML + `grasp_seg/data/transforms.py:apply_augmentations`.
- **Хотите добавить ещё одну метрику** → `grasp_seg/engine/evaluator.py` (положите в возвращаемый dict — Trainer её залогирует и можно будет указать как `target_metric`).
- **Хотите добавить ещё один режим маски** → `grasp_seg/data/grasp_rect.py` (rasterizer) + `grasp_seg/losses/` (loss) + `grasp_seg/models/hrnet.py` (голова) + `grasp_seg/engine/evaluator.py` (evaluator).
- **Хотите захватить свой датасет** → `tools/realsense_capture.py` + `docs/realsense_setup.md`.

### 17.2. Если вы — ИИ-агент

Особенности проекта, которые легко не заметить:

- **Object-wise split** (а не scene-wise): один и тот же `object_id` не может быть и в train, и в val. Делайте сплит **один раз** и переиспользуйте — не пересоздавайте файл при каждом запуске.
- **Compact-polygon mask**: длина `*_grasps.txt`-прямоугольника сжата в 1/3 (`length_scale=0.3333`). Не пытайтесь растеризовать «полный» rect — это сломает интерпретацию.
- **`mask_mode=multitask` ≠ `mask_mode=angle`** в метрических числах: их `miou_fg` ИЗМЕРЯЕТ РАЗНОЕ. Apples-to-apples сравнение возможно только через `val_miou_fg_ang` (см. [docs/multitask_phase2.md](docs/multitask_phase2.md)).
- **Депт нормируется robust 1/99 percentile**, а не min/max — это важно для cross-domain (Cornell, D435).
- **Stereo depth содержит NaN'ы**. Включается через `augmentation.use_stereo_depth_p` — это ключевая аугментация для sim2real. Если NaN'ы мешают на этапе отладки, выставите 0.0 и тренируйте на perfect_depth.
- **Не модифицируйте `colab_train.ipynb`** при работе над локальным/Kaggle ноутбуками — он должен оставаться Colab-совместимым.
- **`configs/*.yaml` — единственная правильная точка изменения гиперпараметров.** Не хардкодьте image_size, batch_size и тому подобное в ноутбуках — параметризуйте через CLI-overrides.
- **Trainer уже умеет всё, что нужно для прод-обучения** (AMP / accum / resume / per-step JSONL / target_metric). Прежде чем писать «свой» train loop, прочитайте `grasp_seg/engine/trainer.py:fit` — велик шанс, что нужный режим уже есть как параметр.
- **PR-ы рекомендуется делать узкими и атомарными.** История PR-ов проекта — в [docs/progress_report_final.md § 8.1](docs/progress_report_final.md#81-pr-история-текущей-стадии); средний размер PR'а — одно явное изменение (один модуль, одна задача).

### 17.3. Быстрая проверка, что у вас всё работает

```bash
python scripts/smoke_test.py              # 30 с, без датасета — должен вывести ALL MODES OK
python tools/prepare_split.py --root <jacquard> --out splits/jacquard_v2.json --val-frac 0.1 --test-frac 0.1 --seed 0
python tools/train.py --config configs/multitask.yaml dataset.splits_path=splits/jacquard_v2.json trainer.epochs=1
```

Если все три шага прошли — у вас работающее окружение.

---

## 18. Ссылки

**Документы проекта:**
- [docs/progress_report_final.md](docs/progress_report_final.md) — финальный отчёт (главный источник цифр)
- [docs/progress_report.md](docs/progress_report.md) / [docs/progress_report_v2.md](docs/progress_report_v2.md) — отчёты ранних этапов
- [docs/multitask_phase2.md](docs/multitask_phase2.md) — переход на multitask + обоснование `miou_fg_ang`
- [docs/visualization.md](docs/visualization.md) — описание viz-модуля и `notebooks/visualize.ipynb`
- [docs/local_setup_windows.md](docs/local_setup_windows.md) — Windows + RTX 3060 (визуализация)
- [docs/local_training_windows.md](docs/local_training_windows.md) — Windows + RTX 4060 (обучение)
- [docs/training_metrics.md](docs/training_metrics.md) — расшифровка всех полей train-лога
- [docs/realsense_setup.md](docs/realsense_setup.md) — D435: установка, форматы, скрипты
- [docs/yandex_datasphere.md](docs/yandex_datasphere.md) — Yandex DataSphere (T4/V100/A100, RUB)

**Внешние:**
- HRNet (статья): https://arxiv.org/abs/1908.07919
- GG-CNN (multitask формулировка): https://arxiv.org/abs/1804.05172
- Jacquard V2: https://jacquard.liris.cnrs.fr/ и https://github.com/lqh12345/Jacquard_V2
- Cornell Grasp Dataset: http://pr.cs.cornell.edu/grasping/rect_data/data.php
- Pretrained backbone: https://huggingface.co/timm/hrnet_w18.ms_aug_in1k
- Intel RealSense D435: https://www.intelrealsense.com/depth-camera-d435/
- Fairino FR5 SDK: https://github.com/FAIR-INNOVATION/fairino-python-sdk
- Архивы Jacquard V2 (Я.Диск, 12 zip): https://disk.yandex.ru/d/Je56nUcC9hiHFQ
