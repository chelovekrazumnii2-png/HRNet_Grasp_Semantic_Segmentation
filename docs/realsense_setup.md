# Intel RealSense D435 — установка и захват датасета

Документ описывает первый шаг интеграции D435 в проект:

* установить `pyrealsense2` на Windows-машине с RTX 3060;
* проверить, что камера работает (`tools/realsense_preview.py`);
* собрать датасет в Cornell-формате (`tools/realsense_capture.py`),
  чтобы существующий `grasp_seg.data.cornell` подхватывал его без
  изменений.

Разметка (положительные / отрицательные grasp-rect'ы), fine-tune
обученных моделей и интеграция с манипулятором Fairino FR5 — отдельные
шаги, описаны в последующих документах / PR.

---

## 1. Камера: что мы получаем

D435 — **активная стерео**: пара IR-сенсоров + IR-проектор паттерна,
плюс отдельный RGB-сенсор.

| Поток | Разрешение по умолчанию | Формат |
|---|---|---|
| Color | 1280×720 @ 30 fps | BGR8 |
| Depth | 1280×720 @ 30 fps | Z16 (16-bit unsigned, миллиметры × `depth_scale`) |

`depth_scale` обычно 0.001 м/unit (получаем из `depth_sensor.get_depth_scale()`),
то есть пиксель `uint16` напрямую — миллиметры.

**Важно: RGB и depth физически разнесены** — у них разная оптика и
небольшой baseline ~15 мм. Чтобы получить «RGB-D как в Jacquard»
(каждый пиксель RGB имеет свою валидную глубину), используем
`pyrealsense2.align(rs.stream.color)` — он перепроецирует depth в
координатный frame RGB через известные intrinsics обоих сенсоров. Все
наши скрипты делают это в одной строке.

Точность глубины — ~2 % от расстояния до 2 м. На столе (0.3–0.8 м)
это ±5–15 мм, чего достаточно для grasp-сегментации.

---

## 2. Установка на Windows

В активированном `.venv` (тот же, в котором вы запускаете обучение):

```powershell
pip install pyrealsense2 tifffile
```

Зависимости `opencv-python` и `numpy` уже установлены проектом. Если
`pyrealsense2` не находит wheel под вашу версию Python — проверьте
[список поддерживаемых версий на PyPI](https://pypi.org/project/pyrealsense2/#files).
Текущая поддержка: CPython 3.7–3.11 на Windows x86_64. Если у вас
Python 3.12+, поставьте параллельный `.venv-rs` под 3.11 специально
для RealSense-скриптов:

```powershell
py -3.11 -m venv .venv-rs
.\.venv-rs\Scripts\Activate.ps1
pip install pyrealsense2 opencv-python numpy tifffile
```

(Тренировочный пайплайн при этом продолжает работать в основном
`.venv` — RealSense нужен только этим двум скриптам.)

### Проверка драйвера

```powershell
python -c "import pyrealsense2 as rs; ctx = rs.context(); print([d.get_info(rs.camera_info.name) for d in ctx.query_devices()])"
```

Ожидаемый вывод — `['Intel RealSense D435', ...]`. Если выводит пустой
список:

* в Диспетчере устройств должен быть `Intel(R) RealSense(TM) Depth Camera 435`
  (без жёлтого восклицательного знака);
* установите [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases)
  для Windows — в нём идут официальные драйверы и утилита
  `Intel RealSense Viewer`, которая удобна для первоначальной проверки.

---

## 3. Скрипты

### 3.1. `tools/realsense_preview.py` — live-предпросмотр

Минимальный smoke-тест: показывает RGB и colormapped depth в двух
окнах, сохраняет один кадр по клавише `s`.

```powershell
python tools\realsense_preview.py --out captures\preview
```

Управление в окне OpenCV:

* `s` — сохранить текущий кадр как `snap_NNNN_r.png` + `snap_NNNN_d.tiff`
  (float32 metres) + `snap_NNNN_d_preview.png` (8-bit jet colormap для
  глаза);
* `q` или `ESC` — выход.

Опции:

* `--width 1280 --height 720 --fps 30` — параметры потоков (по умолчанию).
* `--depth-preset highaccuracy` — пресет качества depth от прошивки D435.
  Варианты: `default`, `highaccuracy` (точнее, реже), `highdensity`
  (плотнее, чуть шумнее), `medium`.

Если этот скрипт работает — RealSense-стек настроен корректно, можно
переходить к структурированному захвату.

### 3.2. `tools/realsense_capture.py` — структурированный захват

Пишет файлы в **Cornell-формате**, чтобы `grasp_seg.data.cornell`
читал датасет без единой строчки нового кода:

```
captures/my_dataset/
├── intrinsics.json          ← intrinsics + depth_scale + serial (1 раз на сессию)
├── capture.log              ← журнал сохранённых сцен
├── 01/
│   ├── pcd0000r.png         ← RGB
│   ├── pcd0000d.tiff        ← depth, float32, метры
│   ├── pcd0000d_preview.png ← cosmetic превью (не читается loader'ом)
│   ├── pcd0001r.png
│   └── ...
├── 02/
│   └── ...
```

Позже файл `pcdNNNNcpos.txt` (положительные grasp-rect'ы) добавит
шаг разметки.

Запуск:

```powershell
# Manual mode: жмёте 's' на каждый сохраняемый кадр
python tools\realsense_capture.py --out captures\my_dataset --subdir 01

# Interval mode: 1 кадр в секунду
python tools\realsense_capture.py --out captures\my_dataset --subdir 02 `
    --mode interval --every 1.0
```

Управление (manual):

* `s` — сохранить текущий aligned-кадр как следующую сцену.
* `b` — burst: захватить 5 кадров подряд (`pcdNNNNr.png`,
  `pcdNNNN_1r.png` … `pcdNNNN_4r.png`). Полезно для медианной фильтрации
  depth офлайн.
* `q` / `ESC` — остановить сессию.

Опции (помимо общих с preview):

* `--subdir 01` — сохранять в подпапку `01/`. Cornell-loader умеет
  обходить и плоскую раскладку, и вложенную (`01/`, `02/` …, скрытые
  и `backgrounds/` пропускаются), так что выбирайте по удобству.
* `--start-id 0` — принудительная стартовая нумерация. Без неё скрипт
  смотрит на существующие файлы в папке и продолжает с первого
  свободного `pcdNNNN`.
* `--burst N` — длина burst-захвата (по умолчанию 5).
* `--no-window` — headless-режим без OpenCV-окон. Используется в
  основном с `--mode interval`.

### 3.3. Что сохраняется

| Файл | Тип | Содержимое |
|---|---|---|
| `pcdNNNNr.png` | uint8, BGR8 | RGB-кадр D435, как пишет OpenCV. |
| `pcdNNNNd.tiff` | float32, m | Depth aligned to color, **в метрах**. Совместим с `_load_depth` Jacquard'а после простой нормировки. |
| `pcdNNNNd_preview.png` | uint8, BGR8 | Колормэп (jet) для глаза. Loader его игнорирует. |
| `intrinsics.json` | JSON | Intrinsics, depth_scale, серийник, версия прошивки. Нужен для калибровки и backprojection. |
| `capture.log` | text | Журнал: timestamp, scene_id, размер кадра. |

---

## 4. Использование с существующим пайплайном

После того как набралось хотя бы 5–10 сцен, можно прямо в ноутбуке
визуализации поставить:

```python
CORNELL_ROOT = r"D:\…\captures\my_dataset"
```

и прогнать секции:

* **1.6** — `figure_cornell_raw` покажет случайную сцену D435 c
  предсказаниями моделей (без grasp-rect'ов, потому что ещё не
  размечено).
* **5.2** — `figure_compare_models_cornell` отрендерит side-by-side для
  всех загруженных моделей.
* **5.3** — top-1 acc метрики потребуют размеченных rect'ов и пока
  работать не будут (вернут пустой DataFrame).
* **8.1** / **8.2** — per-head + Grad-CAM, чтобы посмотреть, что
  модель «видит» на ваших реальных снимках.

Это хороший способ оценить **до разметки**, насколько обученные
Jacquard-модели уже работают в вашем сетапе (освещение / стол / fov).

---

## 5. Что дальше

1. **Разметка** — Flask/canvas-инструмент для Cornell-формата
   `pcdNNNNcpos.txt` / `cneg.txt`. Отдельный PR.
2. **Fine-tune** на размеченных сценах — `configs/finetune_realsense.yaml`,
   `python tools/train.py --resume best.pth ...`.
3. **Интеграция с Fairino FR5** — hand-eye calibration (ChArUco, OpenCV
   `calibrateHandEye`), пайплайн «кадр → grasp → IK → захват». Отдельный
   набор PR'ов.

Подробности по разметке и манипулятору — в последующих документах.
