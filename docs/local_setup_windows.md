# Локальный запуск визуализации — Windows + RTX 3060

## 0. Что должно стоять заранее

| Компонент | Версия | Зачем |
| ---- | ---- | ---- |
| **NVIDIA driver** | актуальный (≥ 552) | поддержка CUDA 12.x для RTX 3060 |
| **Python** | **3.10.x** (рекомендую через [python.org installer](https://www.python.org/downloads/release/python-31011/)) | в этой версии `timm`, `torch`, `numpy` собраны под Windows без сюрпризов |
| **Git** | любой свежий | `git clone` |
| **VS Code** | (опционально) с расширениями `Python` + `Jupyter` | удобный запуск ноутбука |

Проверка после установки (PowerShell):

```powershell
nvidia-smi          # должна быть строка "CUDA Version: 12.x"
python --version    # 3.10.x
git --version
```

> ⚠️ Если у вас стоит Python из Microsoft Store — лучше снести его и поставить с
> python.org. Store-сборка иногда конфликтует с установкой `imagecodecs`.

---

## 1. Клонируем репозиторий

```powershell
cd D:\
git clone https://github.com/chelovekrazumnii2-png/HRNet_Grasp_Semantic_Segmentation.git
cd HRNet_Grasp_Semantic_Segmentation
```

---

## 2. Виртуальное окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> Если `Activate.ps1` ругается «running scripts is disabled» — один раз выполните:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

---

## 3. PyTorch с CUDA (RTX 3060 → CUDA 12.6)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Проверка:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно вывести что-то вроде `2.5.1+cu126 True NVIDIA GeForce RTX 3060`.
Если `False` — у вас в системе живут несколько версий torch / CUDA. Сносим
полностью и ставим заново:

```powershell
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## 4. Остальные зависимости

```powershell
pip install -r requirements.txt
pip install jupyter ipykernel    # чтобы ноутбук был запускаемым из VS Code
python -m ipykernel install --user --name grasp_viz --display-name "Python 3.10 (HRNet viz)"
```

Smoke-тест без датасета (≈30 секунд):

```powershell
python scripts\smoke_test.py
```

Должен закончиться строкой `ALL MODES OK`. Если не закончился — пишите, разберём.

---

## 5. Датасеты

### 5.1 Jacquard V2 (нужен)

У вас **уже распакован** где-то локально или нужно тянуть с Я.Диска?

Если **уже распакован** — просто запомните путь, например `D:\datasets\JacquardV2\`.
Внутри должны быть подпапки вида:

```
JacquardV2_Dataset_0\<object_id>\<idx>_<object_id>_RGB.png
                                 \<idx>_<object_id>_perfect_depth.tiff
                                 \<idx>_<object_id>_grasps.txt
                                 ...
JacquardV2_Dataset_1\...
```
(или `Jacquard_Dataset_0..N\` — мы поддерживаем оба именования).

Если **нужно скачать** — пришлите, какая ссылка / источник у вас был на
обучении (Я.Диск public link?), и я сделаю PowerShell-скрипт.

### 5.2 Cornell Grasp Dataset (для cross-domain)

Нужно подтвердить путь и формат. Внутри ожидаются файлы вида:

```
pcd0100r.png
pcd0100.txt          (point cloud, мы не используем)
pcd0100cpos.txt      (positive grasps)
pcd0100cneg.txt      (negative grasps)
```

Ничего распаковывать не нужно — лоадер берёт прямо из плоской директории.

### 5.3 Splits файл

Есть **готовый** `splits/jacquard_v2.json` (генерировался на обучении) — пришлите его или путь, я положу в репо.
Если **нет** — генерируем заново:

```powershell
python tools\prepare_split.py --root D:\datasets\JacquardV2 --out splits\jacquard_v2.json
```

> Сплиты делаются по object_id (без leakage). Если seed тот же, что был
> использован при обучении — тот же `test`-сплит, что использовали
> вычисления метрик в `metrics.csv`. **Это важно**, иначе картинки
> будут построены не на тех сценах, что считались для отчётов.

---

## 6. Чекпоинты с Google Drive (~3-6 ГБ)

Для каждой из 3 моделей нужен полный набор:

```
runs\<model_name>\
    best.pth
    epoch_001.pth
    epoch_005.pth
    epoch_010.pth
    epoch_015.pth
    epoch_020.pth
    epoch_025.pth
    epoch_030.pth
    metrics.csv
    resolved_config.yaml
```

(остальные `epoch_NNN.pth` для отчёта не нужны — экономим трафик).

Самый удобный способ выкачать выборочно — `gdown`:

```powershell
pip install gdown
```

Я могу сделать готовый PowerShell / Python-скрипт `tools\download_runs.py`,
который из таблицы `RUNS = {name: drive_folder_id_or_url}` сам качает
нужные файлы с Drive. Чтобы он заработал, нужны **ID или ссылки на 3 папки**
в формате (один из):

- ссылка вида `https://drive.google.com/drive/folders/1ABCDEFG...`,
- или ID типа `1ABCDEFG...`.

Альтернатива — открыть папку в браузере, отметить нужные файлы и нажать
«Скачать» (Drive сам соберёт ZIP). Для 3 моделей × 9 файлов это ~27 кликов,
но без `gdown`.

---

## 7. Запуск ноутбука

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

(или открыть `notebooks\visualize.ipynb` в VS Code и выбрать kernel
«Python 3.10 (HRNet viz)»).

В **первой ячейке** ноутбука переключите:

```python
ENV = "local"
```

И в той же ячейке проставьте:

```python
REPO_ROOT     = r"D:\HRNet_Grasp_Semantic_Segmentation"
JACQUARD_ROOT = r"D:\datasets\JacquardV2"
SPLITS_PATH   = os.path.join(REPO_ROOT, r"splits\jacquard_v2.json")
CORNELL_ROOT  = r"D:\datasets\cornell"
RUNS = {
    "angle_rgbd":     r"D:\HRNet\runs\angle_rgbd",
    "multitask_rgb":  r"D:\HRNet\runs\multitask_rgb",
    "multitask_rgbd": r"D:\HRNet\runs\multitask_rgbd",
}
```

Запускаем секции по порядку. **Секция 1 («Датасет») не требует чекпоинтов**
— если она работает, значит пути к Jacquard / splits / Cornell корректные.

---

## 8. Что от вас нужно прямо сейчас

Чтобы я сделал последний штрих (скрипт выкачки + актуализация README под Windows + financial PR), пришлите:

1. **Путь к Jacquard локально** (если уже распакован) — например `D:\datasets\JacquardV2\`. Если ещё не распакован — скажите, и я добавлю шаги по скачиванию.
2. **Путь к Cornell локально** — например `D:\datasets\cornell\`.
3. **Splits файл** — где он лежит сейчас (или регенерировать заново)?
4. **3 ссылки на папки Google Drive** с чекпоинтами (или их ID).

После этого я добавлю `tools\download_runs.py` + Windows-секцию в README, чтобы у вас был один скрипт «всё скачать».
