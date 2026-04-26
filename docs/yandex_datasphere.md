# Запуск обучения на Yandex DataSphere

DataSphere — managed Jupyter в Yandex Cloud. Нативно интегрирован с Я.Диском
и S3-совместимым Object Storage, оплата в рублях. Хорошо подходит для нашей
задачи: датасет уже на Я.Диске, можно обойтись без длительной перезаливки.

> Цены (на момент написания, проверяйте актуальные в консоли):
> - g1.1 (T4, 16 GB VRAM) — ~₽40/ч
> - g1.4 (V100, 32 GB VRAM) — ~₽120/ч
> - g2.1 (A100, 80 GB VRAM) — ~₽300/ч
>
> Полный 80-эпоховый прогон: ~12–15 ч на T4 (~₽500–₽600), ~6–8 ч на V100
> (~₽720–₽960), ~3–4 ч на A100 (~₽900–₽1200).

## Один раз: подготовка облака

1. Создайте платёжный аккаунт в Yandex Cloud (https://console.cloud.yandex.ru).
   Активируйте грант для новых пользователей (~₽4000 на 60 дней) — этого с
   запасом хватит на несколько прогонов на T4/V100.

2. Создайте **community** в DataSphere:
   `Yandex Cloud Console → DataSphere → Создать сообщество`.

3. Внутри community создайте **проект** (он же содержит storage и compute
   квоты).

4. (Опционально, но удобно) Создайте **Object Storage bucket** в том же
   фолдере: `Yandex Cloud Console → Object Storage → Создать бакет`. Имя,
   например, `hrnet-jacquard`. Туда пойдут чекпоинты — DataSphere умеет
   монтировать бакет как datasource.

## Один раз: подготовка датасета

DataSphere **не монтирует Я.Диск напрямую** (это два разных продукта). У вас
два варианта:

### Вариант A — скачать прямо в DataSphere (быстрый, рекомендую)

1. Откройте проект DataSphere → New notebook.
2. Подключите GPU-конфигурацию: правая панель → ВМ → выберите `g1.1` (T4)
   или `g1.4` (V100). VM поднимется за ~30–60 с.
3. Запустите в первой ячейке:

   ```python
   import os, requests, time, glob, zipfile, shutil

   YANDEX_DISK_FOLDER_URL = "https://disk.yandex.ru/d/Je56nUcC9hiHFQ"
   YANDEX_FOLDER_PATH    = "/Jacquard_V2"
   ZIP_DIR  = "/home/jupyter/datasphere/project/Jacquard_V2_zips"
   DATA_DIR = "/home/jupyter/datasphere/project/Jacquard_V2"

   os.makedirs(ZIP_DIR, exist_ok=True)
   API = "https://cloud-api.yandex.net/v1/disk/public/resources"

   def list_folder(public_url, sub_path):
       r = requests.get(API, params={"public_key": public_url, "path": sub_path,
                                       "limit": 200}, timeout=60)
       r.raise_for_status()
       return [(it["name"], it["path"])
               for it in r.json()["_embedded"]["items"] if it["type"] == "file"]

   def download(public_url, inner_path, dest):
       if os.path.exists(dest):
           return
       href = requests.get(API + "/download",
                           params={"public_key": public_url, "path": inner_path},
                           timeout=60).json()["href"]
       with requests.get(href, stream=True, timeout=600) as r:
           r.raise_for_status()
           with open(dest + ".part", "wb") as f:
               for chunk in r.iter_content(8 * 1024 * 1024):
                   f.write(chunk)
           os.rename(dest + ".part", dest)

   for name, path in list_folder(YANDEX_DISK_FOLDER_URL, YANDEX_FOLDER_PATH):
       print("downloading", name)
       download(YANDEX_DISK_FOLDER_URL, path, os.path.join(ZIP_DIR, name))
   ```

   Скорость DataSphere ↔ Я.Диск обычно 30–60 МБ/с — 63 ГБ за **20–35 минут**.

4. Распакуйте:

   ```python
   os.makedirs(DATA_DIR, exist_ok=True)
   for z in sorted(glob.glob(os.path.join(ZIP_DIR, "*.zip"))):
       print("extracting", os.path.basename(z))
       with zipfile.ZipFile(z) as zf:
           zf.extractall(DATA_DIR)
   shards = (sorted(glob.glob(os.path.join(DATA_DIR, "JacquardV2_Dataset_*")))
             + sorted(glob.glob(os.path.join(DATA_DIR, "Jacquard_Dataset_*"))))
   for shard in shards:
       for entry in os.listdir(shard):
           dst = os.path.join(DATA_DIR, entry)
           if not os.path.exists(dst):
               shutil.move(os.path.join(shard, entry), dst)
       try: os.rmdir(shard)
       except OSError: pass
   ```

   Распакованные файлы остаются в `/home/jupyter/datasphere/project/` —
   это **persistent storage проекта**, не теряется при пересоздании VM.

### Вариант B — через Object Storage (если работаете в команде)

1. Скачайте архивы с Я.Диска на свой компьютер.
2. Залейте в созданный бакет через консоль или `aws s3 cp`.
3. В ноутбуке: `s3datasource = "s3://hrnet-jacquard/Jacquard_V2/"`,
   используйте `s3fs` или `aws s3 sync` для загрузки в локальный SSD VM.

Этот путь медленнее (вы платите за исходящий трафик из домашнего интернета),
но удобнее, если несколько человек работают над проектом.

## Один раз: код и зависимости

```bash
!git clone --depth 1 https://github.com/Hesgoryr/HRNet_Grasp_Semantic_Segmentation.git
%cd HRNet_Grasp_Semantic_Segmentation
!pip install -q -r requirements.txt
```

DataSphere preinstall'ит torch с CUDA — `requirements.txt` его не перезаписывает
(там `torch>=2.0.0` без специфики). Проверьте: `python -c "import torch; print(torch.cuda.is_available())"`.

## Запуск обучения

```bash
!python tools/prepare_split.py --root /home/jupyter/datasphere/project/Jacquard_V2 \
    --out splits/jacquard_v2.json --val-frac 0.1 --test-frac 0.1 --seed 0
```

**T4 (g1.1):**
```bash
!python tools/train.py --config configs/default.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    dataset.num_workers=4 \
    trainer.batch_size=16 trainer.accum_steps=1 \
    trainer.save_dir=/home/jupyter/datasphere/project/runs/hrnet_w18_rgbd_angle
```

**V100 (g1.4) — больше VRAM, можно увеличить batch:**
```bash
!python tools/train.py --config configs/default.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    dataset.num_workers=8 \
    trainer.batch_size=32 trainer.accum_steps=1 \
    trainer.save_dir=/home/jupyter/datasphere/project/runs/hrnet_w18_rgbd_angle
```

**A100 (g2.1):**
```bash
!python tools/train.py --config configs/default.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    dataset.num_workers=12 \
    trainer.batch_size=64 trainer.accum_steps=1 \
    trainer.save_dir=/home/jupyter/datasphere/project/runs/hrnet_w18_rgbd_angle
```

## Особенности DataSphere

- **VM не работает в фоне** — закрытие вкладки браузера завершает сессию через
  ~10 минут idle. Чтобы дотренировать в отсутствие пользователя, используйте
  `Run as job`: правый клик по ноутбуку → `Run as Job` → задайте VM-конфиг и
  таймаут (до 24 ч). Job выполняется без открытой вкладки и присылает уведомление.

- **Деньги списываются только за время активной VM** — выключайте её после
  каждого прогона: правая панель → ВМ → Stop.

- **Project storage** (`/home/jupyter/datasphere/project/`) сохраняется
  навсегда (платится за объём, но дёшево); туда кладите чекпоинты. На VM
  стоит ещё локальный SSD `/tmp/` — он сбрасывается между сессиями, но
  быстрее по I/O — туда можно положить распакованный датасет, если вам
  выгоднее перезаливать его раз в сессию вместо хранения на project storage.

- **Возобновление** — то же `--resume`, что и везде:

  ```bash
  !python tools/train.py --config configs/default.yaml \
      dataset.splits_path=splits/jacquard_v2.json \
      trainer.save_dir=/home/jupyter/datasphere/project/runs/hrnet_w18_rgbd_angle \
      --resume /home/jupyter/datasphere/project/runs/hrnet_w18_rgbd_angle/last.pth
  ```

## Как мониторить расход

Yandex Cloud Console → Биллинг → расход по проекту, обновляется ~раз в час.
В DataSphere также отображается current burn rate в правой панели VM.
