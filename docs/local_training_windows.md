# Локальное обучение в VS Code (Windows + RTX 4060)

Целевая конфигурация: **AMD Ryzen 5 5600G + RTX 4060 8 GB + 16 GB RAM, Windows 10/11**.

Ноутбук: [`notebooks/local_train_rgbd_multitask.ipynb`](../notebooks/local_train_rgbd_multitask.ipynb).
Конфиг: [`configs/multitask_local512.yaml`](../configs/multitask_local512.yaml) (RGB-D multitask, 512×512, 150 эпох, чекпоинт раз в 5 эпох, `iter_log.jsonl` каждый шаг).
Расшифровка метрик в логах: [`docs/training_metrics.md`](training_metrics.md).

## 1. Что должно быть установлено заранее

| Компонент | Версия | Где взять |
|---|---|---|
| **NVIDIA Driver** | ≥ 537 (Studio или Game Ready, любой) | https://www.nvidia.com/Download/index.aspx |
| **Python** | 3.11 (рекомендуется) или 3.12 | https://www.python.org/downloads/windows/ — поставь галку «Add to PATH» |
| **Git** | любой | https://git-scm.com/download/win |
| **VS Code** | актуальный | https://code.visualstudio.com/ |
| **VS Code extensions** | Python + Jupyter | в самом VS Code, вкладка Extensions |

Проверка драйвера: открой PowerShell и запусти `nvidia-smi`. Должно показать `RTX 4060`. Если `nvidia-smi` не найдено — драйвер не встал.

## 2. Клонировать репозиторий и создать виртуальное окружение

```powershell
# Куда угодно — например, в Documents
cd $HOME\Documents
git clone https://github.com/chelovekrazumnii2-png/HRNet_Grasp_Semantic_Segmentation.git
cd HRNet_Grasp_Semantic_Segmentation

# venv под этот проект
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# torch с CUDA 12.4 (для RTX 4060)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# остальные зависимости
pip install timm opencv-python tifffile matplotlib pandas pyyaml tqdm `
            requests imagecodecs pynvml ipykernel
```

Проверка:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно вывести что-то вроде:
```
2.5.1+cu124 True NVIDIA GeForce RTX 4060
```

## 3. Открыть ноутбук в VS Code

```powershell
code .
```

В VS Code:
1. Открой `notebooks/local_train_rgbd_multitask.ipynb`.
2. Сверху справа — селектор кернела. Выбери **`.venv (Python 3.11)`** из этого репо. Если его нет — нажми «Select Another Kernel» → «Python Environments» → твой `.venv`.
3. Отредактируй первую ячейку с путями (`DATA_ROOT`, `RUNS_ROOT`, `RUN_NAME`).
4. Запускай ячейки сверху вниз: **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**.

## 4. Перед запуском обучения — обязательно прогони smoke-test

Ячейка 6 делает:
- 1 батч forward+backward с реальными данными;
- замеряет `step_time` и `gpu_mem_peak`;
- печатает оценку времени до конца 150-эпохного рана.

**Если выпадает CUDA OOM:**
1. Сократи `BATCH_SIZE` до 1 в ячейке 0; подними `ACCUM_STEPS` до 16, чтобы эффективный батч остался 16.
2. Или уменьши `IMAGE_SIZE` до 448 (всё равно больше, чем дефолтные 384).
3. Перезапусти ячейку 6.

**Если ETA получается > 5 дней:**
- Уменьши `EPOCHS` до 80–100 (всё равно увидишь плато).
- Или вернись на `IMAGE_SIZE=384` (батч 4 × accum 4 даст ~1.5–2× быстрее).

## 5. Запуск, мониторинг, прерывание

Ячейка 7 запускает `tools/train.py` как subprocess. Логи идут одновременно:
- в вывод ноутбука (видно прямо в VS Code);
- в `SAVE_DIR/train.log` (для post-mortem);
- per-step метрики — в `SAVE_DIR/iter_log.jsonl`;
- per-epoch метрики — в `SAVE_DIR/metrics.csv`;
- чекпоинты — `best.pth` (обновляется каждую эпоху, если `val_miou_fg` улучшился), `last.pth` (каждую эпоху), `epoch_NNN.pth` (раз в 5 эпох + последняя).

**Прерывание:** жмёшь ⏹ (interrupt) в ноутбуке. Это убивает subprocess. Последняя строка `metrics.csv` уже на диске, `last.pth` тоже. При следующем запуске ячейка 7 автоматически подхватит `last.pth` через `--resume`.

**Параллельный мониторинг:** открой второй экземпляр VS Code (или Jupyter Lab), скопируй ячейки 8 и 9 в отдельный ноутбук — они читают `metrics.csv` и `iter_log.jsonl` независимо от тренировочного процесса.

## 6. Что делать при типовых проблемах

| Симптом | Причина | Что делать |
|---|---|---|
| `torch.cuda.is_available() == False`, но `nvidia-smi` работает | CPU-only torch | `pip uninstall torch; pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision` |
| OOM на batch=2 image=512 | RTX 4060 переполнен | batch=1 accum=16 **или** image=448 |
| `dataload_fraction > 0.30` в логах | DataLoader не успевает | `NUM_WORKERS=6`, `PREFETCH_FACTOR=4` (если RAM позволит) |
| Нет CUDA toolkit (`nvcc` not found) | Не нужно для PyTorch | wheels включают свой CUDA runtime; nvcc нужен только для компиляции custom CUDA kernels |
| Kernel падает на `import torch` после `pip install` | Транзитивная зависимость перетянула несовместимые binaries | Перезагрузи kernel (Reset Kernel в VS Code) |
| `pyrealsense2` ругается на Python 3.13 | Колёса под 3.13 пока не выпущены | Установи Python 3.11/3.12 или сделай отдельный venv (см. `docs/realsense_setup.md`) |

## 7. После обучения

- `best.pth` — лучшая эпоха по `val_miou_fg`. Положи её в любой `train_results/<run_name>/best.pth` и `notebooks/visualize.ipynb` сразу её подхватит.
- `metrics.csv` — все per-epoch числа для сравнения с предыдущими ранами (см. секцию 7 в `docs/progress_report_final.md`).
- `iter_log.jsonl` — гладкая кривая loss/lr на каждый optimizer-шаг (визуализация в ячейке 9 ноутбука).
- `resolved_config.yaml` — копия применённого конфига; пригодится, чтобы воспроизвести ран.
