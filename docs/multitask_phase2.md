# Phase 2a — multitask experiment

This document describes the **multitask** training mode added in PR #12 (Plan B
from `docs/progress_report_v2.md` §6, Variant B). It is a contingency
experiment we plan to launch if the running `mask_mode=angle` Colab A100 run
plateaus below the **mIoU_fg ≥ 0.40** target.

## TL;DR

```
# instead of configs/default.yaml
python tools/train.py --config configs/multitask.yaml \
    dataset.splits_path=splits/jacquard_v2.json \
    trainer.batch_size=48 trainer.accum_steps=1 \
    trainer.save_dir=/path/to/runs/hrnet_w18_rgbd_multitask
```

Or in `notebooks/colab_train.ipynb` cell 0, set
`CONFIG_FILE = "configs/multitask.yaml"` and a fresh `RUN_NAME` so the new
experiment writes into its own Drive folder.

## What is multitask mode?

Instead of producing a 19-class mask (background + 18 angle bins), the model
outputs **four dense maps**:

| Channel  | Activation       | Loss               | Meaning                                        |
| -------- | ---------------- | ------------------ | ---------------------------------------------- |
| `pos`    | sigmoid          | BCE + Dice         | Foreground probability (replaces 19-class fg). |
| `cos2t`  | identity         | MSE on positives   | cos(2θ) of grasp angle.                        |
| `sin2t`  | identity         | MSE on positives   | sin(2θ) of grasp angle.                        |
| `width`  | sigmoid          | MSE on positives   | Gripper opening / 150 px (clipped to [0, 1]).  |

This is the **GG-CNN** parametrisation. The angle is recovered at inference
via `θ = 0.5 * atan2(sin2t, cos2t)`, which keeps θ continuous and avoids the
discretisation artefacts of the 18-bin classification head.

## Why this might beat angle mode

- **Continuous angle target** removes the sharp class boundaries of the angle
  mode at every 10° step. Two grasps at 19° and 21° have the same target in
  multitask mode but different classes (1 vs 2) in angle mode.
- **Width is regressed explicitly**, not implicitly via mask shape — so the
  output is directly usable as a grasp pose without polygon post-processing.
- **GG-CNN literature** consistently reports stronger pixel-wise grasp
  detection than 18-bin classification heads on Cornell / Jacquard.

The expected gain for our setup is **+5–10 % mIoU_fg_ang** vs the angle run
(see `docs/progress_report_v2.md` §6 Variant B). If the baseline finishes at
~0.33, multitask should reach ~0.38–0.43.

## Comparing multitask vs angle metrics fairly

This is the part where we have to be careful.

`evaluate_angle` reports `miou_fg` = mean IoU averaged over angle bins 1..18.
`evaluate_multitask` reports `miou_fg` from a **binary** ConfusionMeter
(foreground vs background only). Those two numbers are **not** comparable —
the multitask `miou_fg` will look much higher because foreground-vs-background
is an easier task than 18-way classification.

To make the comparison apples-to-apples we added `miou_fg_ang` /
`dice_fg_ang` to `evaluate_multitask`: at eval time the predicted cos2t /
sin2t are converted back into 18 bins and a 19-class ConfusionMeter is
maintained alongside the binary one. **This is the metric you compare to the
angle run's `val_miou_fg`.**

| Comparison purpose                               | Use this metric        |
| ------------------------------------------------ | ---------------------- |
| Best.pth selection, fg-mask quality              | `val_miou_fg`          |
| Apples-to-apples vs angle-mode `val_miou_fg`     | `val_miou_fg_ang`      |
| How well the cos / sin regression is converging  | `val_ang_mae_deg`      |

`val_ang_mae_deg` is the mean absolute angular error in degrees on positive
pixels (with mod-180° wrap, since grasps are π-symmetric). A useful target:
**< 15°** is competitive, **< 10°** is strong.

## What was changed in the codebase

- **Added** `configs/multitask.yaml` — clones default but flips `mask_mode`
  and points `save_dir` at `outputs/multitask` so it cannot stomp on the
  baseline run's checkpoints.
- **Added** `miou_fg_ang`, `dice_fg_ang`, `ang_mae_deg` to
  `grasp_seg/engine/evaluator.py::evaluate_multitask`.
- **Updated** `notebooks/colab_train.ipynb` — new `CONFIG_FILE` knob in cell 0
  and the train cell uses it instead of hardcoding `configs/default.yaml`.

The training entrypoint (`tools/train.py`), the dataset, the loss
(`MultiTaskGraspLoss`) and the multitask model head (`HRNetMultiTask`) were
already wired in earlier PRs (#1) — no functional changes there.

## Smoke check

`scripts/smoke_test.py` now runs all three modes (binary / angle / multitask)
end-to-end on synthetic Jacquard-like samples. `multitask` also exercises the
new evaluator metrics. Run it with:

```
python scripts/smoke_test.py
```

A clean run prints `ALL MODES OK` after exercising each mode for two epochs
with checkpoint + resume.

## Timing / bottleneck diagnostics (built-in)

Every epoch logs a one-line timing summary, and four new columns land in
`metrics.csv`:

- `train_data_time_s` — wall-clock per step waiting on the DataLoader.
- `train_compute_time_s` — forward + backward + optimizer step.
- `train_step_time_s` — full step (≈ sum of the two above).
- `train_dataload_fraction` — `data_time / step_time`. **Our rule of thumb:**

  | fraction | verdict                                              |
  | -------- | ---------------------------------------------------- |
  | `< 0.10` | GPU-bound — DataLoader is not a bottleneck.          |
  | `0.10–0.30` | balanced — slight headroom, usually fine.         |
  | `> 0.30` | **DataLoader-bound** — raise `num_workers` or `prefetch_factor`, enable `persistent_workers`, or check disk IO. |

Per-epoch log line looks like:

```
[epoch 3] timing: step=612.4ms (data=58.1ms, compute=554.3ms, dataload_frac=0.09) -> GPU-bound (good)
```

Overhead: one `torch.cuda.synchronize()` per optimizer step (~1 ms on A100). Turn
it off with `trainer.profile_timing=false` if you want to benchmark raw
throughput without sync cost.

No parallel notebook cells or terminal access required — the numbers are
captured inline and survive in `metrics.csv` for post-hoc analysis.

## GPU memory + utilization (built-in)

Alongside the timing profiler the trainer also samples GPU memory and
utilization once per step and emits a per-epoch summary line plus three new
columns in `metrics.csv`:

- `train_gpu_mem_alloc_gb` — average live tensor footprint per step (GB).
  Useful to see how close we are to OOM with the chosen `batch_size`.
- `train_gpu_mem_peak_gb` — running peak across the whole epoch
  (`torch.cuda.max_memory_allocated`, reset at epoch start).
- `train_gpu_util_pct` — `nvmlDeviceGetUtilizationRates(...).gpu` —
  this is the same number `nvidia-smi` reports as `Volatile GPU-Util`.
  Only present when `pynvml` imports successfully (it is bundled with the
  cuda PyTorch wheels). When unavailable the column is silently dropped
  and the log line shows `util=N/A`.

Per-epoch log line:

```
[epoch 3] gpu: util=87% mem_alloc=14.32GB mem_peak=15.10GB
```

Heuristics:

| metric | what to look for |
| ------ | ---------------- |
| `gpu_util_pct < 60%` | GPU starved — confirm with `dataload_fraction`; if it's high you are CPU/IO-bound, otherwise your batch / model is too small to saturate. |
| `gpu_mem_peak_gb / total` close to 1.0 | OOM risk — drop `batch_size` or enable `accum_steps`. |
| `gpu_mem_peak_gb` ≪ available | Headroom for a larger `batch_size` (often the cheapest throughput win). |

Turn it off with `trainer.profile_gpu=false`. Memory reads are free
(counter dereferences in the CUDA caching allocator); util% calls into
NVML once per step (~10 µs). No measurable overhead at our step times
(>500 ms on A100).
