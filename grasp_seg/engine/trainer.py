"""Training engine with AMP + gradient accumulation."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TextIO

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..utils.logger import get_logger
from ..utils.meters import MeterDict


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -math.inf
    best_epoch: int = -1


@dataclass
class TrainerConfig:
    epochs: int = 80
    accum_steps: int = 4
    lr: float = 1e-2
    weight_decay: float = 5e-4
    momentum: float = 0.9
    optimizer: str = "sgd"          # "sgd" | "adamw"
    scheduler: str = "poly"         # "poly" | "cosine" | "none"
    poly_power: float = 0.9
    warmup_epochs: float = 1.0
    grad_clip: float = 0.0
    log_interval: int = 50
    save_dir: str = "outputs/run"
    eval_every: int = 1
    target_metric: str = "miou_fg"  # which metric guides best-checkpointing
    save_every_epoch: bool = True   # also write epoch_NNN.pth alongside last/best
    save_every_n_epochs: int = 0    # if >0, only write epoch_NNN.pth every N epochs (overrides save_every_epoch). best.pth + last.pth still update every epoch.
    metrics_csv: str = "metrics.csv"  # appended every epoch under save_dir
    iter_log_path: str = ""         # if non-empty, write one JSONL line per optimizer step into this file (under save_dir if relative)
    profile_timing: bool = True     # measure data/forward/backward wall time (tiny overhead: one torch.cuda.synchronize per step)
    profile_gpu: bool = True        # log per-epoch GPU memory (allocated/peak) and utilization%


def _build_optimizer(model: nn.Module, cfg: TrainerConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def _build_scheduler(opt, cfg: TrainerConfig, total_steps: int):
    warm_steps = int(cfg.warmup_epochs * (total_steps / max(cfg.epochs, 1)))
    warm_steps = max(warm_steps, 1)

    if cfg.scheduler == "poly":
        def lr_lambda(step: int) -> float:
            if step < warm_steps:
                return step / warm_steps
            t = (step - warm_steps) / max(total_steps - warm_steps, 1)
            return max(1.0 - t, 0.0) ** cfg.poly_power
    elif cfg.scheduler == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warm_steps:
                return step / warm_steps
            t = (step - warm_steps) / max(total_steps - warm_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        def lr_lambda(step: int) -> float:
            return 1.0

    return LambdaLR(opt, lr_lambda)


def _move_target(target, device):
    if isinstance(target, dict):
        return {k: v.to(device, non_blocking=True) for k, v in target.items()}
    return target.to(device, non_blocking=True)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: TrainerConfig,
        device: torch.device,
        evaluate_fn: Optional[Callable] = None,
        amp: bool = True,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.evaluate_fn = evaluate_fn
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

        self.optimizer = _build_optimizer(model, cfg)
        steps_per_epoch = max(len(train_loader) // max(cfg.accum_steps, 1), 1)
        total_steps = steps_per_epoch * cfg.epochs
        self.scheduler = _build_scheduler(self.optimizer, cfg, total_steps)

        self.state = TrainState()
        self.logger = get_logger("trainer")
        os.makedirs(cfg.save_dir, exist_ok=True)
        self._csv_path = os.path.join(cfg.save_dir, cfg.metrics_csv) if cfg.metrics_csv else None
        # Per-step iteration log (one JSONL line per optimizer step). Useful for
        # post-hoc plotting of loss/lr/timing at sub-epoch granularity without
        # re-running training. Empty path disables it. Relative paths land
        # under save_dir for tidiness.
        self._iter_log_fp: Optional[TextIO] = None
        if cfg.iter_log_path:
            iter_path = cfg.iter_log_path
            if not os.path.isabs(iter_path):
                iter_path = os.path.join(cfg.save_dir, iter_path)
            os.makedirs(os.path.dirname(iter_path) or ".", exist_ok=True)
            # Append mode so resumed runs keep prior history.
            self._iter_log_fp = open(iter_path, "a", encoding="utf-8")
            self._iter_log_path = iter_path
        self._csv_keys: list[str] = []  # columns lazily expand as new metrics show up
        self._csv_rows: list[dict] = []  # all rows in memory; rewritten each append
        if self._csv_path and os.path.exists(self._csv_path):
            # carry forward existing rows so a resumed run keeps the same file
            # and a column-set expansion (e.g. first eval epoch under
            # eval_every>1) can rewrite the header cleanly.
            with open(self._csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    for k in r:
                        if k not in self._csv_keys:
                            self._csv_keys.append(k)
                    self._csv_rows.append(r)

        # Lazy pynvml import for GPU utilization%. Memory metrics work via
        # torch.cuda alone; util% requires NVML which torch already vendors
        # for newer wheels but is not always exposed. If unavailable we just
        # skip the util column — memory stats still appear.
        self._nvml_handle = None
        if self.device.type == "cuda" and bool(getattr(cfg, "profile_gpu", True)):
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                idx = self.device.index if self.device.index is not None else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._pynvml = pynvml
            except Exception as e:
                self.logger.info(f"pynvml unavailable ({type(e).__name__}); GPU util% will not be logged")
                self._pynvml = None

    def _nvml_utilization(self, handle) -> Optional[float]:
        """Return GPU utilization percent (0-100), or None if NVML failed."""
        try:
            return self._pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception:
            return None

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str, *, load_optim: bool = True) -> None:
        """Restore model + (optionally) optimizer / scheduler / scaler / state.

        ``load_optim=False`` is useful for fine-tuning from a checkpoint while
        starting a fresh optimizer schedule.
        """
        ckpt = torch.load(path, map_location=self.device)
        # Checkpoints are always saved from the underlying (unwrapped) module
        # so they are portable between single-GPU and DataParallel runs.
        state = ckpt["model"]
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # Back-compat: older checkpoints saved from a DataParallel-wrapped
        # model carried a 'module.' prefix on every key; strip it if present.
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        target.load_state_dict(state)
        if load_optim:
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
            if "state" in ckpt:
                for k, v in ckpt["state"].items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
        self.logger.info(f"loaded checkpoint from {path} (resume_epoch={self.state.epoch + 1}, "
                         f"global_step={self.state.global_step})")

    # ------------------------------------------------------------------
    def train_one_epoch(self) -> dict:
        self.model.train()
        meters = MeterDict()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(self.cfg.accum_steps, 1)

        # GPU memory / utilization tracking. Memory is read directly from
        # torch.cuda; utilization% comes from pynvml if available (it is the
        # same number nvidia-smi reports). Both are sampled at end of each
        # step so we can average / take the peak across the epoch.
        profile_gpu = bool(getattr(self.cfg, "profile_gpu", True)) and self.device.type == "cuda"
        if profile_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)
        nvml_handle = self._nvml_handle if profile_gpu else None

        # Timing instrumentation. ``data_time`` is measured as wall time from
        # just after the previous step finishes (post-synchronize) to the
        # moment the next batch arrives. ``compute_time`` covers forward +
        # backward + optimizer step. We call cuda.synchronize once per step
        # (~1 ms on A100) so the split is accurate; if profile_timing is
        # disabled the sync is skipped and the raw wall times are still
        # logged for a rough signal.
        profile = bool(getattr(self.cfg, "profile_timing", True))
        cuda_sync = (self.device.type == "cuda") and profile
        t_step_end = time.perf_counter()  # marks end of previous step
        for step, batch in enumerate(self.train_loader):
            t_batch_ready = time.perf_counter()
            x = batch["input"].to(self.device, non_blocking=True)
            y = _move_target(batch["target"], self.device)
            with torch.amp.autocast('cuda', enabled=self.amp):
                pred = self.model(x)
                loss_dict = self.loss_fn(pred, y)
                loss = loss_dict["loss"] / accum

            self.scaler.scale(loss).backward()
            log = {k: float(v.item() if torch.is_tensor(v) else v)
                   for k, v in loss_dict.items() if torch.is_tensor(v) or isinstance(v, (int, float))}
            meters.update(log, n=x.size(0))

            if (step + 1) % accum == 0:
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.state.global_step += 1

                if self.state.global_step % self.cfg.log_interval == 0:
                    avg = meters.avg()
                    lr = self.optimizer.param_groups[0]["lr"]
                    msg = (f"epoch {self.state.epoch} step {self.state.global_step} "
                           f"lr {lr:.4g} | "
                           + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
                    self.logger.info(msg)

                # Per-step JSONL row (independent of console log_interval —
                # captures *every* optimizer step). Loss/timing values are
                # the *current step's* (last logged into MeterDict above).
                if self._iter_log_fp is not None:
                    last_log = {
                        k: float(v.item() if torch.is_tensor(v) else v)
                        for k, v in loss_dict.items()
                        if torch.is_tensor(v) or isinstance(v, (int, float))
                    }
                    row = {
                        "epoch": self.state.epoch,
                        "step": self.state.global_step,
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        **last_log,
                    }
                    self._iter_log_fp.write(json.dumps(row) + "\n")

            # End-of-step sync + timing.
            if cuda_sync:
                torch.cuda.synchronize()
            t_prev = t_step_end
            t_step_end = time.perf_counter()
            data_time = t_batch_ready - t_prev        # dataloader wait
            compute_time = t_step_end - t_batch_ready  # fwd + bwd + optim
            step_time = t_step_end - t_prev
            # Per-sample averaging over the effective batch so numbers stay
            # comparable across different batch / accum_steps settings.
            timing = {
                "data_time_s": data_time,
                "compute_time_s": compute_time,
                "step_time_s": step_time,
                "dataload_fraction": data_time / max(step_time, 1e-9),
            }
            if profile_gpu:
                # bytes -> GB. memory_allocated is the live tensor footprint
                # at this moment, so averaging across steps is meaningful
                # ("typical pressure during the epoch"). gpu_mem_peak_gb is
                # NOT averaged — it is a running max, captured once at the
                # end of the epoch below.
                timing["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                if nvml_handle is not None:
                    util = self._nvml_utilization(nvml_handle)
                    if util is not None:
                        timing["gpu_util_pct"] = float(util)
            meters.update(timing, n=x.size(0))

        # Final partial accumulation flush
        # (drop the partial gradients to keep step boundaries clean)
        result = meters.avg()
        if profile_gpu:
            # Read the actual peak once, after the loop. Putting this through
            # MeterDict would average a monotonically non-decreasing series
            # and underreport the true peak, defeating the OOM-risk check
            # documented in docs/multitask_phase2.md.
            result["gpu_mem_peak_gb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            )
        return result

    # ------------------------------------------------------------------
    def fit(self) -> TrainState:
        # On resume, ``self.state.epoch`` is the index of the *last completed*
        # epoch (it was saved after that epoch's epoch_NNN.pth was written), so
        # training has to continue from epoch+1. Without the +1 we would re-run
        # the already-completed epoch and advance global_step / scheduler past
        # their checkpointed values.
        start_epoch = self.state.epoch + 1 if self.state.global_step > 0 else 0
        for epoch in range(start_epoch, self.cfg.epochs):
            self.state.epoch = epoch
            train_metrics = self.train_one_epoch()
            self.logger.info(f"[epoch {epoch}] train: " +
                             " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))
            # Human-readable bottleneck summary: if >30% of step time is
            # spent waiting on the dataloader, the run is CPU/IO-bound and
            # num_workers / prefetch_factor should be increased.
            if "step_time_s" in train_metrics and "data_time_s" in train_metrics:
                frac = train_metrics.get("dataload_fraction", 0.0)
                verdict = ("GPU-bound (good)" if frac < 0.10
                           else "balanced" if frac < 0.30
                           else "DATALOADER-BOUND (increase num_workers/prefetch)")
                self.logger.info(
                    f"[epoch {epoch}] timing: step={train_metrics['step_time_s']*1000:.1f}ms "
                    f"(data={train_metrics['data_time_s']*1000:.1f}ms, "
                    f"compute={train_metrics['compute_time_s']*1000:.1f}ms, "
                    f"dataload_frac={frac:.2f}) -> {verdict}"
                )
            # Per-epoch GPU memory + utilization summary line. Memory is in GB
            # (allocated avg across steps + peak); utilization is the average
            # nvidia-smi % over the epoch when pynvml is available.
            if "gpu_mem_alloc_gb" in train_metrics:
                util_str = (f" util={train_metrics['gpu_util_pct']:.0f}%"
                            if "gpu_util_pct" in train_metrics else " util=N/A")
                self.logger.info(
                    f"[epoch {epoch}] gpu:{util_str} "
                    f"mem_alloc={train_metrics['gpu_mem_alloc_gb']:.2f}GB "
                    f"mem_peak={train_metrics['gpu_mem_peak_gb']:.2f}GB"
                )

            val_metrics: dict = {}
            if self.evaluate_fn is not None and self.val_loader is not None \
                    and (epoch + 1) % self.cfg.eval_every == 0:
                val_metrics = self.evaluate_fn(self.model, self.val_loader, self.device, self.amp)
                self.logger.info(f"[epoch {epoch}] val: " +
                                 " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                target = val_metrics.get(self.cfg.target_metric, -math.inf)
                if target > self.state.best_metric:
                    self.state.best_metric = target
                    self.state.best_epoch = epoch
                    self.save_checkpoint("best.pth", val_metrics)

            self.save_checkpoint("last.pth", val_metrics)
            # Per-epoch checkpoints. ``save_every_n_epochs > 0`` takes
            # precedence over ``save_every_epoch`` so long runs (e.g. 150
            # epochs) don't fill the disk with 150 checkpoints.
            if self.cfg.save_every_n_epochs > 0:
                if (epoch + 1) % self.cfg.save_every_n_epochs == 0 \
                        or epoch == self.cfg.epochs - 1:
                    self.save_checkpoint(f"epoch_{epoch:03d}.pth", val_metrics)
            elif self.cfg.save_every_epoch:
                self.save_checkpoint(f"epoch_{epoch:03d}.pth", val_metrics)
            self._append_metrics_row(epoch, train_metrics, val_metrics)
            # Flush per-step log so the file is up-to-date on disk if the
            # process is killed mid-run (or the kernel restarts).
            if self._iter_log_fp is not None:
                self._iter_log_fp.flush()
        if self._iter_log_fp is not None:
            self._iter_log_fp.close()
            self._iter_log_fp = None
        return self.state

    # ------------------------------------------------------------------
    def _append_metrics_row(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        if not self._csv_path:
            return
        row: dict = {"epoch": epoch, "global_step": self.state.global_step,
                     "lr": self.optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        # expand the column set if a new metric appeared this epoch (e.g.
        # eval_every>1 means val_* columns first show up several epochs in)
        for k in row:
            if k not in self._csv_keys:
                self._csv_keys.append(k)
        self._csv_rows.append(row)
        # Rewrite the whole CSV (≤ epochs rows, trivial overhead) so the
        # header always reflects the current column set even after expansion.
        with open(self._csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_keys, extrasaction="ignore")
            w.writeheader()
            for r in self._csv_rows:
                w.writerow(r)

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str, extra: dict) -> None:
        path = os.path.join(self.cfg.save_dir, name)
        # Save the unwrapped module state_dict so checkpoints round-trip
        # between single-GPU and DataParallel sessions.
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save({
            "model": target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": self.state.__dict__,
            "extra": extra,
        }, path)
