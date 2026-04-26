"""Training engine with AMP + gradient accumulation."""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

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
    metrics_csv: str = "metrics.csv"  # appended every epoch under save_dir


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
        self._csv_keys: list[str] = []  # columns lazily expand as new metrics show up

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str, *, load_optim: bool = True) -> None:
        """Restore model + (optionally) optimizer / scheduler / scaler / state.

        ``load_optim=False`` is useful for fine-tuning from a checkpoint while
        starting a fresh optimizer schedule.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
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
        for step, batch in enumerate(self.train_loader):
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

        # Final partial accumulation flush
        # (drop the partial gradients to keep step boundaries clean)
        return meters.avg()

    # ------------------------------------------------------------------
    def fit(self) -> TrainState:
        start_epoch = self.state.epoch if self.state.global_step > 0 else 0
        for epoch in range(start_epoch, self.cfg.epochs):
            self.state.epoch = epoch
            train_metrics = self.train_one_epoch()
            self.logger.info(f"[epoch {epoch}] train: " +
                             " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

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
            if self.cfg.save_every_epoch:
                self.save_checkpoint(f"epoch_{epoch:03d}.pth", val_metrics)
            self._append_metrics_row(epoch, train_metrics, val_metrics)
        return self.state

    # ------------------------------------------------------------------
    def _append_metrics_row(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        if not self._csv_path:
            return
        row = {"epoch": epoch, "global_step": self.state.global_step,
               "lr": self.optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        # expand the column set if a new metric appeared this epoch (rare).
        for k in row:
            if k not in self._csv_keys:
                self._csv_keys.append(k)
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_keys, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str, extra: dict) -> None:
        path = os.path.join(self.cfg.save_dir, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": self.state.__dict__,
            "extra": extra,
        }, path)
