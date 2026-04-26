"""Validation routines for the three mask modes."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.meters import ConfusionMeter, MeterDict


def _to_device(t, device):
    if isinstance(t, dict):
        return {k: v.to(device, non_blocking=True) for k, v in t.items()}
    return t.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device,
                    amp: bool = True, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    cm = ConfusionMeter(num_classes=2, device=device)
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp and device.type == "cuda"):
            logits = model(x)
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        pred = (torch.sigmoid(logits) > threshold).long()
        cm.update(pred, y.long())
    metrics = cm.compute()
    return metrics


@torch.no_grad()
def evaluate_angle(model: nn.Module, loader: DataLoader, device: torch.device,
                   amp: bool = True) -> Dict[str, float]:
    model.eval()
    num_classes = None
    cm = None
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp and device.type == "cuda"):
            logits = model(x)
        if cm is None:
            num_classes = logits.shape[1]
            cm = ConfusionMeter(num_classes=num_classes, device=device)
        pred = logits.argmax(dim=1)
        cm.update(pred, y)
    return cm.compute() if cm is not None else {}


@torch.no_grad()
def evaluate_multitask(model: nn.Module, loader: DataLoader, device: torch.device,
                       amp: bool = True, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    cm = ConfusionMeter(num_classes=2, device=device)
    meters = MeterDict()
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = _to_device(batch["target"], device)
        with torch.amp.autocast('cuda', enabled=amp and device.type == "cuda"):
            pred = model(x)
        pos_pred = (torch.sigmoid(pred["pos"]) > threshold).long()
        cm.update(pos_pred, y["pos"].long())
        # angle errors on positive pixels
        m = y["pos"] > 0.5
        if m.any():
            cos_err = ((pred["cos2t"][m] - y["cos2t"][m]) ** 2).mean().item()
            sin_err = ((pred["sin2t"][m] - y["sin2t"][m]) ** 2).mean().item()
            meters.update({"cos_mse": cos_err, "sin_mse": sin_err}, n=int(m.sum().item()))
    out = cm.compute()
    out.update(meters.avg())
    return out
