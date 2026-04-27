"""Validation routines for the three mask modes."""
from __future__ import annotations

import math
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
                       amp: bool = True, threshold: float = 0.5,
                       num_angle_bins: int = 18) -> Dict[str, float]:
    """Validation for ``mask_mode='multitask'``.

    Returns a metric dict with three groups of keys:

    * Binary foreground vs background (from ``ConfusionMeter`` on ``pos``):
      ``miou``, ``miou_fg``, ``dice``, ``dice_fg``, ``precision_fg``,
      ``recall_fg``. ``miou_fg`` here is the single-class fg IoU used by
      ``Trainer`` to pick best.pth.
    * Angle-binned fg IoU computed by recovering the predicted grasp angle
      from cos2t/sin2t and discretising it into ``num_angle_bins`` bins
      identical to ``mask_mode='angle'``: ``miou_fg_ang``, ``dice_fg_ang``.
      These are directly comparable to ``miou_fg`` reported by the
      ``angle``-mode run (configs/default.yaml).
    * Per-pixel regression sanity metrics on positive pixels:
      ``cos_mse``, ``sin_mse``, ``ang_mae_deg`` (mean absolute angle error
      with mod-180° wrap).
    """
    model.eval()
    cm_bin = ConfusionMeter(num_classes=2, device=device)
    cm_ang = ConfusionMeter(num_classes=num_angle_bins + 1, device=device)
    meters = MeterDict()
    bin_w_rad = math.pi / num_angle_bins  # 180° / K
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = _to_device(batch["target"], device)
        with torch.amp.autocast('cuda', enabled=amp and device.type == "cuda"):
            pred = model(x)
        pos_pred = (torch.sigmoid(pred["pos"]) > threshold).long()
        pos_target = y["pos"].long()
        cm_bin.update(pos_pred, pos_target)

        # ------------------------------------------------------------------
        # Recover predicted angle bin from cos2t / sin2t so we can compute
        # the same multi-class fg-IoU that mask_mode='angle' reports.
        # 2θ = atan2(sin2t, cos2t)  →  θ ∈ [-π/2, π/2)  →  mod π  →  bin.
        # ------------------------------------------------------------------
        theta_p = 0.5 * torch.atan2(pred["sin2t"].float(), pred["cos2t"].float())
        theta_p_mod = theta_p % math.pi
        bin_p = (theta_p_mod / bin_w_rad).long().clamp_(0, num_angle_bins - 1) + 1
        ang_pred = bin_p * pos_pred  # background pixels collapse to class 0

        theta_t = 0.5 * torch.atan2(y["sin2t"].float(), y["cos2t"].float())
        theta_t_mod = theta_t % math.pi
        bin_t = (theta_t_mod / bin_w_rad).long().clamp_(0, num_angle_bins - 1) + 1
        ang_target = bin_t * pos_target

        cm_ang.update(ang_pred, ang_target)

        # angle errors on positive pixels
        m = y["pos"] > 0.5
        if m.any():
            cos_err = ((pred["cos2t"][m] - y["cos2t"][m]) ** 2).mean().item()
            sin_err = ((pred["sin2t"][m] - y["sin2t"][m]) ** 2).mean().item()
            # Mean absolute angular error, wrapped to [0, π/2]: grasps are
            # π-symmetric, so the worst possible error is π/2.
            ang_diff = (theta_p - theta_t).abs()
            ang_diff = torch.minimum(ang_diff, math.pi - ang_diff)
            mae_deg = math.degrees(ang_diff[m].float().mean().item())
            meters.update({"cos_mse": cos_err, "sin_mse": sin_err,
                           "ang_mae_deg": mae_deg}, n=int(m.sum().item()))
    out = cm_bin.compute()
    ang_metrics = cm_ang.compute()
    # Surface only the foreground angle-IoU / Dice; the multi-class "miou"
    # average pulls in the dominant background class and is uninformative.
    out["miou_fg_ang"] = ang_metrics["miou_fg"]
    out["dice_fg_ang"] = ang_metrics["dice_fg"]
    out.update(meters.avg())
    return out
