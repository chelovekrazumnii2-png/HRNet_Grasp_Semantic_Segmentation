"""Lightweight average meters and segmentation metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class MeterDict:
    def __init__(self):
        self._meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self._meters[k].update(float(v), n=n)

    def avg(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------
class ConfusionMeter:
    """Streaming confusion matrix for K-class segmentation."""

    def __init__(self, num_classes: int, device: str | torch.device = "cpu"):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=self.device)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            p = pred.flatten().to(self.device)
            t = target.flatten().to(self.device)
            mask = (t >= 0) & (t < self.num_classes)
            idx = self.num_classes * t[mask] + p[mask]
            bins = torch.bincount(idx, minlength=self.num_classes ** 2)
            self.mat += bins.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        mat = self.mat.float()
        tp = mat.diag()
        fp = mat.sum(dim=0) - tp
        fn = mat.sum(dim=1) - tp
        iou = tp / (tp + fp + fn).clamp(min=1.0)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1.0)
        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        # foreground-only averages (ignore class 0 = background)
        fg = slice(1, self.num_classes) if self.num_classes > 1 else slice(0, 1)
        return {
            "miou": iou.mean().item(),
            "miou_fg": iou[fg].mean().item(),
            "dice": dice.mean().item(),
            "dice_fg": dice[fg].mean().item(),
            "precision_fg": precision[fg].mean().item(),
            "recall_fg": recall[fg].mean().item(),
        }
