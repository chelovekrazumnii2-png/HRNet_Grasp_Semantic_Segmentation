"""Segmentation losses for binary and multi-class angle masks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceBCELoss(nn.Module):
    """BCEWithLogits + soft Dice on a 1-channel logit map."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 smooth: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.dw = dice_weight
        self.bw = bce_weight
        self.smooth = smooth
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(),
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
        )
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(-1, -2))
        denom = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
        dice = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice = dice.mean()
        total = self.bw * bce + self.dw * dice
        return {"loss": total, "bce": bce.detach(), "dice": dice.detach()}


class MultiClassCEDiceLoss(nn.Module):
    """CrossEntropy + multi-class Dice for angle-bin segmentation.

    The Dice term ignores the background class to reduce dominance of the
    very large background area.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1.0,
        ignore_index: int = -100,
        ignore_background_in_dice: bool = True,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dw = dice_weight
        self.cw = ce_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ignore_bg = ignore_background_in_dice
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        ce = F.cross_entropy(
            logits, target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
        prob = F.softmax(logits, dim=1)
        # one-hot target
        valid = (target != self.ignore_index)
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        oh = oh * valid.unsqueeze(1)
        if self.ignore_bg:
            prob = prob[:, 1:]
            oh = oh[:, 1:]
        inter = (prob * oh).sum(dim=(0, 2, 3))
        denom = prob.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        dice_per_class = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice = dice_per_class.mean()
        total = self.cw * ce + self.dw * dice
        return {"loss": total, "ce": ce.detach(), "dice": dice.detach()}
