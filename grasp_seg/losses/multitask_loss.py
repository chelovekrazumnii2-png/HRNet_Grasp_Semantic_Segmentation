"""Multi-task GG-CNN-style loss (pos + cos2t + sin2t + width)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskGraspLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 1.0,
        cos_weight: float = 1.0,
        sin_weight: float = 1.0,
        width_weight: float = 1.0,
        smooth: float = 1.0,
        bce_pos_weight: float | None = None,
    ):
        super().__init__()
        self.pw = pos_weight
        self.cw = cos_weight
        self.sw = sin_weight
        self.ww = width_weight
        self.smooth = smooth
        self.bce_pos_weight = (
            torch.tensor([bce_pos_weight]) if bce_pos_weight is not None else None
        )

    def forward(self, pred: dict, target: dict) -> dict:
        # pos: BCE + Dice
        pos_logits = pred["pos"]
        pos_t = target["pos"].float()
        pw = self.bce_pos_weight.to(pos_logits.device) if self.bce_pos_weight is not None else None
        pos_bce = F.binary_cross_entropy_with_logits(pos_logits, pos_t, pos_weight=pw)
        pos_prob = torch.sigmoid(pos_logits)
        inter = (pos_prob * pos_t).sum(dim=(-1, -2))
        denom = pos_prob.sum(dim=(-1, -2)) + pos_t.sum(dim=(-1, -2))
        pos_dice = (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

        # cos / sin / width: masked MSE on positive pixels
        mask = pos_t > 0.5
        if mask.any():
            cos_loss = F.mse_loss(pred["cos2t"][mask], target["cos2t"][mask])
            sin_loss = F.mse_loss(pred["sin2t"][mask], target["sin2t"][mask])
            # target["width"] is already normalised to [0, 1] in the rasteriser
            width_pred = torch.sigmoid(pred["width"])
            width_loss = F.mse_loss(width_pred[mask], target["width"][mask])
        else:
            cos_loss = pred["cos2t"].sum() * 0.0
            sin_loss = pred["sin2t"].sum() * 0.0
            width_loss = pred["width"].sum() * 0.0

        total = (
            self.pw * (pos_bce + pos_dice)
            + self.cw * cos_loss
            + self.sw * sin_loss
            + self.ww * width_loss
        )
        return {
            "loss": total,
            "pos_bce": pos_bce.detach(),
            "pos_dice": pos_dice.detach(),
            "cos": cos_loss.detach(),
            "sin": sin_loss.detach(),
            "width": width_loss.detach(),
        }
