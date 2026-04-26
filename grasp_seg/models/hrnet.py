"""HRNet-W18 + HRNetV2-style segmentation/multitask head.

Backbone is loaded via ``timm`` (``hrnet_w18`` or ``hrnet_w18_small_v2``)
with multi-scale feature outputs. The first conv is patched to accept
1, 3, or 4 input channels — the 4th channel (depth) is initialised from
the mean of the original RGB weights.
"""
from __future__ import annotations

from typing import List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _patch_first_conv(model: nn.Module, in_channels: int) -> None:
    """Replace the first conv layer to accept ``in_channels`` channels."""
    # timm's HRNet has ``conv1`` (3x3 stride 2, 3 -> 64) at the model root.
    if not hasattr(model, "conv1"):
        raise AttributeError("Backbone has no `conv1` attribute; cannot patch input channels.")
    old = model.conv1
    if old.in_channels == in_channels:
        return
    new = nn.Conv2d(
        in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        w = old.weight  # (out, 3, k, k)
        if in_channels == 3:
            new.weight.copy_(w)
        elif in_channels == 1:
            new.weight.copy_(w.mean(dim=1, keepdim=True))
        elif in_channels >= 3:
            extra = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
            new.weight.copy_(torch.cat([w, extra], dim=1))
        else:
            raise ValueError(f"Unsupported in_channels {in_channels}")
        if new.bias is not None:
            new.bias.zero_()
    model.conv1 = new


class HRNetSeg(nn.Module):
    """HRNet backbone + concatenated multi-scale segmentation head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 4,
        backbone: str = "hrnet_w18",
        pretrained: bool = True,
        head_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.num_classes = num_classes

        bb = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        _patch_first_conv(bb, in_channels)
        self.backbone = bb

        feat_chs: List[int] = list(self.backbone.feature_info.channels())
        sum_ch = sum(feat_chs)
        self.fuse = nn.Sequential(
            nn.Conv2d(sum_ch, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(head_channels, num_classes, kernel_size=1)

    def _fuse_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        H, W = feats[0].shape[-2:]
        ups = [feats[0]]
        for f in feats[1:]:
            ups.append(F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(ups, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._fuse_features(x)
        logits = self.classifier(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class HRNetMultiTask(nn.Module):
    """HRNet backbone + 4-channel head (pos / cos2t / sin2t / width)."""

    def __init__(
        self,
        in_channels: int = 4,
        backbone: str = "hrnet_w18",
        pretrained: bool = True,
        head_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # We share most of the seg architecture and re-purpose ``classifier``
        # as a 4-channel regression head; ``forward`` returns a dict.
        self._seg = HRNetSeg(
            num_classes=4,
            in_channels=in_channels,
            backbone=backbone,
            pretrained=pretrained,
            head_channels=head_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        out = self._seg(x)
        return {
            "pos": out[:, 0],     # logits — apply sigmoid in loss / inference
            "cos2t": out[:, 1],
            "sin2t": out[:, 2],
            "width": out[:, 3],   # logits — apply sigmoid in loss / inference
        }


def build_model(cfg: dict) -> nn.Module:
    """Factory: build a model from a config dict.

    Required keys: ``mask_mode``, ``backbone``, ``in_channels``,
    ``num_angle_bins`` (only for ``mask_mode == 'angle'``), ``pretrained``.
    """
    mask_mode = cfg["mask_mode"]
    backbone = cfg.get("backbone", "hrnet_w18")
    in_channels = cfg["in_channels"]
    pretrained = cfg.get("pretrained", True)

    if mask_mode == "binary":
        num_classes = 1
        return HRNetSeg(num_classes, in_channels, backbone, pretrained)
    if mask_mode == "angle":
        # K bins + background
        num_classes = cfg["num_angle_bins"] + 1
        return HRNetSeg(num_classes, in_channels, backbone, pretrained)
    if mask_mode == "multitask":
        return HRNetMultiTask(in_channels, backbone, pretrained)
    raise ValueError(f"Unknown mask_mode {mask_mode!r}")
