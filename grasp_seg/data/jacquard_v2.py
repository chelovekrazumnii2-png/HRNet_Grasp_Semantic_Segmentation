"""Jacquard V2 dataset for grasp segmentation.

Each item returns a dict with:

- ``input``  : float32 ``(C, H, W)`` tensor (RGB / D / RGB-D depending on mode)
- ``target`` : torch.Tensor — content depends on ``mask_mode``:
    * ``binary``  → ``(H, W)`` float32 in {0, 1}
    * ``angle``   → ``(H, W)`` int64 with values in ``[0, num_angle_bins]``
                    (0 = background, 1..K = angle bins of width 180/K°).
    * ``multitask`` → dict with keys ``pos`` (HxW float32), ``cos2t``, ``sin2t``,
                    ``width`` (all HxW float32).
- ``meta``   : dict with ``grasp_file`` and ``object_id``.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .grasp_rect import Grasp, load_jacquard_grasps, rasterize_grasp_mask
from .transforms import AugConfig, apply_augmentations


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]


def _load_depth(path: str) -> np.ndarray:
    depth = tifffile.imread(path).astype(np.float32)
    if depth.ndim != 2:
        depth = depth.squeeze()
    return depth


def _normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Robust per-image normalisation to [0, 1] using 1/99 percentile clipping."""
    finite = np.isfinite(depth) & (depth > 0)
    if finite.sum() == 0:
        return np.zeros_like(depth, dtype=np.float32)
    vals = depth[finite]
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-6
    out = (depth - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out[~finite] = 0.0
    return out


@dataclass
class DatasetConfig:
    image_size: int = 384
    input_mode: str = "rgbd"        # "rgb" | "depth" | "rgbd"
    mask_mode: str = "angle"        # "binary" | "angle" | "multitask"
    num_angle_bins: int = 18
    length_scale: float = 1.0 / 3.0
    rgb_mean: tuple = (0.485, 0.456, 0.406)
    rgb_std: tuple = (0.229, 0.224, 0.225)
    depth_mean: float = 0.5
    depth_std: float = 0.25
    use_stereo_depth: bool = False  # if True, prefer stereo over perfect depth


class JacquardV2GraspSeg(Dataset):
    def __init__(
        self,
        grasp_files: List[str],
        cfg: DatasetConfig,
        aug: Optional[AugConfig] = None,
        is_training: bool = False,
    ):
        self.grasp_files = list(grasp_files)
        self.cfg = cfg
        self.aug = aug if aug is not None else AugConfig(enable=False)
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.grasp_files)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def _paths(self, grasp_file: str):
        rgb_path = grasp_file.replace("_grasps.txt", "_RGB.png")
        if self.cfg.use_stereo_depth or (
            self.is_training and self.aug.enable and self.aug.use_stereo_depth_p > 0
            and random.random() < self.aug.use_stereo_depth_p
        ):
            depth_path = grasp_file.replace("_grasps.txt", "_stereo_depth.tiff")
            if not os.path.exists(depth_path):
                depth_path = grasp_file.replace("_grasps.txt", "_perfect_depth.tiff")
        else:
            depth_path = grasp_file.replace("_grasps.txt", "_perfect_depth.tiff")
        return rgb_path, depth_path

    def _load(self, grasp_file: str):
        rgb_path, depth_path = self._paths(grasp_file)
        rgb = _load_rgb(rgb_path) if self.cfg.input_mode != "depth" else None
        depth_raw = _load_depth(depth_path)
        depth = _normalise_depth(depth_raw)
        if rgb is None:
            # need a placeholder of the same shape for resizing convenience
            rgb = np.zeros((*depth.shape, 3), dtype=np.float32)

        # Resize to the configured square input size
        S = self.cfg.image_size
        H0, W0 = depth.shape
        if (H0, W0) != (S, S):
            rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (S, S), interpolation=cv2.INTER_LINEAR)
            scale = float(S) / float(H0)
        else:
            scale = 1.0

        grasps = load_jacquard_grasps(grasp_file, scale=scale)
        return rgb, depth, grasps

    # ------------------------------------------------------------------
    # Tensor assembly
    # ------------------------------------------------------------------
    def _input_tensor(self, rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        mean = np.asarray(self.cfg.rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.cfg.rgb_std, dtype=np.float32).reshape(1, 1, 3)
        rgb_n = (rgb - mean) / std
        d_n = (depth - self.cfg.depth_mean) / self.cfg.depth_std
        if self.cfg.input_mode == "rgb":
            arr = rgb_n.transpose(2, 0, 1)
        elif self.cfg.input_mode == "depth":
            arr = d_n[None]
        elif self.cfg.input_mode == "rgbd":
            arr = np.concatenate([rgb_n.transpose(2, 0, 1), d_n[None]], axis=0)
        else:
            raise ValueError(f"Unknown input_mode {self.cfg.input_mode!r}")
        return torch.from_numpy(arr.astype(np.float32))

    def _target(self, grasps: List[Grasp]) -> dict:
        return rasterize_grasp_mask(
            grasps,
            shape=(self.cfg.image_size, self.cfg.image_size),
            mode=self.cfg.mask_mode,
            num_angle_bins=self.cfg.num_angle_bins,
            length_scale=self.cfg.length_scale,
        )

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        grasp_file = self.grasp_files[idx]
        rgb, depth, grasps = self._load(grasp_file)
        if self.is_training:
            rgb, depth, grasps = apply_augmentations(rgb, depth, grasps, self.aug)

        target_np = self._target(grasps)
        if self.cfg.mask_mode == "binary":
            target = torch.from_numpy(target_np["mask"].astype(np.float32))
        elif self.cfg.mask_mode == "angle":
            target = torch.from_numpy(target_np["mask"].astype(np.int64))
        elif self.cfg.mask_mode == "multitask":
            target = {k: torch.from_numpy(v) for k, v in target_np.items()}
        else:
            raise ValueError(self.cfg.mask_mode)

        return {
            "input": self._input_tensor(rgb, depth),
            "target": target,
            "meta": {
                "grasp_file": grasp_file,
                "object_id": os.path.basename(os.path.dirname(grasp_file)),
            },
        }


def collate_fn(batch):
    """Custom collate that handles dict-typed targets (multitask mode)."""
    inputs = torch.stack([b["input"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    first = batch[0]["target"]
    if isinstance(first, dict):
        target = {k: torch.stack([b["target"][k] for b in batch], dim=0) for k in first}
    else:
        target = torch.stack([b["target"] for b in batch], dim=0)
    return {"input": inputs, "target": target, "meta": metas}
