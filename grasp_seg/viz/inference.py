"""Helpers to load a trained checkpoint and run a single forward pass.

The visualisation pipeline only needs *minimal* inference — there is no
DataLoader, batching, AMP scaling, etc. We expose a tiny ``ModelRunner``
that: (1) parses ``resolved_config.yaml`` to figure out the right model
factory + input layout, (2) loads weights from any ``epoch_NNN.pth`` /
``best.pth`` / ``last.pth``, (3) accepts numpy ``rgb`` and ``depth``
images and returns the ready-to-decode prediction maps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import yaml

from ..data.jacquard_v2 import DatasetConfig
from ..models import build_model


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _input_channels(input_mode: str) -> int:
    return {"rgb": 3, "depth": 1, "rgbd": 4}[input_mode]


def load_resolved_config(run_dir: str) -> Dict:
    """Read ``resolved_config.yaml`` from a training output directory."""
    path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"resolved_config.yaml not found in {run_dir!r}. "
            "Pass the directory written by the trainer (it contains best.pth "
            "and metrics.csv as well)."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

@dataclass
class RunnerInfo:
    """Static description of a loaded model — convenient for plot titles."""
    name: str
    mask_mode: str
    input_mode: str
    image_size: int
    num_angle_bins: int
    backbone: str
    in_channels: int
    checkpoint_path: str
    epoch: Optional[int]


class ModelRunner:
    """Wrap a trained checkpoint behind a tiny ``predict`` API.

    Parameters
    ----------
    run_dir
        Path to the trainer ``save_dir`` (must contain ``resolved_config.yaml``).
    checkpoint
        Either an absolute path to a ``.pth`` file or one of the strings
        ``"best"``, ``"last"``, or ``"epoch_NNN"`` (resolved relative to
        ``run_dir``).
    name
        Friendly label used by the visualisation panels.
    device
        ``cuda`` if available, else ``cpu``. Pass explicitly to share a
        single device across runners.
    """

    def __init__(
        self,
        run_dir: str,
        checkpoint: str = "best",
        name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.run_dir = os.path.abspath(run_dir)
        cfg = load_resolved_config(self.run_dir)
        self._cfg = cfg

        self._ds_cfg = DatasetConfig(
            image_size=cfg["dataset"]["image_size"],
            input_mode=cfg["dataset"]["input_mode"],
            mask_mode=cfg["dataset"]["mask_mode"],
            num_angle_bins=cfg["dataset"]["num_angle_bins"],
            length_scale=cfg["dataset"]["length_scale"],
            use_stereo_depth=False,
        )
        self.image_size = self._ds_cfg.image_size
        self.in_channels = _input_channels(self._ds_cfg.input_mode)

        ckpt_path = self._resolve_checkpoint(checkpoint)
        self.checkpoint_path = ckpt_path

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model_cfg = {
            "mask_mode": self._ds_cfg.mask_mode,
            "backbone": cfg["model"]["backbone"],
            "in_channels": self.in_channels,
            "pretrained": False,
            "num_angle_bins": self._ds_cfg.num_angle_bins,
        }
        model = build_model(model_cfg).to(device).eval()
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Strip DataParallel ``module.`` prefixes if present
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                 for k, v in state.items()}
        model.load_state_dict(state)
        self.model = model

        epoch = None
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            epoch = int(ckpt["epoch"])
        elif "epoch_" in os.path.basename(ckpt_path):
            try:
                epoch = int(os.path.basename(ckpt_path).split("epoch_")[1].split(".")[0])
            except ValueError:
                epoch = None

        self.info = RunnerInfo(
            name=name or os.path.basename(self.run_dir),
            mask_mode=self._ds_cfg.mask_mode,
            input_mode=self._ds_cfg.input_mode,
            image_size=self.image_size,
            num_angle_bins=self._ds_cfg.num_angle_bins,
            backbone=cfg["model"]["backbone"],
            in_channels=self.in_channels,
            checkpoint_path=ckpt_path,
            epoch=epoch,
        )

    # ------------------------------------------------------------------
    def _resolve_checkpoint(self, checkpoint: str) -> str:
        if os.path.isabs(checkpoint) and os.path.isfile(checkpoint):
            return checkpoint
        candidate = os.path.join(self.run_dir, checkpoint)
        if os.path.isfile(candidate):
            return candidate
        # Auto-add .pth if user passed "best" / "last" / "epoch_005"
        candidate = os.path.join(self.run_dir, f"{checkpoint}.pth")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"Checkpoint {checkpoint!r} not found under {self.run_dir!r}."
        )

    # ------------------------------------------------------------------
    def preprocess(
        self,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
    ) -> torch.Tensor:
        """Resize + normalise inputs exactly like ``JacquardV2GraspSeg``."""
        S = self.image_size
        if rgb is None and depth is None:
            raise ValueError("At least one of rgb/depth must be provided")

        if rgb is None:
            rgb = np.zeros((S, S, 3), dtype=np.float32)
        if depth is None:
            depth = np.zeros((S, S), dtype=np.float32)
        if rgb.dtype != np.float32 or rgb.max() > 1.5:
            rgb = rgb.astype(np.float32) / 255.0
        if rgb.shape[:2] != (S, S):
            rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_LINEAR)
        depth = depth.astype(np.float32)
        if depth.shape != (S, S):
            depth = cv2.resize(depth, (S, S), interpolation=cv2.INTER_LINEAR)

        cfg = self._ds_cfg
        mean = np.asarray(cfg.rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(cfg.rgb_std, dtype=np.float32).reshape(1, 1, 3)
        rgb_n = (rgb - mean) / std
        d_n = (depth - cfg.depth_mean) / cfg.depth_std
        if cfg.input_mode == "rgb":
            arr = rgb_n.transpose(2, 0, 1)
        elif cfg.input_mode == "depth":
            arr = d_n[None]
        elif cfg.input_mode == "rgbd":
            arr = np.concatenate([rgb_n.transpose(2, 0, 1), d_n[None]], axis=0)
        else:
            raise ValueError(f"Unknown input_mode {cfg.input_mode!r}")
        return torch.from_numpy(arr.astype(np.float32))[None].to(self.device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        return_logits: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run a single forward pass.

        Output keys depend on ``mask_mode``:

        * ``binary``    → ``{"pos": HxW (sigmoid), "fg_mask": HxW uint8}``.
        * ``angle``     → ``{"prob": HxWx(K+1), "argmax": HxW int,
                              "fg_conf": HxW (1-p_bg), "fg_mask": HxW uint8}``.
        * ``multitask`` → ``{"pos": HxW, "cos2t": HxW, "sin2t": HxW,
                              "width": HxW (sigmoid), "fg_mask": HxW uint8,
                              "angle_deg": HxW float}``.
        """
        x = self.preprocess(rgb, depth)
        out = self.model(x)
        mode = self._ds_cfg.mask_mode

        if mode == "binary":
            logits = out
            if logits.dim() == 4 and logits.shape[1] == 1:
                logits = logits[:, 0]
            pos = torch.sigmoid(logits)[0].float().cpu().numpy()
            return {
                "pos": pos,
                "fg_mask": (pos > 0.5).astype(np.uint8),
                **({"logits": logits[0].float().cpu().numpy()} if return_logits else {}),
            }

        if mode == "angle":
            prob = torch.softmax(out, dim=1)[0].float().cpu().numpy()  # (K+1, H, W)
            argmax = prob.argmax(axis=0).astype(np.int64)
            fg_conf = 1.0 - prob[0]
            return {
                "prob": prob.transpose(1, 2, 0),
                "argmax": argmax,
                "fg_conf": fg_conf,
                "fg_mask": (argmax > 0).astype(np.uint8),
                **({"logits": out[0].float().cpu().numpy().transpose(1, 2, 0)}
                   if return_logits else {}),
            }

        if mode == "multitask":
            pos = torch.sigmoid(out["pos"])[0].float().cpu().numpy()
            cos2t = out["cos2t"][0].float().cpu().numpy()
            sin2t = out["sin2t"][0].float().cpu().numpy()
            width = torch.sigmoid(out["width"])[0].float().cpu().numpy()
            angle_deg = np.degrees(0.5 * np.arctan2(sin2t, cos2t))
            return {
                "pos": pos,
                "cos2t": cos2t,
                "sin2t": sin2t,
                "width": width,
                "fg_mask": (pos > 0.5).astype(np.uint8),
                "angle_deg": angle_deg,
            }

        raise ValueError(f"Unknown mask_mode {mode!r}")
