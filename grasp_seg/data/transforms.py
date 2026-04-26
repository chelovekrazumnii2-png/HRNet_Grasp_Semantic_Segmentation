"""Synchronised RGB-D + grasp-list augmentations.

We augment the *grasp list* (not the rasterised mask) so that the angle
information stays correct after rotations / flips. The mask is only
rasterised after all geometric ops have been applied.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .grasp_rect import Grasp


def _affine_grasp(g: Grasp, M: np.ndarray, angle_delta: float, scale: float) -> Grasp:
    """Apply a 2x3 affine matrix to a grasp's center, then update angle/length.

    ``M`` operates in (x, y) image coordinates. ``angle_delta`` is added to
    the grasp angle (in radians) and ``scale`` multiplies the length and
    width.
    """
    yx = g.center
    pt = np.array([yx[1], yx[0], 1.0], dtype=np.float64)  # (x, y, 1)
    new_xy = M @ pt
    new_center = np.array([new_xy[1], new_xy[0]], dtype=np.float64)
    new_angle = ((g.angle + angle_delta + math.pi / 2.0) % math.pi) - math.pi / 2.0
    return Grasp(new_center, new_angle, g.length * scale, g.width * scale)


def _hflip_grasp(g: Grasp, W: int) -> Grasp:
    new_center = np.array([g.center[0], (W - 1) - g.center[1]], dtype=np.float64)
    new_angle = ((-g.angle) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return Grasp(new_center, new_angle, g.length, g.width)


def _vflip_grasp(g: Grasp, H: int) -> Grasp:
    new_center = np.array([(H - 1) - g.center[0], g.center[1]], dtype=np.float64)
    new_angle = ((-g.angle) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return Grasp(new_center, new_angle, g.length, g.width)


@dataclass
class AugConfig:
    enable: bool = True
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate_p: float = 0.8
    rotate_max_deg: float = 180.0
    scale_p: float = 0.7
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translate_p: float = 0.5
    translate_max_frac: float = 0.05
    color_jitter_p: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    rgb_noise_p: float = 0.3
    rgb_noise_std: float = 0.02
    depth_jitter_p: float = 0.5
    depth_jitter_range: Tuple[float, float] = (0.95, 1.05)
    depth_dropout_p: float = 0.3
    depth_dropout_frac: float = 0.02
    use_stereo_depth_p: float = 0.0  # set >0 to randomly swap perfect→stereo


def _color_jitter_rgb(rgb: np.ndarray, cfg: AugConfig) -> np.ndarray:
    """`rgb` is float32 HxWx3 in [0, 1]."""
    out = rgb.copy()
    # brightness
    out = out * (1.0 + random.uniform(-cfg.brightness, cfg.brightness))
    # contrast
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * (1.0 + random.uniform(-cfg.contrast, cfg.contrast)) + mean
    # saturation (luma-preserving)
    luma = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
    out = (out - luma) * (1.0 + random.uniform(-cfg.saturation, cfg.saturation)) + luma
    # hue (cheap channel rotation in HSV)
    if cfg.hue > 0:
        hsv = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-cfg.hue, cfg.hue) * 180.0) % 180.0
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_augmentations(
    rgb: np.ndarray,
    depth: np.ndarray,
    grasps: List[Grasp],
    cfg: AugConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Grasp]]:
    """Apply synchronised RGB+D+grasp augmentations.

    ``rgb`` is float32 HxWx3 in [0,1], ``depth`` is float32 HxW in [0,1].
    """
    if not cfg.enable:
        return rgb, depth, list(grasps)

    H, W = depth.shape

    # Horizontal flip
    if random.random() < cfg.hflip_p:
        rgb = rgb[:, ::-1, :].copy()
        depth = depth[:, ::-1].copy()
        grasps = [_hflip_grasp(g, W) for g in grasps]

    # Vertical flip
    if random.random() < cfg.vflip_p:
        rgb = rgb[::-1, :, :].copy()
        depth = depth[::-1, :].copy()
        grasps = [_vflip_grasp(g, H) for g in grasps]

    # Rotation + scale + translation in a single affine
    apply_affine = False
    angle = 0.0
    scale = 1.0
    tx = 0.0
    ty = 0.0
    if random.random() < cfg.rotate_p:
        angle = random.uniform(-cfg.rotate_max_deg, cfg.rotate_max_deg)
        apply_affine = True
    if random.random() < cfg.scale_p:
        scale = random.uniform(cfg.scale_range[0], cfg.scale_range[1])
        apply_affine = True
    if random.random() < cfg.translate_p:
        tx = random.uniform(-cfg.translate_max_frac, cfg.translate_max_frac) * W
        ty = random.uniform(-cfg.translate_max_frac, cfg.translate_max_frac) * H
        apply_affine = True

    if apply_affine:
        center = (W / 2.0, H / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        depth = cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        angle_delta = math.radians(angle)
        grasps = [_affine_grasp(g, M, angle_delta, scale) for g in grasps]

    # RGB color jitter
    if random.random() < cfg.color_jitter_p:
        rgb = _color_jitter_rgb(rgb, cfg)
    if random.random() < cfg.rgb_noise_p:
        rgb = np.clip(rgb + np.random.normal(0, cfg.rgb_noise_std, rgb.shape).astype(np.float32),
                      0.0, 1.0)

    # Depth jitter / dropout
    if random.random() < cfg.depth_jitter_p:
        depth = depth * random.uniform(*cfg.depth_jitter_range)
        depth = np.clip(depth, 0.0, 1.0)
    if random.random() < cfg.depth_dropout_p:
        mask = np.random.rand(*depth.shape) < cfg.depth_dropout_frac
        depth = depth.copy()
        depth[mask] = 0.0

    # Drop grasps whose centres fell outside the image after the affine
    grasps = [
        g for g in grasps
        if 0.0 <= g.center[0] < H and 0.0 <= g.center[1] < W
    ]

    return rgb.astype(np.float32), depth.astype(np.float32), grasps
