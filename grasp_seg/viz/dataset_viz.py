"""Dataset-level visualisation panels.

These are the figures that go into the "Dataset" section of the report:

- :func:`figure_raw_with_grasps`  — original RGB + all GT rectangles.
- :func:`figure_resize_pipeline`  — original (with rect) vs 384×384/512x512 input vs
  rasterised compact-polygon target.
- :func:`figure_mask_modes`       — same scene rasterised under
  binary / angle / multitask masks (side-by-side) for both RGB-only and
  RGB-D inputs.
- :func:`figure_compact_vs_full`  — compact-polygon (length_scale=1/3) vs
  full grasp rectangle as the training target.
- :func:`figure_augmentation_steps` — RGB+depth before/after each major
  augmentation step (hflip, vflip, rotation, scale, translate, color
  jitter, RGB noise, depth jitter, depth dropout).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..data.grasp_rect import Grasp, load_jacquard_grasps, rasterize_grasp_mask
from ..data.jacquard_v2 import _load_depth, _load_rgb, _normalise_depth
from ..data.transforms import AugConfig
from . import draw, palette

IMG_RES = 512 #384 512

# ---------------------------------------------------------------------------
# Loading helpers (re-exported for the notebook)
# ---------------------------------------------------------------------------

def load_jacquard_scene(grasp_file: str, image_size: Optional[int] = None):
    """Return ``(rgb, depth, grasps, scale)`` for a Jacquard ``*_grasps.txt``.

    If ``image_size`` is ``None`` we return the raw 1024×1024 input;
    otherwise the trio is resized exactly like the training dataset.
    """
    rgb_path = grasp_file.replace("_grasps.txt", "_RGB.png")
    depth_path = grasp_file.replace("_grasps.txt", "_perfect_depth.tiff")
    rgb = _load_rgb(rgb_path)
    depth_raw = _load_depth(depth_path)
    depth = _normalise_depth(depth_raw)

    if image_size is not None and rgb.shape[:2] != (image_size, image_size):
        scale = float(image_size) / float(rgb.shape[0])
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
    grasps = load_jacquard_grasps(grasp_file, scale=scale)
    return rgb, depth, grasps, scale


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


# def figure_raw_with_grasps(
#     grasp_file: str,
#     max_grasps: int = 30,
#     title: Optional[str] = None,
# ):
#     """RGB at native resolution with all GT grasp rectangles drawn."""
#     rgb, _depth, grasps, _ = load_jacquard_scene(grasp_file, image_size=None)
#     overlay = draw.draw_grasp_list(rgb, grasps, max_n=max_grasps,
#                                    color=(0.1, 1.0, 0.2), thickness=2)
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#     ax.imshow(np.clip(overlay, 0, 1))
#     ax.set_axis_off()
#     ax.set_title(
#         title or f"Исходное изображение (1024×1024) + {len(grasps)} GT-захватов",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     return fig

def figure_raw_with_grasps(
    grasp_file: str,
    max_grasps: int = 10,
    title: Optional[str] = None,
):
    """RGB at native resolution - left: original, right: with GT grasp rectangles."""
    rgb, _depth, grasps, _ = load_jacquard_scene(grasp_file, image_size=None)
    
    # Создаём изображение с захватами
    overlay = draw.draw_grasp_list(rgb, grasps, max_n=max_grasps,
                                   color=(0.1, 1.0, 0.2), thickness=2)
    
    # Создаём фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Первое изображение - исходное без прямоугольников
    ax1.imshow(np.clip(rgb, 0, 1))
    ax1.set_axis_off()
    ax1.set_title("Исходное изображение", fontsize=11)
    
    # Второе изображение - с захватами
    ax2.imshow(np.clip(overlay, 0, 1))
    ax2.set_axis_off()
    ax2.set_title(
        title or f"GT-захваты ({len(grasps)} шт.)",
        fontsize=11,
    )
    
    fig.tight_layout()
    return fig


def figure_resize_pipeline(
    grasp_file: str,
    image_size: int = IMG_RES,
    num_angle_bins: int = 18,
    length_scale: float = 1.0 / 3.0,
):
    """Show how a sample passes through resize → mask rasterisation."""
    rgb_raw, _d, grasps_raw, _ = load_jacquard_scene(grasp_file, image_size=None)
    rgb_res, depth_res, grasps_res, _ = load_jacquard_scene(grasp_file, image_size=image_size)

    raw_overlay = draw.draw_grasp_list(rgb_raw, grasps_raw, max_n=20)
    res_overlay = draw.draw_grasp_list(rgb_res, grasps_res, max_n=20)
    target = rasterize_grasp_mask(
        grasps_res, (image_size, image_size), mode="angle",
        num_angle_bins=num_angle_bins, length_scale=length_scale,
    )["mask"]
    target_overlay = draw.overlay_angle_mask(rgb_res, target, num_angle_bins, alpha=0.6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(np.clip(raw_overlay, 0, 1))
    axes[0].set_title(f"Исходное RGB ({rgb_raw.shape[1]}×{rgb_raw.shape[0]})\n"
                      f"+ {len(grasps_raw)} GT-захватов")
    axes[1].imshow(np.clip(res_overlay, 0, 1))
    axes[1].set_title(f"Resize → {image_size}×{image_size}\n"
                      f"+ те же {len(grasps_res)} захватов в новой системе координат")
    axes[2].imshow(np.clip(target_overlay, 0, 1))
    axes[2].set_title(
        f"Целевая маска (angle, {num_angle_bins} классов)\n"
        f"compact-polygon, length_scale={length_scale:.3f}"
    )
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def figure_mask_modes(
    grasp_file: str,
    image_size: int = IMG_RES,
    num_angle_bins: int = 18,
    length_scale: float = 1.0 / 3.0,
):
    """Same scene rasterised under all three mask modes (binary/angle/multitask).

    Multitask is shown as ``pos`` overlay + the recovered angle map (in
    degrees) so the multi-channel head is visible at a glance.
    """
    rgb, depth, grasps, _ = load_jacquard_scene(grasp_file, image_size=image_size)
    bin_t = rasterize_grasp_mask(grasps, (image_size, image_size), mode="binary",
                                 length_scale=length_scale)["mask"]
    ang_t = rasterize_grasp_mask(grasps, (image_size, image_size), mode="angle",
                                 num_angle_bins=num_angle_bins,
                                 length_scale=length_scale)["mask"]
    mt = rasterize_grasp_mask(grasps, (image_size, image_size), mode="multitask",
                              length_scale=length_scale)
    pos = mt["pos"]
    ang_deg = np.degrees(0.5 * np.arctan2(mt["sin2t"], mt["cos2t"]))
    ang_deg_disp = np.where(pos > 0.5, (ang_deg + 90.0) % 180.0, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title("RGB-вход")
    axes[0, 1].imshow(depth, cmap="viridis")
    axes[0, 1].set_title("Глубина (нормированная)")
    axes[0, 2].imshow(np.clip(draw.draw_grasp_list(rgb, grasps, max_n=30), 0, 1))
    axes[0, 2].set_title(f"GT-прямоугольники ({len(grasps)})")

    axes[1, 0].imshow(np.clip(draw.overlay_binary_mask(rgb, bin_t,
                                                       color=(1.0, 0.2, 0.2),
                                                       alpha=0.55), 0, 1))
    axes[1, 0].set_title("binary (BCE+Dice)\n«graspable»")
    axes[1, 1].imshow(np.clip(draw.overlay_angle_mask(rgb, ang_t, num_angle_bins,
                                                      alpha=0.6), 0, 1))
    axes[1, 1].set_title(f"angle ({num_angle_bins} классов угла)\nCE + Dice")
    im = axes[1, 2].imshow(ang_deg_disp, cmap="hsv", vmin=0.0, vmax=180.0)
    axes[1, 2].set_title("multitask: pos + угол θ°\n(GG-CNN: pos / cos2θ / sin2θ / w)")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label="θ°")

    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(
        "Целевые маски, подаваемые на разные режимы обучения "
        "(одна и та же сцена)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def figure_compact_vs_full(
    grasp_file: str,
    image_size: int = IMG_RES,
    num_angle_bins: int = 18,
):
    """Compact-polygon vs full grasp rectangle as the training target."""
    rgb, _d, grasps, _ = load_jacquard_scene(grasp_file, image_size=image_size)
    full = rasterize_grasp_mask(grasps, (image_size, image_size), mode="angle",
                                num_angle_bins=num_angle_bins, length_scale=1.0)["mask"]
    compact = rasterize_grasp_mask(grasps, (image_size, image_size), mode="angle",
                                   num_angle_bins=num_angle_bins,
                                   length_scale=1.0 / 3.0)["mask"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("RGB-вход")
    axes[1].imshow(np.clip(draw.overlay_angle_mask(rgb, full, num_angle_bins,
                                                   alpha=0.6), 0, 1))
    axes[1].set_title("Полный прямоугольник\n(length_scale = 1.0)")
    axes[2].imshow(np.clip(draw.overlay_angle_mask(rgb, compact, num_angle_bins,
                                                   alpha=0.6), 0, 1))
    axes[2].set_title("Compact-polygon\n(length_scale = 1/3, режим обучения)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("Почему мы обучаемся на компактной центральной полоске,"
                 " а не на полном прямоугольнике", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Augmentation walkthrough
# ---------------------------------------------------------------------------

@dataclass
class AugStep:
    name_ru: str
    apply: callable  # (rgb, depth, grasps) -> (rgb, depth, grasps)


def _aug_hflip(rgb, depth, grasps):
    from ..data.transforms import _hflip_grasp
    H, W = depth.shape
    return rgb[:, ::-1, :].copy(), depth[:, ::-1].copy(), [_hflip_grasp(g, W) for g in grasps]


def _aug_vflip(rgb, depth, grasps):
    from ..data.transforms import _vflip_grasp
    H, W = depth.shape
    return rgb[::-1, :, :].copy(), depth[::-1, :].copy(), [_vflip_grasp(g, H) for g in grasps]


def _aug_rotate(angle_deg: float):
    from ..data.transforms import _affine_grasp

    def _f(rgb, depth, grasps):
        H, W = depth.shape
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
        rgb_o = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        depth_o = cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        gs = [_affine_grasp(g, M, math.radians(angle_deg), 1.0) for g in grasps]
        return rgb_o, depth_o, gs
    return _f


def _aug_scale_translate(scale: float, tx_frac: float, ty_frac: float):
    from ..data.transforms import _affine_grasp

    def _f(rgb, depth, grasps):
        H, W = depth.shape
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), 0.0, scale)
        M[0, 2] += tx_frac * W
        M[1, 2] += ty_frac * H
        rgb_o = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        depth_o = cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        gs = [_affine_grasp(g, M, 0.0, scale) for g in grasps]
        return rgb_o, depth_o, gs
    return _f


def _aug_color_jitter(seed: int = 0):
    def _f(rgb, depth, grasps):
        rng = np.random.RandomState(seed)
        out = rgb.copy()
        out *= 1.0 + rng.uniform(-0.2, 0.2)            # brightness
        m = out.mean(axis=(0, 1), keepdims=True)
        out = (out - m) * (1.0 + rng.uniform(-0.2, 0.2)) + m  # contrast
        return np.clip(out, 0, 1).astype(np.float32), depth, grasps
    return _f


def _aug_rgb_noise(std: float = 0.04, seed: int = 0):
    def _f(rgb, depth, grasps):
        rng = np.random.RandomState(seed)
        n = rng.normal(0.0, std, rgb.shape).astype(np.float32)
        return np.clip(rgb + n, 0, 1), depth, grasps
    return _f


def _aug_depth_jitter(scale: float = 0.95):
    def _f(rgb, depth, grasps):
        return rgb, np.clip(depth * scale, 0, 1), grasps
    return _f


def _aug_depth_dropout(frac: float = 0.05, seed: int = 0):
    def _f(rgb, depth, grasps):
        rng = np.random.RandomState(seed)
        m = rng.rand(*depth.shape) < frac
        d = depth.copy()
        d[m] = 0.0
        return rgb, d, grasps
    return _f


def figure_augmentation_steps(
    grasp_file: str,
    image_size: int = IMG_RES,
    seed: int = 0,
):
    """Walk through every major augmentation in :class:`AugConfig`."""
    random.seed(seed)
    np.random.seed(seed)
    rgb, depth, grasps, _ = load_jacquard_scene(grasp_file, image_size=image_size)

    steps: List[AugStep] = [
        AugStep("Исходник (после resize)", lambda r, d, g: (r, d, g)),
        AugStep("Горизонтальный флип", _aug_hflip),
        AugStep("Вертикальный флип", _aug_vflip),
        AugStep("Поворот на +30°", _aug_rotate(30.0)),
        AugStep("Поворот на −60°", _aug_rotate(-60.0)),
        AugStep("Масштаб 0.85 + сдвиг (+5%, −3%)",
                _aug_scale_translate(0.85, 0.05, -0.03)),
        AugStep("Color jitter (яркость+контраст)", _aug_color_jitter(seed=seed)),
        AugStep("Гауссов шум RGB (σ=0.04)", _aug_rgb_noise(0.04, seed=seed)),
        AugStep("Jitter глубины (×0.95)", _aug_depth_jitter(0.95)),
        AugStep("Dropout глубины (5% пикселей → 0)", _aug_depth_dropout(0.05, seed=seed)),
    ]

    n = len(steps)
    n_cols = 5
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows * 2))
    axes = np.atleast_2d(axes)

    for i, step in enumerate(steps):
        r, d, gs = step.apply(rgb.copy(), depth.copy(), list(grasps))
        rr = (i // n_cols) * 2
        cc = i % n_cols
        ax_rgb = axes[rr, cc]
        ax_d = axes[rr + 1, cc]
        ax_rgb.imshow(np.clip(draw.draw_grasp_list(r, gs, max_n=15), 0, 1))
        ax_rgb.set_title(step.name_ru, fontsize=10)
        ax_d.imshow(d, cmap="viridis", vmin=0, vmax=1)
        ax_d.set_title("(карта глубины)", fontsize=9, color="0.4")
        ax_rgb.set_axis_off()
        ax_d.set_axis_off()

    # Hide unused cells
    for j in range(n, n_rows * n_cols):
        rr = (j // n_cols) * 2
        cc = j % n_cols
        axes[rr, cc].set_axis_off()
        axes[rr + 1, cc].set_axis_off()

    fig.suptitle(
        "Аугментации: каждый шаг применяется к RGB и к глубине синхронно,"
        " GT-захваты пересчитываются под новую геометрию",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def figure_cornell_raw(scene, max_grasps: int = 20):
    """Cornell scene with positive grasps + depth (if available)."""
    overlay = draw.draw_grasp_list(scene.rgb, scene.pos_grasps, max_n=max_grasps,
                                   color=(0.1, 1.0, 0.2), thickness=2)
    has_depth = scene.depth is not None
    n_cols = 2 if has_depth else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5.0))
    if not has_depth:
        axes = [axes]
    axes[0].imshow(np.clip(overlay, 0, 1))
    axes[0].set_title(
        f"RGB + GT-захваты\n(сцена {scene.scene_id}, "
        f"{len(scene.pos_grasps)} позитивных)"
    )
    axes[0].set_axis_off()
    if has_depth:
        axes[1].imshow(scene.depth, cmap="viridis")
        axes[1].set_title("Depth (нормированная, pcdNNNNd.tiff)")
        axes[1].set_axis_off()
    fig.tight_layout()
    return fig
