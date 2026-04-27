"""Side-by-side comparison of multiple trained models on the same scenes.

Used in two contexts:

1. **Jacquard test split** — qualitative comparison of all 3 trained
   heads (angle / multitask-RGB / multitask-RGB-D) on a few test scenes.
2. **Cornell Grasp Dataset** — same models, but the dataset distribution
   is different (real photos, no per-pixel GT mask). We draw the
   Cornell positive-grasp rectangles instead of a mask overlay.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..data.grasp_rect import Grasp
from . import dataset_viz, decoder, draw
from .inference import ModelRunner


def _pad_to_square(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad ``img`` (H×W or H×W×C) with zeros to a square of side ``max(H, W)``.

    Returns ``(padded, pad_top, pad_left)``. The padding goes on bottom/right
    if the dimension is shorter, otherwise top/left — i.e. the original image
    is anchored at ``(pad_top, pad_left)`` in the padded canvas.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bot = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    if img.ndim == 3:
        pad = ((pad_top, pad_bot), (pad_left, pad_right), (0, 0))
    else:
        pad = ((pad_top, pad_bot), (pad_left, pad_right))
    return np.pad(img, pad, mode="constant", constant_values=0), pad_top, pad_left


def _scene_to_model_space(
    rgb: np.ndarray,
    depth: Optional[np.ndarray],
    grasps: Sequence[Grasp],
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Grasp], float, int, int]:
    """Pad+resize a non-square scene to ``(target_size, target_size)``.

    Returns ``(rgb_m, depth_m, grasps_m, scale, pad_top, pad_left)`` where:
    - ``rgb_m`` and ``depth_m`` are the padded+resized RGB and depth in
      ``(target_size, target_size)``;
    - ``grasps_m`` are the input grasps with centers shifted by the padding
      and uniformly scaled so they live in the model coordinate frame.

    Aspect ratio is preserved (we pad to square first, then uniform-resize),
    so grasp angles are unchanged — only ``center`` and ``length``/``width``
    need scaling.
    """
    rgb_padded, pad_top, pad_left = _pad_to_square(rgb)
    side = rgb_padded.shape[0]
    scale = target_size / side
    rgb_m = cv2.resize(rgb_padded, (target_size, target_size),
                       interpolation=cv2.INTER_LINEAR)

    if depth is None:
        depth_m = np.zeros((target_size, target_size), dtype=np.float32)
    else:
        depth_padded, _, _ = _pad_to_square(depth)
        depth_m = cv2.resize(depth_padded, (target_size, target_size),
                             interpolation=cv2.INTER_LINEAR)

    shift = np.array([pad_top, pad_left], dtype=np.float64)
    grasps_m: List[Grasp] = []
    for g in grasps:
        new_center = (g.center + shift) * scale
        grasps_m.append(Grasp(
            center=new_center,
            angle=g.angle,
            length=g.length * scale,
            width=g.width * scale,
        ))
    return rgb_m, depth_m, grasps_m, scale, pad_top, pad_left


def _predict_overlay(
    runner: ModelRunner,
    rgb: np.ndarray,
    depth: np.ndarray,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    overlay_alpha: float = 0.55,
    top_k: int = 5,
) -> tuple:
    pred = runner.predict(rgb=rgb, depth=depth)
    base = np.clip(rgb, 0, 1)
    if runner.info.mask_mode == "angle":
        overlay = draw.overlay_angle_mask(base, pred["argmax"],
                                           runner.info.num_angle_bins,
                                           alpha=overlay_alpha)
        decoded = decoder.decode_angle(pred["fg_conf"], pred["argmax"],
                                        runner.info.num_angle_bins, cfg=decode_cfg)
    elif runner.info.mask_mode == "multitask":
        overlay = draw.overlay_heatmap(base, pred["pos"], cmap="magma",
                                        alpha=overlay_alpha)
        decoded = decoder.decode_multitask(pred["pos"], pred["cos2t"],
                                            pred["sin2t"], pred["width"],
                                            cfg=decode_cfg)
    else:
        overlay = draw.overlay_binary_mask(base, pred["fg_mask"], alpha=overlay_alpha)
        decoded = []

    overlay = draw.draw_grasp_list(
        overlay, [g for g, _ in decoded[:top_k]],
        color=(0.1, 1.0, 0.2), plate_color=(1.0, 0.2, 0.2), thickness=2,
    )
    return overlay, decoded, pred


def figure_compare_models_jacquard(
    runners: Sequence[ModelRunner],
    grasp_files: Sequence[str],
    *,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    title: str = "Сравнение моделей на Jacquard V2 (test-сплит)",
):
    """Rows = scenes, columns = [GT, model 1, model 2, …]."""
    n_rows = len(grasp_files)
    n_cols = 1 + len(runners)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    axes = np.atleast_2d(axes)

    img_size = runners[0].image_size
    for r, gf in enumerate(grasp_files):
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=img_size)
        gt = draw.draw_grasp_list(rgb, grasps, max_n=15,
                                   color=(0.1, 1.0, 0.2),
                                   plate_color=(1.0, 0.2, 0.2),
                                   thickness=2)
        axes[r, 0].imshow(np.clip(gt, 0, 1))
        if r == 0:
            axes[r, 0].set_title("GT-захваты", fontsize=10)
        axes[r, 0].set_axis_off()

        for c, runner in enumerate(runners, start=1):
            overlay, decoded, _pred = _predict_overlay(runner, rgb, depth,
                                                        decode_cfg=decode_cfg)
            top1_ok = False
            if decoded:
                top1_ok, _, _ = decoder.jacquard_match(decoded[0][0], grasps)
            axes[r, c].imshow(np.clip(overlay, 0, 1))
            if r == 0:
                axes[r, c].set_title(runner.info.name, fontsize=10)
            axes[r, c].set_axis_off()
            tag = "top-1 ✓" if top1_ok else "top-1 ×"
            axes[r, c].text(
                3, 18, tag, color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
            )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def figure_compare_models_cornell(
    runners: Sequence[ModelRunner],
    cornell_scenes,
    *,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    title: str = "Сравнение моделей на Cornell Grasp Dataset (cross-domain)",
):
    """Same layout but Cornell scenes (no depth, no per-pixel GT)."""
    n_rows = len(cornell_scenes)
    n_cols = 1 + len(runners)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    axes = np.atleast_2d(axes)

    # Cornell is 480×640 (4:3) while the model expects square input at
    # ``runner.image_size``. We pad each scene to a square (preserving
    # aspect ratio) and resize uniformly so that:
    #   - per-pixel predictions and the RGB/depth panels share the same
    #     spatial size (no broadcast errors when overlaying heatmaps);
    #   - decoded grasp rectangles and GT grasps live in the same
    #     coordinate frame (so ``jacquard_match`` produces a meaningful
    #     top-1 indicator).
    img_size = runners[0].image_size

    for r, scene in enumerate(cornell_scenes):
        rgb_m, depth_m, gt_m, _, _, _ = _scene_to_model_space(
            scene.rgb, scene.depth, scene.pos_grasps, img_size,
        )

        gt_img = draw.draw_grasp_list(rgb_m, gt_m, max_n=15,
                                       color=(0.1, 1.0, 0.2),
                                       plate_color=(1.0, 0.2, 0.2),
                                       thickness=2)
        axes[r, 0].imshow(np.clip(gt_img, 0, 1))
        axes[r, 0].set_axis_off()
        if r == 0:
            axes[r, 0].set_title(f"GT-захваты Cornell\n(сцена {scene.scene_id})",
                                  fontsize=10)

        for c, runner in enumerate(runners, start=1):
            overlay, decoded, _ = _predict_overlay(
                runner, rgb_m, depth_m, decode_cfg=decode_cfg,
            )
            ok = False
            if decoded and gt_m:
                ok, _, _ = decoder.jacquard_match(decoded[0][0], gt_m)
            axes[r, c].imshow(np.clip(overlay, 0, 1))
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(runner.info.name, fontsize=10)
            tag = "top-1 ✓" if ok else "top-1 ×"
            axes[r, c].text(
                3, 18, tag, color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
            )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
