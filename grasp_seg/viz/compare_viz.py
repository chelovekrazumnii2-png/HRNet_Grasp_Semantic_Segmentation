"""Side-by-side comparison of multiple trained models on the same scenes.

Used in two contexts:

1. **Jacquard test split** — qualitative comparison of all 3 trained
   heads (angle / multitask-RGB / multitask-RGB-D) on a few test scenes.
2. **Cornell Grasp Dataset** — same models, but the dataset distribution
   is different (real photos, no per-pixel GT mask). We draw the
   Cornell positive-grasp rectangles instead of a mask overlay.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import dataset_viz, decoder, draw
from .inference import ModelRunner


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

    for r, scene in enumerate(cornell_scenes):
        rgb_native = scene.rgb
        gt_img = draw.draw_grasp_list(rgb_native, scene.pos_grasps, max_n=15,
                                       color=(0.1, 1.0, 0.2),
                                       plate_color=(1.0, 0.2, 0.2),
                                       thickness=2)
        axes[r, 0].imshow(np.clip(gt_img, 0, 1))
        axes[r, 0].set_axis_off()
        if r == 0:
            axes[r, 0].set_title(f"GT-захваты Cornell\n(сцена {scene.scene_id})",
                                  fontsize=10)

        for c, runner in enumerate(runners, start=1):
            # Cornell has no depth → models with input_mode='rgbd' get a
            # zero depth channel; those trained on depth-only would just
            # see zeros (we still run them so we never silently skip).
            overlay, decoded, _ = _predict_overlay(
                runner, rgb_native, np.zeros(rgb_native.shape[:2], dtype=np.float32),
                decode_cfg=decode_cfg,
            )
            ok = False
            if decoded and scene.pos_grasps:
                ok, _, _ = decoder.jacquard_match(decoded[0][0], scene.pos_grasps)
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
