"""Best-epoch qualitative + quantitative panels.

Two top-level entry points:

- :func:`figure_best_epoch_scenes`: for one model, draw a row per scene
  with [input, GT-mask, predicted-mask, GT vs predicted rectangles,
  error map].

- :func:`figure_per_class_iou`: for one ``mask_mode='angle'`` model,
  bar-chart of per-bin IoU on the validation/test split.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.grasp_rect import Grasp, rasterize_grasp_mask
from . import dataset_viz, decoder, draw, palette
from .inference import ModelRunner


def _gt_mask(runner: ModelRunner, grasps: List[Grasp]) -> dict:
    return rasterize_grasp_mask(
        grasps, (runner.image_size, runner.image_size),
        mode=runner.info.mask_mode,
        num_angle_bins=runner.info.num_angle_bins,
    )


def _pred_overlay(runner: ModelRunner, rgb: np.ndarray, pred: dict, alpha: float = 0.55) -> np.ndarray:
    if runner.info.mask_mode == "angle":
        return draw.overlay_angle_mask(rgb, pred["argmax"], runner.info.num_angle_bins, alpha=alpha)
    if runner.info.mask_mode == "multitask":
        return draw.overlay_heatmap(rgb, pred["pos"], cmap="magma", alpha=alpha)
    return draw.overlay_binary_mask(rgb, pred["fg_mask"], alpha=alpha)


def _gt_overlay(runner: ModelRunner, rgb: np.ndarray, target: dict, alpha: float = 0.55) -> np.ndarray:
    if runner.info.mask_mode == "angle":
        return draw.overlay_angle_mask(rgb, target["mask"], runner.info.num_angle_bins, alpha=alpha)
    if runner.info.mask_mode == "multitask":
        return draw.overlay_heatmap(rgb, target["pos"], cmap="magma", alpha=alpha)
    return draw.overlay_binary_mask(rgb, target["mask"], alpha=alpha)


def figure_best_epoch_scenes(
    runner: ModelRunner,
    grasp_files: Sequence[str],
    *,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    title: Optional[str] = None,
):
    """One row per scene; columns: input, GT, prediction, decoded vs GT, errors."""
    n_rows = len(grasp_files)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.3 * n_rows))
    axes = np.atleast_2d(axes)

    col_titles = [
        "Вход (RGB)",
        "GT-маска",
        "Предсказание модели",
        "GT-прямоугольники vs. предсказанные",
        "Ошибки (TP/FP/FN)",
    ]

    for r, gf in enumerate(grasp_files):
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=runner.image_size)
        gt = _gt_mask(runner, grasps)
        pred = runner.predict(rgb=rgb, depth=depth)

        # Decode predicted grasps
        if runner.info.mask_mode == "angle":
            decoded = decoder.decode_angle(pred["fg_conf"], pred["argmax"],
                                            runner.info.num_angle_bins, cfg=decode_cfg)
        elif runner.info.mask_mode == "multitask":
            decoded = decoder.decode_multitask(pred["pos"], pred["cos2t"], pred["sin2t"],
                                                pred["width"], cfg=decode_cfg)
        else:
            decoded = []

        # 1) input
        axes[r, 0].imshow(np.clip(rgb, 0, 1))
        # 2) GT mask overlay
        axes[r, 1].imshow(np.clip(_gt_overlay(runner, rgb, gt), 0, 1))
        # 3) predicted mask overlay
        axes[r, 2].imshow(np.clip(_pred_overlay(runner, rgb, pred), 0, 1))
        # 4) GT rectangles vs predicted (top-5)
        rect_img = draw.draw_grasp_list(rgb, grasps, max_n=20,
                                         color=(0.1, 1.0, 0.2),
                                         plate_color=(0.1, 1.0, 0.2),
                                         thickness=2)
        rect_img = draw.draw_grasp_list(rect_img, [g for g, _ in decoded[:5]],
                                         color=(1.0, 0.2, 0.2),
                                         plate_color=(1.0, 0.2, 0.2),
                                         thickness=2)
        axes[r, 3].imshow(np.clip(rect_img, 0, 1))
        # 5) error map
        if "fg_mask" in pred:
            gt_fg = (gt.get("mask", gt.get("pos", np.zeros((runner.image_size,) * 2))) > 0).astype(np.uint8)
            err = draw.overlay_error_map(rgb, pred["fg_mask"], gt_fg, alpha=0.5)
            axes[r, 4].imshow(np.clip(err, 0, 1))

        # Compute Jacquard accuracy on this scene
        if decoded:
            top1_ok, _, _ = decoder.jacquard_match(decoded[0][0], grasps)
            top5_ok = any(
                decoder.jacquard_match(g, grasps)[0] for g, _ in decoded[:5]
            )
            tag = f"top-1: {'✓' if top1_ok else '×'}, top-5: {'✓' if top5_ok else '×'}"
        else:
            tag = "нет предсказаний"
        axes[r, 0].set_ylabel(
            f"{tag}\n{len(grasps)} GT-захватов",
            fontsize=9, rotation=0, labelpad=70, ha="right", va="center",
        )

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            for c, t in enumerate(col_titles):
                axes[r, c].set_title(t, fontsize=10)

    title = title or f"Лучшая эпоха — {runner.info.name}"
    fig.suptitle(
        f"{title}  (mask_mode={runner.info.mask_mode}, "
        f"input={runner.info.input_mode})",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-class IoU bar chart (angle mode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_class_iou(
    runner: ModelRunner,
    grasp_files: Sequence[str],
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute per-bin IoU on a sample of grasp files. Shape ``(K + 1,)``."""
    K = runner.info.num_angle_bins
    inter = np.zeros(K + 1, dtype=np.int64)
    union = np.zeros(K + 1, dtype=np.int64)
    files = list(grasp_files)
    if max_samples is not None:
        files = files[:max_samples]
    for gf in files:
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=runner.image_size)
        target = rasterize_grasp_mask(
            grasps, (runner.image_size, runner.image_size), mode="angle",
            num_angle_bins=K,
        )["mask"]
        pred = runner.predict(rgb=rgb, depth=depth)
        if runner.info.mask_mode == "angle":
            pmask = pred["argmax"]
        elif runner.info.mask_mode == "multitask":
            # Bin the predicted angle into the same K bins
            angles = 0.5 * np.arctan2(pred["sin2t"], pred["cos2t"])
            angles_mod = (angles + np.pi / 2.0) % np.pi
            bin_w = np.pi / K
            bins = np.clip((angles_mod / bin_w).astype(np.int64), 0, K - 1) + 1
            pmask = np.where(pred["fg_mask"] > 0, bins, 0)
        else:
            continue
        for c in range(K + 1):
            t = target == c
            p = pmask == c
            inter[c] += int(np.logical_and(t, p).sum())
            union[c] += int(np.logical_or(t, p).sum())
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou


def figure_per_class_iou(
    runners: Sequence[ModelRunner],
    grasp_files: Sequence[str],
    max_samples: Optional[int] = 200,
):
    """Bar chart: IoU per angle-bin on a sample of files, for one or more models."""
    K_all = [r.info.num_angle_bins for r in runners]
    if len(set(K_all)) > 1:
        raise ValueError("All runners must share num_angle_bins for this plot")
    K = K_all[0]
    centres = palette.angle_bin_centers_deg(K)

    fig, ax = plt.subplots(1, 1, figsize=(0.6 * K + 4, 4.5))
    width = 0.8 / max(len(runners), 1)
    x = np.arange(K)
    colors = draw.colors_for_n(len(runners))

    for i, runner in enumerate(runners):
        iou = per_class_iou(runner, grasp_files, max_samples=max_samples)
        ax.bar(x + (i - (len(runners) - 1) / 2) * width, iou[1:], width,
               label=runner.info.name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0f}°" for c in centres], rotation=45)
    ax.set_xlabel("Угол захвата θ (центр бина)")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-bin IoU (на {max_samples or 'всех'} test-сценах)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
