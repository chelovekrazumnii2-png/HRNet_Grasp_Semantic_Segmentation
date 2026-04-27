"""Extra (bonus) figures requested for the report:

- :func:`figure_iou_vs_angle` — histogram of foreground IoU vs the
  dominant ground-truth angle bin (where does the model fail by angle?).
- :func:`figure_depth_contribution` — pixel-wise difference between
  RGB-D and RGB-only ``pos`` predictions on the same scene, overlaid as
  a red/blue heat-map (red = depth helped, blue = depth hurt).
- :func:`figure_failure_catalog` — top ``N`` worst predictions on a
  test sample, with simple heuristic annotations explaining likely
  causes (low contrast, small object, depth dropout, etc.).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..data.grasp_rect import rasterize_grasp_mask
from . import dataset_viz, decoder, draw, palette
from .inference import ModelRunner


# ---------------------------------------------------------------------------
# IoU × angle histogram
# ---------------------------------------------------------------------------

def figure_iou_vs_angle(
    runner: ModelRunner,
    grasp_files: Sequence[str],
    *,
    max_samples: Optional[int] = 200,
    title: Optional[str] = None,
):
    """Per-scene foreground IoU bucketed by the scene's dominant angle bin."""
    K = runner.info.num_angle_bins
    centres = palette.angle_bin_centers_deg(K)
    bin_iou: List[List[float]] = [[] for _ in range(K)]

    files = list(grasp_files)
    if max_samples is not None:
        files = files[:max_samples]

    for gf in files:
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=runner.image_size)
        if not grasps:
            continue
        target = rasterize_grasp_mask(
            grasps, (runner.image_size, runner.image_size), mode="angle",
            num_angle_bins=K,
        )["mask"]
        # Dominant GT bin = mode of foreground pixels
        fg_pixels = target[target > 0]
        if fg_pixels.size == 0:
            continue
        dom_bin = int(np.bincount(fg_pixels, minlength=K + 1).argmax())
        if dom_bin == 0:
            continue

        pred = runner.predict(rgb=rgb, depth=depth)
        pmask = pred["fg_mask"]
        gt_fg = (target > 0).astype(np.uint8)
        inter = int(np.logical_and(pmask, gt_fg).sum())
        union = int(np.logical_or(pmask, gt_fg).sum())
        iou = inter / union if union > 0 else 0.0
        bin_iou[dom_bin - 1].append(iou)

    fig, ax = plt.subplots(1, 1, figsize=(0.55 * K + 4, 4.5))
    means = [np.mean(b) if b else np.nan for b in bin_iou]
    counts = [len(b) for b in bin_iou]
    x = np.arange(K)
    bars = ax.bar(x, means, color=draw.colors_for_n(K, cmap="hsv"), alpha=0.85,
                   edgecolor="0.2")
    for i, (b, m, c) in enumerate(zip(bars, means, counts)):
        if c > 0:
            ax.text(i, (m or 0) + 0.02, f"n={c}", ha="center", fontsize=8, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0f}°" for c in centres], rotation=45)
    ax.set_xlabel("Доминирующий GT-угол сцены, θ")
    ax.set_ylabel("Средний IoU (foreground)")
    ax.set_ylim(0, 1)
    ax.set_title(title or
                  f"IoU как функция угла GT — {runner.info.name}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Depth contribution heat-map
# ---------------------------------------------------------------------------

def figure_depth_contribution(
    runner_rgb: ModelRunner,
    runner_rgbd: ModelRunner,
    grasp_files: Sequence[str],
    *,
    title: str = "Вклад глубины: |pred(RGB-D) − pred(RGB)|",
):
    """For each scene draw the raw `pos` for both heads + their difference."""
    n_rows = len(grasp_files)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.0, 3.4 * n_rows))
    axes = np.atleast_2d(axes)

    img_size = runner_rgbd.image_size
    for r, gf in enumerate(grasp_files):
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=img_size)
        # RGB-only model: pass zeros for depth (it ignores the channel).
        pos_rgb = _pos_map(runner_rgb, rgb, depth_or_zero(rgb, depth, runner_rgb))
        pos_rgbd = _pos_map(runner_rgbd, rgb, depth)
        diff = pos_rgbd - pos_rgb

        axes[r, 0].imshow(np.clip(rgb, 0, 1))
        axes[r, 0].set_title("Вход", fontsize=10) if r == 0 else None
        axes[r, 1].imshow(pos_rgb, cmap="magma", vmin=0, vmax=1)
        axes[r, 1].set_title("pos (RGB-only)", fontsize=10) if r == 0 else None
        axes[r, 2].imshow(pos_rgbd, cmap="magma", vmin=0, vmax=1)
        axes[r, 2].set_title("pos (RGB-D)", fontsize=10) if r == 0 else None
        im = axes[r, 3].imshow(diff, cmap="seismic", vmin=-0.6, vmax=0.6)
        axes[r, 3].set_title("RGB-D − RGB", fontsize=10) if r == 0 else None
        if r == 0:
            plt.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04,
                          label="Δ pos")

        for c in range(4):
            axes[r, c].set_axis_off()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def _pos_map(runner: ModelRunner, rgb, depth) -> np.ndarray:
    pred = runner.predict(rgb=rgb, depth=depth)
    if "pos" in pred:
        return pred["pos"]
    if "fg_conf" in pred:
        return pred["fg_conf"]
    return pred["fg_mask"].astype(np.float32)


def depth_or_zero(rgb, depth, runner: ModelRunner) -> np.ndarray:
    """If runner is RGB-only, replace depth with zeros to mirror its training."""
    if runner.info.input_mode == "rgb":
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    return depth


# ---------------------------------------------------------------------------
# Failure case catalog
# ---------------------------------------------------------------------------

@dataclass
class _Failure:
    grasp_file: str
    iou: float
    matched_top1: bool
    angle_err_deg: float
    annotation_ru: str
    rgb: np.ndarray
    depth: np.ndarray
    grasps: list
    pred: dict
    decoded: list


def _annotate_failure(rgb: np.ndarray, depth: np.ndarray, grasps: list) -> str:
    """Cheap heuristics to label likely failure reasons."""
    notes: List[str] = []
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if luma.mean() < 0.30:
        notes.append("низкая яркость")
    if luma.std() < 0.08:
        notes.append("низкий контраст")
    fg_area = 0.0
    if grasps:
        sample = rasterize_grasp_mask(
            grasps, rgb.shape[:2], mode="binary",
        )["mask"]
        fg_area = float(sample.mean())
    if 0 < fg_area < 0.005:
        notes.append("малая площадь захвата")
    if depth is not None:
        zero_frac = float((depth == 0).mean())
        if zero_frac > 0.10:
            notes.append("большие пропуски в глубине")
    if not notes:
        notes.append("нет очевидной причины")
    return ", ".join(notes)


def _per_scene_iou(runner: ModelRunner, rgb, depth, grasps) -> Tuple[float, dict, list]:
    pred = runner.predict(rgb=rgb, depth=depth)
    target = rasterize_grasp_mask(
        grasps, (runner.image_size, runner.image_size), mode="binary"
    )["mask"]
    p = pred["fg_mask"]
    inter = int(np.logical_and(p, target).sum())
    union = int(np.logical_or(p, target).sum())
    iou = inter / union if union > 0 else 0.0
    return iou, pred, []


def figure_failure_catalog(
    runner: ModelRunner,
    grasp_files: Sequence[str],
    *,
    n_show: int = 8,
    max_samples: int = 200,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    title: Optional[str] = None,
):
    """Top-N worst predictions, ranked by ``1 − foreground IoU``."""
    files = list(grasp_files)[:max_samples]
    failures: List[_Failure] = []
    for gf in files:
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=runner.image_size)
        if not grasps:
            continue
        iou, pred, _ = _per_scene_iou(runner, rgb, depth, grasps)

        if runner.info.mask_mode == "angle":
            decoded = decoder.decode_angle(pred["fg_conf"], pred["argmax"],
                                            runner.info.num_angle_bins, cfg=decode_cfg)
        elif runner.info.mask_mode == "multitask":
            decoded = decoder.decode_multitask(pred["pos"], pred["cos2t"],
                                                pred["sin2t"], pred["width"],
                                                cfg=decode_cfg)
        else:
            decoded = []

        matched = False
        ang_err = 180.0
        if decoded:
            matched, _, ang_err = decoder.jacquard_match(decoded[0][0], grasps)

        failures.append(_Failure(
            grasp_file=gf, iou=iou, matched_top1=matched, angle_err_deg=ang_err,
            annotation_ru=_annotate_failure(rgb, depth, grasps),
            rgb=rgb, depth=depth, grasps=grasps, pred=pred, decoded=decoded,
        ))

    failures.sort(key=lambda f: (f.iou, -f.angle_err_deg))
    failures = failures[:n_show]

    n_cols = 4
    n_rows = int(math.ceil(len(failures) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 4.0 * n_rows))
    axes = np.atleast_2d(axes)

    for i, f in enumerate(failures):
        ax = axes[i // n_cols, i % n_cols]
        # Overlay GT (green) + decoded top-3 (red) on the input
        img = draw.draw_grasp_list(f.rgb, f.grasps, max_n=10,
                                    color=(0.1, 1.0, 0.2),
                                    plate_color=(0.1, 1.0, 0.2),
                                    thickness=2)
        img = draw.draw_grasp_list(img, [g for g, _ in f.decoded[:3]],
                                    color=(1.0, 0.2, 0.2),
                                    plate_color=(1.0, 0.2, 0.2),
                                    thickness=2)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_axis_off()
        ax.set_title(
            f"IoU={f.iou:.2f}, top-1: {'✓' if f.matched_top1 else '×'}\n"
            f"причина: {f.annotation_ru}",
            fontsize=9,
        )

    for j in range(len(failures), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_axis_off()

    fig.suptitle(
        title or f"Каталог худших предсказаний ({len(failures)} сцен) — {runner.info.name}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
