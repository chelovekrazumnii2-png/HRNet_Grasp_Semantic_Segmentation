"""Visualise how a model "sees" a fixed scene as it trains.

For each saved ``epoch_NNN.pth`` we run inference on the same 1–3 scenes
and lay them out as a (scene × epoch) grid. Epoch 1 + every 5th epoch up
to the final one is the default — for a 30-epoch run this gives 7 cells
per scene, plus a leading column with the ground-truth target so you can
see "starting point → training direction → final result".
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..data.grasp_rect import rasterize_grasp_mask
from . import dataset_viz, decoder, draw, palette
from .inference import ModelRunner


def _list_epoch_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """Find ``epoch_NNN.pth`` files sorted by epoch number."""
    out: List[Tuple[int, str]] = []
    for name in os.listdir(run_dir):
        if name.startswith("epoch_") and name.endswith(".pth"):
            try:
                ep = int(name[len("epoch_"):-len(".pth")])
            except ValueError:
                continue
            out.append((ep, os.path.join(run_dir, name)))
    return sorted(out, key=lambda t: t[0])


def select_milestones(
    run_dir: str,
    step: int = 5,
    include_first: bool = True,
) -> List[Tuple[int, str]]:
    """Return ``[(epoch, ckpt_path), ...]`` for visualisation.

    Picks: epoch 1 (if ``include_first``), then every ``step``-th epoch
    up to the largest available, plus the largest itself if it's not
    already in the list.
    """
    avail = _list_epoch_checkpoints(run_dir)
    if not avail:
        return []
    avail_dict = dict(avail)
    selected: List[int] = []
    if include_first and 1 in avail_dict:
        selected.append(1)
    for ep, _ in avail:
        if ep > 1 and ep % step == 0 and ep not in selected:
            selected.append(ep)
    last_ep = avail[-1][0]
    if last_ep not in selected:
        selected.append(last_ep)
    return [(ep, avail_dict[ep]) for ep in selected]


def _render_prediction(
    runner: ModelRunner,
    rgb: np.ndarray,
    depth: np.ndarray,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    overlay_alpha: float = 0.55,
) -> np.ndarray:
    """Run a forward pass and return an RGB overlay with predicted rectangles."""
    pred = runner.predict(rgb=rgb, depth=depth)
    mode = runner.info.mask_mode
    base = np.clip(rgb, 0, 1)

    if mode == "angle":
        overlay = draw.overlay_angle_mask(
            base, pred["argmax"], runner.info.num_angle_bins, alpha=overlay_alpha
        )
        grasps = decoder.decode_angle(
            pred["fg_conf"], pred["argmax"], runner.info.num_angle_bins,
            cfg=decode_cfg,
        )
    elif mode == "multitask":
        overlay = draw.overlay_heatmap(base, pred["pos"], cmap="magma",
                                        alpha=overlay_alpha)
        grasps = decoder.decode_multitask(
            pred["pos"], pred["cos2t"], pred["sin2t"], pred["width"],
            cfg=decode_cfg,
        )
    elif mode == "binary":
        overlay = draw.overlay_heatmap(base, pred["pos"], cmap="magma",
                                        alpha=overlay_alpha)
        grasps = []
    else:
        raise ValueError(mode)

    # Top-K rectangles
    overlay = draw.draw_grasp_list(
        overlay, [g for g, _ in grasps[:5]],
        color=(0.1, 1.0, 0.2), plate_color=(1.0, 0.2, 0.2), thickness=2,
    )
    return overlay


def figure_epoch_evolution(
    run_dir: str,
    grasp_files: Sequence[str],
    *,
    step: int = 5,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    image_size: Optional[int] = None,
    title: Optional[str] = None,
):
    """Draw a (scene × epoch) grid for one model.

    Layout: rows are scenes, columns are
    ``[GT, epoch 1, epoch 5, epoch 10, …, last]``.
    """
    milestones = select_milestones(run_dir, step=step, include_first=True)
    if not milestones:
        raise RuntimeError(f"No epoch_NNN.pth checkpoints found in {run_dir}")

    # Load runners lazily — we keep one in memory at a time to limit VRAM,
    # then evict before loading the next epoch.
    n_rows = len(grasp_files)
    n_cols = 1 + len(milestones)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 2.7 * n_rows))
    axes = np.atleast_2d(axes)

    # Build a "header" row of titles (left = GT, then epochs)
    col_titles = ["Целевая маска (GT)"] + [f"эпоха {ep}" for ep, _ in milestones]

    # Pre-load scenes once (resize to model image_size from first runner)
    runner0 = ModelRunner(run_dir, checkpoint=milestones[0][1])
    img_size = image_size or runner0.image_size
    scenes: List[Tuple[np.ndarray, np.ndarray, list]] = []
    for gf in grasp_files:
        rgb, depth, grasps, _ = dataset_viz.load_jacquard_scene(gf, image_size=img_size)
        scenes.append((rgb, depth, grasps))

    # GT column (using each scene's mask_mode-specific target rasterisation)
    for r, (rgb, _depth, grasps) in enumerate(scenes):
        if runner0.info.mask_mode == "angle":
            tgt = rasterize_grasp_mask(grasps, (img_size, img_size), mode="angle",
                                       num_angle_bins=runner0.info.num_angle_bins)["mask"]
            gt_img = draw.overlay_angle_mask(rgb, tgt, runner0.info.num_angle_bins, alpha=0.55)
        elif runner0.info.mask_mode == "multitask":
            tgt = rasterize_grasp_mask(grasps, (img_size, img_size), mode="multitask")
            gt_img = draw.overlay_heatmap(rgb, tgt["pos"], cmap="magma", alpha=0.55)
        else:
            tgt = rasterize_grasp_mask(grasps, (img_size, img_size), mode="binary")["mask"]
            gt_img = draw.overlay_binary_mask(rgb, tgt, alpha=0.55)
        gt_img = draw.draw_grasp_list(gt_img, grasps, max_n=8,
                                       color=(0.1, 1.0, 0.2),
                                       plate_color=(1.0, 0.2, 0.2),
                                       thickness=2)
        axes[r, 0].imshow(np.clip(gt_img, 0, 1))
        axes[r, 0].set_axis_off()
        if r == 0:
            axes[r, 0].set_title(col_titles[0], fontsize=10)

    # Epoch columns
    for c, (ep, ckpt) in enumerate(milestones, start=1):
        runner = runner0 if c == 1 else ModelRunner(run_dir, checkpoint=ckpt,
                                                     device=runner0.device)
        for r, (rgb, depth, _g) in enumerate(scenes):
            ax = axes[r, c]
            overlay = _render_prediction(runner, rgb, depth, decode_cfg=decode_cfg)
            ax.imshow(np.clip(overlay, 0, 1))
            ax.set_axis_off()
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
        # free CUDA memory between epochs (if not the first runner)
        if runner is not runner0:
            del runner
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    fig.suptitle(
        title or f"Эволюция предсказаний по эпохам — {os.path.basename(run_dir)}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig
