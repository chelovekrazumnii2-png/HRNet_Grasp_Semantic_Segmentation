"""Heatmap visualisations of *what the model sees*.

Two families of figures live here:

1. **Per-head decomposition** — for one scene, show every output head of
   the model as a separate heatmap (with a colorbar). For multitask
   models that means ``pos``, ``cos2θ``, ``sin2θ``, ``width``; for
   ``angle`` models it means ``fg_conf`` (1 − p_bg) and the argmax
   angle bin colored by :func:`palette.angle_cmap`.

2. **Grad-CAM** — for one or several scenes, compute a class-activation
   map by hooking the last shared feature stack of HRNet (``self.fuse``)
   and back-propagating the spatial mean of the foreground / ``pos``
   logit. The result is upsampled to the model image size, normalised
   to ``[0, 1]``, and rendered as a standalone colormapped heatmap
   alongside the RGB (no underlay).

Both figures use :func:`compare_viz._scene_to_model_space` for Cornell
inputs to preserve aspect ratio (pad-to-square + uniform resize),
matching the rest of the visualisation pipeline.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..data.cornell import CornellSample
from . import dataset_viz, palette
from .inference import ModelRunner

Source = Union[str, CornellSample, Tuple[np.ndarray, Optional[np.ndarray]]]


# ---------------------------------------------------------------------------
# Input normalisation: Jacquard path, CornellSample, or precomputed (rgb, depth)
# ---------------------------------------------------------------------------

def _prepare_inputs(
    runner: ModelRunner,
    source: Source,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return ``(rgb_m, depth_m, label)`` in model coordinate frame.

    Cornell scenes are pad-to-square + uniform-resized so aspect ratio is
    preserved; Jacquard scenes are already square and only need a uniform
    resize down to ``runner.image_size``.
    """
    S = runner.image_size
    if isinstance(source, str):
        rgb, depth, _, _ = dataset_viz.load_jacquard_scene(source, image_size=S)
        label = os.path.basename(source).replace("_grasps.txt", "")
        return rgb, depth, label

    if isinstance(source, CornellSample):
        # Local import avoids a circular dependency: compare_viz already
        # imports heatmap_viz indirectly via grasp_seg.viz package init.
        from .compare_viz import _scene_to_model_space
        rgb_m, depth_m, _, _, _, _ = _scene_to_model_space(
            source.rgb, source.depth, [], S,
        )
        return rgb_m, depth_m, f"cornell {source.scene_id}"

    rgb, depth = source
    if rgb.shape[:2] != (S, S):
        rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_LINEAR)
    if depth is not None and depth.shape != (S, S):
        depth = cv2.resize(depth, (S, S), interpolation=cv2.INTER_LINEAR)
    return rgb, depth if depth is not None else np.zeros((S, S), dtype=np.float32), ""


# ---------------------------------------------------------------------------
# Per-head decomposition
# ---------------------------------------------------------------------------

def _imshow_heatmap(ax, data: np.ndarray, *, cmap: str, vmin=None, vmax=None,
                     title: Optional[str] = None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def figure_per_head_heatmap(
    runner: ModelRunner,
    source: Source,
    *,
    title: Optional[str] = None,
):
    """Show every output head of ``runner`` for one scene.

    Layout depends on ``runner.info.mask_mode``:

    * ``multitask`` — 2×3 grid: RGB | depth | pos | cos2θ | sin2θ | width.
    * ``angle``     — 1×4 grid: RGB | depth | fg_conf | argmax (colored).
    * ``binary``    — 1×3 grid: RGB | depth | pos.
    """
    rgb_m, depth_m, label = _prepare_inputs(runner, source)
    pred = runner.predict(rgb=rgb_m, depth=depth_m)
    rgb_disp = np.clip(rgb_m, 0, 1)
    mode = runner.info.mask_mode

    if mode == "multitask":
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(rgb_disp); axes[0, 0].set_axis_off()
        axes[0, 0].set_title("RGB", fontsize=10)
        _imshow_heatmap(axes[0, 1], depth_m, cmap="viridis", title="depth (norm)")
        _imshow_heatmap(axes[0, 2], pred["pos"], cmap="magma",
                        vmin=0.0, vmax=1.0, title="pos (graspability)")
        _imshow_heatmap(axes[1, 0], pred["cos2t"], cmap="RdBu",
                        vmin=-1.0, vmax=1.0, title="cos 2θ")
        _imshow_heatmap(axes[1, 1], pred["sin2t"], cmap="RdBu",
                        vmin=-1.0, vmax=1.0, title="sin 2θ")
        _imshow_heatmap(axes[1, 2], pred["width"], cmap="viridis",
                        vmin=0.0, vmax=1.0, title="width (norm)")

    elif mode == "angle":
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        axes[0].imshow(rgb_disp); axes[0].set_axis_off()
        axes[0].set_title("RGB", fontsize=10)
        _imshow_heatmap(axes[1], depth_m, cmap="viridis", title="depth (norm)")
        _imshow_heatmap(axes[2], pred["fg_conf"], cmap="magma",
                        vmin=0.0, vmax=1.0, title="fg_conf (1 − p_bg)")
        K = runner.info.num_angle_bins
        cmap = palette.angle_cmap(K)
        im = axes[3].imshow(pred["argmax"], cmap=cmap, vmin=0, vmax=K)
        axes[3].set_axis_off()
        axes[3].set_title("argmax angle", fontsize=10)
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02,
                     ticks=[0, K // 2, K])

    elif mode == "binary":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        axes[0].imshow(rgb_disp); axes[0].set_axis_off()
        axes[0].set_title("RGB", fontsize=10)
        _imshow_heatmap(axes[1], depth_m, cmap="viridis", title="depth (norm)")
        _imshow_heatmap(axes[2], pred["pos"], cmap="magma",
                        vmin=0.0, vmax=1.0, title="pos (sigmoid)")
    else:
        raise ValueError(f"Unsupported mask_mode {mode!r}")

    fig.suptitle(
        title or f"{runner.info.name} | {label}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def _grad_cam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last shared conv stack — what we hook for Grad-CAM."""
    if hasattr(model, "_seg"):  # HRNetMultiTask
        return model._seg.fuse
    if hasattr(model, "fuse"):  # HRNetSeg (binary / angle)
        return model.fuse
    raise AttributeError(
        f"Could not locate a 'fuse' module on {type(model).__name__}; "
        "Grad-CAM expects an HRNet-style architecture."
    )


def _target_scalar(out, mode: str) -> torch.Tensor:
    """Single scalar to back-propagate from, per ``mask_mode``.

    Higher = "more grasp evidence". Sigmoid / softmax are monotonic in
    these logits, so it's fine (and cheaper) to differentiate the raw
    logits rather than the activated output.
    """
    if mode == "multitask":
        return out["pos"].mean()
    if mode == "binary":
        if out.dim() == 4 and out.shape[1] == 1:
            out = out[:, 0]
        return out.mean()
    if mode == "angle":
        # sum of foreground logits across angle bins (skip index 0 = bg)
        return out[:, 1:].mean()
    raise ValueError(f"Unsupported mask_mode {mode!r}")


def compute_grad_cam(
    runner: ModelRunner,
    rgb: np.ndarray,
    depth: Optional[np.ndarray],
) -> np.ndarray:
    """Return a ``(image_size, image_size)`` Grad-CAM map in ``[0, 1]``.

    The map is non-negative (we ReLU after the channel-weighted sum) and
    upsampled bilinearly from the ``self.fuse`` resolution to model
    image size.
    """
    model = runner.model
    target_layer = _grad_cam_target_layer(model)

    saved = {}

    def fwd_hook(_module, _inp, output):
        saved["A"] = output

    def bwd_hook(_module, _grad_in, grad_out):
        saved["G"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    was_training = model.training
    requires_grad_state = [p.requires_grad for p in model.parameters()]

    try:
        model.eval()
        # We don't need parameter gradients — only grads through the
        # activations the hook captured. Set the input to require_grad
        # so the autograd graph still gets built end-to-end.
        for p in model.parameters():
            p.requires_grad_(False)

        x = runner.preprocess(rgb, depth).clone().detach()
        x.requires_grad_(True)
        with torch.enable_grad():
            out = model(x)
            score = _target_scalar(out, runner.info.mask_mode)
            score.backward()

        A = saved["A"].detach()  # (1, C, h, w)
        G = saved["G"].detach()  # (1, C, h, w)
        weights = G.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * A).sum(dim=1)              # (1, h, w)
        cam = torch.relu(cam)

        cam = F.interpolate(cam[None],
                            size=(runner.image_size, runner.image_size),
                            mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam.float().cpu().numpy()
        max_val = float(cam_np.max())
        if max_val > 0:
            cam_np = cam_np / max_val
        return cam_np
    finally:
        h1.remove()
        h2.remove()
        if was_training:
            model.train()
        for p, rg in zip(model.parameters(), requires_grad_state):
            p.requires_grad_(rg)


def figure_grad_cam(
    runner: ModelRunner,
    sources: Sequence[Source],
    *,
    n_cols: int = 3,
    alpha: float = 0.55,
    cmap: str = "jet",
    title: Optional[str] = None,
):
    """Grad-CAM grid: for every scene render RGB + standalone heatmap.

    Each panel pair is ``[RGB][CAM]``; the right panel shows the
    colormapped Grad-CAM map by itself (no RGB underlay) so the
    saliency is unambiguous to read. Rows wrap at ``n_cols`` panel
    pairs (visual width = ``2 * n_cols``).

    ``alpha`` is kept for backward-compat but no longer used.
    """
    del alpha  # standalone heatmap; underlay was removed by request
    if not sources:
        raise ValueError("sources must be non-empty")

    panels: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for src in sources:
        rgb_m, depth_m, label = _prepare_inputs(runner, src)
        cam = compute_grad_cam(runner, rgb_m, depth_m)
        panels.append((np.clip(rgb_m, 0, 1), cam, label))

    n_rows = max(1, int(math.ceil(len(panels) / n_cols)))
    fig, axes = plt.subplots(
        n_rows, 2 * n_cols,
        figsize=(2.8 * 2 * n_cols, 2.8 * n_rows),
    )
    axes = np.atleast_2d(axes)

    for i, (rgb_disp, cam, label) in enumerate(panels):
        r = i // n_cols
        c = i % n_cols
        ax_rgb = axes[r, 2 * c]
        ax_cam = axes[r, 2 * c + 1]
        ax_rgb.imshow(rgb_disp); ax_rgb.set_axis_off()
        ax_rgb.set_title(label or "RGB", fontsize=9)

        ax_cam.imshow(cam, cmap=cmap, vmin=0.0, vmax=1.0)
        ax_cam.set_axis_off()
        ax_cam.set_title("Grad-CAM", fontsize=9)

    # Hide unused panel pairs.
    for j in range(len(panels), n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, 2 * c].set_axis_off()
        axes[r, 2 * c + 1].set_axis_off()

    fig.suptitle(
        title or f"Grad-CAM — {runner.info.name} ({runner.info.mask_mode})",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
