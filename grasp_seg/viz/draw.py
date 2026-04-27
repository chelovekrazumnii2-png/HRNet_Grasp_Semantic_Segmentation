"""Primitive drawing routines used by the higher-level viz panels.

Conventions:
- All RGB images are float32 ``HxWx3`` in ``[0, 1]`` unless noted.
- All grasp coordinates follow :class:`grasp_seg.data.grasp_rect.Grasp`
  (centre is ``(y, x)``, angle in radians, length along the gripper axis).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..data.grasp_rect import Grasp
from .palette import angle_palette


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert a float [0,1] HxWx3 image to uint8 ``HxWx3`` for cv2/imshow."""
    arr = np.clip(rgb, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def colorize_depth(depth: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Return an RGB float32 ``HxWx3`` colour map of a normalised depth image."""
    import matplotlib.cm as cm
    d = np.clip(depth, 0.0, 1.0)
    rgb = cm.get_cmap(cmap)(d)[..., :3]
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Oriented-rectangle drawing
# ---------------------------------------------------------------------------

def draw_grasp_rectangle(
    img: np.ndarray,
    grasp: Grasp,
    color: Tuple[float, float, float] = (0.1, 1.0, 0.2),
    thickness: int = 2,
    plate_color: Optional[Tuple[float, float, float]] = (1.0, 0.2, 0.2),
    draw_axis: bool = True,
) -> np.ndarray:
    """Draw a single oriented grasp rectangle (4 sides + axis line) on ``img``.

    The two short sides ("plates") are drawn in ``plate_color`` to match the
    Jacquard / Cornell convention where they represent the gripper jaws.
    """
    H, W = img.shape[:2]
    out = img.copy()
    poly = grasp.as_polygon()  # (4, 2) in (y, x)
    pts = np.stack([poly[:, 1], poly[:, 0]], axis=1).astype(np.int32)  # (x, y)

    long_pairs = [(0, 1), (2, 3)]   # along grasp axis
    short_pairs = [(1, 2), (3, 0)]  # gripper plates
    for a, b in long_pairs:
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for a, b in short_pairs:
        cv2.line(
            out,
            tuple(pts[a]),
            tuple(pts[b]),
            plate_color if plate_color is not None else color,
            thickness + 1,
            cv2.LINE_AA,
        )

    if draw_axis:
        c_xy = (int(round(grasp.center[1])), int(round(grasp.center[0])))
        # Tick mark the centre so the viewer sees where the gripper closes
        cv2.circle(out, c_xy, max(2, thickness), color, -1, cv2.LINE_AA)
    return out


def draw_grasp_list(
    img: np.ndarray,
    grasps: Iterable[Grasp],
    color: Tuple[float, float, float] = (0.1, 1.0, 0.2),
    plate_color: Optional[Tuple[float, float, float]] = (1.0, 0.2, 0.2),
    thickness: int = 2,
    max_n: Optional[int] = None,
) -> np.ndarray:
    """Draw multiple grasps. Optionally cap to ``max_n`` (shuffled subset)."""
    out = img.copy()
    grasps = list(grasps)
    if max_n is not None and len(grasps) > max_n:
        idx = np.linspace(0, len(grasps) - 1, max_n).astype(int)
        grasps = [grasps[i] for i in idx]
    for g in grasps:
        out = draw_grasp_rectangle(out, g, color=color, plate_color=plate_color,
                                    thickness=thickness)
    return out


# ---------------------------------------------------------------------------
# Mask overlays
# ---------------------------------------------------------------------------

def overlay_binary_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a binary mask on top of ``img`` with the given colour."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    m = (mask > 0).astype(np.float32)[..., None]
    tint = np.broadcast_to(np.asarray(color, dtype=np.float32), img.shape)
    return img * (1.0 - alpha * m) + tint * (alpha * m)


def overlay_angle_mask(
    img: np.ndarray,
    mask: np.ndarray,
    num_bins: int,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay an angle-class mask using :func:`palette.angle_palette`.

    ``mask`` is integer-valued in ``[0, num_bins]`` (0 = background).
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    pal = angle_palette(num_bins)
    color = pal[np.clip(mask, 0, num_bins).astype(np.int64)]  # HxWx3
    fg = (mask > 0).astype(np.float32)[..., None]
    return img * (1.0 - alpha * fg) + color * (alpha * fg)


def overlay_heatmap(
    img: np.ndarray,
    heat: np.ndarray,
    cmap: str = "magma",
    alpha: float = 0.55,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Overlay a continuous ``[vmin, vmax]`` heat-map (e.g. sigmoid pos)."""
    import matplotlib.cm as cm
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h = np.clip((heat - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    color = cm.get_cmap(cmap)(h)[..., :3].astype(np.float32)
    weight = (h * alpha)[..., None]
    return img * (1.0 - weight) + color * weight


def overlay_error_map(
    img: np.ndarray,
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    fp_color: Tuple[float, float, float] = (1.0, 0.3, 0.3),
    fn_color: Tuple[float, float, float] = (0.3, 0.5, 1.0),
    tp_color: Tuple[float, float, float] = (0.3, 1.0, 0.3),
    alpha: float = 0.5,
) -> np.ndarray:
    """Highlight TP / FP / FN of a binary foreground prediction."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    pred = (pred_fg > 0).astype(np.float32)
    gt = (gt_fg > 0).astype(np.float32)
    tp = pred * gt
    fp = pred * (1.0 - gt)
    fn = (1.0 - pred) * gt
    out = img.copy()
    for m, c in [(tp, tp_color), (fp, fp_color), (fn, fn_color)]:
        m = m[..., None]
        tint = np.broadcast_to(np.asarray(c, dtype=np.float32), img.shape)
        out = out * (1.0 - alpha * m) + tint * (alpha * m)
    return out


# ---------------------------------------------------------------------------
# Captions / titles directly on the image (used when matplotlib axis titles
# are not enough — e.g. when saving stand-alone PNGs from the CLI).
# ---------------------------------------------------------------------------

def add_caption(
    img: np.ndarray,
    text: str,
    pos: str = "top",
    bg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    pad: int = 6,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Render a Russian caption strip above/below the image (cv2 text)."""
    out = img.copy()
    H, W = out.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    strip_h = th + 2 * pad
    strip = np.ones((strip_h, W, 3), dtype=np.float32) * np.asarray(bg, dtype=np.float32)
    cv2.putText(strip, text, (pad, th + pad - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, 1, cv2.LINE_AA)
    if pos == "top":
        return np.concatenate([strip, out], axis=0)
    return np.concatenate([out, strip], axis=0)


def grid(
    images: Sequence[np.ndarray],
    n_cols: int,
    pad: int = 4,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Tile equally-sized RGB images into a grid with ``pad`` px gutters."""
    if len(images) == 0:
        return np.zeros((1, 1, 3), dtype=np.float32)
    H, W = images[0].shape[:2]
    n_rows = int(np.ceil(len(images) / n_cols))
    canvas = np.ones(
        (n_rows * H + (n_rows + 1) * pad,
         n_cols * W + (n_cols + 1) * pad, 3),
        dtype=np.float32,
    ) * np.asarray(bg, dtype=np.float32)
    for i, im in enumerate(images):
        r, c = divmod(i, n_cols)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = im
    return canvas


def colors_for_n(n: int, cmap: str = "tab10") -> List[Tuple[float, float, float]]:
    """N visually-distinct RGB colours for plotting model overlays."""
    import matplotlib.cm as cm
    base = cm.get_cmap(cmap)
    return [tuple(base(i % base.N)[:3]) for i in range(n)]
