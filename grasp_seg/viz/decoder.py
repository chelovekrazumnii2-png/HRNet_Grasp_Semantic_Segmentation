"""Decode predicted masks back into oriented grasp rectangles.

Two model families are supported:

* ``mode='angle'``    — input is an ``HxW`` integer label map with values
  ``0..K`` (0 = background, 1..K = angle bin); we additionally consume the
  raw foreground confidence (per-pixel maximum softmax over angle classes,
  i.e. ``1 - softmax[..., 0]``) when available so peak-finding is sharper.

* ``mode='multitask'`` — input is the GG-CNN-style 4-channel head:
  ``pos`` (logits), ``cos2t``, ``sin2t``, ``width`` (logits or ``[0,1]``).
  Angle is recovered as ``θ = ½·atan2(sin2θ, cos2θ)`` and width as
  ``sigmoid(width)·width_scale``.

In both cases we follow the GR-ConvNet / GG-CNN recipe:

1. Smooth the confidence map (Gaussian, σ ≈ 2 px on 384-px input).
2. Find local maxima above a threshold ``conf_thresh``.
3. Rank by confidence and apply NMS in ``(centre, angle)`` space — two
   peaks are merged if their centres are within ``nms_dist`` px AND the
   absolute angle difference (mod π) is below ``nms_angle_deg``.
4. Build a :class:`grasp_seg.data.grasp_rect.Grasp` for each survivor.

For the ``angle`` mode the gripper width and length aren't predicted, so
we use defaults consistent with how the training masks were rasterised
(``length_scale=1/3`` of a fixed ``length_default`` px, ``width_default``
px wide). These defaults can be tuned for visual readability without
changing the training pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from ..data.grasp_rect import Grasp


# ---------------------------------------------------------------------------
# Decoding configuration
# ---------------------------------------------------------------------------

@dataclass
class DecodeConfig:
    smooth_sigma: float = 2.0
    conf_thresh: float = 0.4
    nms_dist_px: float = 12.0
    nms_angle_deg: float = 15.0
    max_grasps: int = 20
    # Used only when the decoder has no width/length channel (angle mode).
    default_length_px: float = 80.0
    default_width_px: float = 30.0
    # Width-channel rescaling for multitask: predicted (sigmoid) width in
    # [0, 1] is mapped to [0, max_width_px]. Matches the rasterisation
    # convention where target widths come straight from grasp_rect annotations.
    max_width_px: float = 150.0
    # Length factor for multitask: GG-CNN predicts width along the gripper
    # axis only via the ``width`` channel. We assume length ≈ 2.5×width
    # for visualisation (this is purely a display heuristic and never
    # affects matching/IoU scoring of GT rectangles).
    length_to_width_ratio: float = 2.5


# ---------------------------------------------------------------------------
# Peak finding + NMS
# ---------------------------------------------------------------------------

def _local_maxima(
    score: np.ndarray,
    min_value: float,
    nbhd_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(yy, xx)`` of local maxima ≥ ``min_value`` in ``score``."""
    nbhd = max(int(round(nbhd_px)), 1)
    pooled = maximum_filter(score, size=2 * nbhd + 1)
    mask = (score == pooled) & (score >= min_value)
    yy, xx = np.where(mask)
    return yy, xx


def _angle_aware_nms(
    centers_yx: np.ndarray,
    angles_rad: np.ndarray,
    scores: np.ndarray,
    dist_px: float,
    angle_deg: float,
    max_keep: int,
) -> np.ndarray:
    """Suppress overlapping peaks in (centre, angle) space.

    Returns the kept indices, ordered by descending ``scores``.
    """
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: List[int] = []
    angle_thresh = math.radians(angle_deg)
    for i in order:
        ok = True
        for j in keep:
            dy = centers_yx[i, 0] - centers_yx[j, 0]
            dx = centers_yx[i, 1] - centers_yx[j, 1]
            if dy * dy + dx * dx > dist_px * dist_px:
                continue
            d_ang = abs(angles_rad[i] - angles_rad[j])
            d_ang = min(d_ang, math.pi - d_ang)
            if d_ang < angle_thresh:
                ok = False
                break
        if ok:
            keep.append(int(i))
            if len(keep) >= max_keep:
                break
    return np.asarray(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Public decoders
# ---------------------------------------------------------------------------

def decode_angle(
    confidence: np.ndarray,
    angle_classes: np.ndarray,
    num_bins: int,
    cfg: Optional[DecodeConfig] = None,
) -> List[Tuple[Grasp, float]]:
    """Decode an ``angle``-mode prediction.

    Parameters
    ----------
    confidence
        ``HxW`` foreground confidence in ``[0, 1]`` (e.g. ``1 - p_bg``).
    angle_classes
        ``HxW`` integer angle-bin labels in ``[0, num_bins]`` (0 = background).
    num_bins
        ``num_angle_bins`` in the model config.
    cfg
        Decoding hyper-parameters. ``None`` uses defaults.
    """
    cfg = cfg or DecodeConfig()
    score = gaussian_filter(confidence.astype(np.float32), sigma=cfg.smooth_sigma)
    yy, xx = _local_maxima(score, cfg.conf_thresh, nbhd_px=int(cfg.nms_dist_px))
    if len(yy) == 0:
        return []

    bin_w_rad = math.pi / num_bins
    bins = angle_classes[yy, xx]
    valid = bins > 0
    if valid.sum() == 0:
        return []
    yy, xx = yy[valid], xx[valid]
    bins = bins[valid]
    # bin centre in [0, π); shift to (-π/2, π/2] using the same convention
    # as :class:`Grasp` (positive-y rotation).
    centres_rad = (bins.astype(np.float32) - 0.5) * bin_w_rad
    centres_rad = (centres_rad + math.pi / 2.0) % math.pi - math.pi / 2.0
    confs = score[yy, xx]
    centers_yx = np.stack([yy, xx], axis=1).astype(np.float32)

    keep = _angle_aware_nms(
        centers_yx, centres_rad, confs,
        dist_px=cfg.nms_dist_px,
        angle_deg=cfg.nms_angle_deg,
        max_keep=cfg.max_grasps,
    )

    grasps: List[Tuple[Grasp, float]] = []
    for i in keep:
        g = Grasp(
            center=np.array([centers_yx[i, 0], centers_yx[i, 1]], dtype=np.float64),
            angle=float(centres_rad[i]),
            length=float(cfg.default_length_px),
            width=float(cfg.default_width_px),
        )
        grasps.append((g, float(confs[i])))
    return grasps


def decode_multitask(
    pos_sigmoid: np.ndarray,
    cos2t: np.ndarray,
    sin2t: np.ndarray,
    width_sigmoid: np.ndarray,
    cfg: Optional[DecodeConfig] = None,
) -> List[Tuple[Grasp, float]]:
    """Decode a ``multitask``-mode prediction (GG-CNN style)."""
    cfg = cfg or DecodeConfig()
    pos = gaussian_filter(pos_sigmoid.astype(np.float32), sigma=cfg.smooth_sigma)
    yy, xx = _local_maxima(pos, cfg.conf_thresh, nbhd_px=int(cfg.nms_dist_px))
    if len(yy) == 0:
        return []

    cos_v = cos2t[yy, xx]
    sin_v = sin2t[yy, xx]
    angles = 0.5 * np.arctan2(sin_v, cos_v)
    angles = (angles + math.pi / 2.0) % math.pi - math.pi / 2.0

    widths_n = np.clip(width_sigmoid[yy, xx], 0.0, 1.0)
    widths_px = widths_n * cfg.max_width_px
    lengths_px = widths_px * cfg.length_to_width_ratio
    confs = pos[yy, xx]

    centers_yx = np.stack([yy, xx], axis=1).astype(np.float32)
    keep = _angle_aware_nms(
        centers_yx, angles.astype(np.float32), confs.astype(np.float32),
        dist_px=cfg.nms_dist_px,
        angle_deg=cfg.nms_angle_deg,
        max_keep=cfg.max_grasps,
    )

    grasps: List[Tuple[Grasp, float]] = []
    for i in keep:
        g = Grasp(
            center=np.array([centers_yx[i, 0], centers_yx[i, 1]], dtype=np.float64),
            angle=float(angles[i]),
            length=max(float(lengths_px[i]), 4.0),
            width=max(float(widths_px[i]), 2.0),
        )
        grasps.append((g, float(confs[i])))
    return grasps


# ---------------------------------------------------------------------------
# Grasp-rectangle IoU + Jacquard / Cornell standard match
# ---------------------------------------------------------------------------

def _polygon_area(poly_yx: np.ndarray) -> float:
    """Shoelace area of a 4-corner polygon (rows are (y, x))."""
    y = poly_yx[:, 0]
    x = poly_yx[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_clip(subj: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman polygon clipping. Polygons are CCW in (y, x)."""
    out = subj.tolist()
    if len(out) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    cp1 = clip[-1]
    for cp2 in clip:
        inp = out
        out = []
        if not inp:
            break
        s = inp[-1]
        edge_dy = cp2[0] - cp1[0]
        edge_dx = cp2[1] - cp1[1]

        def _inside(p):
            # Inside if p is to the left of the directed edge cp1→cp2
            return (cp2[1] - cp1[1]) * (p[0] - cp1[0]) - (cp2[0] - cp1[0]) * (p[1] - cp1[1]) >= 0.0

        def _intersect(a, b):
            x1, y1 = a[1], a[0]
            x2, y2 = b[1], b[0]
            x3, y3 = cp1[1], cp1[0]
            x4, y4 = cp2[1], cp2[0]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-12:
                return a  # degenerate, fall back to source vertex
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return [iy, ix]

        for e in inp:
            if _inside(e):
                if not _inside(s):
                    out.append(_intersect(s, e))
                out.append(e)
            elif _inside(s):
                out.append(_intersect(s, e))
            s = e
        cp1 = cp2
    return np.asarray(out, dtype=np.float64)


def grasp_iou(a: Grasp, b: Grasp) -> float:
    """Polygon IoU between two oriented grasp rectangles."""
    pa = a.as_polygon().astype(np.float64)
    pb = b.as_polygon().astype(np.float64)
    inter = _polygon_clip(pa, pb)
    if inter.shape[0] < 3:
        return 0.0
    inter_area = _polygon_area(inter)
    union = _polygon_area(pa) + _polygon_area(pb) - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def angle_diff_deg(a: Grasp, b: Grasp) -> float:
    """Absolute angle difference in degrees, wrapped to [0, 90]°."""
    d = abs(a.angle - b.angle)
    d = min(d, math.pi - d)
    return math.degrees(d)


def jacquard_match(
    pred: Grasp,
    ground_truth: Sequence[Grasp],
    iou_thresh: float = 0.25,
    angle_thresh_deg: float = 30.0,
) -> Tuple[bool, float, float]:
    """Jacquard / Cornell standard: a predicted grasp is "correct" if there
    exists at least one GT grasp with ``IoU > iou_thresh`` and angular
    error ``< angle_thresh_deg``.

    Returns ``(matched, best_iou, best_angle_err_deg)``.
    """
    best_iou = 0.0
    best_ang = 180.0
    matched = False
    for gt in ground_truth:
        ang = angle_diff_deg(pred, gt)
        if ang > angle_thresh_deg:
            continue
        iou = grasp_iou(pred, gt)
        if iou > best_iou:
            best_iou = iou
            best_ang = ang
        if iou > iou_thresh and ang < angle_thresh_deg:
            matched = True
    return matched, float(best_iou), float(best_ang)


def topk_correct_rate(
    predictions: Iterable[Tuple[Grasp, float]],
    ground_truth: Sequence[Grasp],
    k: int = 1,
    **kw,
) -> float:
    """Return 1.0 if any of the top-``k`` predictions matches a GT grasp.

    Predictions must be supplied as ``(grasp, score)`` pairs, ordered
    arbitrarily — we re-sort by score descending.
    """
    sorted_preds = sorted(predictions, key=lambda gs: -gs[1])[:k]
    for g, _ in sorted_preds:
        ok, _, _ = jacquard_match(g, ground_truth, **kw)
        if ok:
            return 1.0
    return 0.0
