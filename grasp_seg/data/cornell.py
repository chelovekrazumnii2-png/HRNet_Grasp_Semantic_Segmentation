"""Cornell Grasp Dataset reader (Jiang et al., 2011).

The Cornell dataset is laid out as a flat directory of ``pcdNNNN*`` files.
For each scene index ``NNNN`` we expect:

* ``pcdNNNNr.png``    — RGB image (640×480 or similar).
* ``pcdNNNN.txt``     — point-cloud (we don't use it for visualisation).
* ``pcdNNNNcpos.txt`` — positive grasp rectangles, 4 corner ``(x, y)``
  pairs per rectangle (each rectangle takes 4 lines).
* ``pcdNNNNcneg.txt`` — negative grasp rectangles (same format).

The dataset has no per-pixel ground-truth segmentation mask. We expose
the rectangles as :class:`grasp_seg.data.grasp_rect.Grasp` instances
(``angle`` recovered from the long edge of the corner polygon and
length/width inferred from edge magnitudes), and an optional helper to
*rasterise* them with the same compact-polygon (``length_scale=1/3``)
trick used by :class:`grasp_seg.data.grasp_rect.rasterize_grasp_mask`.

Use this loader strictly for inference / cross-domain visualisation —
the project trains exclusively on Jacquard V2.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .grasp_rect import Grasp, rasterize_grasp_mask


_SCENE_RE = re.compile(r"pcd(\d+)r\.png$")


def _grasp_from_corners(corners: np.ndarray) -> Optional[Grasp]:
    """Build a :class:`Grasp` from a 4×2 ``(y, x)`` corner array.

    Cornell rectangles are not guaranteed to be axis-aligned and may
    contain ``NaN``s for partial annotations — those are skipped.
    """
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return None
    p0, p1, p2, p3 = corners
    e01 = p1 - p0  # (dy, dx)
    e12 = p2 - p1
    len_a = float(np.linalg.norm(e01))
    len_b = float(np.linalg.norm(e12))
    if len_a < 1.0 or len_b < 1.0:
        return None
    # The "length" axis is along the gripper opening; in Cornell it is
    # the long edge. We pick whichever of ``e01`` / ``e12`` is longer.
    if len_a >= len_b:
        long_edge = e01
        length = len_a
        width = len_b
    else:
        long_edge = e12
        length = len_b
        width = len_a
    # Angle of the long edge (atan2 takes (dy, dx)). Rotation matrix in
    # :class:`Grasp` rotates *vertical* corners around the centre; the
    # default polygon has the long axis along *x*, so we want
    # ``angle = atan2(dy, dx)`` — wrapped to (-π/2, π/2].
    theta = float(np.arctan2(long_edge[0], long_edge[1]))
    theta = (theta + np.pi / 2.0) % np.pi - np.pi / 2.0
    centre = corners.mean(axis=0)
    return Grasp(center=centre.astype(np.float64), angle=theta,
                 length=float(length), width=float(width))


def _read_corners_file(path: str) -> List[np.ndarray]:
    """Parse a Cornell ``cpos`` / ``cneg`` file into a list of 4×2 arrays."""
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rects: List[np.ndarray] = []
    for i in range(0, len(lines) - 3, 4):
        try:
            xy = np.array([
                [float(v) for v in ln.split()] for ln in lines[i:i + 4]
            ], dtype=np.float64)  # (4, 2) as (x, y)
        except ValueError:
            continue
        if xy.shape != (4, 2):
            continue
        # Convert to (y, x) to match the rest of the project.
        yx = np.stack([xy[:, 1], xy[:, 0]], axis=1)
        rects.append(yx)
    return rects


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CornellSample:
    scene_id: str
    rgb_path: str
    rgb: np.ndarray            # float32 HxWx3 in [0, 1]
    pos_grasps: List[Grasp]
    neg_grasps: List[Grasp]


def list_scenes(root: str) -> List[str]:
    """Return a sorted list of scene IDs (e.g. ``["0100", "0101", ...]``)."""
    out = []
    for name in os.listdir(root):
        m = _SCENE_RE.match(name)
        if m is not None:
            out.append(m.group(1))
    out.sort()
    return out


def load_scene(root: str, scene_id: str) -> CornellSample:
    """Load a single Cornell scene by its 4-digit ID."""
    rgb_path = os.path.join(root, f"pcd{scene_id}r.png")
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pos_path = os.path.join(root, f"pcd{scene_id}cpos.txt")
    neg_path = os.path.join(root, f"pcd{scene_id}cneg.txt")
    pos = [_grasp_from_corners(c) for c in _read_corners_file(pos_path)]
    neg = [_grasp_from_corners(c) for c in _read_corners_file(neg_path)]
    return CornellSample(
        scene_id=scene_id,
        rgb_path=rgb_path,
        rgb=rgb,
        pos_grasps=[g for g in pos if g is not None],
        neg_grasps=[g for g in neg if g is not None],
    )


def iter_scenes(root: str, scene_ids: Optional[List[str]] = None):
    """Iterate ``CornellSample`` instances over the dataset."""
    ids = scene_ids if scene_ids is not None else list_scenes(root)
    for sid in ids:
        yield load_scene(root, sid)


def rasterize_cornell_mask(
    grasps: List[Grasp],
    shape: Tuple[int, int],
    mode: str = "binary",
    num_angle_bins: int = 18,
    length_scale: float = 1.0 / 3.0,
) -> dict:
    """Rasterise Cornell positive grasps into a pseudo ground-truth mask.

    This is a convenience used by the visualisation panels — Cornell does
    not officially publish per-pixel masks. Predicted-vs-rasterised IoU
    therefore carries an inherent rasterisation bias and should be read
    as a *qualitative* check; the canonical Cornell metric is the
    Jacquard-style grasp-rectangle accuracy implemented in
    :func:`grasp_seg.viz.decoder.jacquard_match`.
    """
    return rasterize_grasp_mask(
        grasps, shape, mode=mode, num_angle_bins=num_angle_bins,
        length_scale=length_scale,
    )
