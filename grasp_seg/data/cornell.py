"""Cornell Grasp Dataset reader (Jiang et al., 2011).

For each scene index ``NNNN`` we expect:

* ``pcdNNNNr.png``    — RGB image (640×480 or similar).
* ``pcdNNNN.txt``     — point-cloud (we don't use it for visualisation).
* ``pcdNNNNcpos.txt`` — positive grasp rectangles, 4 corner ``(x, y)``
  pairs per rectangle (each rectangle takes 4 lines).
* ``pcdNNNNcneg.txt`` — negative grasp rectangles (same format).

Two dataset layouts are supported transparently:

* a **flat** directory containing all ``pcdNNNN*`` files;
* the **original** layout with 10 sub-directories ``01/`` … ``10/`` (and an
  optional ``backgrounds/`` folder, which we skip).

The loader is recursive: pass any directory that *contains* the scene
files — either flat or nested — and it will find them.

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

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .grasp_rect import Grasp, rasterize_grasp_mask


_SCENE_RE = re.compile(r"pcd(\d+)r\.png$")

# Sub-directories ignored during the recursive walk.
_IGNORED_DIRS = {"backgrounds"}


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


def _index_scenes(root: str) -> Dict[str, str]:
    """Walk ``root`` recursively and map ``scene_id → absolute RGB path``.

    Skips ``backgrounds/`` sub-directories. Hidden directories (``.git``,
    ``__pycache__``…) are also skipped.
    """
    index: Dict[str, str] = {}
    for cur, dirs, files in os.walk(root):
        # Mutate ``dirs`` in place to prune the walk.
        dirs[:] = [d for d in dirs
                   if d.lower() not in _IGNORED_DIRS
                   and not d.startswith(".")]
        for name in files:
            m = _SCENE_RE.match(name)
            if m is not None:
                index[m.group(1)] = os.path.join(cur, name)
    return index


def index_scenes(root: str) -> Dict[str, str]:
    """Public alias of :func:`_index_scenes`: ``scene_id → RGB path``.

    Useful for callers that want to load many scenes — build the index
    once and pass it to :func:`load_scene` to avoid re-walking the tree.
    """
    return _index_scenes(root)


def list_scenes(root: str, *, index: Optional[Dict[str, str]] = None) -> List[str]:
    """Return a sorted list of scene IDs (e.g. ``["0100", "0101", ...]``).

    Works for both the flat layout and the original 10-sub-directory
    Cornell layout (``01/`` … ``10/`` — ``backgrounds/`` is ignored).
    Pass a pre-built ``index`` (from :func:`index_scenes`) to avoid the
    ``os.walk`` if you already have one.
    """
    if index is None:
        index = _index_scenes(root)
    return sorted(index.keys())


def _load_scene_from_path(scene_id: str, rgb_path: str) -> CornellSample:
    """Internal worker: read RGB + cpos/cneg from a resolved RGB path."""
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    scene_dir = os.path.dirname(rgb_path)
    pos_path = os.path.join(scene_dir, f"pcd{scene_id}cpos.txt")
    neg_path = os.path.join(scene_dir, f"pcd{scene_id}cneg.txt")
    pos = [_grasp_from_corners(c) for c in _read_corners_file(pos_path)]
    neg = [_grasp_from_corners(c) for c in _read_corners_file(neg_path)]
    return CornellSample(
        scene_id=scene_id,
        rgb_path=rgb_path,
        rgb=rgb,
        pos_grasps=[g for g in pos if g is not None],
        neg_grasps=[g for g in neg if g is not None],
    )


def load_scene(
    root: str,
    scene_id: str,
    *,
    index: Optional[Dict[str, str]] = None,
) -> CornellSample:
    """Load a single Cornell scene by its 4-digit ID.

    Pass a pre-built ``index`` to skip the recursive ``os.walk``. When
    iterating over many scenes, build the index once and reuse it across
    calls (or use :func:`iter_scenes`, which already does this).
    """
    if index is not None:
        rgb_path = index.get(scene_id)
    else:
        # Fast path for a single call: try the flat layout first; only
        # walk the tree if the file isn't right at ``root``.
        flat = os.path.join(root, f"pcd{scene_id}r.png")
        rgb_path = flat if os.path.isfile(flat) else _index_scenes(root).get(scene_id)
    if rgb_path is None:
        raise FileNotFoundError(
            f"Cornell scene pcd{scene_id}r.png not found under {root}"
        )
    return _load_scene_from_path(scene_id, rgb_path)


def iter_scenes(root: str, scene_ids: Optional[List[str]] = None):
    """Iterate ``CornellSample`` instances over the dataset.

    Builds the recursive index once and reuses it across all scenes (no
    redundant ``os.walk`` per scene).
    """
    index = _index_scenes(root)
    ids = scene_ids if scene_ids is not None else sorted(index)
    for sid in ids:
        rgb_path = index.get(sid)
        if rgb_path is None:
            raise FileNotFoundError(
                f"Cornell scene pcd{sid}r.png not found under {root}"
            )
        yield _load_scene_from_path(sid, rgb_path)


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
