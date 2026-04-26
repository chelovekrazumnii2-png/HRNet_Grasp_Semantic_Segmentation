"""Grasp rectangle utilities adapted from the official Jacquard V2 toolbox.

Each line of a Jacquard `*_grasps.txt` file encodes a single grasp pose as
``x;y;theta;w;h`` with ``theta`` in degrees and ``(x, y)`` the grasp center
in pixel coordinates of the original 1024x1024 image.

We expose a small object-oriented wrapper around these annotations plus a
single function ``rasterize_grasp_mask`` that turns the grasp list into the
training target our HRNet head consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from skimage.draw import polygon


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass
class Grasp:
    """A single grasp parametrised by center, angle (rad), length (w), width (h)."""

    center: np.ndarray  # shape (2,), order (y, x)
    angle: float        # radians, in [-pi/2, pi/2)
    length: float       # gripper opening, along the grasp axis
    width: float        # gripper plate height, perpendicular

    def as_polygon(self) -> np.ndarray:
        """Return 4x2 corner array (rows are (y, x))."""
        l, w = self.length / 2.0, self.width / 2.0
        # corners in local frame (axis-aligned)
        corners = np.array([
            [-w, -l],
            [-w,  l],
            [ w,  l],
            [ w, -l],
        ], dtype=np.float64)  # (y, x)
        R = _rotation_matrix(self.angle)
        # rotate (note: our angle convention follows the Jacquard toolbox)
        rotated = corners @ R.T
        return rotated + self.center  # broadcast (y, x)

    def compact(self, length_scale: float = 1.0 / 3.0) -> "Grasp":
        return Grasp(self.center, self.angle, self.length * length_scale, self.width)

    def angle_deg_mod180(self) -> float:
        """Angle in degrees, wrapped to [0, 180)."""
        return float((np.degrees(self.angle) % 180.0))


def _parse_jacquard_line(line: str) -> Grasp:
    parts = line.strip().split(";")
    if len(parts) != 5:
        raise ValueError(f"Bad grasp line: {line!r}")
    x, y, theta, w, h = (float(v) for v in parts)
    # Jacquard angle convention is flipped (see the official toolbox):
    angle = -np.deg2rad(theta)
    return Grasp(center=np.array([y, x], dtype=np.float64), angle=angle, length=w, width=h)


def load_jacquard_grasps(fname: str, scale: float = 1.0) -> List[Grasp]:
    grasps: List[Grasp] = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = _parse_jacquard_line(line)
            except ValueError:
                continue
            if scale != 1.0:
                g = Grasp(g.center * scale, g.angle, g.length * scale, g.width * scale)
            grasps.append(g)
    return grasps


def _polygon_pixels(grasp: Grasp, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    poly = grasp.as_polygon()
    rr, cc = polygon(poly[:, 0], poly[:, 1], shape=shape)
    return rr, cc


def rasterize_grasp_mask(
    grasps: List[Grasp],
    shape: Tuple[int, int],
    mode: str = "angle",
    num_angle_bins: int = 18,
    length_scale: float = 1.0 / 3.0,
    width_norm: float = 150.0,
) -> dict:
    """Rasterise a list of grasps into the training target.

    Parameters
    ----------
    grasps : list[Grasp]
        Already scaled to ``shape``.
    shape : (H, W)
        Output spatial size.
    mode : {"binary", "angle", "multitask"}
        - ``binary``: returns ``{"mask": HxW uint8 (0/1)}`` — Q-map.
        - ``angle`` (default): returns ``{"mask": HxW int64}`` with values
          ``0`` for background and ``1..K`` for angle bins of width
          ``180/K`` degrees.
        - ``multitask``: returns ``{"pos": HxW float32, "cos2t": HxW float32,
          "sin2t": HxW float32, "width": HxW float32}`` — GG-CNN-style.
    num_angle_bins : int
        Only used for ``mode="angle"``.
    length_scale : float
        Compact-polygon shrinking factor along the grasp axis. Default
        ``1/3`` matches the official Jacquard toolbox.
    """
    H, W = shape
    if mode == "binary":
        m = np.zeros((H, W), dtype=np.uint8)
        for g in grasps:
            rr, cc = _polygon_pixels(g.compact(length_scale), (H, W))
            m[rr, cc] = 1
        return {"mask": m}

    if mode == "angle":
        if num_angle_bins <= 0:
            raise ValueError("num_angle_bins must be positive")
        bin_width_deg = 180.0 / num_angle_bins
        m = np.zeros((H, W), dtype=np.int64)
        # Sort by area descending so larger grasps are written first and
        # smaller ones overwrite — keeps small but specific grasps visible.
        sorted_grasps = sorted(grasps, key=lambda g: g.length * g.width, reverse=True)
        for g in sorted_grasps:
            rr, cc = _polygon_pixels(g.compact(length_scale), (H, W))
            cls = int(g.angle_deg_mod180() // bin_width_deg) + 1
            cls = min(cls, num_angle_bins)  # clamp degenerate 180-degree
            m[rr, cc] = cls
        return {"mask": m}

    if mode == "multitask":
        pos = np.zeros((H, W), dtype=np.float32)
        cos2t = np.zeros((H, W), dtype=np.float32)
        sin2t = np.zeros((H, W), dtype=np.float32)
        widthm = np.zeros((H, W), dtype=np.float32)
        for g in grasps:
            rr, cc = _polygon_pixels(g.compact(length_scale), (H, W))
            pos[rr, cc] = 1.0
            cos2t[rr, cc] = float(np.cos(2.0 * g.angle))
            sin2t[rr, cc] = float(np.sin(2.0 * g.angle))
            widthm[rr, cc] = float(np.clip(g.length, 0.0, width_norm) / width_norm)
        return {"pos": pos, "cos2t": cos2t, "sin2t": sin2t, "width": widthm}

    raise ValueError(f"Unknown mask mode: {mode!r}")
