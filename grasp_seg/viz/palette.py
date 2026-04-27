"""Colour palettes for grasp-segmentation visualisations.

All entries return ``(N, 3)`` ``float32`` RGB arrays in [0, 1] so they
plug into matplotlib (``ListedColormap``) and ``cv2.addWeighted`` overlays
identically.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap, hsv_to_rgb


def angle_palette(num_bins: int) -> np.ndarray:
    """Return an ``(num_bins + 1, 3)`` palette for ``mask_mode='angle'``.

    Index 0 is reserved for *background* (transparent black). Indices
    ``1..num_bins`` walk the hue circle so visually adjacent angle bins
    get visually adjacent colours.
    """
    palette = np.zeros((num_bins + 1, 3), dtype=np.float32)
    hues = np.linspace(0.0, 1.0, num_bins, endpoint=False)
    hsv = np.stack([hues, np.full_like(hues, 0.85), np.full_like(hues, 0.95)], axis=-1)
    palette[1:] = hsv_to_rgb(hsv).astype(np.float32)
    return palette


def angle_cmap(num_bins: int) -> ListedColormap:
    """Matplotlib colormap matching :func:`angle_palette` (background black)."""
    return ListedColormap(angle_palette(num_bins), name=f"angle_{num_bins}")


def angle_bin_centers_deg(num_bins: int) -> np.ndarray:
    """Return centres of each angle bin in degrees, in ``[0, 180)``."""
    bin_w = 180.0 / num_bins
    return (np.arange(num_bins, dtype=np.float32) + 0.5) * bin_w


def angle_label_ru(num_bins: int) -> list:
    """Russian labels for the legend: ``Угол ≈ NN°``."""
    centers = angle_bin_centers_deg(num_bins)
    return ["Фон"] + [f"≈ {c:.0f}°" for c in centers]


# Friendly two-tone overlay for the binary-mode pos map (background → red→yellow).
POS_HEATMAP = "magma"


# Channel labels for the multitask head.
MULTITASK_CHANNELS_RU = {
    "pos": "Позиция захвата (sigmoid)",
    "cos2t": "cos(2θ)",
    "sin2t": "sin(2θ)",
    "width": "Ширина гриппера",
    "angle_deg": "Угол захвата θ, °",
}
