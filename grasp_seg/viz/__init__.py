"""Visualisation utilities for the HRNet grasp-segmentation pipeline.

The submodules are kept small and independent so they can be used either
from the CLI (``tools/visualize.py``) or interactively from the demo
notebook (``notebooks/visualize.ipynb``):

- :mod:`palette`       — colour tables for angle bins / multitask channels
- :mod:`draw`          — oriented-rectangle and mask-overlay rendering
- :mod:`decoder`       — predicted mask → grasp rectangles (+ NMS, IoU match)
- :mod:`inference`     — checkpoint loading and a unified ``predict`` API
- :mod:`dataset_viz`   — raw sample / resized / masks / augmentation panels
- :mod:`metrics_viz`   — train/val plots from ``metrics.csv``
- :mod:`epoch_evolution` — predictions across saved epoch checkpoints
- :mod:`eval_viz`      — best-epoch qualitative + per-class IoU panels
- :mod:`compare_viz`   — side-by-side comparison of multiple models
- :mod:`extra_viz`     — depth-contribution / IoU×angle / failure catalog
- :mod:`cornell_eval`  — quantitative top-1 grasp accuracy on Cornell
"""

from . import (
    compare_viz,
    cornell_eval,
    dataset_viz,
    decoder,
    draw,
    epoch_evolution,
    eval_viz,
    extra_viz,
    inference,
    metrics_viz,
    palette,
)

__all__ = [
    "palette",
    "draw",
    "decoder",
    "inference",
    "dataset_viz",
    "metrics_viz",
    "epoch_evolution",
    "eval_viz",
    "compare_viz",
    "extra_viz",
    "cornell_eval",
]
