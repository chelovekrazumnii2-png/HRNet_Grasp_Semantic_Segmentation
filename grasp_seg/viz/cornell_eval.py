"""Quantitative evaluation of grasp models on Cornell Grasp Dataset.

The standard Jacquard / Cornell metric:
- A predicted grasp is "correct" if there exists at least one positive
  GT grasp with ``IoU > 0.25`` (``iou_thresh``) and angular error
  ``< 30°`` (``angle_thresh_deg``).

We report:
- ``top1_acc``: top-1 prediction accuracy (the standard headline metric).
- ``topk_any_acc``: at least one of top-``k`` predictions matches.
- ``mean_top1_iou``: mean IoU between top-1 prediction and best GT,
  ignoring the angle gate — a useful diagnostic.
- ``mean_top1_angle_err_deg``: mean angular error of top-1 prediction.

All quantities are computed in the **model coordinate frame** — each
Cornell scene is padded to square and uniformly resized to
``runner.image_size`` via :func:`grasp_seg.viz.compare_viz._scene_to_model_space`,
which keeps grasp angles intact and only scales centers / lengths /
widths uniformly.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.cornell import CornellSample
from ..data.grasp_rect import Grasp
from . import decoder
from .compare_viz import _scene_to_model_space
from .inference import ModelRunner


def _decode_pred(
    runner: ModelRunner,
    pred: dict,
    cfg: Optional[decoder.DecodeConfig] = None,
) -> List[Tuple[Grasp, float]]:
    """Dispatch to the right decoder based on ``runner.info.mask_mode``."""
    if runner.info.mask_mode == "angle":
        return decoder.decode_angle(pred["fg_conf"], pred["argmax"],
                                    runner.info.num_angle_bins, cfg=cfg)
    if runner.info.mask_mode == "multitask":
        return decoder.decode_multitask(pred["pos"], pred["cos2t"],
                                        pred["sin2t"], pred["width"], cfg=cfg)
    return []


def evaluate_cornell(
    runner: ModelRunner,
    scenes: Sequence[CornellSample],
    *,
    decode_cfg: Optional[decoder.DecodeConfig] = None,
    iou_thresh: float = 0.25,
    angle_thresh_deg: float = 30.0,
    top_k: int = 5,
) -> List[dict]:
    """Run ``runner`` on each Cornell scene; return per-scene records.

    Each record is a dict with keys::

        scene_id           — Cornell 4-digit scene id
        n_gt               — number of positive GT grasps in the scene
        n_decoded          — number of grasps the decoder produced
        top1_ok            — True if top-1 grasp passes IoU + angle gates
        topk_any_ok        — True if any of top-k grasps passes
        top1_iou           — IoU of top-1 with best GT (ignoring angle gate)
        top1_angle_err_deg — angular error of top-1 vs that best-IoU GT

    Records are returned in the same order as ``scenes``.
    """
    img_size = runner.image_size
    records: List[dict] = []
    for scene in scenes:
        rgb_m, depth_m, gt_m, _, _, _ = _scene_to_model_space(
            scene.rgb, scene.depth, scene.pos_grasps, img_size,
        )
        pred = runner.predict(rgb=rgb_m, depth=depth_m)
        decoded = _decode_pred(runner, pred, decode_cfg)
        decoded_sorted = sorted(decoded, key=lambda gs: -gs[1])

        rec = {
            "scene_id": scene.scene_id,
            "n_gt": len(gt_m),
            "n_decoded": len(decoded_sorted),
            "top1_ok": False,
            "topk_any_ok": False,
            "top1_iou": 0.0,
            "top1_angle_err_deg": 90.0,
        }
        if not gt_m or not decoded_sorted:
            records.append(rec)
            continue

        top1, _ = decoded_sorted[0]
        ious = [decoder.grasp_iou(top1, gt) for gt in gt_m]
        best_idx = int(np.argmax(ious))
        rec["top1_iou"] = float(ious[best_idx])
        rec["top1_angle_err_deg"] = float(decoder.angle_diff_deg(top1, gt_m[best_idx]))

        ok, _, _ = decoder.jacquard_match(
            top1, gt_m,
            iou_thresh=iou_thresh, angle_thresh_deg=angle_thresh_deg,
        )
        rec["top1_ok"] = bool(ok)

        for g, _score in decoded_sorted[:top_k]:
            okk, _, _ = decoder.jacquard_match(
                g, gt_m,
                iou_thresh=iou_thresh, angle_thresh_deg=angle_thresh_deg,
            )
            if okk:
                rec["topk_any_ok"] = True
                break

        records.append(rec)
    return records


def summarize_cornell(records: Sequence[dict]) -> dict:
    """Aggregate per-scene records into top-line numbers.

    Returns a dict with ``n``, ``top1_acc``, ``topk_any_acc``,
    ``mean_top1_iou``, ``mean_top1_angle_err_deg``. All percentages are
    in [0, 1].
    """
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "top1_acc": 0.0,
            "topk_any_acc": 0.0,
            "mean_top1_iou": 0.0,
            "mean_top1_angle_err_deg": 0.0,
        }
    return {
        "n": n,
        "top1_acc": float(np.mean([r["top1_ok"] for r in records])),
        "topk_any_acc": float(np.mean([r["topk_any_ok"] for r in records])),
        "mean_top1_iou": float(np.mean([r["top1_iou"] for r in records])),
        "mean_top1_angle_err_deg": float(
            np.mean([r["top1_angle_err_deg"] for r in records])
        ),
    }
