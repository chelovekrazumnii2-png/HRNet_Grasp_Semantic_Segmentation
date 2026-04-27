"""CLI driver for the visualisation pipeline.

Generates every figure used in the project report (dataset / training /
best epoch / model comparison / cross-domain Cornell) into a single
output directory tree::

    outputs/viz/<RUN_TAG>/
        dataset/
            01_raw_with_grasps.png
            02_resize_pipeline.png
            03_mask_modes.png
            04_compact_vs_full.png
            05_augmentation_steps.png
        training/
            <model_name>_curves.png
            compare_runs.png
        epoch_evolution/
            <model_name>.png
        best_epoch/
            <model_name>_scenes.png
            per_class_iou.png
        compare/
            jacquard_test.png
            cornell_test.png
        extras/
            iou_vs_angle_<model>.png
            depth_contribution.png
            failures_<model>.png

All figures are saved as PNGs at ``--dpi`` resolution. Pass
``--show`` to also call ``plt.show`` (useful when running interactively).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

# Use the non-interactive backend by default — the notebook overrides this.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grasp_seg.data.splits import load_split
from grasp_seg.data import cornell as cornell_data
from grasp_seg.viz import (
    compare_viz,
    dataset_viz,
    decoder,
    epoch_evolution,
    eval_viz,
    extra_viz,
    inference,
    metrics_viz,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Сгенерировать набор визуализаций для отчёта по проекту.",
        allow_abbrev=False,
    )
    # Models
    p.add_argument("--run", action="append", default=[],
                   help="Кортеж 'NAME=PATH' к директории обучения "
                        "(можно указывать многократно).")
    # Datasets
    p.add_argument("--jacquard-root", default=None)
    p.add_argument("--splits-path", default=None)
    p.add_argument("--cornell-root", default=None,
                   help="Локальный путь к Cornell Grasp Dataset (опционально).")
    # Selection
    p.add_argument("--num-evolution-scenes", type=int, default=2)
    p.add_argument("--num-best-scenes", type=int, default=4)
    p.add_argument("--num-compare-scenes", type=int, default=4)
    p.add_argument("--num-cornell-scenes", type=int, default=4)
    p.add_argument("--epoch-step", type=int, default=5)
    p.add_argument("--max-iou-samples", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    # Output
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "outputs", "viz", "report"))
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--sections", nargs="*", default=None,
                   help="Подмножество секций: dataset / training / "
                        "epoch_evolution / best_epoch / compare / extras")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _save(fig, path: str, dpi: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_run_kv(specs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--run expects NAME=PATH, got {s!r}")
        name, path = s.split("=", 1)
        out[name] = path
    return out


def _pick(files: List[str], n: int, rng: random.Random) -> List[str]:
    if n >= len(files):
        return list(files)
    return rng.sample(files, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse()
    rng = random.Random(args.seed)
    runs = _parse_run_kv(args.run)
    sections = set(args.sections) if args.sections else {
        "dataset", "training", "epoch_evolution",
        "best_epoch", "compare", "extras",
    }
    os.makedirs(args.out, exist_ok=True)

    # Load split if available
    split = None
    if args.splits_path and os.path.isfile(args.splits_path):
        split = load_split(args.splits_path)
        print(f"[ok] loaded split: train={len(split.train)} "
              f"val={len(split.val)} test={len(split.test)}")
    elif "dataset" in sections or "epoch_evolution" in sections \
            or "best_epoch" in sections or "compare" in sections \
            or "extras" in sections:
        print("[warn] --splits-path not provided — sections that need scenes "
              "will be skipped.")

    test_files: List[str] = list(split.test) if split is not None else []

    # ------------------------------------------------------------------
    # Section: dataset
    # ------------------------------------------------------------------
    if "dataset" in sections and test_files:
        out = os.path.join(args.out, "dataset")
        sample = _pick(test_files, 1, rng)[0]
        print(f"[dataset] visualising scene: {sample}")
        _save(dataset_viz.figure_raw_with_grasps(sample),
              os.path.join(out, "01_raw_with_grasps.png"), args.dpi)
        _save(dataset_viz.figure_resize_pipeline(sample),
              os.path.join(out, "02_resize_pipeline.png"), args.dpi)
        _save(dataset_viz.figure_mask_modes(sample),
              os.path.join(out, "03_mask_modes.png"), args.dpi)
        _save(dataset_viz.figure_compact_vs_full(sample),
              os.path.join(out, "04_compact_vs_full.png"), args.dpi)
        _save(dataset_viz.figure_augmentation_steps(sample),
              os.path.join(out, "05_augmentation_steps.png"), args.dpi)
        if args.cornell_root:
            ids = cornell_data.list_scenes(args.cornell_root)
            if ids:
                cs = cornell_data.load_scene(args.cornell_root, ids[0])
                _save(dataset_viz.figure_cornell_raw(cs),
                      os.path.join(out, "06_cornell_raw.png"), args.dpi)

    # ------------------------------------------------------------------
    # Section: training curves
    # ------------------------------------------------------------------
    if "training" in sections and runs:
        out = os.path.join(args.out, "training")
        csv_paths: Dict[str, str] = {}
        for name, run_dir in runs.items():
            csv = os.path.join(run_dir, "metrics.csv")
            if os.path.isfile(csv):
                csv_paths[name] = csv
                _save(metrics_viz.figure_single_run(csv, title=name),
                      os.path.join(out, f"{name}_curves.png"), args.dpi)
        if len(csv_paths) > 1:
            _save(metrics_viz.figure_compare_runs(csv_paths),
                  os.path.join(out, "compare_runs.png"), args.dpi)

    # Build runners (best.pth) ------------------------------------------------
    runners: List[inference.ModelRunner] = []
    if any(s in sections for s in ("epoch_evolution", "best_epoch", "compare", "extras")) \
            and runs:
        for name, run_dir in runs.items():
            try:
                runners.append(inference.ModelRunner(run_dir, checkpoint="best", name=name))
                print(f"[ok] loaded model {name!r}: {runners[-1].info}")
            except Exception as e:
                print(f"[warn] cannot load {name} from {run_dir}: {e}")

    # ------------------------------------------------------------------
    # Section: epoch evolution
    # ------------------------------------------------------------------
    if "epoch_evolution" in sections and runs and test_files:
        out = os.path.join(args.out, "epoch_evolution")
        scenes = _pick(test_files, args.num_evolution_scenes, rng)
        for name, run_dir in runs.items():
            print(f"[epoch_evolution] {name} on {scenes}")
            try:
                fig = epoch_evolution.figure_epoch_evolution(
                    run_dir, scenes, step=args.epoch_step,
                    title=f"Эволюция предсказаний — {name}",
                )
                _save(fig, os.path.join(out, f"{name}.png"), args.dpi)
            except Exception as e:
                print(f"[warn] epoch evolution failed for {name}: {e}")

    # ------------------------------------------------------------------
    # Section: best epoch
    # ------------------------------------------------------------------
    if "best_epoch" in sections and runners and test_files:
        out = os.path.join(args.out, "best_epoch")
        scenes = _pick(test_files, args.num_best_scenes, rng)
        for runner in runners:
            print(f"[best_epoch] {runner.info.name} on {len(scenes)} scenes")
            fig = eval_viz.figure_best_epoch_scenes(runner, scenes)
            _save(fig, os.path.join(out, f"{runner.info.name}_scenes.png"), args.dpi)
        # Per-class IoU bar chart for angle-mode models (multitask is also OK).
        compatible = [r for r in runners
                      if r.info.mask_mode in ("angle", "multitask")]
        if compatible:
            try:
                fig = eval_viz.figure_per_class_iou(
                    compatible, test_files, max_samples=args.max_iou_samples,
                )
                _save(fig, os.path.join(out, "per_class_iou.png"), args.dpi)
            except Exception as e:
                print(f"[warn] per_class_iou failed: {e}")

    # ------------------------------------------------------------------
    # Section: compare (jacquard + cornell)
    # ------------------------------------------------------------------
    if "compare" in sections and runners:
        out = os.path.join(args.out, "compare")
        if test_files:
            scenes = _pick(test_files, args.num_compare_scenes, rng)
            fig = compare_viz.figure_compare_models_jacquard(runners, scenes)
            _save(fig, os.path.join(out, "jacquard_test.png"), args.dpi)
        if args.cornell_root:
            ids = cornell_data.list_scenes(args.cornell_root)
            if ids:
                ids = rng.sample(ids, min(args.num_cornell_scenes, len(ids)))
                cs = [cornell_data.load_scene(args.cornell_root, i) for i in ids]
                fig = compare_viz.figure_compare_models_cornell(runners, cs)
                _save(fig, os.path.join(out, "cornell_test.png"), args.dpi)

    # ------------------------------------------------------------------
    # Section: extras
    # ------------------------------------------------------------------
    if "extras" in sections and runners and test_files:
        out = os.path.join(args.out, "extras")
        # IoU × angle per model
        for runner in runners:
            try:
                fig = extra_viz.figure_iou_vs_angle(
                    runner, test_files, max_samples=args.max_iou_samples,
                )
                _save(fig, os.path.join(out, f"iou_vs_angle_{runner.info.name}.png"),
                      args.dpi)
            except Exception as e:
                print(f"[warn] iou_vs_angle failed for {runner.info.name}: {e}")
        # Depth contribution: needs one RGB-only and one RGB-D runner with same mask_mode
        rgb_runs = [r for r in runners if r.info.input_mode == "rgb"]
        rgbd_runs = [r for r in runners if r.info.input_mode == "rgbd"]
        if rgb_runs and rgbd_runs:
            rgb_r = rgb_runs[0]
            rgbd_r = next((r for r in rgbd_runs if r.info.mask_mode == rgb_r.info.mask_mode),
                          rgbd_runs[0])
            scenes = _pick(test_files, 3, rng)
            try:
                fig = extra_viz.figure_depth_contribution(rgb_r, rgbd_r, scenes)
                _save(fig, os.path.join(out, "depth_contribution.png"), args.dpi)
            except Exception as e:
                print(f"[warn] depth_contribution failed: {e}")
        # Failure catalog per model
        for runner in runners:
            try:
                fig = extra_viz.figure_failure_catalog(runner, test_files,
                                                        n_show=8, max_samples=args.max_iou_samples)
                _save(fig, os.path.join(out, f"failures_{runner.info.name}.png"),
                      args.dpi)
            except Exception as e:
                print(f"[warn] failure_catalog failed for {runner.info.name}: {e}")

    print(f"[done] all figures saved under: {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
