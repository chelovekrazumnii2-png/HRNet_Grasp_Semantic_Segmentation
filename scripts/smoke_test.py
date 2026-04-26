"""Smoke test that exercises the whole pipeline on synthetic samples.

Generates 8 fake Jacquard-like scenes in a temp directory, builds the
dataloader, runs a couple of training steps for each mask mode and logs
parameter shapes. Useful as a CI sanity check before downloading the full
~51 GB dataset.

Usage::

    python scripts/smoke_test.py
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from typing import List

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp_seg.data import (AugConfig, DatasetConfig, JacquardV2GraspSeg,
                            collate_fn, discover_dataset, make_split,
                            save_split, load_split)
from grasp_seg.engine import (Trainer, TrainerConfig, evaluate_angle,
                              evaluate_binary, evaluate_multitask)
from grasp_seg.losses import (BinaryDiceBCELoss, MultiClassCEDiceLoss,
                              MultiTaskGraspLoss)
from grasp_seg.models import build_model
from grasp_seg.utils.logger import get_logger


def _make_fake_dataset(root: str, num_objects: int = 4, scenes_per_object: int = 2) -> None:
    """Write fake RGB / perfect_depth / stereo_depth / mask / grasps files
    in the same layout the loader expects."""
    rng = np.random.default_rng(0)
    for o in range(num_objects):
        obj_id = f"obj_{o:08d}"
        obj_dir = os.path.join(root, "JacquardV2_Dataset_0", obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        for s in range(scenes_per_object):
            stem = f"{s}_{obj_id}"
            H = W = 1024
            rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
            depth = rng.uniform(0.5, 1.5, size=(H, W)).astype(np.float32)
            mask = (rng.random((H, W)) > 0.7).astype(np.uint8) * 255
            Image.fromarray(rgb, mode="RGB").save(
                os.path.join(obj_dir, f"{stem}_RGB.png"))
            tifffile.imwrite(
                os.path.join(obj_dir, f"{stem}_perfect_depth.tiff"), depth)
            tifffile.imwrite(
                os.path.join(obj_dir, f"{stem}_stereo_depth.tiff"),
                depth + rng.normal(0, 0.02, depth.shape).astype(np.float32))
            Image.fromarray(mask, mode="L").save(
                os.path.join(obj_dir, f"{stem}_mask.png"))
            with open(os.path.join(obj_dir, f"{stem}_grasps.txt"), "w") as f:
                # 3-5 random grasps per scene
                for _ in range(int(rng.integers(3, 6))):
                    x = rng.uniform(200, W - 200)
                    y = rng.uniform(200, H - 200)
                    theta = rng.uniform(-90, 90)
                    w = rng.uniform(60, 220)
                    h = rng.uniform(40, 100)
                    f.write(f"{x:.2f};{y:.2f};{theta:.2f};{w:.2f};{h:.2f}\n")


def _run_one_mode(mask_mode: str, root: str, splits_path: str, device: torch.device) -> None:
    logger = get_logger(f"smoke[{mask_mode}]")
    logger.info("starting…")
    aug = AugConfig(enable=True)
    ds_cfg = DatasetConfig(
        image_size=128, input_mode="rgbd", mask_mode=mask_mode,
        num_angle_bins=8, length_scale=1.0 / 3.0,
    )
    split = load_split(splits_path)
    train_ds = JacquardV2GraspSeg(split.train, ds_cfg, aug=aug, is_training=True)
    val_ds = JacquardV2GraspSeg(split.val, ds_cfg, aug=AugConfig(enable=False),
                                is_training=False)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

    model_cfg = {
        "mask_mode": mask_mode,
        "backbone": "hrnet_w18_small_v2",  # smaller for the smoke test
        "in_channels": 4,
        "pretrained": False,
        "num_angle_bins": 8,
    }
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_params/1e6:.2f}M")

    if mask_mode == "binary":
        loss_fn = BinaryDiceBCELoss(pos_weight=5.0)
        eval_fn = evaluate_binary
    elif mask_mode == "angle":
        loss_fn = MultiClassCEDiceLoss(num_classes=9)
        eval_fn = evaluate_angle
    else:
        loss_fn = MultiTaskGraspLoss(bce_pos_weight=5.0)
        eval_fn = evaluate_multitask

    save_dir = os.path.join(root, f"out_{mask_mode}")
    trainer_cfg = TrainerConfig(
        epochs=2, accum_steps=1, lr=1e-3, optimizer="adamw",
        scheduler="poly", warmup_epochs=0.0, log_interval=1,
        save_dir=save_dir, eval_every=1,
        save_every_epoch=True, metrics_csv="metrics.csv",
    )
    trainer = Trainer(model, loss_fn, train_loader, val_loader, trainer_cfg,
                      device, evaluate_fn=eval_fn, amp=device.type == "cuda")
    trainer.fit()
    logger.info(f"done, best metric={trainer.state.best_metric:.4f}")

    # --- verify checkpoint side-effects ---
    assert os.path.exists(os.path.join(save_dir, "epoch_000.pth"))
    assert os.path.exists(os.path.join(save_dir, "epoch_001.pth"))
    assert os.path.exists(os.path.join(save_dir, "last.pth"))
    csv_path = os.path.join(save_dir, "metrics.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as fh:
        rows = fh.readlines()
    assert len(rows) == 3, f"expected header + 2 rows, got {len(rows)}"

    # --- verify resume from epoch_001 round-trips state ---
    resume_model = build_model(model_cfg).to(device)
    resume_trainer = Trainer(resume_model, loss_fn, train_loader, val_loader,
                              trainer_cfg, device, evaluate_fn=eval_fn,
                              amp=device.type == "cuda")
    resume_trainer.load_checkpoint(os.path.join(save_dir, "epoch_001.pth"))
    assert resume_trainer.state.epoch == 1, resume_trainer.state.epoch

    # --- verify resume continues at epoch+1, not the same epoch ---
    extended_cfg = TrainerConfig(
        epochs=4, accum_steps=1, lr=1e-3, optimizer="adamw",
        scheduler="poly", warmup_epochs=0.0, log_interval=1,
        save_dir=save_dir, eval_every=1,
        save_every_epoch=True, metrics_csv="metrics.csv",
    )
    extended_model = build_model(model_cfg).to(device)
    extended = Trainer(extended_model, loss_fn, train_loader, val_loader,
                       extended_cfg, device, evaluate_fn=eval_fn,
                       amp=device.type == "cuda")
    extended.load_checkpoint(os.path.join(save_dir, "epoch_001.pth"))
    extended.fit()
    # epochs 0,1 already done -> 2,3 should be the new ones
    assert os.path.exists(os.path.join(save_dir, "epoch_002.pth"))
    assert os.path.exists(os.path.join(save_dir, "epoch_003.pth"))
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    epochs_seen = sorted(int(r["epoch"]) for r in rows)
    assert epochs_seen == [0, 1, 2, 3], epochs_seen
    logger.info("checkpoint + resume verified")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep", action="store_true",
                   help="Keep the temp directory for inspection")
    args = p.parse_args()

    tmp_root = tempfile.mkdtemp(prefix="jv2_smoke_")
    print(f"working dir: {tmp_root}")
    try:
        _make_fake_dataset(tmp_root, num_objects=4, scenes_per_object=2)
        objs = discover_dataset(tmp_root)
        split = make_split(objs, val_frac=0.25, test_frac=0.25, seed=0)
        splits_path = os.path.join(tmp_root, "split.json")
        save_split(split, splits_path)
        print(f"split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for mode in ("binary", "angle", "multitask"):
            _run_one_mode(mode, tmp_root, splits_path, device)
        print("ALL MODES OK")
    finally:
        if not args.keep:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
