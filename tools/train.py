"""Train HRNet-W18 on Jacquard V2 for grasp segmentation.

Examples
--------
::

    # 1. Build splits once
    python tools/prepare_split.py --root /data/JacquardV2_Dataset \
        --out splits/jacquard_v2.json

    # 2. Train (default config — angle mode, RGB-D, 384x384, batch=2, accum=4)
    python tools/train.py --config configs/default.yaml \
        dataset.root=/data/JacquardV2_Dataset

Any leaf-level config field can be overriden on the CLI as ``key.path=value``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from typing import Any, Dict

# Make the package importable when this script is launched directly from the
# repo root (e.g. ``python tools/train.py``) without setting PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from grasp_seg.data import (AugConfig, DatasetConfig, JacquardV2GraspSeg,
                            collate_fn, load_split)
from grasp_seg.engine import (Trainer, TrainerConfig, evaluate_angle,
                              evaluate_binary, evaluate_multitask)
from grasp_seg.losses import (BinaryDiceBCELoss, MultiClassCEDiceLoss,
                              MultiTaskGraspLoss)
from grasp_seg.models import build_model
from grasp_seg.utils.logger import get_logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _override(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    sub = cfg
    for k in keys[:-1]:
        if k not in sub:
            raise KeyError(f"Unknown config path: {path}")
        sub = sub[k]
    leaf = keys[-1]
    if leaf not in sub:
        raise KeyError(f"Unknown config path: {path}")
    # cast to existing type when possible
    cur = sub[leaf]
    if isinstance(cur, bool):
        v = str(value).lower() in ("1", "true", "yes")
    elif isinstance(cur, int) and not isinstance(cur, bool):
        v = int(value)
    elif isinstance(cur, float):
        v = float(value)
    elif isinstance(cur, list):
        v = yaml.safe_load(value)
    else:
        v = value
    sub[leaf] = v


def _parse_cli_overrides(argv):
    """Pull ``a.b.c=value`` overrides off the argv tail."""
    overrides = {}
    rest = []
    for a in argv:
        if "=" in a and not a.startswith("--"):
            k, v = a.split("=", 1)
            overrides[k] = v
        else:
            rest.append(a)
    return overrides, rest


def _input_channels(input_mode: str) -> int:
    return {"rgb": 3, "depth": 1, "rgbd": 4}[input_mode]


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    overrides, argv_rest = _parse_cli_overrides(sys.argv[1:])
    args = parser.parse_args(argv_rest)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        _override(cfg, k, v)

    _set_seed(int(cfg.get("seed", 0)))
    logger = get_logger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"config: {cfg}")

    ds_cfg = DatasetConfig(
        image_size=cfg["dataset"]["image_size"],
        input_mode=cfg["dataset"]["input_mode"],
        mask_mode=cfg["dataset"]["mask_mode"],
        num_angle_bins=cfg["dataset"]["num_angle_bins"],
        length_scale=cfg["dataset"]["length_scale"],
        use_stereo_depth=cfg["dataset"]["use_stereo_depth"],
    )
    aug_cfg = AugConfig(**cfg["augmentation"])

    split = load_split(cfg["dataset"]["splits_path"])
    train_ds = JacquardV2GraspSeg(split.train, ds_cfg, aug=aug_cfg, is_training=True)
    val_aug = AugConfig(enable=False)
    val_ds = JacquardV2GraspSeg(split.val, ds_cfg, aug=val_aug, is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"]["pin_memory"],
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["trainer"]["batch_size"], shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"]["pin_memory"],
        collate_fn=collate_fn,
    )

    model_cfg = {
        "mask_mode": ds_cfg.mask_mode,
        "backbone": cfg["model"]["backbone"],
        "in_channels": _input_channels(ds_cfg.input_mode),
        "pretrained": cfg["model"]["pretrained"],
        "num_angle_bins": ds_cfg.num_angle_bins,
    }
    model = build_model(model_cfg)

    if ds_cfg.mask_mode == "binary":
        loss_fn = BinaryDiceBCELoss(
            dice_weight=cfg["loss"]["dice_weight"],
            bce_weight=cfg["loss"]["bce_weight"],
            pos_weight=cfg["loss"].get("bce_pos_weight"),
        )
        eval_fn = evaluate_binary
    elif ds_cfg.mask_mode == "angle":
        loss_fn = MultiClassCEDiceLoss(
            num_classes=ds_cfg.num_angle_bins + 1,
            dice_weight=cfg["loss"]["dice_weight"],
            ce_weight=cfg["loss"]["ce_weight"],
            ignore_background_in_dice=cfg["loss"]["ignore_background_in_dice"],
        )
        eval_fn = evaluate_angle
    else:  # multitask
        loss_fn = MultiTaskGraspLoss(
            pos_weight=cfg["loss"]["pos_weight"],
            cos_weight=cfg["loss"]["cos_weight"],
            sin_weight=cfg["loss"]["sin_weight"],
            width_weight=cfg["loss"]["width_weight"],
            bce_pos_weight=cfg["loss"].get("bce_pos_weight"),
        )
        eval_fn = evaluate_multitask

    trainer_cfg = TrainerConfig(
        epochs=cfg["trainer"]["epochs"],
        accum_steps=cfg["trainer"]["accum_steps"],
        lr=cfg["trainer"]["lr"],
        weight_decay=cfg["trainer"]["weight_decay"],
        momentum=cfg["trainer"]["momentum"],
        optimizer=cfg["trainer"]["optimizer"],
        scheduler=cfg["trainer"]["scheduler"],
        poly_power=cfg["trainer"]["poly_power"],
        warmup_epochs=cfg["trainer"]["warmup_epochs"],
        grad_clip=cfg["trainer"]["grad_clip"],
        log_interval=cfg["trainer"]["log_interval"],
        save_dir=cfg["trainer"]["save_dir"],
        eval_every=cfg["trainer"]["eval_every"],
        target_metric="miou_fg",
    )
    os.makedirs(trainer_cfg.save_dir, exist_ok=True)
    with open(os.path.join(trainer_cfg.save_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=trainer_cfg,
        device=device,
        evaluate_fn=eval_fn,
        amp=cfg["trainer"]["amp"],
    )
    state = trainer.fit()
    logger.info(f"Done. Best {trainer_cfg.target_metric}={state.best_metric:.4f} "
                f"at epoch {state.best_epoch}")


if __name__ == "__main__":
    main()
