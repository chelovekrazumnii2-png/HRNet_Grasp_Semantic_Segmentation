"""Evaluate a trained HRNet checkpoint on a Jacquard V2 split."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

# Make the package importable when launched directly from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from torch.utils.data import DataLoader

from grasp_seg.data import (AugConfig, DatasetConfig, JacquardV2GraspSeg,
                            collate_fn, load_split)
from grasp_seg.engine import evaluate_angle, evaluate_binary, evaluate_multitask
from grasp_seg.models import build_model
from grasp_seg.utils.logger import get_logger


def _input_channels(input_mode: str) -> int:
    return {"rgb": 3, "depth": 1, "rgbd": 4}[input_mode]


def _parse_cli_overrides(argv):
    """Pull ``a.b.c=value`` overrides off the argv tail.

    Mirrors the helper in tools/train.py so the Colab notebook can pass the
    same ``dataset.splits_path=...`` style flags to either entry point.
    """
    overrides: Dict[str, str] = {}
    rest = []
    for a in argv:
        if "=" in a and not a.startswith("--"):
            k, v = a.split("=", 1)
            overrides[k] = v
        else:
            rest.append(a)
    return overrides, rest


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
    cur = sub[leaf]
    # cast to the existing type when possible so "true"/"42" round-trip cleanly
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


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    overrides, argv_rest = _parse_cli_overrides(sys.argv[1:])
    args = p.parse_args(argv_rest)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for path, value in overrides.items():
        _override(cfg, path, value)
    if overrides:
        get_logger("eval").info("applied %d CLI override(s): %s",
                                len(overrides), overrides)

    ds_cfg = DatasetConfig(
        image_size=cfg["dataset"]["image_size"],
        input_mode=cfg["dataset"]["input_mode"],
        mask_mode=cfg["dataset"]["mask_mode"],
        num_angle_bins=cfg["dataset"]["num_angle_bins"],
        length_scale=cfg["dataset"]["length_scale"],
        use_stereo_depth=cfg["dataset"]["use_stereo_depth"],
    )
    split = load_split(cfg["dataset"]["splits_path"])
    files = getattr(split, args.split)
    ds = JacquardV2GraspSeg(files, ds_cfg, aug=AugConfig(enable=False), is_training=False)
    loader = DataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=False,
                        num_workers=cfg["dataset"]["num_workers"], collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = {
        "mask_mode": ds_cfg.mask_mode,
        "backbone": cfg["model"]["backbone"],
        "in_channels": _input_channels(ds_cfg.input_mode),
        "pretrained": False,
        "num_angle_bins": ds_cfg.num_angle_bins,
    }
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    eval_fn = {
        "binary": evaluate_binary,
        "angle": evaluate_angle,
        "multitask": evaluate_multitask,
    }[ds_cfg.mask_mode]
    metrics = eval_fn(model, loader, device, amp=cfg["trainer"]["amp"])
    logger = get_logger("eval")
    logger.info("split=%s metrics=%s", args.split, metrics)


if __name__ == "__main__":
    main()
