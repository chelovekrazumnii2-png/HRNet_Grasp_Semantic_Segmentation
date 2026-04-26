"""Create object-wise train/val/test splits for Jacquard V2.

Usage::

    python tools/prepare_split.py --root /data/JacquardV2_Dataset \
        --out splits/jacquard_v2.json --val-frac 0.1 --test-frac 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when launched directly from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grasp_seg.data.splits import discover_dataset, make_split, save_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to the unzipped Jacquard V2 dataset root")
    p.add_argument("--out", required=True, help="Where to write the JSON split file")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    objs = discover_dataset(args.root)
    print(f"Discovered {len(objs)} objects, "
          f"{sum(len(v) for v in objs.values())} grasp files.")
    split = make_split(objs, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    save_split(split, args.out)
    print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
