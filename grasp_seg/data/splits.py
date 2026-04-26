"""Object-wise train/val/test splits for Jacquard V2.

Each grasp scene lives in a folder named after its object id; multiple
images per object share the object id. To avoid leakage we split by
object id (not by image).
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


def discover_dataset(root: str) -> Dict[str, List[str]]:
    """Walk a dataset root and return ``{object_id: [grasp_file, ...]}``.

    The directory layout produced by extracting the Jacquard V2 archives is::

        root/JacquardV2_Dataset_{0..3}/<object_id>/<idx>_<object_id>_grasps.txt
        root/Jacquard_Dataset_{0..11}/<object_id>/<idx>_<object_id>_grasps.txt

    (The upstream release uses ``JacquardV2_Dataset_N``; some community
    mirrors — including the Yandex Disk share we reference in the Colab
    notebook — use the shorter ``Jacquard_Dataset_N`` with more shards.)

    We support a flatter layout too (objects directly under ``root``).
    """
    pattern_a = os.path.join(root, "Jacquard*Dataset_*", "*", "*_grasps.txt")
    pattern_b = os.path.join(root, "*", "*_grasps.txt")
    files = sorted(glob.glob(pattern_a)) or sorted(glob.glob(pattern_b))
    if not files:
        raise FileNotFoundError(
            f"No '*_grasps.txt' files found under {root!r}. "
            "Expected the unzipped Jacquard V2 directory structure."
        )

    objects: Dict[str, List[str]] = {}
    for f in files:
        obj_id = os.path.basename(os.path.dirname(f))
        objects.setdefault(obj_id, []).append(f)
    return objects


@dataclass
class Split:
    train: List[str]
    val: List[str]
    test: List[str]

    def to_dict(self) -> dict:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, d: dict) -> "Split":
        return cls(d["train"], d["val"], d["test"])


def make_split(
    objects: Dict[str, List[str]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Split:
    rng = random.Random(seed)
    obj_ids = sorted(objects.keys())
    rng.shuffle(obj_ids)
    n = len(obj_ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_objs = set(obj_ids[:n_test])
    val_objs = set(obj_ids[n_test:n_test + n_val])

    train, val, test = [], [], []
    for obj_id, files in objects.items():
        if obj_id in test_objs:
            test.extend(files)
        elif obj_id in val_objs:
            val.extend(files)
        else:
            train.extend(files)
    train.sort()
    val.sort()
    test.sort()
    return Split(train=train, val=val, test=test)


def save_split(split: Split, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(split.to_dict(), f)


def load_split(path: str) -> Split:
    with open(path, "r") as f:
        return Split.from_dict(json.load(f))
