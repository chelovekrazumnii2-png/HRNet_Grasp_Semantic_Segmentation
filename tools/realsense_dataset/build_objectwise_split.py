from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    meta_csv = root / 'meta.csv'
    if not meta_csv.exists():
        raise SystemExit('meta.csv not found. capture_session.py should create it.')

    by_object = {}
    with meta_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_name = row['object_name']
            scene_base = row['scene_base']
            subdir = row['subdir']
            rel = f'{subdir}/{scene_base}' if subdir else scene_base
            by_object.setdefault(object_name, []).append(rel)

    objects = sorted(by_object)
    rng = random.Random(args.seed)
    n_test = max(1, round(len(objects) * args.test_frac))
    test_objects = set(rng.sample(objects, n_test))
    train_items, test_items = [], []
    for obj, items in by_object.items():
        (test_items if obj in test_objects else train_items).extend(items)

    (root / 'train.txt').write_text(''.join(sorted(train_items)), encoding='utf-8')
    (root / 'test.txt').write_text(''.join(sorted(test_items)), encoding='utf-8')
    print(f'objects_total={len(objects)} train_objects={len(objects)-len(test_objects)} test_objects={len(test_objects)}')
    print(f'train_images={len(train_items)} test_images={len(test_items)}')
    print(f'test_objects={sorted(test_objects)}')


if __name__ == '__main__':
    main()
