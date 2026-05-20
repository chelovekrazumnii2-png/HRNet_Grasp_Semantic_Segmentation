from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import cv2


def load_rects(path: Path):
    if not path.exists():
        return []
    lines = [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    rects = []
    for i in range(0, len(lines), 4):
        pts = []
        for j in range(4):
            x, y = map(float, lines[i+j].split())
            pts.append(np.array([x, y], dtype=float))
        rects.append(pts)
    return rects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != 'backgrounds']
    if not subdirs:
        subdirs = [root]

    total_rgb = total_pos = total_neg = 0
    missing_depth = []
    rows = []
    for d in sorted(subdirs):
        for rgb in sorted(d.glob('pcd*r.png')):
            if rgb.name.endswith('d_preview.png'):
                continue
            base = rgb.name[:-5]
            depth = d / f'{base}d.tiff'
            pos = d / f'{base}cpos.txt'
            neg = d / f'{base}cneg.txt'
            pos_n = len(load_rects(pos))
            neg_n = len(load_rects(neg))
            total_rgb += 1
            total_pos += pos_n
            total_neg += neg_n
            if not depth.exists():
                missing_depth.append(str(depth))
            rows.append((d.name, base, pos_n, neg_n, int(depth.exists())))

    print(f'images={total_rgb}')
    print(f'positive_rectangles={total_pos}')
    print(f'negative_rectangles={total_neg}')
    print(f'missing_depth={len(missing_depth)}')
    if missing_depth:
        print('first_missing_depth=', missing_depth[:5])

    out_csv = root / 'dataset_summary.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subdir', 'scene_base', 'n_pos', 'n_neg', 'has_depth'])
        writer.writerows(rows)
    print(f'saved_summary={out_csv}')


if __name__ == '__main__':
    main()
