"""Interactive Cornell grasp annotation tool for RGB images.

Creates:
- pcdNNNNcpos.txt
- pcdNNNNcneg.txt

Each rectangle is stored as four lines of x y coordinates, compatible with
Cornell Grasp Dataset and the repository loader expectations.

Rectangle construction logic (3-click, always orthogonal):
  Click 1 (P0), Click 2 (P1) — two vertices of one side (gripper jaw width).
      The vector P0→P1 defines direction and length of that side.
  Click 3 (P2) — any point; only its perpendicular distance to the line
      through P0-P1 is used as the rectangle height h.

      unit vector along P0→P1:  u = (P1-P0) / |P1-P0|
      inward normal (90° CCW):   n = (-u.y, u.x)
      signed height:             h = (P2 - P0) · n
      opposite vertices:         P3 = P0 + n*h
                                 P4 = P1 + n*h
      stored order:              [P0, P1, P4, P3]  (clockwise, Cornell convention)

All four angles are exactly 90° regardless of where P2 was placed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

LINE_WIDTH = 1
POINT_SIZE = 2

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def build_rect(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Return four vertices of an axis-aligned-to-p0p1 rectangle.

    Parameters
    ----------
    p0, p1 : ndarray shape (2,)
        Two vertices of the first side (gripper width side).
    p2 : ndarray shape (2,)
        Any point whose perpendicular distance to the line through p0-p1
        defines the rectangle height.

    Returns
    -------
    list of four ndarray shape (2,) in clockwise order: [p0, p1, p4, p3]
    """
    edge = p1 - p0
    length = np.linalg.norm(edge)
    if length < 1e-6:
        # degenerate: p0 == p1, return a tiny square as fallback
        return [p0, p1, p1 + np.array([1.0, 0.0]), p0 + np.array([1.0, 0.0])]

    u = edge / length                        # unit vector along p0→p1
    n = np.array([-u[1], u[0]], dtype=float) # 90° CCW normal

    h = float(np.dot(p2 - p0, n))           # signed perpendicular distance
    # enforce minimum height of 1 px so we never store a line
    if abs(h) < 1.0:
        h = 1.0 if h >= 0 else -1.0

    p3 = p0 + n * h   # opposite vertex from p0
    p4 = p1 + n * h   # opposite vertex from p1

    # Cornell clockwise order starting from p0: p0 → p1 → p4 → p3
    return [p0.copy(), p1.copy(), p4, p3]


def rect_params(rect):
    """Return (cx, cy, theta_deg, w, h) from four vertices."""
    pts = np.array(rect)
    cx, cy = pts.mean(axis=0)
    d01 = np.linalg.norm(pts[1] - pts[0])
    d12 = np.linalg.norm(pts[2] - pts[1])
    w, h = max(d01, d12), min(d01, d12)
    dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    theta = float(np.degrees(np.arctan2(dy, dx))) % 180.0
    return cx, cy, theta, w, h


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_rects(path: Path):
    if not path.exists():
        return []
    lines = [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    rects = []
    for i in range(0, len(lines) - 3, 4):
        pts = []
        for j in range(4):
            x, y = map(float, lines[i + j].split())
            pts.append(np.array([x, y], dtype=float))
        rects.append(pts)
    return rects


def save_rects(path: Path, rects):
    with path.open('w', encoding='utf-8') as f:
        for rect in rects:
            for pt in rect:
                f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw(img: np.ndarray, pos_rects, neg_rects, current, mode: str) -> np.ndarray:
    vis = img.copy()

    # Saved positive rects — green solid
    for rect in pos_rects:
        pts = np.array(rect, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), LINE_WIDTH)
        cx, cy, th, w, h = rect_params(rect)
        # cv2.putText(vis, f"{w:.0f}x{h:.0f} {th:.0f}deg",
        #             (int(cx) + 4, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 0), 1)

    # Saved negative rects — red dashed look (thin)
    for rect in neg_rects:
        pts = np.array(rect, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 1)

    # Current clicks
    for i, pt in enumerate(current):
        cv2.circle(vis, tuple(np.int32(pt)), POINT_SIZE, (255, 255, 0), -1)
        label = ['P0 (jaw start)', 'P1 (jaw end)', 'P2 (height)'][i]
        cv2.putText(vis, label, tuple(np.int32(pt + np.array([8, 4]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # Live side preview after 2 clicks
    if len(current) == 2:
        cv2.line(vis, tuple(np.int32(current[0])), tuple(np.int32(current[1])),
                 (255, 200, 0), LINE_WIDTH)

    # Live rectangle preview after 3 clicks — guaranteed orthogonal
    if len(current) == 3:
        rect_pts = build_rect(current[0], current[1], current[2])
        poly = np.array(rect_pts, dtype=np.int32)
        cv2.polylines(vis, [poly], True, (255, 165, 0), LINE_WIDTH)
        cx, cy, th, w, h = rect_params(rect_pts)
        cv2.putText(vis, f"PREVIEW  {w:.0f}x{h:.0f}px  {th:.1f}deg",
                    (int(cx) - 40, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    color_mode = (0, 255, 0) if mode == 'pos' else (0, 0, 255)
    cv2.putText(vis,
                f"mode={mode.upper()}  "
                "| p=switch  s=save  c=clear  z=undo  n=next  q=quit",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_mode, 1)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Annotate Cornell-style grasp rectangles (always orthogonal)."
    )
    ap.add_argument('--root', required=True, help='Cornell-style dataset root')
    ap.add_argument('--subdir', default='01', help='Sub-directory within root (e.g. 01)')
    args = ap.parse_args()

    scene_dir = Path(args.root) / args.subdir
    rgb_files = sorted(scene_dir.glob('pcd*r.png'))
    rgb_files = [p for p in rgb_files if 'd_preview' not in p.name]
    if not rgb_files:
        raise SystemExit(f"[ERROR] No pcd*r.png files found in {scene_dir}")

    mode = 'pos'
    current: list[np.ndarray] = []
    idx = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN and len(current) < 3:
            current.append(np.array([x, y], dtype=float))

    cv2.namedWindow('annotate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('annotate', 1280, 960)
    cv2.setMouseCallback('annotate', on_mouse)

    print(f"[annotate] {len(rgb_files)} images found in {scene_dir}")
    print("  Click P0 (jaw start) → P1 (jaw end) → P2 (height point)")
    print("  s=save  p=switch pos/neg  c=clear clicks  z=undo last rect  n=next  q=quit")

    while idx < len(rgb_files):
        rgb_path = rgb_files[idx]
        base = rgb_path.name[:-5]   # strip 'r.png'
        img = cv2.imread(str(rgb_path))
        if img is None:
            print(f"[WARN] Cannot read {rgb_path}, skipping.")
            idx += 1
            continue

        pos_path = scene_dir / f'{base}cpos.txt'
        neg_path = scene_dir / f'{base}cneg.txt'
        pos_rects = load_rects(pos_path)
        neg_rects = load_rects(neg_path)
        current = []

        print(f"\n[{idx+1}/{len(rgb_files)}] {base}  "
              f"(pos={len(pos_rects)}, neg={len(neg_rects)})")

        while True:
            vis = draw(img, pos_rects, neg_rects, current, mode)
            cv2.imshow('annotate', vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('p'):
                mode = 'neg' if mode == 'pos' else 'pos'
                print(f"  Mode → {mode.upper()}")

            elif key == ord('c'):
                current = []

            elif key == ord('z'):
                if mode == 'pos' and pos_rects:
                    pos_rects.pop()
                    print(f"  Undo last pos. Remaining: {len(pos_rects)}")
                elif mode == 'neg' and neg_rects:
                    neg_rects.pop()
                    print(f"  Undo last neg. Remaining: {len(neg_rects)}")

            elif key == ord('s'):
                if len(current) == 3:
                    rect = build_rect(current[0], current[1], current[2])
                    _, _, th, w, h = rect_params(rect)
                    (pos_rects if mode == 'pos' else neg_rects).append(rect)
                    print(f"  Saved {mode} rect: {w:.1f}×{h:.1f}px  θ={th:.1f}°  "
                          f"(pos={len(pos_rects)}, neg={len(neg_rects)})")
                    current = []
                else:
                    print(f"  [!] Need 3 clicks to save (have {len(current)})")

            elif key == ord('n'):
                save_rects(pos_path, pos_rects)
                save_rects(neg_path, neg_rects)
                print(f"  Saved → {pos_path.name} ({len(pos_rects)} rects), "
                      f"{neg_path.name} ({len(neg_rects)} rects)")
                idx += 1
                break

            elif key in (ord('q'), 27):
                save_rects(pos_path, pos_rects)
                save_rects(neg_path, neg_rects)
                print(f"  Saved and quit.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n[annotate] All images processed.")


if __name__ == '__main__':
    main()
