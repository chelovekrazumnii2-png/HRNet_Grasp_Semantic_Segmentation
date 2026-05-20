"""Structured RealSense D435 capture for Cornell-style grasp dataset.

This script is an extension of tools/realsense_capture.py for the dataset
creation phase. It keeps the same depth conventions as the repository:

- depth is aligned to color;
- depth is saved as float32 TIFF in metres;
- intrinsics/session metadata are stored once per session;
- optional per-scene metadata are appended to meta.csv for later object-wise split.

Manual mode is the intended default for dataset collection.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    sys.stderr.write("pyrealsense2 is not installed. See docs/realsense_setup.md")
    raise SystemExit(1) from exc

try:
    import tifffile
except ImportError as exc:
    sys.stderr.write("tifffile is not installed. Run: pip install tifffile")
    raise SystemExit(1) from exc

PRESETS = {"default": 0, "highaccuracy": 3, "highdensity": 4, "medium": 5}


def make_pipeline(width: int, height: int, fps: int, depth_preset: str):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        try:
            depth_sensor.set_option(rs.option.visual_preset, PRESETS[depth_preset])
        except RuntimeError:
            pass
    depth_scale = float(depth_sensor.get_depth_scale())
    aligner = rs.align(rs.stream.color)
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    meta = {
        "device_name": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "width": color_intr.width,
        "height": color_intr.height,
        "fps": fps,
        "depth_scale": depth_scale,
        "color_intrinsics": {
            "fx": color_intr.fx, "fy": color_intr.fy,
            "ppx": color_intr.ppx, "ppy": color_intr.ppy,
            "coeffs": list(color_intr.coeffs),
            "model": str(color_intr.model),
        },
        "depth_intrinsics": {
            "fx": depth_intr.fx, "fy": depth_intr.fy,
            "ppx": depth_intr.ppx, "ppy": depth_intr.ppy,
            "coeffs": list(depth_intr.coeffs),
            "model": str(depth_intr.model),
        },
    }
    return pipeline, aligner, depth_scale, meta


def next_scene_id(scene_dir: Path) -> int:
    used = []
    if scene_dir.exists():
        for p in scene_dir.glob('pcd*r.png'):
            stem = p.name.split('r.png')[0]
            digits = ''.join(ch for ch in stem if ch.isdigit())
            if digits:
                used.append(int(digits))
    return max(used) + 1 if used else 0


def save_scene(scene_dir: Path, scene_id: int, rgb_bgr: np.ndarray, depth_m: np.ndarray):
    base = scene_dir / f"pcd{scene_id:04d}"
    cv2.imwrite(str(base) + 'r.png', rgb_bgr)
    tifffile.imwrite(str(base) + 'd.tiff', depth_m.astype(np.float32))
    d_clip = np.clip(depth_m, 0.0, 1.5)
    d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    preview[depth_m <= 0] = 0
    cv2.imwrite(str(base) + 'd_preview.png', preview)
    return base.name


def ensure_meta_csv(path: Path):
    if not path.exists():
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'subdir', 'scene_base', 'object_name', 'object_instance',
                'scene_note', 'distance_m', 'lighting', 'background'
            ])


def append_meta(path: Path, row: list[str]):
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='captures/my_dataset')
    ap.add_argument('--subdir', default='01')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--depth-preset', default='highaccuracy', choices=list(PRESETS))
    ap.add_argument('--object-name', default='unknown_object')
    ap.add_argument('--object-instance', default='01')
    ap.add_argument('--distance-m', default='')
    ap.add_argument('--lighting', default='default')
    ap.add_argument('--background', default='table')
    ap.add_argument('--scene-note', default='')
    args = ap.parse_args()

    out_root = Path(args.out)
    scene_dir = out_root / args.subdir
    scene_dir.mkdir(parents=True, exist_ok=True)

    pipeline, aligner, depth_scale, intr_meta = make_pipeline(args.width, args.height, args.fps, args.depth_preset)

    intr_path = out_root / 'intrinsics.json'
    if not intr_path.exists():
        intr_path.write_text(json.dumps(intr_meta, indent=2), encoding='utf-8')

    meta_csv = out_root / 'meta.csv'
    ensure_meta_csv(meta_csv)
    log_path = out_root / 'capture.log'
    next_id = next_scene_id(scene_dir)

    print(f'[capture] output={scene_dir} next_scene_id={next_id}')
    print("Controls: s=save, q/ESC=quit")

    win_rgb = 'RealSense Dataset Capture — RGB'
    win_depth = 'RealSense Dataset Capture — Depth'
    cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = aligner.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale
            d_clip = np.clip(depth_m, 0.0, 1.5)
            d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
            depth_color[depth_raw == 0] = 0

            cv2.putText(rgb_bgr, f"obj={args.object_name} inst={args.object_instance}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
            cv2.imshow(win_rgb, rgb_bgr)
            cv2.imshow(win_depth, depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                scene_base = save_scene(scene_dir, next_id, rgb_bgr, depth_m)
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                append_meta(meta_csv, [
                    ts, args.subdir, scene_base, args.object_name, args.object_instance,
                    args.scene_note, args.distance_m, args.lighting, args.background,
                ])
                with log_path.open('a', encoding='utf-8') as f:
                    f.write(f"{ts}	{args.subdir}	{scene_base}	{args.object_name}")
                print(f'[saved] {scene_base} ({args.object_name})')
                next_id += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
