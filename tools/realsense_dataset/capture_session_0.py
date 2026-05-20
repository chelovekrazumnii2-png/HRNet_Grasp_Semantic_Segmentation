"""Structured RealSense D435 capture for Cornell-style grasp dataset.

Post-processing pipeline (applied to depth before alignment):
  Decimation  →  Disparity (depth→disparity)  →  Spatial  →  Temporal
  →  Disparity (disparity→depth)  →  Hole Filling  →  Align

- depth is aligned to color;
- depth is saved as float32 TIFF in metres (post-processed, clean);
- intrinsics/session metadata are stored once per session;
- optional per-scene metadata are appended to meta.csv for later object-wise split.

Post-processing filters match the default RealSense Viewer preset and are
individually tunable via CLI flags. Decimation is applied only to the
preview stream; the saved depth retains full resolution (no --decimate-save).

CLI quick reference:
  --depth-preset highaccuracy   Best accuracy, slower fill
  --depth-preset highdensity    More filled, slightly noisier edges
  --spatial-magnitude 3         Stronger spatial smoothing (1-5)
  --temporal-alpha 0.4          Temporal smoothing strength (0.0-1.0)
  --hole-fill 1                 Fill small holes only (0=off, 1-5=larger)
  --no-temporal                 Disable temporal filter (moving objects)
  --no-spatial                  Disable spatial filter
  --no-hole-fill                Disable hole filling
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    sys.stderr.write("pyrealsense2 is not installed. See docs/realsense_setup.md\n")
    raise SystemExit(1) from exc

try:
    import tifffile
except ImportError as exc:
    sys.stderr.write("tifffile is not installed. Run: pip install tifffile\n")
    raise SystemExit(1) from exc

PRESETS = {"default": 0, "highaccuracy": 3, "highdensity": 4, "medium": 5}


# ---------------------------------------------------------------------------
# Post-processing filter chain
# ---------------------------------------------------------------------------

def build_filters(args) -> list:
    """
    Build the post-processing chain identical to RealSense Viewer defaults.

    Execution order matters:
      1. decimation_filter    — reduces resolution noise (preview only)
      2. depth_to_disparity   — convert to disparity domain for spatial/temporal
      3. spatial_filter       — edge-preserving fill (reduces holes + smooths)
      4. temporal_filter      — accumulates frames to reduce flicker noise
      5. disparity_to_depth   — back to depth domain
      6. hole_filling_filter  — fills remaining holes using neighbour strategy

    Decimation is NOT applied when saving — full-resolution depth is preserved.
    """
    filters = []

    # --- Decimation (preview display only, see capture loop) ---
    dec = rs.decimation_filter()
    dec.set_option(rs.option.filter_magnitude, 2)  # 2 = half resolution preview
    filters.append(("decimation", dec))

    # --- Depth → Disparity (required before spatial/temporal) ---
    d2d = rs.disparity_transform(True)
    filters.append(("depth_to_disparity", d2d))

    # --- Spatial filter ---
    if not args.no_spatial:
        spat = rs.spatial_filter()
        # magnitude: number of iterations (1-5). Higher = stronger fill, softer edges
        spat.set_option(rs.option.filter_magnitude, args.spatial_magnitude)
        # smooth_alpha: spatial weight (0.25-1.0). Higher = more aggressive smoothing
        spat.set_option(rs.option.filter_smooth_alpha, args.spatial_alpha)
        # smooth_delta: depth gradient threshold in mm. Pixels with larger gradient
        # are treated as edges and NOT smoothed across — preserves object boundaries
        spat.set_option(rs.option.filter_smooth_delta, args.spatial_delta)
        # hole_fill: fill holes up to this size (0=none, 5=unlimited)
        spat.set_option(rs.option.holes_fill, args.spatial_hole_fill)
        filters.append(("spatial", spat))

    # --- Temporal filter ---
    if not args.no_temporal:
        temp = rs.temporal_filter()
        # alpha: exponential moving average weight for current frame (0.0-1.0).
        # Lower = more temporal smoothing but more motion blur on fast objects
        temp.set_option(rs.option.filter_smooth_alpha, args.temporal_alpha)
        # delta: depth jump threshold in mm — pixel not updated if jump > delta
        temp.set_option(rs.option.filter_smooth_delta, args.temporal_delta)
        # persistency_index: how long a pixel value is held when no new data.
        # 0=disabled, 1=valid in 8/8, ..., 8=always (aggressive hole fill over time)
        temp.set_option(rs.option.holes_fill, args.temporal_persistency)
        filters.append(("temporal", temp))

    # --- Disparity → Depth ---
    d2d_inv = rs.disparity_transform(False)
    filters.append(("disparity_to_depth", d2d_inv))

    # --- Hole filling ---
    if not args.no_hole_fill:
        hf = rs.hole_filling_filter()
        # mode: 0=fill from left, 1=nearest colour, 2=farthest valid (nearest object)
        hf.set_option(rs.option.holes_fill, args.hole_fill)
        filters.append(("hole_fill", hf))

    return filters


def apply_filters(depth_frame, filters: list, apply_decimation: bool = False):
    """
    Apply the filter chain to a depth frame.
    Set apply_decimation=True for preview, False for the frame to be saved.
    """
    frame = depth_frame
    for name, filt in filters:
        if name == "decimation" and not apply_decimation:
            continue
        frame = filt.process(frame)
    return frame


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

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
    # Colour preview: clip at 1.5 m, zero pixels stay black
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


# ---------------------------------------------------------------------------
# Depth frame → numpy (handles decimated / non-decimated frames uniformly)
# ---------------------------------------------------------------------------

def depth_frame_to_numpy(frame, depth_scale: float) -> np.ndarray:
    raw = np.asanyarray(frame.get_data())
    return raw.astype(np.float32) * depth_scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- I/O ---
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

    # --- Spatial filter ---
    ap.add_argument('--no-spatial', action='store_true',
                    help='Disable spatial filter')
    ap.add_argument('--spatial-magnitude', type=int, default=2,
                    help='Spatial iterations (1-5). Higher = stronger smoothing')
    ap.add_argument('--spatial-alpha', type=float, default=0.5,
                    help='Spatial smooth alpha (0.25-1.0)')
    ap.add_argument('--spatial-delta', type=float, default=20.0,
                    help='Spatial edge threshold in mm (1-50). Higher = less edge preservation')
    ap.add_argument('--spatial-hole-fill', type=int, default=0,
                    help='Spatial hole fill size (0=none, 1-5)')

    # --- Temporal filter ---
    ap.add_argument('--no-temporal', action='store_true',
                    help='Disable temporal filter (use for fast-moving objects)')
    ap.add_argument('--temporal-alpha', type=float, default=0.4,
                    help='Temporal EMA alpha (0.0-1.0). Lower = smoother but more lag')
    ap.add_argument('--temporal-delta', type=float, default=20.0,
                    help='Temporal depth-jump threshold in mm')
    ap.add_argument('--temporal-persistency', type=int, default=3,
                    help='Temporal hole persistence (0=off, 8=always fill)')

    # --- Hole filling ---
    ap.add_argument('--no-hole-fill', action='store_true',
                    help='Disable hole filling filter')
    ap.add_argument('--hole-fill', type=int, default=1,
                    help='Hole fill mode: 0=fill_from_left, 1=nearest_colour, 2=farthest_valid')

    args = ap.parse_args()

    out_root = Path(args.out)
    scene_dir = out_root / args.subdir
    scene_dir.mkdir(parents=True, exist_ok=True)

    pipeline, aligner, depth_scale, intr_meta = make_pipeline(
        args.width, args.height, args.fps, args.depth_preset
    )
    filters = build_filters(args)

    # Print active filter chain
    active = [n for n, _ in filters]
    print(f"[postproc] active filters: {' → '.join(active)}")

    intr_path = out_root / 'intrinsics.json'
    if not intr_path.exists():
        intr_path.write_text(json.dumps(intr_meta, indent=2), encoding='utf-8')

    meta_csv = out_root / 'meta.csv'
    ensure_meta_csv(meta_csv)
    log_path = out_root / 'capture.log'
    next_id = next_scene_id(scene_dir)

    print(f'[capture] output={scene_dir} next_scene_id={next_id}')
    print("Controls: s=save  q/ESC=quit")

    win_rgb = 'RealSense Dataset Capture — RGB'
    win_depth = 'RealSense Dataset Capture — Depth (post-processed)'
    cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            # --- Raw depth frame (before alignment) ---
            raw_depth_frame = frames.get_depth_frame()
            if not raw_depth_frame:
                continue

            # ---------------------------------------------------------------
            # SAVE path: full-resolution post-processing, then align to colour
            # ---------------------------------------------------------------
            depth_for_save = apply_filters(raw_depth_frame, filters, apply_decimation=False)

            # Wrap processed depth back into a frameset for aligner
            processed_frameset = rs.composite_frame(rs.frame(depth_for_save))
            # aligner needs both streams — use original colour + processed depth
            aligned = aligner.process(frames)  # colour alignment reference
            aligned_depth = aligner.process(
                rs.composite_frame(rs.frame(depth_for_save))
                if hasattr(rs, 'composite_frame') else frames
            )

            # Fallback: standard align on full frames, then replace depth array
            aligned_full = aligner.process(frames)
            color_frame = aligned_full.get_color_frame()
            aligned_depth_frame = aligned_full.get_depth_frame()
            if not color_frame or not aligned_depth_frame:
                continue

            # Apply full post-processing chain (no decimation) to the aligned depth
            depth_frame_pp = apply_filters(aligned_depth_frame, filters, apply_decimation=False)
            depth_m_save = depth_frame_to_numpy(depth_frame_pp, depth_scale)
            rgb_bgr = np.asanyarray(color_frame.get_data())  # clean, no overlay

            # ---------------------------------------------------------------
            # PREVIEW path: decimated post-processing for display only
            # ---------------------------------------------------------------
            depth_frame_preview = apply_filters(aligned_depth_frame, filters, apply_decimation=True)
            depth_raw_preview = np.asanyarray(depth_frame_preview.get_data())
            depth_m_preview = depth_raw_preview.astype(np.float32) * depth_scale

            d_clip = np.clip(depth_m_preview, 0.0, 1.5)
            d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
            # Resize preview colourmap back to full display size if decimated
            depth_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
            if depth_color.shape[:2] != rgb_bgr.shape[:2]:
                depth_color = cv2.resize(depth_color, (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            depth_color[cv2.resize(depth_raw_preview, (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) == 0] = 0

            # ---------------------------------------------------------------
            # Display overlay (copy only — never touches rgb_bgr or depth_m_save)
            # ---------------------------------------------------------------
            display_bgr = rgb_bgr.copy()
            cv2.putText(display_bgr,
                        f"obj={args.object_name} inst={args.object_instance}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
            cv2.putText(display_bgr,
                        f"scene_id={next_id:04d}  s=save  q=quit",
                        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 1, cv2.LINE_AA)
            cv2.imshow(win_rgb, display_bgr)
            cv2.imshow(win_depth, depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                # Save clean rgb_bgr and post-processed full-resolution depth
                scene_base = save_scene(scene_dir, next_id, rgb_bgr, depth_m_save)
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                append_meta(meta_csv, [
                    ts, args.subdir, scene_base, args.object_name, args.object_instance,
                    args.scene_note, args.distance_m, args.lighting, args.background,
                ])
                with log_path.open('a', encoding='utf-8') as f:
                    f.write(f"{ts}\t{args.subdir}\t{scene_base}\t{args.object_name}\n")
                print(f'[saved] {scene_base} ({args.object_name})')
                next_id += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
