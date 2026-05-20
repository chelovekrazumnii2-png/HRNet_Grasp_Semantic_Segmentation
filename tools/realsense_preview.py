"""Live preview from an Intel RealSense D435 (RGB + aligned depth).

Two windows side-by-side:

* **RGB** — raw color frame, 1280×720 by default.
* **Depth (colormap)** — depth aligned to color, displayed via
  ``cv2.applyColorMap`` for human-readable visualisation. The colormap
  is purely cosmetic; the underlying values are 16-bit millimetres.

Keyboard shortcuts (focus must be on one of the OpenCV windows):

* ``s`` — save the *current* aligned frame pair into ``--out`` as
  ``snap_NNNN_r.png`` (RGB) + ``snap_NNNN_d.tiff`` (float32 metres) +
  ``snap_NNNN_d_preview.png`` (8-bit colormap, for quick eyeballing).
* ``q`` / ``ESC`` — quit.

This script is intentionally minimal: it's the smoke-test for the
RealSense Python stack on Windows. Use ``tools/realsense_capture.py``
for structured dataset capture.

Requirements (Windows, RTX 3060 box):

    pip install pyrealsense2 opencv-python numpy tifffile

Run:

    python tools/realsense_preview.py --out captures/preview
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "pyrealsense2 is not installed. On Windows run\n"
        "    pip install pyrealsense2\n"
        "If pip can't find a wheel for your Python version, see\n"
        "docs/realsense_setup.md for alternatives.\n"
    )
    raise SystemExit(1) from exc

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - import guard
    sys.stderr.write("tifffile is not installed. Run: pip install tifffile\n")
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

def _make_pipeline(width: int, height: int, fps: int,
                   depth_preset: str) -> tuple[rs.pipeline, rs.align, float]:
    """Start a RealSense pipeline with RGB + aligned depth.

    Returns ``(pipeline, aligner, depth_scale_in_metres)``.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    # Pick a depth-quality preset. D435 firmware exposes a small set of
    # presets keyed by integer indices; we map the most useful ones to
    # human-readable names. Falls back silently when the preset is
    # unsupported.
    presets = {
        "default":      0,
        "highaccuracy": 3,
        "highdensity":  4,
        "medium":       5,
    }
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_preset.lower() in presets and depth_sensor.supports(
            rs.option.visual_preset):
        try:
            depth_sensor.set_option(rs.option.visual_preset,
                                    presets[depth_preset.lower()])
        except RuntimeError:
            pass  # firmware refused — keep default

    depth_scale = float(depth_sensor.get_depth_scale())  # metres per unit
    aligner = rs.align(rs.stream.color)

    intr = profile.get_stream(rs.stream.color) \
        .as_video_stream_profile().get_intrinsics()
    print(f"[realsense] color stream  : {intr.width}x{intr.height} @ {fps}fps")
    print(f"[realsense] depth scale   : {depth_scale * 1000:.4f} mm/unit")
    print(f"[realsense] depth preset  : {depth_preset}")
    print(f"[realsense] color intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    return pipeline, aligner, depth_scale


# ---------------------------------------------------------------------------
# Frame saving
# ---------------------------------------------------------------------------

def _next_index(out_dir: str, prefix: str = "snap_") -> int:
    """Return the next free 4-digit index for ``snap_NNNN_*`` files.

    Always ``max(existing) + 1`` (or 0 when empty) so we never overwrite —
    the main loop increments the index blindly after each save, so a
    gap-filling strategy would clobber the next existing file.
    """
    if not os.path.isdir(out_dir):
        return 0
    used = set()
    for name in os.listdir(out_dir):
        if name.startswith(prefix) and len(name) >= len(prefix) + 4:
            tail = name[len(prefix):len(prefix) + 4]
            if tail.isdigit():
                used.add(int(tail))
    return max(used) + 1 if used else 0


def _save_pair(out_dir: str, idx: int, rgb_bgr: np.ndarray,
               depth_metres: np.ndarray) -> None:
    """Save BGR + depth metres + a colormap preview."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"snap_{idx:04d}")
    cv2.imwrite(base + "_r.png", rgb_bgr)
    tifffile.imwrite(base + "_d.tiff", depth_metres.astype(np.float32))
    # Cosmetic 8-bit preview of the depth (clipped to 1 m for readability).
    d_clip = np.clip(depth_metres, 0.0, 1.5)
    d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    cv2.imwrite(base + "_d_preview.png", preview)
    print(f"[saved] {base}_r.png + _d.tiff + _d_preview.png")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="captures/preview",
                   help="Directory to write saved frames into.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--depth-preset", default="highaccuracy",
                   choices=["default", "highaccuracy", "highdensity",
                            "medium"],
                   help="Depth-quality preset for the D435 firmware.")
    args = p.parse_args()

    pipeline, aligner, depth_scale = _make_pipeline(
        args.width, args.height, args.fps, args.depth_preset,
    )

    next_idx = _next_index(args.out)
    print(f"[realsense] writing to    : {args.out} (next idx = {next_idx})")
    print("Controls: 's' = save frame  |  'q' / ESC = quit")

    win_rgb = "RealSense D435 - RGB"
    win_depth = "RealSense D435 - Depth (jet colormap)"
    cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    fps_t0 = time.time()
    fps_n = 0
    fps_smooth = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = aligner.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())  # (H, W, 3) BGR
            depth_raw = np.asanyarray(depth_frame.get_data())  # (H, W) uint16
            depth_m = depth_raw.astype(np.float32) * depth_scale  # metres

            # Cosmetic colormap (clip to 1.5 m for readability).
            d_clip = np.clip(depth_m, 0.0, 1.5)
            d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
            # Mark missing depth (raw == 0) as black so it's visible.
            depth_color[depth_raw == 0] = 0

            # FPS HUD on the RGB frame.
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 0.5:
                fps_smooth = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0
            hud = rgb_bgr.copy()
            cv2.putText(hud, f"{fps_smooth:5.1f} fps",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (40, 220, 40), 2, cv2.LINE_AA)

            cv2.imshow(win_rgb, hud)
            cv2.imshow(win_depth, depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("s"):
                _save_pair(args.out, next_idx, rgb_bgr, depth_m)
                next_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
