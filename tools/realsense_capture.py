"""Capture a Cornell-format dataset with an Intel RealSense D435.

For each captured scene we write:

* ``pcdNNNNr.png``        — RGB image (BGR8 → PNG via OpenCV).
* ``pcdNNNNd.tiff``       — depth aligned to color, **float32 metres**
  (matches what :func:`grasp_seg.data.jacquard_v2._load_depth` /
  :func:`_normalise_depth` expect, modulo the post-load percentile clip).
* ``pcdNNNNd_preview.png`` — cosmetic 8-bit colormap of the depth so
  you can eyeball it in a regular image viewer (not used by the loader).

Per session we additionally write:

* ``intrinsics.json``     — color stream intrinsics (``fx``, ``fy``,
  ``ppx``, ``ppy``, distortion model + coefficients) and the depth
  scale. Needed later for hand-eye calibration / pixel→3D backprojection
  in the manipulator phase.
* ``capture.log``         — append-only log of saved scenes.

The output directory layout matches the *flat* Cornell layout that
:func:`grasp_seg.data.cornell._index_scenes` already understands. To use
sub-directories ``01/``, ``02/`` … pass ``--subdir 01`` (we'll create
``<out>/01/pcd0100r.png`` etc.).

Two capture modes:

* **manual**    — press ``s`` to save the current aligned frame pair.
* **interval**  — auto-save one frame every ``--every`` seconds.

Common controls:

* ``s`` (manual mode) — save the current frame as the next scene.
* ``b``               — save **burst**: grab N frames quickly in a row
  (5 by default) so you can median-filter depth offline.
* ``q`` / ESC         — stop the session.

Annotation (positive ``cpos.txt`` / negative ``cneg.txt``) is **not**
done here — that's a separate UI step (Flask annotation tool, coming
in a follow-up PR).

Requirements (Windows):

    pip install pyrealsense2 opencv-python numpy tifffile

Examples::

    # Manual mode, save into captures/my_dataset/01/
    python tools/realsense_capture.py \\
        --out captures/my_dataset --subdir 01 --start-id 0

    # Interval mode, 1 frame/sec
    python tools/realsense_capture.py \\
        --out captures/my_dataset --subdir 02 \\
        --mode interval --every 1.0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from typing import Optional

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


_PCD_RE = re.compile(r"pcd(\d+)r\.png$")


# ---------------------------------------------------------------------------
# RealSense helpers (shared shape with realsense_preview.py)
# ---------------------------------------------------------------------------

def _make_pipeline(width: int, height: int, fps: int,
                   depth_preset: str) -> tuple[rs.pipeline, rs.align,
                                                float, dict]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

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
            pass

    depth_scale = float(depth_sensor.get_depth_scale())
    aligner = rs.align(rs.stream.color)

    color_intr = profile.get_stream(rs.stream.color) \
        .as_video_stream_profile().get_intrinsics()

    intr_dict = {
        "width":  color_intr.width,
        "height": color_intr.height,
        "fx":     float(color_intr.fx),
        "fy":     float(color_intr.fy),
        "ppx":    float(color_intr.ppx),
        "ppy":    float(color_intr.ppy),
        "model":  str(color_intr.model),
        "coeffs": [float(c) for c in color_intr.coeffs],
        "depth_scale_metres": depth_scale,
        "depth_preset":       depth_preset,
        "fps":                fps,
        "device": {
            "name":   profile.get_device().get_info(rs.camera_info.name),
            "serial": profile.get_device().get_info(
                rs.camera_info.serial_number),
            "fw":     profile.get_device().get_info(
                rs.camera_info.firmware_version),
        },
    }
    print(f"[realsense] device  : {intr_dict['device']['name']} "
          f"(serial {intr_dict['device']['serial']})")
    print(f"[realsense] color   : {color_intr.width}x{color_intr.height}"
          f" @ {fps}fps  fx={color_intr.fx:.1f}")
    print(f"[realsense] depth   : scale {depth_scale*1000:.4f} mm/unit, "
          f"preset {depth_preset}")
    return pipeline, aligner, depth_scale, intr_dict


# ---------------------------------------------------------------------------
# Output helpers (Cornell layout)
# ---------------------------------------------------------------------------

def _scene_dir(out_root: str, subdir: Optional[str]) -> str:
    """Return ``out_root/<subdir>`` (or ``out_root`` if ``subdir`` is None)."""
    target = os.path.join(out_root, subdir) if subdir else out_root
    os.makedirs(target, exist_ok=True)
    return target


def _next_pcd_id(scene_dir: str, start_id: Optional[int]) -> int:
    """Return the next ``pcdNNNN`` id to use, scanning existing files.

    Always returns ``max(existing) + 1`` (or 0 when empty) so we never
    overwrite a file. Filling gaps would be tempting but the main loop
    increments the id blindly after each save, which would clobber the
    next existing file on the second save.
    """
    if start_id is not None:
        return int(start_id)
    used = set()
    if os.path.isdir(scene_dir):
        for name in os.listdir(scene_dir):
            m = _PCD_RE.search(name)
            if m:
                used.add(int(m.group(1)))
    return max(used) + 1 if used else 0


def _save_scene(scene_dir: str, idx: int, rgb_bgr: np.ndarray,
                depth_metres: np.ndarray, log_path: str) -> str:
    """Write ``pcdNNNNr.png``, ``pcdNNNNd.tiff`` and a preview PNG.

    Returns the scene id ``"NNNN"`` as a zero-padded string.
    """
    sid = f"{idx:04d}"
    base = os.path.join(scene_dir, f"pcd{sid}")
    cv2.imwrite(base + "r.png", rgb_bgr)
    tifffile.imwrite(base + "d.tiff", depth_metres.astype(np.float32))

    d_clip = np.clip(depth_metres, 0.0, 1.5)
    d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    preview[depth_metres == 0] = 0
    cv2.imwrite(base + "d_preview.png", preview)

    with open(log_path, "a", encoding="utf-8") as f:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        f.write(f"{ts}\tpcd{sid}\t{rgb_bgr.shape[1]}x{rgb_bgr.shape[0]}\n")

    print(f"[saved] {base}r.png + d.tiff + d_preview.png  "
          f"depth: nz={int((depth_metres > 0).sum())}/{depth_metres.size}, "
          f"median={float(np.median(depth_metres[depth_metres > 0])):.3f}m"
          if (depth_metres > 0).any() else
          f"[saved] {base}* (no valid depth pixels!)")
    return sid


def _save_burst(scene_dir: str, idx: int, frames_rgb: list,
                frames_depth: list, log_path: str) -> str:
    """Save N frames as ``pcdNNNN_kr.png`` / ``pcdNNNN_kd.tiff``.

    The first frame is also saved as the canonical ``pcdNNNNr.png`` /
    ``pcdNNNNd.tiff`` so the Cornell loader can index the scene; the
    rest are kept alongside for offline median-filtering.
    """
    sid = f"{idx:04d}"
    # canonical pair
    _save_scene(scene_dir, idx, frames_rgb[0], frames_depth[0], log_path)
    # extra frames
    for k, (rgb, depth) in enumerate(zip(frames_rgb[1:], frames_depth[1:]),
                                     start=1):
        base = os.path.join(scene_dir, f"pcd{sid}_{k}")
        cv2.imwrite(base + "r.png", rgb)
        tifffile.imwrite(base + "d.tiff", depth.astype(np.float32))
    return sid


# ---------------------------------------------------------------------------
# Frame grabbing
# ---------------------------------------------------------------------------

def _grab_aligned(pipeline: rs.pipeline, aligner: rs.align,
                  depth_scale: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rgb_bgr, depth_metres, depth_raw_uint16)`` aligned to color."""
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    aligned = aligner.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("Dropped frame from RealSense pipeline")
    rgb_bgr = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * depth_scale
    return rgb_bgr, depth_m, depth_raw


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True,
                   help="Output dataset root, Cornell layout.")
    p.add_argument("--subdir", default=None,
                   help="Optional sub-directory under --out (e.g. '01').")
    p.add_argument("--start-id", type=int, default=None,
                   help="Force the first pcd ID instead of auto-detecting.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--depth-preset", default="highaccuracy",
                   choices=["default", "highaccuracy", "highdensity",
                            "medium"])
    p.add_argument("--mode", default="manual",
                   choices=["manual", "interval"],
                   help="manual = save on 's'; interval = save every N sec.")
    p.add_argument("--every", type=float, default=1.0,
                   help="Interval-mode save period (seconds).")
    p.add_argument("--burst", type=int, default=5,
                   help="Number of frames in a burst capture (key 'b').")
    p.add_argument("--no-window", action="store_true",
                   help="Don't open OpenCV preview windows (headless).")
    args = p.parse_args()

    # Headless mode has no keyboard input, so manual save/quit is impossible.
    # Reject the combination explicitly rather than silently looping forever.
    if args.no_window and args.mode == "manual":
        p.error("--no-window requires --mode interval "
                "(manual mode needs a window to read 's'/'b'/'q' keys)")

    pipeline, aligner, depth_scale, intr = _make_pipeline(
        args.width, args.height, args.fps, args.depth_preset,
    )

    scene_dir = _scene_dir(args.out, args.subdir)
    next_id = _next_pcd_id(scene_dir, args.start_id)
    # intrinsics.json + capture.log live at the dataset root so a single
    # session covering multiple subdirs (01/, 02/, ...) shares them, and
    # the layout matches docs/realsense_setup.md.
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "capture.log")
    intr_path = os.path.join(args.out, "intrinsics.json")
    if not os.path.isfile(intr_path):
        with open(intr_path, "w", encoding="utf-8") as f:
            json.dump(intr, f, indent=2)
        print(f"[realsense] wrote {intr_path}")

    print(f"[realsense] saving to {scene_dir}, next id = {next_id:04d}")
    if args.mode == "manual":
        print("Controls: 's' = save  |  'b' = burst (5 frames)  "
              "|  'q' / ESC = quit")
    else:
        print(f"Controls: 'q' / ESC = quit. Auto-saving every "
              f"{args.every:.2f}s.")

    win_rgb = win_depth = None
    if not args.no_window:
        win_rgb = "RealSense D435 — RGB"
        win_depth = "RealSense D435 — Depth"
        cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    last_auto_save = time.time()

    try:
        while True:
            try:
                rgb_bgr, depth_m, depth_raw = _grab_aligned(
                    pipeline, aligner, depth_scale)
            except RuntimeError as e:
                print(f"[realsense] {e}, retrying...")
                continue

            if not args.no_window:
                d_clip = np.clip(depth_m, 0.0, 1.5)
                d_u8 = (d_clip / 1.5 * 255.0).astype(np.uint8)
                depth_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                depth_color[depth_raw == 0] = 0

                hud = rgb_bgr.copy()
                cv2.putText(hud, f"next: pcd{next_id:04d}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (40, 220, 40), 2, cv2.LINE_AA)
                cv2.imshow(win_rgb, hud)
                cv2.imshow(win_depth, depth_color)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0

            if key in (ord("q"), 27):
                break

            do_save = False
            do_burst = False
            if args.mode == "manual":
                if key == ord("s"):
                    do_save = True
                elif key == ord("b"):
                    do_burst = True
            else:  # interval
                now = time.time()
                if now - last_auto_save >= args.every:
                    do_save = True
                    last_auto_save = now

            if do_burst:
                bursts_rgb = [rgb_bgr]
                bursts_depth = [depth_m]
                for _ in range(max(1, args.burst) - 1):
                    rgb2, d2, _ = _grab_aligned(pipeline, aligner, depth_scale)
                    bursts_rgb.append(rgb2)
                    bursts_depth.append(d2)
                _save_burst(scene_dir, next_id,
                            bursts_rgb, bursts_depth, log_path)
                next_id += 1
            elif do_save:
                _save_scene(scene_dir, next_id, rgb_bgr, depth_m, log_path)
                next_id += 1

    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
