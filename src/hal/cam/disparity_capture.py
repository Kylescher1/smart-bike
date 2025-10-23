#!/usr/bin/env python3
"""
disparity_capture.py

Single-file stereo capture + disparity pipeline with consistent full-resolution frames.

Features:
- Uses camera-native resolution everywhere (no downsampling, no cropping).
- Saves synchronized outputs every 150 ms:
    - Compressed left/right images (.jpg)
    - Full-fidelity disparity map (.npz) with metadata
    - Optional colorized disparity visualization (.jpg) when preview enabled
- Optional live preview window (off by default; enable with --preview).
- Optional tuner UI for SGBM parameters (off by default; enable with --tuner).

Dependencies:
- OpenCV (cv2), NumPy
- Your project’s calibration and camera utilities:
    - src.hal.cam.calibrate.calib.load_calibration
    - src.hal.cam.Camera.open_stereo_pair
"""

from __future__ import annotations

import os
import cv2
import json
import time
import argparse
import threading
import numpy as np
from datetime import datetime

# Optional ximgproc WLS post-filter
try:
    import cv2.ximgproc as xip  # type: ignore
    HAS_XIMGPROC = True
except Exception:
    HAS_XIMGPROC = False

# Project-specific imports (must exist in your environment)
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Camera import open_stereo_pair


# ---------------------------
# Utility: performance tracker
# ---------------------------
class PerfTracker:
    def __init__(self) -> None:
        self.times = {}
        self.last = time.perf_counter()

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self.times[label] = (now - self.last) * 1000.0  # ms
        self.last = now

    def summary(self) -> str:
        total = sum(self.times.values())
        parts = " | ".join(f"{k}:{v:.1f}ms" for k, v in self.times.items())
        return f"{parts} | total:{total:.1f}ms"


# ---------------------------
# Camera threading wrapper
# ---------------------------
class ThreadedCamera:
    """
    Continuous background grab. get() returns the most recent frame.
    """
    def __init__(self, cam) -> None:
        self.cam = cam
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.th = threading.Thread(target=self._update, daemon=True)
        self.th.start()

    def _update(self) -> None:
        while self.running:
            f = self.cam.read_frame()
            if f is not None:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.002)

    def get(self):
        with self.lock:
            return self.frame

    def close(self) -> None:
        self.running = False
        try:
            self.th.join(timeout=0.5)
        except Exception:
            pass
        self.cam.close()


from queue import Queue

save_queue = Queue(maxsize=4)

def save_worker():
    while True:
        item = save_queue.get()
        if item is None:  # shutdown signal
            break
        save_outputs(**item)
        save_queue.task_done()

# ---------------------------
# Disparity computation utils
# ---------------------------
ROOT = os.path.join(os.path.dirname(__file__), "../../..")
SETTINGS_FILE = os.path.join(ROOT, "disparity_settings.json")
PROFILE_DIR = os.path.join(ROOT, "disparity_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "minDisparity": 0,
    "numDisparities": 4,
    "blockSize": 16,
    "preFilterCap": 31,
    "uniquenessRatio": 5,
    "speckleWindowSize": 160,
    "speckleRange": 7,
    "disp12MaxDiff": 7,
    "medianBlurK": 0,
    "downSample": 100,
    "crop": 0,
    "farEnhance": 50,
    "nearCutoff": 0,
    "useMorph": 1,
    "morphIter": 1,
    "useBilateral": 1,
    "bilateralStrength": 8,
    "useWLS": 0,
    "wlsLambda": 4000,
    "wlsSigma": 1.0,
    "profileName": "default"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

def load_profile(name):
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"⚠️ Profile '{name}' not found.")
        return None
    with open(path) as f:
        return {**DEFAULT_SETTINGS, **json.load(f)}



def ensure_odd(n: int, min_val: int = 3) -> int:
    n = max(min_val, n)
    return n if (n % 2 == 1) else (n + 1)


def rectify_pair(left_bgr: np.ndarray,
                 right_bgr: np.ndarray,
                 calib):
    """
    Remap color frames using precomputed rectification maps.
    """
    leftMapX, leftMapY, rightMapX, rightMapY, _, _ = calib
    rectL = cv2.remap(left_bgr, leftMapX, leftMapY, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_bgr, rightMapX, rightMapY, cv2.INTER_LINEAR)
    return rectL, rectR


def compute_disparity_map(gray_left: np.ndarray,
                          gray_right: np.ndarray,
                          s: dict):
    """
    StereoSGBM disparity at full resolution. Returns (disp_float32, num_disp_int).
    """
    min_disp = int(s["minDisparity"])
    num_disp = 16 * max(1, int(s["numDisparitiesK"]))
    block_size = ensure_odd(int(s["blockSize"]))

    P1 = 8 * block_size * block_size
    P2 = 32 * block_size * block_size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        preFilterCap=int(s["preFilterCap"]),
        uniquenessRatio=int(s["uniquenessRatio"]),
        speckleWindowSize=int(s["speckleWindowSize"]),
        speckleRange=int(s["speckleRange"]),
        disp12MaxDiff=int(s["disp12MaxDiff"]),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disp, num_disp

def disparity_to_depth_opencv(disp: np.ndarray, calib):
    """
    Convert disparity to real-world depth (Z in meters) using calibration reprojection matrix Q.
    Expects calib to include Q as the 5th element of the tuple returned by load_calibration().
    """
    # calib = (leftMapX, leftMapY, rightMapX, rightMapY, Q, extras)
    Q = calib[4]
    # 3D reprojected points: shape (H, W, 3)
    points_3d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    depth = points_3d[:, :, 2]  # Z coordinate in meters
    depth[~np.isfinite(depth)] = 0.0
    return depth


def preprocess_images(grayL: np.ndarray, grayR: np.ndarray, s: dict):
    """
    Apply downsampling and cropping before disparity computation.
    """
    scale = max(0.1, s.get("downSample", 100) / 100.0)
    if scale < 0.999:
        grayL = cv2.resize(grayL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grayR = cv2.resize(grayR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c = int(s.get("crop", 0))
    if c > 0:
        h, w = grayL.shape[:2]
        grayL = grayL[c:h - c, c:w - c]
        grayR = grayR[c:h - c, c:w - c]
    return grayL, grayR


def post_filter_strong(disp: np.ndarray,
                       guide_gray: np.ndarray,
                       s: dict) -> np.ndarray:
    """
    Optional morphological closing, bilateral smoothing, and WLS (if available).
    """
    out = disp
    if s.get("useMorph", 1):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                               iterations=int(s.get("morphIter", 1)))

    if s.get("useBilateral", 1) and int(s.get("bilateralStrength", 0)) > 0:
        b = int(s["bilateralStrength"])
        # Bilateral on float disparity; OpenCV supports this if data is CV_32F
        out = cv2.bilateralFilter(out, 5, b, b)

    if s.get("useWLS", 0) and HAS_XIMGPROC and int(s.get("wlsLambda", 0)) > 0:
        wls = xip.createDisparityWLSFilterGeneric(False)
        wls.setLambda(float(s["wlsLambda"]))
        wls.setSigmaColor(float(s.get("wlsSigma", 1.0)))
        out = wls.filter(out, guide_gray)

    return out

def post_filter_weak(disp: np.ndarray, s: dict) -> np.ndarray:
    """
    Remove very near pixels (large disparity) and clean isolated noise.
    """
    out = disp.copy()

    # Cutoff for very near objects
    near_cut = float(s.get("nearCutoff", 0))
    if near_cut > 0:
        out[out > near_cut] = 0.0

    # Remove isolated remnants by neighborhood tally
    mask = (out > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(3):
        neighbors = cv2.filter2D(mask, -1, kernel)
        isolated = (neighbors <= 3) & (mask == 1)
        if not np.any(isolated):
            break
        mask[isolated] = 0

    out[mask == 0] = 0.0
    return out


def visualize_disparity(disp: np.ndarray,
                        num_disp: int,
                        far_enhance: int = 50) -> np.ndarray:
    """
    Colorized visualization tuned towards far field when far_enhance > 0.
    Returns BGR uint8 image.
    """
    disp = np.clip(disp, 0, num_disp).astype(np.float32)

    valid = disp[disp > 0]
    if valid.size > 0:
        bias = np.clip(far_enhance / 200.0, 0.0, 1.0)
        low = np.percentile(valid, (1 - bias) * 80)
        high = np.percentile(valid, 100 - (1 - bias) * 10)
        if high <= low:
            high = low + 1.0
        vis = np.clip((disp - low) / (high - low), 0.0, 1.0)
    else:
        vis = np.zeros_like(disp)

    norm = (vis * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return color


# ---------------------------
# Optional tuner UI
# ---------------------------
def nothing(_):  # trackbar callback
    pass


def create_tuner_window(params: dict) -> None:
    """
    Creates a window with trackbars controlling SGBM and filter parameters.
    """
    cv2.namedWindow("Disparity Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Disparity Tuner", 420, 480)

    cv2.createTrackbar("minDisp", "Disparity Tuner", int(params["minDisparity"]), 64, nothing)
    cv2.createTrackbar("numDispK", "Disparity Tuner", int(params["numDisparitiesK"]), 32, nothing)
    cv2.createTrackbar("blockSize", "Disparity Tuner", int(params["blockSize"]), 31, nothing)
    cv2.createTrackbar("uniqueness", "Disparity Tuner", int(params["uniquenessRatio"]), 50, nothing)
    cv2.createTrackbar("preFilterCap", "Disparity Tuner", int(params["preFilterCap"]), 63, nothing)
    cv2.createTrackbar("speckleRange", "Disparity Tuner", int(params["speckleRange"]), 50, nothing)

    cv2.createTrackbar("useMorph", "Disparity Tuner", int(params["useMorph"]), 1, nothing)
    cv2.createTrackbar("morphIter", "Disparity Tuner", int(params["morphIter"]), 3, nothing)
    cv2.createTrackbar("useBilateral", "Disparity Tuner", int(params["useBilateral"]), 1, nothing)
    cv2.createTrackbar("bilatStr", "Disparity Tuner", int(params["bilateralStrength"]), 32, nothing)
    cv2.createTrackbar("useWLS", "Disparity Tuner", int(params["useWLS"]), 1, nothing)
    cv2.createTrackbar("wlsLambda", "Disparity Tuner", int(params["wlsLambda"]), 10000, nothing)
    cv2.createTrackbar("wlsSigmaX10", "Disparity Tuner", int(params["wlsSigma"] * 10), 50, nothing)

    cv2.createTrackbar("farEnh", "Disparity Tuner", int(params["farEnhance"]), 200, nothing)
    cv2.createTrackbar("nearCut", "Disparity Tuner", int(params["nearCutoff"]), 200, nothing)

    cv2.createTrackbar("downSample%", "Disparity Tuner", int(params.get("downSample", 100)), 100, nothing)
    cv2.createTrackbar("crop(px)", "Disparity Tuner", int(params.get("crop", 0)), 200, nothing)



def read_tuner_params() -> dict:
    """
    Read current trackbar values into a parameter dict.
    """
    s = {
        "minDisparity": cv2.getTrackbarPos("minDisp", "Disparity Tuner"),
        "numDisparitiesK": max(1, cv2.getTrackbarPos("numDispK", "Disparity Tuner")),
        "blockSize": ensure_odd(cv2.getTrackbarPos("blockSize", "Disparity Tuner")),
        "preFilterCap": cv2.getTrackbarPos("preFilterCap", "Disparity Tuner"),
        "uniquenessRatio": cv2.getTrackbarPos("uniqueness", "Disparity Tuner"),
        "speckleRange": cv2.getTrackbarPos("speckleRange", "Disparity Tuner"),
        "speckleWindowSize": 160,
        "disp12MaxDiff": 7,
        "useMorph": cv2.getTrackbarPos("useMorph", "Disparity Tuner"),
        "morphIter": cv2.getTrackbarPos("morphIter", "Disparity Tuner"),
        "useBilateral": cv2.getTrackbarPos("useBilateral", "Disparity Tuner"),
        "bilateralStrength": cv2.getTrackbarPos("bilatStr", "Disparity Tuner"),
        "useWLS": cv2.getTrackbarPos("useWLS", "Disparity Tuner"),
        "wlsLambda": cv2.getTrackbarPos("wlsLambda", "Disparity Tuner"),
        "wlsSigma": cv2.getTrackbarPos("wlsSigmaX10", "Disparity Tuner") / 10.0,
        "farEnhance": cv2.getTrackbarPos("farEnh", "Disparity Tuner"),
        "nearCutoff": cv2.getTrackbarPos("nearCut", "Disparity Tuner"),
        "downSample": max(10, cv2.getTrackbarPos("downSample%", "Disparity Tuner")),
        "crop": cv2.getTrackbarPos("crop(px)", "Disparity Tuner"),

    }
    return s


# ---------------------------
# Saving utilities
# ---------------------------
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision


def save_outputs(
    out_dir: str,
    ts: str,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    depth: np.ndarray,
    num_disp: int,
    settings: dict,
    jpeg_quality: int,
    save_vis_jpg: bool
) -> None:
    """
    Save synchronized pair + depth. JPEG for images, NPZ for disparity.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Left/Right compressed JPEGs
    cv2.imwrite(
        os.path.join(out_dir, f"left_{ts}.jpg"),
        left_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    )
    cv2.imwrite(
        os.path.join(out_dir, f"right_{ts}.jpg"),
        right_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    )

    # Lossless disparity with metadata
    # Use compressed npz to reduce disk bandwidth while preserving exact values
    np.savez_compressed(
        os.path.join(out_dir, f"depthmap_{ts}.npz"),
        depth=disp.astype(np.float32),  # now holds meters
        num_disp=int(num_disp),
        settings=json.dumps(settings)
    )

    # Optional visualization image for quick glance
    if save_vis_jpg:
        vis = visualize_disparity(disp, num_disp, int(settings.get("farEnhance", 50)))
        cv2.imwrite(
            os.path.join(out_dir, f"disp_vis_{ts}.jpg"),
            vis,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        )


# ---------------------------
# Main application
# ---------------------------
def run(args,
        preview: bool,
        tuner: bool,
        save_interval_s: float,
        out_dir: str,
        jpeg_quality: int) -> None:


    # Load calibration and open cameras
    calib = load_calibration()
    left_cam_raw, right_cam_raw = open_stereo_pair()
    left_cam = ThreadedCamera(left_cam_raw)
    right_cam = ThreadedCamera(right_cam_raw)

    params = load_settings()

    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()

    # Backward-compatibility: convert old key names if needed
    if "numDisparitiesK" not in params and "numDisparities" in params:
        # old tuner stored raw disparity count multiplier (4, 8, etc.)
        params["numDisparitiesK"] = params.pop("numDisparities")

    # Optionally override with a saved profile
    if args.profile:
        prof = load_profile(args.profile)
        if prof:
            # same normalization for loaded profile
            if "numDisparitiesK" not in prof and "numDisparities" in prof:
                prof["numDisparitiesK"] = prof.pop("numDisparities")
            params = prof
            print(f"Loaded profile: {args.profile}")
        else:
            print(f"Profile '{args.profile}' not found. Using last saved settings.")

    # UI initialization (off by default)
    if tuner:
        create_tuner_window(params)

    if preview:
        cv2.namedWindow("Disparity Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Disparity Preview", 1280, 720)

    last_save = time.perf_counter()

    try:
        while True:
            tracker = PerfTracker()

            # Capture most recent frames
            left = left_cam.get()
            right = right_cam.get()
            tracker.mark("capture")
            if left is None or right is None:
                time.sleep(0.001)
                continue

            # Rectify full-resolution color
            rectL_color, rectR_color = rectify_pair(left, right, calib)
            tracker.mark("rectify")

            # Convert to grayscale for disparity
            grayL = cv2.cvtColor(rectL_color, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectR_color, cv2.COLOR_BGR2GRAY)
            tracker.mark("grayscale")
            grayL, grayR = preprocess_images(grayL, grayR, params)
            tracker.mark("preproc")

            # Update params from tuner if enabled
            if tuner:
                params = {**params, **read_tuner_params()}

            # Compute disparity at full resolution
            disp, num_disp = compute_disparity_map(grayL, grayR, params)
            tracker.mark("sgbm")

            # Post-filters
            disp = post_filter_strong(disp, grayL, params)
            disp = post_filter_weak(disp, params)
            tracker.mark("filters")

            # Convert disparity to depth
            depth = disparity_to_depth_opencv(disp, calib)
            tracker.mark("depth_convert")

            # Periodic save
            now = time.perf_counter()
            if args.saveframes and (now - last_save >= save_interval_s):
                ts = timestamp()
                save_queue.put({
                    "out_dir": out_dir,
                    "ts": ts,
                    "left_bgr": rectL_color.copy(),
                    "right_bgr": rectR_color.copy(),
                    "depth": disp.copy(),
                    "num_disp": num_disp,
                    "settings": params.copy(),
                    "jpeg_quality": jpeg_quality,
                    "save_vis_jpg": preview
                })
                last_save = now


            # Optional live preview
            if preview:
                vis = visualize_disparity(disp, num_disp, int(params.get("farEnhance", 50)))
                # Fit window for display only; underlying data remains full-res
                h, w = vis.shape[:2]
                target_w = 1280
                scale = target_w / float(w)
                vis_display = cv2.resize(vis, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

                # HUD text
                txt = f"numDisp={num_disp}  blk={params['blockSize']}  uniq={params['uniquenessRatio']}  " \
                      f"bilat={'Y' if params['useBilateral'] else 'N'}  WLS={'Y' if params['useWLS'] and HAS_XIMGPROC else 'N'}"
                cv2.putText(vis_display, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Disparity Preview", vis_display)
                # Allow key handling only when preview/tuner windows exist
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Headless: tiny sleep to avoid pegging CPU
                time.sleep(0.001)

            # Log performance summary to stdout
            print(tracker.summary())

    except KeyboardInterrupt:
        pass

    finally:
        # Persist last-used disparity settings
        try:
            save_settings(params)
            print("Settings saved to disparity_settings.json")
        except Exception as e:
            print("Warning: could not save settings:", e)

        # ---- Saving thread shutdown ----
        try:
            save_queue.put(None)  # signal worker to stop
            save_thread.join(timeout=5)
            print("Background save thread stopped.")
        except Exception as e:
            print("Warning: could not join save thread:", e)

        # Cleanup
        try:
            left_cam.close()
            right_cam.close()
        except Exception as e:
            print("Warning: could not close cameras:", e)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass



# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Full-resolution stereo disparity capture with periodic saving.")
    p.add_argument("--preview", action="store_true", help="Enable live preview window. Default off.")
    p.add_argument("--tuner", action="store_true", help="Enable tuner UI for SGBM and filters. Default off.")
    p.add_argument("--save-interval", type=float, default=0.15, help="Save interval in seconds. Default 0.15 (150 ms).")
    p.add_argument("--out", type=str, default="./images", help="Output directory. Default ./images")
    p.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 1..100. Default 85")
    p.add_argument("--profile", type=str,
                   help="Profile name to load from ./disparity_profiles (optional)")
    p.add_argument("--saveframes", action="store_true",
               help="Enable saving of .jpg and .npz outputs. Default off for max performance.")

    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    run(
        args,
        preview=args.preview,
        tuner=args.tuner,
        save_interval_s=max(0.01, args.save_interval),
        out_dir=args.out,
        jpeg_quality=int(np.clip(args.jpeg_quality, 1, 100))
    )

