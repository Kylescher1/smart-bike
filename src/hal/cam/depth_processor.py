#!/usr/bin/env python3
"""
Depth.py

Importable, class-structured stereo disparity and depth computation API.

Key points:
- Accepts stereo frames (left/right BGR), calibration, and settings.
- Provides depth map output and a method to save a .npz file.
- Allows runtime tuning via setters (e.g., colormap, downsample, filters).
- No camera I/O, CLI, GUI, or background threads in this module.

Calibration scaling:
- When downsampling/cropping is applied, the reprojection matrix Q is
  scaled to maintain accurate depth calculations.
"""

from __future__ import annotations

import os
import cv2
import json
import time
import numpy as np
from datetime import datetime

# Optional ximgproc WLS post-filter
try:
    import cv2.ximgproc as xip  # type: ignore
    HAS_XIMGPROC = True
except Exception:
    HAS_XIMGPROC = False

# Project-specific imports should be performed by the caller. This module
# does not open cameras or load calibration on its own.


# ---------------------------
# Constants and Configuration
# ---------------------------
ROOT = os.path.join(os.path.dirname(__file__), "../../..")
SETTINGS_FILE = os.path.join(ROOT, "disparity_settings.json")
PROFILE_DIR = os.path.join(ROOT, "disparity_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "minDisparity": 0,
    "numDisparitiesK": 4,
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

# (Performance tracker removed; keep module focused on API.)


# (ThreadedCamera removed; no I/O in this module.)


class StereoSGBMCache:
    """Cache StereoSGBM objects to avoid recreating them every frame."""
    def __init__(self):
        self.stereo = None
        self.last_params = None
    
    def get_or_create(self, params: dict) -> cv2.StereoSGBM:
        """Get cached stereo object or create new one if parameters changed."""
        current_params = (
            params["minDisparity"],
            params["numDisparitiesK"],
            params["blockSize"],
            params["preFilterCap"],
            params["uniquenessRatio"],
            params["speckleWindowSize"],
            params["speckleRange"],
            params["disp12MaxDiff"]
        )
        
        if self.stereo is None or self.last_params != current_params:
            min_disp = int(params["minDisparity"])
            num_disp = 16 * max(1, int(params["numDisparitiesK"]))
            block_size = ensure_odd(int(params["blockSize"]))
            
            P1 = 8 * block_size * block_size
            P2 = 32 * block_size * block_size
            
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=P1, P2=P2,
                preFilterCap=int(params["preFilterCap"]),
                uniquenessRatio=int(params["uniquenessRatio"]),
                speckleWindowSize=int(params["speckleWindowSize"]),
                speckleRange=int(params["speckleRange"]),
                disp12MaxDiff=int(params["disp12MaxDiff"]),
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            self.last_params = current_params
        
        return self.stereo


class RectificationCache:
    """Cache unpacked rectification maps to avoid tuple unpacking overhead."""
    def __init__(self, calib):
        self.leftMapX = calib[0]
        self.leftMapY = calib[1]
        self.rightMapX = calib[2]
        self.rightMapY = calib[3]


# (Background save worker removed; saving is synchronous via API.)


# ---------------------------
# Disparity computation utils
# ---------------------------

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS.copy()

def save_settings(s: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

def load_profile(name: str) -> dict | None:
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
                 rect_cache: RectificationCache):
    """
    Remap color frames using precomputed rectification maps.
    """
    rectL = cv2.remap(left_bgr, rect_cache.leftMapX, rect_cache.leftMapY, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_bgr, rect_cache.rightMapX, rect_cache.rightMapY, cv2.INTER_LINEAR)
    return rectL, rectR


def compute_disparity_map(gray_left: np.ndarray,
                          gray_right: np.ndarray,
                          s: dict,
                          stereo_cache: StereoSGBMCache):
    """
    StereoSGBM disparity computation with cached stereo object.
    Returns (disp_float32, num_disp_int).
    """
    stereo = stereo_cache.get_or_create(s)
    num_disp = 16 * max(1, int(s["numDisparitiesK"]))
    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disp, num_disp

def scale_calibration_for_downsampling(calib, scale_factor: float, crop_pixels: int = 0):
    """
    Scale calibration parameters for downsampled and cropped images.
    
    Args:
        calib: Original calibration tuple (leftMapX, leftMapY, rightMapX, rightMapY, imageSize, Q)
        scale_factor: Downsampling scale factor (e.g., 0.5 for 50% size)
        crop_pixels: Number of pixels cropped from each edge
        
    Returns:
        Scaled calibration tuple with adjusted Q matrix
    """
    leftMapX, leftMapY, rightMapX, rightMapY, imageSize, Q = calib
    
    # Create a copy of Q to avoid modifying the original
    Q_scaled = Q.copy().astype(np.float64)
    
    if scale_factor != 1.0 or crop_pixels > 0:
        # Scale the focal length and principal point in Q matrix
        # Q[0,3] = -fx * cx, Q[1,3] = -fy * cy, Q[2,3] = fx, Q[3,2] = -fx * baseline
        # We need to scale fx, fy, cx, cy by the scale factor
        
        # Extract and scale focal lengths and principal points
        fx = Q_scaled[2, 3] * scale_factor
        fy = Q_scaled[2, 3] * scale_factor  # Assuming fx = fy for stereo
        cx = Q_scaled[0, 3] / Q_scaled[2, 3] * scale_factor - crop_pixels
        cy = Q_scaled[1, 3] / Q_scaled[2, 3] * scale_factor - crop_pixels
        
        # Update Q matrix with scaled parameters
        Q_scaled[0, 3] = -fx * cx
        Q_scaled[1, 3] = -fy * cy  
        Q_scaled[2, 3] = fx
        # Q[3,2] (baseline) remains unchanged as it's in world units
        
    return (leftMapX, leftMapY, rightMapX, rightMapY, imageSize, Q_scaled)


def disparity_to_depth_opencv(disp: np.ndarray, calib):
    """
    Convert disparity to real-world depth (Z in meters) using calibration reprojection matrix Q.
    Expects Q as the last element in the calibration tuple.
    """
    Q = calib[-1]  # last element, not calib[4]

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, dtype=np.float64)
    else:
        Q = Q.astype(np.float64)

    if Q.shape != (4, 4):
        raise ValueError(f"Invalid Q shape: expected (4,4), got {Q.shape}")

    points_3d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    depth = points_3d[:, :, 2]
    depth[~np.isfinite(depth)] = 0.0
    return depth


def preprocess_images(grayL: np.ndarray, grayR: np.ndarray, s: dict) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Apply downsampling and cropping before disparity computation.
    
    Returns:
        Tuple of (processed_left, processed_right, scale_factor, crop_pixels)
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
    return grayL, grayR, scale, c


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
    Deprecated: GUI tuner removed from API module.
    """
    raise NotImplementedError("Tuner UI is not available in this API module.")



def read_tuner_params() -> dict:
    """
    Deprecated: GUI tuner removed from API module.
    """
    raise NotImplementedError("Tuner UI is not available in this API module.")


# ---------------------------
# (Saving utilities moved to Vision.py)
# ---------------------------


# ---------------------------
# Class-based API
# ---------------------------
class DisparityDepthCapture:
    """
    Class wrapper for stereo disparity and depth computation.

    Usage:
        engine = DisparityDepthCapture(calibration, settings=None, default_profile="CDR")
        result = engine.process(left_bgr, right_bgr)
        engine.save_npz(path, result["depth"], result["num_disp"], result["meta"])
    """

    def __init__(self, calibration, settings: dict | None = None, default_profile: str | None = "CDR") -> None:
        self.calibration = calibration
        self.rect_cache = RectificationCache(calibration)
        self.stereo_cache = StereoSGBMCache()

        # Start from defaults; optionally overlay a hard-coded profile, then overlay caller settings
        base = DEFAULT_SETTINGS.copy()
        if default_profile:
            prof = load_profile(default_profile)
            if prof:
                base.update(prof)
        if settings:
            base.update(settings)
        self.settings = base

        # Visualization colormap state
        self._colormap = "jet"  # jet | bw | bone

    # ---------------------------
    # Public configuration API
    # ---------------------------
    def colormap(self, name: str) -> "DisparityDepthCapture":
        name = name.lower()
        if name not in {"jet", "bw", "bone"}:
            raise ValueError("Unsupported colormap. Use 'bw', 'jet', or 'bone'.")
        self._colormap = name
        return self

    def downsample(self, x: int, y: int) -> "DisparityDepthCapture":
        """
        Set downsample based on target size (x=width, y=height) relative to calibration image size.
        """
        imageSize = self.calibration[4]  # (w, h)
        if not imageSize or len(imageSize) != 2:
            raise ValueError("Calibration must include imageSize=(w,h)")
        w0, h0 = imageSize
        if w0 <= 0 or h0 <= 0 or x <= 0 or y <= 0:
            raise ValueError("Invalid dimensions for downsample().")
        sx = float(x) / float(w0)
        sy = float(y) / float(h0)
        # Keep aspect ratio by using the smaller scale
        scale = max(0.1, min(sx, sy))
        self.settings["downSample"] = int(round(scale * 100))
        return self

    def set_downsample_percent(self, percent: int) -> "DisparityDepthCapture":
        self.settings["downSample"] = int(max(10, min(100, percent)))
        return self

    def crop(self, pixels: int) -> "DisparityDepthCapture":
        self.settings["crop"] = int(max(0, pixels))
        return self

    def set_profile(self, name: str) -> "DisparityDepthCapture":
        prof = load_profile(name)
        if not prof:
            raise ValueError(f"Profile '{name}' not found.")
        self.settings.update(prof)
        return self

    def update_settings(self, overrides: dict) -> "DisparityDepthCapture":
        if not isinstance(overrides, dict):
            raise TypeError("overrides must be a dict")
        self.settings.update(overrides)
        return self

    def get_settings(self) -> dict:
        return self.settings.copy()

    def enable_wls(self, lambda_: float = 4000, sigma: float = 1.0) -> "DisparityDepthCapture":
        self.settings["useWLS"] = 1
        self.settings["wlsLambda"] = float(lambda_)
        self.settings["wlsSigma"] = float(sigma)
        return self

    def disable_wls(self) -> "DisparityDepthCapture":
        self.settings["useWLS"] = 0
        return self

    def enable_bilateral(self, strength: int = 8) -> "DisparityDepthCapture":
        self.settings["useBilateral"] = 1
        self.settings["bilateralStrength"] = int(max(0, strength))
        return self

    def disable_bilateral(self) -> "DisparityDepthCapture":
        self.settings["useBilateral"] = 0
        return self

    def enable_morph(self, iterations: int = 1) -> "DisparityDepthCapture":
        self.settings["useMorph"] = 1
        self.settings["morphIter"] = int(max(1, iterations))
        return self

    def disable_morph(self) -> "DisparityDepthCapture":
        self.settings["useMorph"] = 0
        return self

    def set_near_cutoff(self, value: float) -> "DisparityDepthCapture":
        self.settings["nearCutoff"] = float(max(0.0, value))
        return self

    def set_far_enhance(self, value: int) -> "DisparityDepthCapture":
        self.settings["farEnhance"] = int(max(0, value))
        return self

    # ---------------------------
    # Core processing
    # ---------------------------
    def process(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> dict:
        s = self.settings

        rectL_color, rectR_color = rectify_pair(left_bgr, right_bgr, self.rect_cache)

        grayL = cv2.cvtColor(rectL_color, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR_color, cv2.COLOR_BGR2GRAY)

        grayL, grayR, scale_factor, crop_pixels = preprocess_images(grayL, grayR, s)

        disp, num_disp = compute_disparity_map(grayL, grayR, s, self.stereo_cache)

        disp = post_filter_strong(disp, grayL, s)
        disp = post_filter_weak(disp, s)

        scaled_calib = scale_calibration_for_downsampling(self.calibration, scale_factor, crop_pixels)
        depth = disparity_to_depth_opencv(disp, scaled_calib)

        meta = {
            "scale_factor": float(scale_factor),
            "crop_pixels": int(crop_pixels),
            "settings_snapshot": json.dumps(s)
        }

        return {
            "depth": depth,
            "disp": disp,
            "num_disp": int(num_disp),
            "meta": meta,
        }




# (Application/CLI removed; this module is import-only.)



# No __main__ entry point.

