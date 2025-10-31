"""High level vision orchestrator for the Smart Bike pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .cam.Camera import Camera, DEFAULT_CAMERA_CONFIG
from .cam.depth_processor import DepthProcessor, DepthResult
from .cam.stereo_capture import capture_stereo_pair
from .cam.calibrate.calib import load_calibration
from .cam.depth_profile import (
    load_profile,
    load_settings,
    StereoSGBMCache,
    RectificationCache,
    rectify_pair,
    preprocess_images,
    compute_disparity_map,
    post_filter_strong,
    post_filter_weak,
    scale_calibration_for_downsampling,
    disparity_to_depth_opencv,
)
from .config import LEFT_INDEX, RIGHT_INDEX, SWAP_LR, PROFILE_NAME


def default_calibration_file() -> Path:
    return Path(__file__).resolve().parent / "cam" / "calibrate" / "data" / "stereo_calib.npz"


def _default_camera_config() -> dict:
    return DEFAULT_CAMERA_CONFIG.copy()


def _default_output_dir() -> Path:
    return Path("./data/depth_maps")


@dataclass
class VisionSystem:
    """Owns the stereo cameras and depth pipeline."""

    calibration_file: Path = field(default_factory=default_calibration_file)
    output_dir: Path = field(default_factory=_default_output_dir)
    camera_config: dict = field(default_factory=_default_camera_config)
    left_index: int = LEFT_INDEX
    right_index: int = RIGHT_INDEX
    object_distance_threshold_mm: float = 1500.0
    profile_name: str = PROFILE_NAME
    _profile_params: Optional[dict] = field(default=None, repr=False)
    _stereo_cache: Optional[StereoSGBMCache] = field(default=None, repr=False)
    _rect_cache: Optional[RectificationCache] = field(default=None, repr=False)
    _calib_tuple: Optional[tuple] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.left_camera = Camera(self.left_index, self.camera_config, name="left")
        self.right_camera = Camera(self.right_index, self.camera_config, name="right")
        self.depth_processor: Optional[DepthProcessor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        self.left_camera.open_camera()
        self.right_camera.open_camera()
        self.depth_processor = DepthProcessor(
            calibration_file=self.calibration_file, output_dir=self.output_dir
        )
        calib = load_calibration(str(self.calibration_file))
        self._calib_tuple = calib
        self._rect_cache = RectificationCache(calib)
        self._stereo_cache = StereoSGBMCache()
        params = load_profile(self.profile_name)
        if params is None:
            params = load_settings()
        self._profile_params = params

    def close(self) -> None:
        self.left_camera.close_camera()
        self.right_camera.close_camera()
        self.depth_processor = None

    # ------------------------------------------------------------------
    # Capture utilities
    # ------------------------------------------------------------------
    def capture_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not (self.left_camera.grab_frame() and self.right_camera.grab_frame()):
            return None
        left_frame = self.left_camera.retrieve_frame()
        right_frame = self.right_camera.retrieve_frame()
        if left_frame is None or right_frame is None:
            return None
        return left_frame, right_frame

    def capture_and_save(self, output_dir: str | Path) -> Tuple[Path, Path]:
        return capture_stereo_pair(self.left_camera, self.right_camera, output_dir)

    # ------------------------------------------------------------------
    # Processing utilities
    # ------------------------------------------------------------------
    def compute_depth(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        if (
            self._rect_cache is None
            or self._stereo_cache is None
            or self._profile_params is None
            or self._calib_tuple is None
        ):
            raise RuntimeError("VisionSystem must be opened before computing depth.")

        if SWAP_LR:
            left, right = right, left
        rectL_color, rectR_color = rectify_pair(left, right, self._rect_cache)
        grayL = cv2.cvtColor(rectL_color, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR_color, cv2.COLOR_BGR2GRAY)

        params = self._profile_params
        grayL_p, grayR_p, scale_factor, crop_pixels = preprocess_images(grayL, grayR, params)
        disp, num_disp = compute_disparity_map(grayL_p, grayR_p, params, self._stereo_cache)
        disp = post_filter_strong(disp, grayL_p, params)
        disp = post_filter_weak(disp, params)

        scaled_calib = scale_calibration_for_downsampling(self._calib_tuple, scale_factor, crop_pixels)
        depth = disparity_to_depth_opencv(disp, scaled_calib)

        metadata = {"profile": self.profile_name, "num_disparities": int(num_disp)}
        return DepthResult(depth, disp, int(num_disp), None, metadata)

    @staticmethod
    def edge_map_from_depth(depth_map: np.ndarray) -> np.ndarray:
        if depth_map.size == 0:
            return np.zeros_like(depth_map, dtype=np.uint8)
        depth = depth_map.astype(np.float32)
        if not np.any(depth):
            return np.zeros_like(depth_map, dtype=np.uint8)
        norm = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        return cv2.Canny(norm, 75, 150)

    def is_object_close(self, depth_map: np.ndarray) -> bool:
        if depth_map.size == 0:
            return False
        valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
        if valid.size == 0:
            return False
        return float(np.min(valid)) <= self.object_distance_threshold_mm

    def warn_rider(self) -> None:
        print("⚠️  Object detected close to the rider!", flush=True)


__all__ = ["VisionSystem", "default_calibration_file"]
