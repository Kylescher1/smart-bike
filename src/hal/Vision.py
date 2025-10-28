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
    left_index: int = 1
    right_index: int = 3
    object_distance_threshold_mm: float = 1500.0

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
        if self.depth_processor is None:
            raise RuntimeError("VisionSystem must be opened before computing depth.")
        return self.depth_processor.process(left, right)

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
