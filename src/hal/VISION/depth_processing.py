import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple


class DepthProcessor:
    """Encapsulates stereo rectification, disparity, and depth conversion."""

    def __init__(
        self,
        left_config: Dict,
        right_config: Dict,
        stereo_matcher: Optional[cv2.StereoSGBM] = None,
        q_matrix: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> None:
        self.left_config = left_config
        self.right_config = right_config
        self.stereo_matcher = stereo_matcher
        self.q_matrix = q_matrix
        self.debug = debug

    def update_matcher(self, stereo_matcher: Optional[cv2.StereoSGBM]) -> None:
        self.stereo_matcher = stereo_matcher

    def update_q_matrix(self, q_matrix: Optional[np.ndarray]) -> None:
        self.q_matrix = q_matrix

    def rectify(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_map_x = self.left_config.get("map_x")
        left_map_y = self.left_config.get("map_y")
        right_map_x = self.right_config.get("map_x")
        right_map_y = self.right_config.get("map_y")

        missing_maps = [
            name
            for name, value in (
                ("left.map_x", left_map_x),
                ("left.map_y", left_map_y),
                ("right.map_x", right_map_x),
                ("right.map_y", right_map_y),
            )
            if value is None
        ]

        if missing_maps:
            raise RuntimeError(f"Calibration maps missing: {', '.join(missing_maps)}")

        rect_left = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def compute_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        if self.stereo_matcher is None:
            raise RuntimeError("Stereo matcher not initialized. Call start() first.")

        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity < 0] = 0
        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        if self.q_matrix is None:
            raise RuntimeError("Calibration Q matrix not loaded. Call start() first.")

        points = cv2.reprojectImageTo3D(disparity, self.q_matrix)
        depth = points[:, :, 2]
        depth[~np.isfinite(depth)] = 0
        depth = np.maximum(depth, 0)
        return depth

    def process_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        rect_left, rect_right = self.rectify(left_frame, right_frame)
        disparity = self.compute_disparity(rect_left, rect_right)
        depth_map = self.disparity_to_depth(disparity)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_disparities": int(self.stereo_matcher.getNumDisparities()) if self.stereo_matcher else 0,
        }

        return depth_map, metadata

