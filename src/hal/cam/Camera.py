"""Camera I/O primitives for the Smart Bike vision pipeline.

This module keeps the camera interaction surface very small and predictable:
 - ``open_camera``/``close`` wrap ``cv2.VideoCapture`` lifecycle handling.
 - ``grab_frame`` grabs the next frame without decoding so that stereo pairs can
   be synchronised using ``Camera.grab_frame`` on both cameras before calling
   ``retrieve_frame``.
 - ``retrieve_frame`` decodes the last grabbed frame.
 - ``save_frame`` persists a BGR frame to disk.

The goal is to keep the low level camera contract crystal clear for the higher
level pipeline components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import copy

# Centralised configuration defaults – callers may override when instantiating
# a ``Camera``.
DEFAULT_CAMERA_CONFIG: Dict[str, int | str] = {
    "backend": cv2.CAP_V4L2,
    "width": 1024,
    "height": 768,
    "fps": 90,
    "fourcc": "MJPG",
}


def _ensure_open(cap: Optional[cv2.VideoCapture]) -> cv2.VideoCapture:
    if cap is None or not cap.isOpened():
        raise RuntimeError("Camera handle is not open. Call open_camera() first.")
    return cap


@dataclass
class Camera:
    """Thin wrapper around cv2.VideoCapture with explicit lifecycle calls."""

    index: int
    config: Dict[str, int | str] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CAMERA_CONFIG))
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def open_camera(self) -> None:
        """Open the camera with the configured backend and stream settings."""
        if self.cap and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.index, self.config["backend"])
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.index}.")

        # Apply stream configuration
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(self.config["fourcc"])) )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config["width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config["height"]))
        self.cap.set(cv2.CAP_PROP_FPS, int(self.config["fps"]))

    def close_camera(self) -> None:
        """Release the underlying ``cv2.VideoCapture`` handle."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    # ------------------------------------------------------------------
    # Frame acquisition helpers
    # ------------------------------------------------------------------
    def grab_frame(self) -> bool:
        """Grab the next frame without decoding it."""
        cap = _ensure_open(self.cap)
        return bool(cap.grab())

    def retrieve_frame(self) -> Optional[np.ndarray]:
        """Retrieve the frame that was previously grabbed."""
        cap = _ensure_open(self.cap)
        success, frame = cap.retrieve()
        return frame if success else None

    def get_frame(self) -> Optional[np.ndarray]:
        """Convenience helper that performs a grab followed by retrieve."""
        if not self.grab_frame():
            return None
        return self.retrieve_frame()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def save_frame(frame: np.ndarray, path: str | Path) -> Path:
        """Persist a BGR frame to ``path`` and return the ``Path`` instance."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise IOError(f"Failed to write frame to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def is_open(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def close(self) -> None:  # Backwards compatibility alias
        self.close_camera()


def open_stereo_pair(
    left_index: int = 1,
    right_index: int = 3,
    config: Dict[str, int | str] = DEFAULT_CAMERA_CONFIG,
) -> Tuple[Camera, Camera]:
    """Open a left/right stereo pair and return the camera handles."""

    left = Camera(index=left_index, config=config, name="left")
    right = Camera(index=right_index, config=config, name="right")
    left.open_camera()
    right.open_camera()
    return left, right
