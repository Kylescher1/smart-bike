"""Stereo capture utilities.

This module coordinates a left/right ``Camera`` pair so that captures are
performed in lockstep. The resulting frames are persisted to disk making it
suitable for building calibration image sets.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2

from .Camera import Camera


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def capture_stereo_pair(
    left: Camera,
    right: Camera,
    output_dir: str | Path,
    prefix: str | None = None,
    ext: str = "png",
) -> Tuple[Path, Path]:
    """Capture a synchronised stereo pair and save them to ``output_dir``.

    Returns the file paths of the saved left/right frames.
    """
    if not left.is_open() or not right.is_open():
        raise RuntimeError("Both cameras must be opened before capturing.")

    if not left.grab_frame() or not right.grab_frame():
        raise RuntimeError("Failed to grab frames from stereo pair.")

    frame_left = left.retrieve_frame()
    frame_right = right.retrieve_frame()
    if frame_left is None or frame_right is None:
        raise RuntimeError("Failed to retrieve frames from stereo pair.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = prefix or _timestamp()
    left_path = output_path / f"{prefix}_left.{ext}"
    right_path = output_path / f"{prefix}_right.{ext}"

    if not cv2.imwrite(str(left_path), frame_left):
        raise IOError(f"Failed to save left frame to {left_path}")
    if not cv2.imwrite(str(right_path), frame_right):
        raise IOError(f"Failed to save right frame to {right_path}")

    return left_path, right_path
