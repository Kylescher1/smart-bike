"""Stereo capture utilities.

This module coordinates a left/right ``Camera`` pair so that captures are
performed in lockstep. The resulting frames are persisted to disk making it
suitable for building calibration image sets.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple
import argparse
import shutil
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
    """Capture a synchronised stereo pair and save them to ``output_dir``."""
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


if __name__ == "__main__":
    from src.hal.cam.Camera import open_stereo_pair
    from src.hal.cam.stereo_capture import capture_stereo_pair

    parser = argparse.ArgumentParser(description="Stereo capture tool")
    parser.add_argument(
        "-rmp",
        action="store_true",
        help="Remove all existing images in the stereo_pairs folder before starting",
    )
    args = parser.parse_args()

    output_dir = Path("src/hal/cam/calibrate/data/stereo_pairs")

    # Remove previous images if requested
    if args.rmp and output_dir.exists():
        print(f"🧹 Removing all files in {output_dir.resolve()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open both cameras
    left, right = open_stereo_pair()

    try:
        cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right", cv2.WINDOW_NORMAL)

        while True:
            lf = left.get_frame()
            rf = right.get_frame()
            if lf is None or rf is None:
                continue

            cv2.imshow("Left", lf)
            cv2.imshow("Right", rf)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                capture_stereo_pair(left, right, output_dir)
                print(f"Saved stereo pair to {output_dir.resolve()}")
            elif key == ord("q"):
                break
    finally:
        left.close()
        right.close()
        cv2.destroyAllWindows()
