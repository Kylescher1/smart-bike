"""Stereo calibration helpers.

The module consumes stereo image pairs captured with ``stereo_capture`` and
produces the calibration artefacts consumed by :mod:`depth_processor`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

CHECKERBOARD_DEFAULT = (7, 10)  # (columns, rows) inner corners
SQUARE_SIZE_MM = 20.0


class CalibrationError(RuntimeError):
    pass


def _prepare_object_points(checkerboard: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = checkerboard
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def _collect_image_points(
    left_images: Sequence[Path],
    right_images: Sequence[Path],
    checkerboard: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    objpoints: List[np.ndarray] = []
    imgpoints_left: List[np.ndarray] = []
    imgpoints_right: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    obj_template = _prepare_object_points(checkerboard, 1.0)

    for left_path, right_path in zip(left_images, right_images):
        img_left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if img_left is None or img_right is None:
            print(f"⚠️ Unable to read pair: {left_path.name} / {right_path.name}")
            continue

        if img_left.shape != img_right.shape:
            print("⚠️ Skipping pair with mismatched resolution")
            continue

        ret_left, corners_left = cv2.findChessboardCorners(img_left, checkerboard)
        ret_right, corners_right = cv2.findChessboardCorners(img_right, checkerboard)
        if not (ret_left and ret_right):
            print(f"⚠️ Chessboard not found in pair {left_path.name} / {right_path.name}")
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)

        objpoints.append(obj_template.copy())
        imgpoints_left.append(corners_left.reshape(-1, 2))
        imgpoints_right.append(corners_right.reshape(-1, 2))
        image_size = (img_left.shape[1], img_left.shape[0])

    if not objpoints or image_size is None:
        raise CalibrationError("No valid stereo pairs found for calibration.")

    return objpoints, imgpoints_left, imgpoints_right, image_size


def stereo_calibrate(
    pairs_dir: str | Path,
    checkerboard: Tuple[int, int] = CHECKERBOARD_DEFAULT,
    square_size: float = SQUARE_SIZE_MM,
    output_dir: str | Path | None = None,
) -> Path:
    """Calibrate the stereo rig and persist the rectification maps."""
    pairs_path = Path(pairs_dir)
    if not pairs_path.exists():
        raise CalibrationError(f"Pairs directory does not exist: {pairs_path}")

    left_images = sorted(pairs_path.glob("*_left.*"))
    right_images = sorted(pairs_path.glob("*_right.*"))
    if not left_images or not right_images:
        raise CalibrationError("No stereo pairs found in directory.")

    objpoints, imgpoints_left, imgpoints_right, image_size = _collect_image_points(
        left_images, right_images, checkerboard
    )

    objpoints = [op * square_size for op in objpoints]
    objpoints_np = [op.reshape(-1, 1, 3).astype(np.float32) for op in objpoints]
    imgpoints_left_np = [ip.reshape(-1, 1, 2).astype(np.float32) for ip in imgpoints_left]
    imgpoints_right_np = [ip.reshape(-1, 1, 2).astype(np.float32) for ip in imgpoints_right]

    # Individual camera calibration
    _, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
        objpoints_np, imgpoints_left_np, image_size, None, None
    )
    _, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
        objpoints_np, imgpoints_right_np, image_size, None, None
    )

    # Stereo calibration
    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC
    (_, _, _, _, _, R, T, E, F) = cv2.stereoCalibrate(
        objpoints_np,
        imgpoints_left_np,
        imgpoints_right_np,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        image_size,
        criteria=stereocalib_criteria,
        flags=flags,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, image_size, R, T, alpha=0
    )

    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, image_size, cv2.CV_32FC1
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, image_size, cv2.CV_32FC1
    )

    # Persist calibration artefacts
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "calibrate" / "data"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calib_file = output_path / "stereo_calib.npz"

    np.savez_compressed(
        calib_file,
        imageSize=np.array(image_size, dtype=np.int32),
        K1=mtx_left,
        D1=dist_left,
        K2=mtx_right,
        D2=dist_right,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        leftMapX=left_map_x,
        leftMapY=left_map_y,
        rightMapX=right_map_x,
        rightMapY=right_map_y,
    )

    return calib_file


__all__ = ["stereo_calibrate", "CalibrationError"]
