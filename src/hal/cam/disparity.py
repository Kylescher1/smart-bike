"""
Minimal disparity output script.

This file computes and displays the raw disparity map from the stereo pair.
It is intentionally small and free of tuning UI or depth reprojecting.

Usage: run the module. Press 'q' to quit.
"""

from __future__ import annotations

import cv2
import numpy as np
from src.hal.cam.depth import rectify_pair
from src.hal.cam.calib import load_calibration
from src.hal.cam.Camera import open_stereo_pair


DEFAULT_SETTINGS = {
    "numDisparities": 6,  # multiplied by 16 internally
    "blockSize": 5,
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1,
    "medianBlurK": 0,
}


def compute_disparity_map(gray_left: np.ndarray, gray_right: np.ndarray, settings: dict | None = None) -> tuple[np.ndarray, int]:
    """Compute raw disparity map (float32) and return (disparity, num_disparities).

    The disparity values are returned in pixels (float32). Invalid/negative disparities
    are left as-is (can be <= 0). For visualization use visualize_disparity.
    """
    s = (DEFAULT_SETTINGS.copy() if settings is None else {**DEFAULT_SETTINGS, **settings})
    num_disp = 16 * max(1, int(s["numDisparities"]))
    block = max(3, int(s["blockSize"]) | 1)
    cn = 1
    P1 = 8 * cn * block * block
    P2 = 32 * cn * block * block

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1,
        P2=P2,
        preFilterCap=int(s["preFilterCap"]),
        uniquenessRatio=int(s["uniquenessRatio"]),
        speckleWindowSize=int(s["speckleWindowSize"]),
        speckleRange=int(s["speckleRange"]),
        disp12MaxDiff=int(s["disp12MaxDiff"]),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    if s.get("medianBlurK", 0) >= 3:
        k = int(s["medianBlurK"])
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            disp = cv2.medianBlur(disp, k)

    return disp, num_disp


def visualize_disparity(disp: np.ndarray, num_disp: int) -> np.ndarray:
    """Return an 8-bit colorized visualization of disparity for display."""
    # Normalize ignoring negative/invalid disparities
    disp_vis = disp.copy()
    disp_vis[disp_vis < 0] = 0
    vis = np.clip(disp_vis / float(max(1, num_disp)) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return color


def main() -> None:
    calib = load_calibration()
    left_cam, right_cam = open_stereo_pair()

    try:
        while True:
            left = left_cam.read_frame()
            right = right_cam.read_frame()
            if left is None or right is None:
                # camera may be warming up; skip until frames available
                continue

            rectL, rectR = rectify_pair(left, right, calib)
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY) if rectL.ndim == 3 else rectL
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY) if rectR.ndim == 3 else rectR

            disp, num_disp = compute_disparity_map(grayL, grayR)

            vis = visualize_disparity(disp, num_disp)
            cv2.imshow("Disparity (color)", vis)

            # For some consumers it may be useful to see raw disparity as text (optional)
            # cv2.imshow("Disparity (raw)", (disp / num_disp * 255).astype(np.uint8))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
