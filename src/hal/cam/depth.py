# src/hal/cam/depth.py
import cv2
import numpy as np
from typing import Tuple

def load_calibration(filename: str = "src/hal/cam/calibrate/data/stereo_calib_fisheye.npz"):
    data = np.load(filename, allow_pickle=True)
    return {
        "leftMapX": data["leftMapX"],
        "leftMapY": data["leftMapY"],
        "rightMapX": data["rightMapX"],
        "rightMapY": data["rightMapY"],
        "imageSize": tuple(data["imageSize"]),
        "Q": data["Q"]
    }

def rectify_pair(left_frame, right_frame, calib: dict) -> Tuple[np.ndarray, np.ndarray]:
    rectL = cv2.remap(left_frame, calib["leftMapX"], calib["leftMapY"], cv2.INTER_LINEAR)
    rectR = cv2.remap(right_frame, calib["rightMapX"], calib["rightMapY"], cv2.INTER_LINEAR)
    return rectL, rectR

def compute_depth_map(left_frame, right_frame, calib: dict):
    rectL, rectR = rectify_pair(left_frame, right_frame, calib)

    # Stereo matcher
    numDisparities = 16 * 6  # multiple of 16
    blockSize = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * blockSize ** 2,
        P2=32 * 3 * blockSize ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(rectL, rectR).astype(np.float32) / 16.0

    # Normalize for display
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    return disp_vis
