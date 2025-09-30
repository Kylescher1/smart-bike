# src/hal/cam/depth.py
import cv2
import numpy as np
import os, json
from typing import Tuple

SETTINGS_FILE = "stereo_settings.json"

# Defaults if no tuner file
DEFAULTS = {
    "numDisparities": 6,  # multiplier of 16
    "blockSize": 5,
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1
}

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

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return DEFAULTS.copy()

def rectify_pair(left_frame, right_frame, calib: dict) -> Tuple[np.ndarray, np.ndarray]:
    rectL = cv2.remap(left_frame, calib["leftMapX"], calib["leftMapY"], cv2.INTER_LINEAR)
    rectR = cv2.remap(right_frame, calib["rightMapX"], calib["rightMapY"], cv2.INTER_LINEAR)
    return rectL, rectR

def compute_depth_map(left_frame, right_frame, calib: dict):
    rectL, rectR = rectify_pair(left_frame, right_frame, calib)
    settings = load_settings()

    numDisparities = 16 * max(1, settings["numDisparities"])
    blockSize = max(3, settings["blockSize"] | 1)  # must be odd and >=3

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * blockSize ** 2,
        P2=32 * 3 * blockSize ** 2,
        preFilterCap=settings["preFilterCap"],
        uniquenessRatio=settings["uniquenessRatio"],
        speckleWindowSize=settings["speckleWindowSize"],
        speckleRange=settings["speckleRange"],
        disp12MaxDiff=settings["disp12MaxDiff"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(rectL, rectR).astype(np.float32) / 16.0
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(disp_vis)
