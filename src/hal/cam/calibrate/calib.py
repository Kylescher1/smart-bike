import os
import numpy as np
import cv2

def load_calibration(filename=None):
    if filename is None:
        # this file is /home/radxa/smart-bike/src/hal/cam/calibrate/calib.py
        base = os.path.dirname(__file__)
        filename = os.path.join(base, "data", "stereo_calib.npz")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Calibration file not found: {filename}")

    data = np.load(filename, allow_pickle=True)
    return (
        data["leftMapX"],
        data["leftMapY"],
        data["rightMapX"],
        data["rightMapY"],
        tuple(data["imageSize"]),
        data["Q"],
    )
