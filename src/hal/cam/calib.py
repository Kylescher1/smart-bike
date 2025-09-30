# src/hal/cam/calib.py
import os, numpy as np, cv2

def load_calibration(filename=None):
    if filename is None:
        # path of this file: /src/hal/cam/calib.py
        base = os.path.dirname(__file__)
        filename = os.path.join(base, "calibrate", "data", "stereo_calib_fisheye.npz")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Calibration file not found: {filename}")

    data = np.load(filename, allow_pickle=True)
    return (data["leftMapX"], data["leftMapY"],
            data["rightMapX"], data["rightMapY"],
            tuple(data["imageSize"]), data["Q"])

def rectify_pair(left, right, calib):
    leftMapX, leftMapY, rightMapX, rightMapY, _, _ = calib
    rectL = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR)
    return rectL, rectR
