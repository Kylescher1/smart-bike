# src/hal/cam/depth_tuner.py
import cv2
import json
import os
import numpy as np
from src.hal.cam.depth import load_calibration, rectify_pair
from src.hal.cam.Camera import Camera

SETTINGS_FILE = "stereo_settings.json"

# Default parameters
DEFAULTS = {
    "numDisparities": 6,  # multiplier of 16
    "blockSize": 5,
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return DEFAULTS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

def nothing(x): pass

def create_tuner_window(settings):
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuner", 500, 400)
    cv2.createTrackbar("numDisparities", "Tuner", settings["numDisparities"], 20, nothing)
    cv2.createTrackbar("blockSize", "Tuner", settings["blockSize"], 21, nothing)
    cv2.createTrackbar("preFilterCap", "Tuner", settings["preFilterCap"], 63, nothing)
    cv2.createTrackbar("uniquenessRatio", "Tuner", settings["uniquenessRatio"], 50, nothing)
    cv2.createTrackbar("speckleWindowSize", "Tuner", settings["speckleWindowSize"], 200, nothing)
    cv2.createTrackbar("speckleRange", "Tuner", settings["speckleRange"], 50, nothing)
    cv2.createTrackbar("disp12MaxDiff", "Tuner", settings["disp12MaxDiff"], 25, nothing)

def get_settings_from_trackbar():
    vals = {
        "numDisparities": cv2.getTrackbarPos("numDisparities", "Tuner"),
        "blockSize": cv2.getTrackbarPos("blockSize", "Tuner"),
        "preFilterCap": cv2.getTrackbarPos("preFilterCap", "Tuner"),
        "uniquenessRatio": cv2.getTrackbarPos("uniquenessRatio", "Tuner"),
        "speckleWindowSize": cv2.getTrackbarPos("speckleWindowSize", "Tuner"),
        "speckleRange": cv2.getTrackbarPos("speckleRange", "Tuner"),
        "disp12MaxDiff": cv2.getTrackbarPos("disp12MaxDiff", "Tuner"),
    }
    # Ensure constraints
    vals["numDisparities"] = max(1, vals["numDisparities"])
    vals["blockSize"] = max(3, vals["blockSize"] | 1)  # must be odd and >=3
    return vals

def main():
    calib = load_calibration()

    left_cam = Camera(index=1, backend=cv2.CAP_ANY, width=640, height=480, fps=30)
    right_cam = Camera(index=3, backend=cv2.CAP_ANY, width=640, height=480, fps=30)
    left_cam.open(); right_cam.open()

    settings = load_settings()
    create_tuner_window(settings)

    try:
        while True:
            frameL = left_cam.read_frame()
            frameR = right_cam.read_frame()
            if frameL is None or frameR is None:
                continue

            rectL, rectR = rectify_pair(frameL, frameR, calib)
            params = get_settings_from_trackbar()

            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16 * params["numDisparities"],
                blockSize=params["blockSize"],
                P1=8 * 3 * params["blockSize"] ** 2,
                P2=32 * 3 * params["blockSize"] ** 2,
                preFilterCap=params["preFilterCap"],
                uniquenessRatio=params["uniquenessRatio"],
                speckleWindowSize=params["speckleWindowSize"],
                speckleRange=params["speckleRange"],
                disp12MaxDiff=params["disp12MaxDiff"],
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            disparity = stereo.compute(rectL, rectR).astype(np.float32) / 16.0
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)

            cv2.imshow("Depth Map", disp_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                save_settings(params)
                break
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
