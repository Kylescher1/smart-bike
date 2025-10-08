"""
Minimal disparity output script with interactive tuning.
Adds fine-grained downSample (10–100%) and crop sliders.
Creates/loads disparity_settings.json in smart-bike root folder.
Press 'q' to quit.
"""

from __future__ import annotations
import cv2, json, os, numpy as np
from src.hal.cam.depth import rectify_pair
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Camera import open_stereo_pair

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "../../..", "disparity_settings.json")

DEFAULT_SETTINGS = {
    "numDisparities": 4,
    "blockSize": 16,
    "preFilterCap": 31,
    "uniquenessRatio": 5,
    "speckleWindowSize": 160,
    "speckleRange": 7,
    "disp12MaxDiff": 7,
    "medianBlurK": 0,
    "downSample": 100,  # percentage (10–100)
    "crop": 0           # pixels trimmed from edges
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

def nothing(x): pass

def create_tuner_window(s):
    cv2.namedWindow("Disparity Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Disparity Tuner", 420, 400)
    cv2.createTrackbar("numDisp", "Disparity Tuner", s["numDisparities"], 20, nothing)
    cv2.createTrackbar("blockSize", "Disparity Tuner", s["blockSize"], 21, nothing)
    cv2.createTrackbar("preFilterCap", "Disparity Tuner", s["preFilterCap"], 63, nothing)
    cv2.createTrackbar("uniquenessRatio", "Disparity Tuner", s["uniquenessRatio"], 50, nothing)
    cv2.createTrackbar("speckleRange", "Disparity Tuner", s["speckleRange"], 50, nothing)
    cv2.createTrackbar("medianBlurK", "Disparity Tuner", s["medianBlurK"], 7, nothing)
    cv2.createTrackbar("downSample%", "Disparity Tuner", s["downSample"], 100, nothing)
    cv2.createTrackbar("crop(px)", "Disparity Tuner", s["crop"], 200, nothing)

def read_trackbar():
    s = {
        "numDisparities": max(1, cv2.getTrackbarPos("numDisp", "Disparity Tuner")),
        "blockSize": max(3, cv2.getTrackbarPos("blockSize", "Disparity Tuner") | 1),
        "preFilterCap": cv2.getTrackbarPos("preFilterCap", "Disparity Tuner"),
        "uniquenessRatio": cv2.getTrackbarPos("uniquenessRatio", "Disparity Tuner"),
        "speckleRange": cv2.getTrackbarPos("speckleRange", "Disparity Tuner"),
        "speckleWindowSize": 160,
        "disp12MaxDiff": 7,
        "medianBlurK": cv2.getTrackbarPos("medianBlurK", "Disparity Tuner"),
        "downSample": max(10, cv2.getTrackbarPos("downSample%", "Disparity Tuner")),  # enforce ≥10%
        "crop": cv2.getTrackbarPos("crop(px)", "Disparity Tuner")
    }
    if s["medianBlurK"] % 2 == 0:
        s["medianBlurK"] = max(0, s["medianBlurK"] - 1)
    return s

def compute_disparity_map(gray_left, gray_right, settings):
    num_disp = 16 * settings["numDisparities"]
    blk = settings["blockSize"]
    P1, P2 = 8 * blk * blk, 32 * blk * blk
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=blk,
        P1=P1, P2=P2,
        preFilterCap=settings["preFilterCap"],
        uniquenessRatio=settings["uniquenessRatio"],
        speckleWindowSize=settings["speckleWindowSize"],
        speckleRange=settings["speckleRange"],
        disp12MaxDiff=settings["disp12MaxDiff"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    k = settings["medianBlurK"]
    if k >= 3:
        disp = cv2.medianBlur(disp, k)
    return disp, num_disp

def visualize_disparity(disp, num_disp):
    disp_vis = np.clip(disp, 0, num_disp)
    norm = (disp_vis / float(max(1, num_disp)) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_BONE)

def preprocess_images(grayL, grayR, s):
    # fine-grained downsample (10–100%)
    scale = max(0.1, s["downSample"] / 100.0)
    if scale < 0.999:
        grayL = cv2.resize(grayL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grayR = cv2.resize(grayR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # crop edges
    c = s["crop"]
    if c > 0:
        h, w = grayL.shape[:2]
        grayL = grayL[c:h - c, c:w - c]
        grayR = grayR[c:h - c, c:w - c]
    return grayL, grayR

def main():
    calib = load_calibration()
    left_cam, right_cam = open_stereo_pair()
    s = load_settings()
    create_tuner_window(s)

    try:
        while True:
            left = left_cam.read_frame()
            right = right_cam.read_frame()
            if left is None or right is None:
                continue

            rectL, rectR = rectify_pair(left, right, calib)
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

            s = read_trackbar()
            grayL, grayR = preprocess_images(grayL, grayR, s)
            disp, num_disp = compute_disparity_map(grayL, grayR, s)
            vis = visualize_disparity(disp, num_disp)

            cv2.putText(vis, f"Scale={s['downSample']}%  Crop={s['crop']}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Disparity (color)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                save_settings(s)
                break
    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
