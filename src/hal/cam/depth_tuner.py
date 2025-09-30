# src/hal/cam/depth_tuner.py
import cv2, json, os, numpy as np
from src.hal.cam.depth import load_calibration, rectify_pair
from src.hal.cam.Camera import open_stereo_pair

SETTINGS_FILE = "stereo_settings.json"
DEFAULTS = {
    "numDisparities": 6,  # -> used as 16 * value
    "blockSize": 5,       # odd, >=3
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1
}

def load_settings():
    return json.load(open(SETTINGS_FILE)) if os.path.exists(SETTINGS_FILE) else DEFAULTS.copy()

def save_settings(s): json.dump(s, open(SETTINGS_FILE, "w"), indent=2)
def nothing(x): pass

def create_tuner_window(s):
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL); cv2.resizeWindow("Tuner", 500, 400)
    cv2.createTrackbar("numDisparities", "Tuner", s["numDisparities"], 20, nothing)
    cv2.createTrackbar("blockSize", "Tuner", s["blockSize"], 21, nothing)
    cv2.createTrackbar("preFilterCap", "Tuner", s["preFilterCap"], 63, nothing)
    cv2.createTrackbar("uniquenessRatio", "Tuner", s["uniquenessRatio"], 50, nothing)
    cv2.createTrackbar("speckleWindowSize", "Tuner", s["speckleWindowSize"], 200, nothing)
    cv2.createTrackbar("speckleRange", "Tuner", s["speckleRange"], 50, nothing)
    cv2.createTrackbar("disp12MaxDiff", "Tuner", s["disp12MaxDiff"], 25, nothing)

def read_trackbar():
    v = {
        "numDisparities": cv2.getTrackbarPos("numDisparities", "Tuner"),
        "blockSize":      cv2.getTrackbarPos("blockSize", "Tuner"),
        "preFilterCap":   cv2.getTrackbarPos("preFilterCap", "Tuner"),
        "uniquenessRatio":cv2.getTrackbarPos("uniquenessRatio", "Tuner"),
        "speckleWindowSize": cv2.getTrackbarPos("speckleWindowSize", "Tuner"),
        "speckleRange":   cv2.getTrackbarPos("speckleRange", "Tuner"),
        "disp12MaxDiff":  cv2.getTrackbarPos("disp12MaxDiff", "Tuner"),
    }
    # enforce constraints
    v["numDisparities"] = max(1, v["numDisparities"])
    v["blockSize"] = max(3, v["blockSize"] | 1)  # odd and >=3
    return v

def main():
    calib = load_calibration()
    left_cam, right_cam = open_stereo_pair()  # already opened
    settings = load_settings()
    create_tuner_window(settings)

    try:
        while True:
            L = left_cam.read_frame(); R = right_cam.read_frame()
            if L is None or R is None: continue
            rectL, rectR = rectify_pair(L, R, calib)

            # Use grayscale for SGBM
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY) if rectL.ndim == 3 else rectL
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY) if rectR.ndim == 3 else rectR

            p = read_trackbar()
            numDisp = 16 * p["numDisparities"]
            blk = p["blockSize"]
            cn = 1
            P1 = 8 * cn * blk * blk
            P2 = 32 * cn * blk * blk

            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=numDisp,
                blockSize=blk,
                P1=P1, P2=P2,
                preFilterCap=p["preFilterCap"],
                uniquenessRatio=p["uniquenessRatio"],
                speckleWindowSize=p["speckleWindowSize"],
                speckleRange=p["speckleRange"],
                disp12MaxDiff=p["disp12MaxDiff"],
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

            disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.putText(vis, f"numDisp={numDisp} block={blk}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Depth Map", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                save_settings(p)
                break
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
