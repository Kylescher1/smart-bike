# src/hal/cam/depth_tuner.py
import cv2, json, os, numpy as np
from src.hal.cam.depth import load_calibration, rectify_pair
from src.hal.cam.Camera import open_stereo_pair

SETTINGS_FILE = "stereo_settings.json"

DEFAULTS = {
    "numDisparities": 6,
    "blockSize": 5,
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1,
    "medianBlurK": 0,
    "alpha": 40,
    "minDistance": 0,
    "maxDepth": 20,
    "farEnhance": 50    # new: emphasize far-field contrast
}

def load_settings():
    return json.load(open(SETTINGS_FILE)) if os.path.exists(SETTINGS_FILE) else DEFAULTS.copy()

def save_settings(s): json.dump(s, open(SETTINGS_FILE, "w"), indent=2)
def nothing(x): pass

def create_tuner_window(s):
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL); cv2.resizeWindow("Tuner", 500, 650)
    cv2.createTrackbar("numDisparities", "Tuner", s["numDisparities"], 20, nothing)
    cv2.createTrackbar("blockSize", "Tuner", s["blockSize"], 21, nothing)
    cv2.createTrackbar("preFilterCap", "Tuner", s["preFilterCap"], 63, nothing)
    cv2.createTrackbar("uniquenessRatio", "Tuner", s["uniquenessRatio"], 50, nothing)
    cv2.createTrackbar("speckleWindowSize", "Tuner", s["speckleWindowSize"], 200, nothing)
    cv2.createTrackbar("speckleRange", "Tuner", s["speckleRange"], 50, nothing)
    cv2.createTrackbar("disp12MaxDiff", "Tuner", s["disp12MaxDiff"], 25, nothing)
    cv2.createTrackbar("medianBlurK", "Tuner", s.get("medianBlurK", 0), 7, nothing)
    cv2.createTrackbar("alpha", "Tuner", s.get("alpha", 40), 100, nothing)
    cv2.createTrackbar("minDistance", "Tuner", s.get("minDistance", 0), 50, nothing)
    cv2.createTrackbar("maxDepth", "Tuner", s.get("maxDepth", 20), 50, nothing)
    cv2.createTrackbar("farEnhance", "Tuner", s.get("farEnhance", 50), 100, nothing)

def read_trackbar():
    v = {
        "numDisparities": cv2.getTrackbarPos("numDisparities", "Tuner"),
        "blockSize":      cv2.getTrackbarPos("blockSize", "Tuner"),
        "preFilterCap":   cv2.getTrackbarPos("preFilterCap", "Tuner"),
        "uniquenessRatio":cv2.getTrackbarPos("uniquenessRatio", "Tuner"),
        "speckleWindowSize": cv2.getTrackbarPos("speckleWindowSize", "Tuner"),
        "speckleRange":   cv2.getTrackbarPos("speckleRange", "Tuner"),
        "disp12MaxDiff":  cv2.getTrackbarPos("disp12MaxDiff", "Tuner"),
        "medianBlurK":    cv2.getTrackbarPos("medianBlurK", "Tuner"),
        "alpha":          cv2.getTrackbarPos("alpha", "Tuner"),
        "minDistance":    cv2.getTrackbarPos("minDistance", "Tuner"),
        "maxDepth":       max(1, cv2.getTrackbarPos("maxDepth", "Tuner")),
        "farEnhance":     cv2.getTrackbarPos("farEnhance", "Tuner")
    }
    v["numDisparities"] = max(1, v["numDisparities"])
    v["blockSize"] = max(3, v["blockSize"] | 1)
    if v["medianBlurK"] % 2 == 0:
        v["medianBlurK"] = max(0, v["medianBlurK"]-1)
    return v

def compute_disparity(grayL, grayR, p):
    numDisp = 16 * p["numDisparities"]
    blk = p["blockSize"]
    P1, P2 = 8*blk*blk, 32*blk*blk
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=numDisp, blockSize=blk,
        P1=P1, P2=P2,
        preFilterCap=p["preFilterCap"],
        uniquenessRatio=p["uniquenessRatio"],
        speckleWindowSize=p["speckleWindowSize"],
        speckleRange=p["speckleRange"],
        disp12MaxDiff=p["disp12MaxDiff"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return np.clip(disp, 0, numDisp).astype(np.float32), numDisp

def visualize_depth(disp, Q, p, prev_vis):
    # convert disparity to depth in meters
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    Z = points_3d[:,:,2] / 1000.0

    # apply cutoff
    mask = (disp <= 0) | ~np.isfinite(Z) | (Z <= 0) | (Z > p["maxDepth"])
    Z[mask] = 0

    # apply non-linear mapping for far-field contrast
    # farEnhance=0 → linear, farEnhance=100 → strongly emphasize far range
    gamma = 1.0 + (p["farEnhance"] / 50.0)   # range ~1.0–3.0
    Z_norm = np.clip(Z / p["maxDepth"], 0, 1) ** (1.0/gamma)
    vis = (Z_norm * 255).astype(np.uint8)

    # median blur
    if p["medianBlurK"] >= 3:
        vis = cv2.medianBlur(vis, p["medianBlurK"])

    # temporal smoothing
    if prev_vis is None: prev_vis = vis.copy()
    alpha = p["alpha"] / 100.0
    vis = cv2.addWeighted(prev_vis, alpha, vis, 1-alpha, 0)

    return vis, Z, vis.copy()

def draw_histogram(Z, maxDepth):
    mask = (Z > 0) & np.isfinite(Z) & (Z < maxDepth)
    Z_valid = Z[mask]
    hist_img = np.zeros((200,400,3),dtype=np.uint8)
    avg_depth = 0.0
    if Z_valid.size > 0:
        hist, bins = np.histogram(Z_valid, bins=40, range=(0,maxDepth))
        maxh = hist.max() if hist.max() > 0 else 1
        for i in range(1,len(hist)):
            x1 = int((i-1)/len(hist)*400)
            x2 = int(i/len(hist)*400)
            y1 = 200-int(hist[i-1]/maxh*200)
            y2 = 200-int(hist[i]/maxh*200)
            cv2.line(hist_img,(x1,y1),(x2,y2),(255,255,255),1)
        avg_depth = float(np.mean(Z_valid))
    cv2.putText(hist_img,f"Depth Histogram 0-{maxDepth}m",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    cv2.putText(hist_img,f"Avg Depth: {avg_depth:.2f} m",(10,190),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    cv2.imshow("Depth Histogram", hist_img)

def main():
    calib = load_calibration()
    left_cam, right_cam = open_stereo_pair()
    settings = load_settings()
    create_tuner_window(settings)
    *_, Q = calib

    prev_vis = None
    try:
        while True:
            L = left_cam.read_frame()
            R = right_cam.read_frame()
            if L is None or R is None:
                continue

            rectL, rectR = rectify_pair(L, R, calib)
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY) if rectL.ndim == 3 else rectL
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY) if rectR.ndim == 3 else rectR

            p = read_trackbar()
            disp, _ = compute_disparity(grayL, grayR, p)

            vis, Z, prev_vis = visualize_depth(disp, Q, p, prev_vis)

            # --- Depth Map: grayscale ---
            cv2.putText(vis, f"farEnhance={p['farEnhance']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Depth Map", vis)

            # --- Disparity Map: color ---
            disp_color = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
            disp_color = cv2.applyColorMap(disp_color.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Disparity Map", disp_color)

            draw_histogram(Z, p["maxDepth"])

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                save_settings(p)
                break
    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
