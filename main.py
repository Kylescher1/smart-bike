# src/main.py
import time, os, json, cv2, numpy as np, matplotlib.pyplot as plt
from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.calib import load_calibration
from src.hal.cam.depth import rectify_pair, disparity_to_points

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "stereo_settings.json")

DEFAULTS = {
    "numDisparities": 6,
    "blockSize": 5,
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            return json.load(open(SETTINGS_FILE))
        except Exception:
            pass
    return DEFAULTS.copy()

def setup_system():
    calib = load_calibration()
    left, right = open_stereo_pair()
    if not left or not right:
        raise RuntimeError("❌ Could not open stereo cameras")

    s = load_settings()
    numDisp = 16 * max(1, s.get("numDisparities", 6))
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisp,
        blockSize=max(3, s.get("blockSize", 5) | 1),
        preFilterCap=s.get("preFilterCap", 31),
        uniquenessRatio=s.get("uniquenessRatio", 15),
        speckleWindowSize=s.get("speckleWindowSize", 100),
        speckleRange=s.get("speckleRange", 32),
        disp12MaxDiff=s.get("disp12MaxDiff", 1),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return calib, left, right, matcher, numDisp

def compute_points(left, right, calib, matcher, numDisp):
    L, R = left.read_frame(), right.read_frame()
    if L is None or R is None:
        return None, None
    rectL, rectR = rectify_pair(L, R, calib)
    gL, gR = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    disp = matcher.compute(gL, gR).astype("float32") / 16.0
    disp = np.clip(disp, 0, numDisp)
    points = disparity_to_points(disp, calib)
    return disp, points

def visualize(disp, points, numDisp, scatter, ax):
    # Depth map
    vis = (disp / numDisp * 255).astype(np.uint8)
    color_vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imshow("Depth Map", color_vis)

    # Top-down: X (left/right), Z (forward up to 15 m)
    mask = (
        np.isfinite(points[:,:,2]) &
        (points[:,:,2] > 0) &
        (points[:,:,2] <= 15000)   # trim at 15 m
    )
    pts = points[mask]
    if len(pts) > 0:
        sample = pts[::800]
        scatter.set_offsets(np.c_[sample[:,0], sample[:,2]])
        plt.draw()
        plt.pause(0.001)




def main():
    calib, left, right, matcher, numDisp = setup_system()

    # prepare matplotlib scatter
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=1)
    ax.set_xlabel("X (m left/right)")
    ax.set_ylabel("Z (m forward)")
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(0, 4600)

    try:
        while True:
            disp, points = compute_points(left, right, calib, matcher, numDisp)
            if disp is not None:
                visualize(disp, points, numDisp, sc, ax)  # <-- pass ax too
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            time.sleep(0.25)

    finally:
        left.close(); right.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
