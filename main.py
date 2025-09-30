from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.calib import load_calibration
from src.hal.cam.depth import disparity_to_points, rectify_pair

import os, cv2, time, numpy as np, matplotlib.pyplot as plt, json

SETTINGS_FILE = "stereo_settings.json"

def load_settings(path=SETTINGS_FILE):
    return json.load(open(path)) if os.path.exists(path) else {}

def build_matcher(p):
    numDisp = 16 * max(1, p.get("numDisparities", 6))
    blk = max(3, p.get("blockSize", 5) | 1)  # odd >=3
    cn = 1
    P1, P2 = 8*cn*blk*blk, 32*cn*blk*blk
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisp,
        blockSize=blk,
        P1=P1, P2=P2,
        preFilterCap=p.get("preFilterCap", 31),
        uniquenessRatio=p.get("uniquenessRatio", 15),
        speckleWindowSize=p.get("speckleWindowSize", 100),
        speckleRange=p.get("speckleRange", 32),
        disp12MaxDiff=p.get("disp12MaxDiff", 1),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    ), numDisp

def main():
    time.sleep(2)
    calib = load_calibration()
    left_cam, right_cam = open_stereo_pair()
    if not left_cam or not right_cam:
        print("❌ Could not open stereo cameras")
        return

    settings = load_settings()
    stereo, numDisp = build_matcher(settings)

    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z forward (m)")

    # lock to 15 ft (≈4.6 m) forward and ±2 m sideways
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(00, 4600)

    plt.show()


    prev_vis = None
    try:
        while True:
            L = left_cam.read_frame(); R = right_cam.read_frame()
            if L is None or R is None:
                continue

            rectL, rectR = rectify_pair(L, R, calib)
            gL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

            disp = stereo.compute(gL, gR).astype(np.float32) / 16.0
            disp = np.clip(disp, 0, numDisp).astype(np.float32)

            vis = (disp / numDisp * 255).astype(np.uint8)

            # median blur on vis
            k = settings.get("medianBlurK", 0)
            if k >= 3 and k % 2 == 1:
                vis = cv2.medianBlur(vis, k)

            # temporal smoothing with alpha
            alpha = settings.get("alpha", 40) / 100.0
            if prev_vis is None:
                prev_vis = vis.copy()
            vis = cv2.addWeighted(prev_vis, alpha, vis, 1 - alpha, 0)
            prev_vis = vis.copy()

            color_vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", color_vis)

            # points for scatter
            points = disparity_to_points(disp, calib)
            mask = np.isfinite(points[:,:,2]) & (points[:,:,2] > 0)
            pts = points[mask]
            if len(pts) > 0:
                sample = pts[::800]
                sc.set_offsets(np.c_[sample[:,0], sample[:,2]])

                fig.canvas.draw()
                fig.canvas.flush_events()

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
