# src/hal/cam/depth.py
import cv2, numpy as np
from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.calibrate.calib import load_calibration


DEFAULTS = {
    "numDisparities": 6,
    "blockSize": 5,
    "preFilterCap": 31,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1,
    "medianBlurK": 0
}

def rectify_pair(left, right, calib):
    leftMapX, leftMapY, rightMapX, rightMapY, _, _ = calib
    rectL = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR)
    return rectL, rectR

def compute_depth_map(left, right, calib, settings=DEFAULTS):
    rectL, rectR = rectify_pair(left, right, calib)
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY) if rectL.ndim == 3 else rectL
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY) if rectR.ndim == 3 else rectR

    numDisp = 16 * max(1, settings["numDisparities"])
    blk = max(3, settings["blockSize"] | 1)
    cn = 1
    P1, P2 = 8*cn*blk*blk, 32*cn*blk*blk

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisp,
        blockSize=blk,
        P1=P1, P2=P2,
        preFilterCap=settings["preFilterCap"],
        uniquenessRatio=settings["uniquenessRatio"],
        speckleWindowSize=settings["speckleWindowSize"],
        speckleRange=settings["speckleRange"],
        disp12MaxDiff=settings["disp12MaxDiff"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp = np.clip(disp, 0, numDisp)

    if settings["medianBlurK"] >= 3 and settings["medianBlurK"] % 2 == 1:
        disp = cv2.medianBlur(disp, settings["medianBlurK"])

    # Normalize for visualization
    vis = (disp / numDisp * 255).astype(np.uint8)
    color_vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    return disp, color_vis  # disp = depth values, color_vis = display image

def get_static_depth(show=False):
    calib = load_calibration()
    left_cam, right_cam = open_stereo_pair()
    try:
        L = left_cam.read_frame(); R = right_cam.read_frame()
        if L is None or R is None:
            return None, None
        disp, vis = compute_depth_map(L, R, calib)
        if show:
            cv2.imshow("Static Depth Map", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return disp, vis
    finally:
        left_cam.close(); right_cam.close()

def disparity_to_points(disp, calib):
    """Convert disparity map to 3D point cloud using calibration matrix Q."""
    _, _, _, _, _, Q = calib
    points_3D = cv2.reprojectImageTo3D(disp, Q)
    return points_3D


if __name__ == "__main__":
    disp, vis = get_static_depth(show=True)
    if vis is not None:
        cv2.imshow("Static Depth Map", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
