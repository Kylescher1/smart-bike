import cv2
import numpy as np
import glob
import os

def main():
    # Load calibration
    calib = np.load("stereo_calib.npz")
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    RL, RR, PL, PR, Q = calib["RL"], calib["RR"], calib["PL"], calib["PR"], calib["Q"]

    # Load first stereo pair
    left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left_*.png")))
    right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right_*.png")))

    if not left_images or not right_images:
        print("❌ No stereo images found.")
        return

    imgL = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)

    h, w = imgL.shape[:2]

    # Rectification maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, (w, h), cv2.CV_32FC1)

    rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Stereo matcher (SGBM = better quality than BM)
    window_size = 5
    min_disp = 0
    num_disp = 16 * 8  # must be divisible by 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=7,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

    # Normalize for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow("Left Rectified", rectifiedL)
    cv2.imshow("Right Rectified", rectifiedR)
    cv2.imshow("Disparity Map", disp_vis)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
