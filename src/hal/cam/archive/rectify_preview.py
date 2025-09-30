import cv2
import numpy as np
import glob
import os

def main():
    # Load fisheye calibration
    calib = np.load("stereo_calib_fisheye.npz")
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]
    R1, R2, P1, P2, Q = calib["R1"], calib["R2"], calib["P1"], calib["P2"], calib["Q"]

    # Load one stereo pair
    left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left_*.png")))
    right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right_*.png")))

    if not left_images or not right_images:
        print("❌ No stereo images found.")
        return

    imgL = cv2.imread(left_images[0])
    imgR = cv2.imread(right_images[0])
    h, w = imgL.shape[:2]

    # Compute rectification maps
    mapLx, mapLy = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Side by side with horizontal lines
    combined = np.hstack((rectifiedL, rectifiedR))
    for y in range(0, combined.shape[0], 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv2.imshow("Rectified Pair (Fisheye)", combined)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
