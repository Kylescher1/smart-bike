import cv2
import numpy as np
import glob
import os

def main():
    # Load calibration results
    calib = np.load("stereo_calib.npz")
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    RL, RR, PL, PR, Q = calib["RL"], calib["RR"], calib["PL"], calib["PR"], calib["Q"]

    # Get first stereo pair for testing
    left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left_*.png")))
    right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right_*.png")))

    if not left_images or not right_images:
        print("❌ No stereo images found.")
        return

    imgL = cv2.imread(left_images[0])
    imgR = cv2.imread(right_images[0])

    h, w = imgL.shape[:2]

    # Compute rectification maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, (w, h), cv2.CV_32FC1)

    # Apply rectification
    rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Combine side by side for visualization
    combined = np.hstack((rectifiedL, rectifiedR))

    # Draw horizontal lines every 40 pixels
    for y in range(0, combined.shape[0], 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv2.imshow("Rectified Pair", combined)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
