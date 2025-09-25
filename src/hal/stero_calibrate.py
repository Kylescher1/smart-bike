import cv2
import numpy as np
import glob
import os

# Checkerboard settings
CHECKERBOARD = (7, 10)   # number of inner corners (width, height)
SQUARE_SIZE = 15.0       # mm per square

def main():
    # Prepare object points for the checkerboard pattern
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Arrays to store object points and image points
    objpoints = []   # 3d points in real world
    imgpointsL = []  # 2d points in left image
    imgpointsR = []  # 2d points in right image

    # Load image pairs
    left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left_*.png")))
    right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right_*.png")))

    if len(left_images) != len(right_images):
        print("❌ Mismatch in number of left/right images.")
        return

    print(f"Found {len(left_images)} stereo pairs.")

    for left_img, right_img in zip(left_images, right_images):
        imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

        # Find chessboard corners
        retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, None)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, None)

        if retL and retR:
            objpoints.append(objp)

            # Refine corners
            cornersL = cv2.cornerSubPix(
                imgL, cornersL, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cornersR = cv2.cornerSubPix(
                imgR, cornersR, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

            # Draw and show (optional, helps debug)
            cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
            cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
            cv2.imshow("Left", imgL)
            cv2.imshow("Right", imgR)
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("❌ Not enough valid stereo pairs found for calibration.")
        return

    print(f"✅ Using {len(objpoints)} valid pairs for calibration.")

    # Calibrate individual cameras
    retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, imgL.shape[::-1], None, None)
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, imgR.shape[::-1], None, None)

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR,
        imgL.shape[::-1], criteria=criteria, flags=flags
    )

    print("Calibration RMS error:", retval)

    # Stereo rectification
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        mtxL, distL, mtxR, distR,
        imgL.shape[::-1], R, T, alpha=0
    )

    # Save results
    np.savez("stereo_calib.npz",
             mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R=R, T=T, E=E, F=F,
             RL=RL, RR=RR, PL=PL, PR=PR, Q=Q)

    print("💾 Calibration saved to stereo_calib.npz")

if __name__ == "__main__":
    main()
