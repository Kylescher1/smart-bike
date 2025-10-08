# src/hal/cam/calibrate/stereo_calibrate_fisheye.py
import cv2
import numpy as np
import glob
import os

# Checkerboard parameters
CHECKERBOARD = (7, 10)   # inner corners (cols, rows)
SQUARE_SIZE = 20.0       # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-6)

def check_camera_similarity(imgL, imgR, brightness_thresh=10, contrast_thresh=20, hist_thresh=0.5):
    if imgL.shape != imgR.shape:
        print("⚠️ Resolution mismatch")
        return False
    meanL, stdL = cv2.meanStdDev(imgL)
    meanR, stdR = cv2.meanStdDev(imgR)
    brightness_diff = abs(meanL[0][0] - meanR[0][0])
    contrast_diff = abs(stdL[0][0] - stdR[0][0])
    histL = cv2.calcHist([imgL], [0], None, [64], [0, 256])
    histR = cv2.calcHist([imgR], [0], None, [64], [0, 256])
    histL = cv2.normalize(histL, histL).flatten()
    histR = cv2.normalize(histR, histR).flatten()
    hist_corr = cv2.compareHist(histL, histR, cv2.HISTCMP_CORREL)
    if brightness_diff > brightness_thresh:
        print(f"⚠️ Brightness mismatch ({brightness_diff:.1f})")
        return False
    if contrast_diff > contrast_thresh:
        print(f"⚠️ Contrast mismatch ({contrast_diff:.1f})")
        return False
    if hist_corr < hist_thresh:
        print(f"⚠️ Histogram mismatch (corr={hist_corr:.2f})")
        return False
    return True


def main():
    base_dir = os.path.dirname(__file__)
    pairs_dir = os.path.join(base_dir, "stereo_pairs")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpointsL, imgpointsR = [], [], []

    left_images = sorted(glob.glob(os.path.join(pairs_dir, "left_*.png")))
    right_images = sorted(glob.glob(os.path.join(pairs_dir, "right_*.png")))

    for left_img, right_img in zip(left_images, right_images):
        print(f"Processing pair: {os.path.basename(left_img)} , {os.path.basename(right_img)}")
        imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            print("⚠️ Could not read one of the images")
            continue
        if not check_camera_similarity(imgL, imgR):
            print("🚫 Skipping this pair due to mismatch\n")
            continue

        retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, None)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, None)
        print(f"  Found corners L={retL}, R={retR}")

        if retL and retR:
            cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)

            visL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
            visR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(visL, CHECKERBOARD, cornersL, True)
            cv2.drawChessboardCorners(visR, CHECKERBOARD, cornersR, True)
            combo = np.hstack((visL, visR))
            cv2.imshow("Detected Corners (L | R)", combo)
            key = cv2.waitKey(300) & 0xFF
            if key == ord('q'):
                break

            objpoints.append(objp)
            imgpointsL.append(cornersL.reshape(1, -1, 2))
            imgpointsR.append(cornersR.reshape(1, -1, 2))
        else:
            print("🚫 Skipping pair due to missing corners\n")

    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs")
    if N_OK < 5:
        print("❌ Not enough valid pairs")
        cv2.destroyAllWindows()
        return

    img_shape = imgL.shape[::-1]

    # Initialize intrinsics
    K1 = np.eye(3)
    D1 = np.zeros((4, 1))
    K2 = np.eye(3)
    D2 = np.zeros((4, 1))

    print("\n--- Stereo Calibration (Fisheye) ---")

    # Fisheye stereo calibration
    rms, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1, D1, K2, D2,
        img_shape,
        criteria=criteria,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    )

    print(f"\nRMS reprojection error (fisheye): {rms:.4f}")
    print("\nLeft Camera Intrinsics:\n", K1)
    print("Left Distortion Coefficients:\n", D1.ravel())
    print("\nRight Camera Intrinsics:\n", K2)
    print("Right Distortion Coefficients:\n", D2.ravel())
    print("\nRotation (R):\n", R)
    print("Translation (T):\n", T.ravel())

    # Stereo rectification (fisheye)
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2,
        img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.7,
        fov_scale=1.2
    )

    leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, img_shape, cv2.CV_32FC1
    )
    rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, img_shape, cv2.CV_32FC1
    )

    # Save calibration
    out_file = os.path.join(data_dir, "stereo_calib.npz")
    np.savez_compressed(
        out_file,
        imageSize=img_shape,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        leftMapX=leftMapX, leftMapY=leftMapY,
        rightMapX=rightMapX, rightMapY=rightMapY,
        rms=rms
    )

    print(f"\n💾 Saved fisheye calibration to {out_file}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
