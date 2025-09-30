import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (7, 10)   # inner corners (cols, rows)
SQUARE_SIZE = 15.0       # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

def main():
    base_dir = os.path.dirname(__file__)          # /src/hal/cam/calibrate
    pairs_dir = os.path.join(base_dir, "stereo_pairs")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Prepare a single set of object points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []   # 3D points
    imgpointsL = []  # 2D points (left)
    imgpointsR = []  # 2D points (right)

    # Load stereo images
    left_images = sorted(glob.glob(os.path.join(pairs_dir, "left_*.png")))
    right_images = sorted(glob.glob(os.path.join(pairs_dir, "right_*.png")))

    for left_img, right_img in zip(left_images, right_images):
        print(f"Processing pair: {left_img}, {right_img}")
        imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

        if imgL is None or imgR is None:
            print(f"⚠️ Could not read {left_img} or {right_img}")
            continue

        # Optional cropping
        h, w = imgL.shape
        crop = 20
        imgL = imgL[crop:h-crop, crop:w-crop]
        imgR = imgR[crop:h-crop, crop:w-crop]

        retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, None)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, None)
        print(f"  Found corners L={retL}, R={retR}")

        if retL and retR:
            cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)

            # Draw for visual confirmation
            cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
            cv2.imshow("Corners L", imgL)
            cv2.waitKey(200)

            # Ensure correct shape
            objpoints.append(objp)  # (1, N, 3)
            imgpointsL.append(cornersL.reshape(1, -1, 2))
            imgpointsR.append(cornersR.reshape(1, -1, 2))

    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs")

    if N_OK < 5:
        print("❌ Not enough valid pairs")
        cv2.destroyAllWindows()
        return

    # Debug: check shapes
    print("objpoints[0].shape =", objpoints[0].shape)
    print("imgpointsL[0].shape =", imgpointsL[0].shape)
    print("imgpointsR[0].shape =", imgpointsR[0].shape)

    img_shape = imgL.shape[::-1]  # (width, height)

    # Init intrinsics
    K1 = np.eye(3)
    D1 = np.zeros((4, 1))
    K2 = np.eye(3)
    D2 = np.zeros((4, 1))

    # Stereo calibration (fisheye model)
    rms, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1, D1,
        K2, D2,
        img_shape,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        criteria=criteria
    )

    print("RMS error:", rms)

    # Stereo rectification
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2,
        img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0,
        fov_scale=1.0
    )

    # Rectification maps
    leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, img_shape, cv2.CV_32FC1
    )
    rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, img_shape, cv2.CV_32FC1
    )

    # Save calibration
    out_file = os.path.join(data_dir, "stereo_calib_fisheye.npz")
    np.savez_compressed(
        out_file,
        imageSize=img_shape,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        leftMapX=leftMapX, leftMapY=leftMapY,
        rightMapX=rightMapX, rightMapY=rightMapY
    )

    print(f"💾 Saved calibration to {out_file}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
