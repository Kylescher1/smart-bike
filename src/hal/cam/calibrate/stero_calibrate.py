import cv2
import numpy as np
import glob
import os

# Checkerboard parameters
CHECKERBOARD = (7, 10)   # inner corners (cols, rows)
SQUARE_SIZE = 20.0       # mm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

def main():
    base_dir = os.path.dirname(__file__)  # /src/hal/cam/calibrate
    pairs_dir = os.path.join(base_dir, "stereo_pairs")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # prepare one template of object points
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # storage for calibration
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # load stereo images
    left_images = sorted(glob.glob(os.path.join(pairs_dir, "left_*.png")))
    right_images = sorted(glob.glob(os.path.join(pairs_dir, "right_*.png")))

    for left_img, right_img in zip(left_images, right_images):
        print(f"Processing pair: {os.path.basename(left_img)} , {os.path.basename(right_img)}")
        imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            print("⚠️ Could not read one of the images")
            continue

        retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, None)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, None)
        print(f"  Found corners L={retL}, R={retR}")

        if retL and retR:
            cornersL = cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), criteria)
            cornersR = cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), criteria)

            # visual check
            cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
            cv2.imshow("Corners L", imgL)
            cv2.waitKey(200)

            # append correctly shaped
            objpoints.append(objp.reshape(-1,1,3))
            imgpointsL.append(cornersL.reshape(-1,1,2))
            imgpointsR.append(cornersR.reshape(-1,1,2))

    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs")
    if N_OK < 5:
        print("❌ Not enough valid pairs")
        cv2.destroyAllWindows()
        return

    img_shape = imgL.shape[::-1]  # (w,h)

    # init intrinsics
    K1 = np.eye(3)
    D1 = np.zeros((4,1))
    K2 = np.eye(3)
    D2 = np.zeros((4,1))

    try:
        # stereo calibration (fisheye model)
        rms, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR,
            K1, D1, K2, D2,
            img_shape,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            criteria=criteria
        )
        print("RMS error (fisheye):", rms)

        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2,
            img_shape, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0, fov_scale=1.0
        )

        leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, P1, img_shape, cv2.CV_32FC1
        )
        rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R2, P2, img_shape, cv2.CV_32FC1
        )

    except cv2.error as e:
        print("⚠️ fisheye calibration failed, retrying with pinhole model")
        rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR,
            None, None, None, None,
            img_shape,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=criteria
        )
        print("RMS error (pinhole):", rms)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_shape, R, T, flags=0
        )

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, img_shape, cv2.CV_32FC1
        )
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, img_shape, cv2.CV_32FC1
        )

    # save calibration
    out_file = os.path.join(data_dir, "stereo_calib.npz")
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
