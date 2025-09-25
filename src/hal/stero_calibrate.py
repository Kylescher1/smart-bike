import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (7, 10)   # inner corners
SQUARE_SIZE = 15.0       # mm

def main():
    # Prepare object points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE


    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # Load images
    left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left_*.png")))
    right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right_*.png")))

    for left_img, right_img in zip(left_images, right_images):
        imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

        retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, None)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, None)

        if retL and retR:
            objpoints.append(objp)

            imgpointsL.append(cornersL.reshape(1, -1, 2))
            imgpointsR.append(cornersR.reshape(1, -1, 2))


    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs")

    if N_OK < 5:
        print("❌ Not enough valid pairs")
        return

    # Calibration
    K1 = np.zeros((3, 3))
    D1 = np.zeros((4, 1))
    K2 = np.zeros((3, 3))
    D2 = np.zeros((4, 1))
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))

    img_shape = imgL.shape[::-1]

    rms, _, _, _, _, R, T = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1, D1,
        K2, D2,
        img_shape,
        R, T,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    print("RMS error:", rms)

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2,
        img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0,
        fov_scale=1.0
    )



    # Save
    np.savez("stereo_calib_fisheye.npz",
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    print("💾 Saved calibration to stereo_calib_fisheye.npz")

if __name__ == "__main__":
    main()
