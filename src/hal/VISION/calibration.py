import cv2
import numpy as np
from typing import List, Tuple


def run_calibration(vision, checkerboard=(7, 10), square_size=20.0, min_pairs=5):
    """Perform stereo calibration using the active cameras on the provided vision instance."""

    if not vision.connected:
        print(f"{vision.name}: Starting cameras for calibration...")
        vision.start()

    CHECKERBOARD = checkerboard
    SQUARE_SIZE = square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-6)

    print("\n" + "=" * 60)
    print("STEREO CALIBRATION - IMAGE CAPTURE")
    print("=" * 60)
    print("Instructions:")
    print("  - Press 's' to capture a stereo pair")
    print("  - Press 'q' to finish capturing and proceed with calibration")
    print(f"  - You need at least {min_pairs} valid pairs with detected checkerboards")
    print("=" * 60 + "\n")

    captured_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    pair_count = 0

    try:
        while True:
            left_frame = vision.left_camera.read_frame()
            right_frame = vision.right_camera.read_frame()

            if left_frame is None or right_frame is None:
                print("⚠️ Failed to grab one or both frames. Retrying...")
                continue

            preview_left = cv2.resize(left_frame, (800, 600))
            preview_right = cv2.resize(right_frame, (800, 600))

            status_text = f"Pairs captured: {len(captured_pairs)}/{min_pairs}"
            cv2.putText(preview_left, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview_right, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Left Camera - Press 's' to capture, 'q' to finish", preview_left)
            cv2.imshow("Right Camera - Press 's' to capture, 'q' to finish", preview_right)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(f"\n✅ Finished capturing. Total pairs captured: {len(captured_pairs)}")
                break
            elif key == ord("s"):
                gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                retL, cornersL = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
                retR, cornersR = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

                if retL and retR:
                    cornersL_refined = cv2.cornerSubPix(gray_left, cornersL, (11, 11), (-1, -1), criteria)
                    cornersR_refined = cv2.cornerSubPix(gray_right, cornersR, (11, 11), (-1, -1), criteria)

                    captured_pairs.append((gray_left.copy(), gray_right.copy(), cornersL_refined, cornersR_refined))
                    print(f"✅ Captured pair {len(captured_pairs)}: Checkerboard detected in both images")
                else:
                    print(f"⚠️ Pair {pair_count + 1}: Checkerboard not detected in both images. Skipping...")
                pair_count += 1

    except KeyboardInterrupt:
        print("\n⚠️ Capture interrupted by user")
    finally:
        cv2.destroyAllWindows()

    if len(captured_pairs) < min_pairs:
        raise RuntimeError(f"Not enough valid pairs captured ({len(captured_pairs)}). Need at least {min_pairs}.")

    print(f"\n📸 Processing {len(captured_pairs)} captured stereo pairs...")

    vision.img_shape = captured_pairs[0][0].shape[::-1]

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpointsL, imgpointsR = [], [], []

    for i, (imgL, imgR, cornersL, cornersR) in enumerate(captured_pairs):
        objpoints.append(objp)
        imgpointsL.append(cornersL.reshape(1, -1, 2))
        imgpointsR.append(cornersR.reshape(1, -1, 2))
        print(f"  ✓ Processed pair {i + 1}/{len(captured_pairs)}")

    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs for calibration")

    if N_OK < min_pairs:
        raise RuntimeError(f"Not enough valid pairs ({N_OK}). Need at least {min_pairs}.")

    K1 = np.eye(3)
    D1 = np.zeros((4, 1))
    K2 = np.eye(3)
    D2 = np.zeros((4, 1))

    print("\n--- Stereo Calibration (Fisheye) ---")

    rms, K1, D1, K2, D2, R, T, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1,
        D1,
        K2,
        D2,
        vision.img_shape,
        criteria=criteria,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    )

    print(f"RMS reprojection error: {rms:.4f}")

    R1, R2, P1, P2, vision.Q = cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        vision.img_shape,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0,
        fov_scale=1.2,
    )

    vision.left["map_x"], vision.left["map_y"] = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, vision.img_shape, cv2.CV_32FC1
    )
    vision.right["map_x"], vision.right["map_y"] = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, vision.img_shape, cv2.CV_32FC1
    )

    print(f"\n💾 Calibration complete")
    print(f"   Maps shape: {vision.left['map_x'].shape}")
    print(f"   Image size: {vision.img_shape}")

    return {vision.name: vars(vision)}

