import cv2
import numpy as np
from typing import List, Tuple
import dill
import sys, os
import importlib.util
import time

config_path = r"config.dill"


def check_maps(map_x: np.ndarray, map_y: np.ndarray, size: Tuple[int, int], name: str) -> None:
    """Validate rectification maps before persisting/using them."""
    width, height = size
    expected_shape = (height, width)

    if map_x.shape != expected_shape or map_y.shape != expected_shape:
        raise ValueError(f"{name}: map shape mismatch {map_x.shape}/{map_y.shape} vs {expected_shape}")

    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise ValueError(f"{name}: map contains NaN/Inf values (likely wrong projection matrix or size)")

    if (
        map_x.min() < -1
        or map_x.max() > width
        or map_y.min() < -1
        or map_y.max() > height
    ):
        print(
            f"{name}: warning, map samples extend beyond source image "
            f"(min/max x: {map_x.min():.2f}/{map_x.max():.2f}, "
            f"y: {map_y.min():.2f}/{map_y.max():.2f})"
        )

def detect_corners(gray, pattern):
    if hasattr(cv2, "findChessboardCornersSB"):
        return cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    return cv2.findChessboardCorners(gray, pattern, flags)


def run_calibration(vision, checkerboard=(7, 10), square_size=20.0, min_pairs=5):
    """Perform stereo calibration using the active cameras on the provided vision instance."""

    if not vision.connected:
        print(f"{vision.name}: Starting cameras for calibration...")
        vision.start()

    CHECKERBOARD = checkerboard
    SQUARE_SIZE = square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    print("\n" + "=" * 60)
    print("STEREO CALIBRATION - IMAGE CAPTURE")
    print("=" * 60)
    print("Instructions:")
    print("  - Pictures will be captured automatically every 5 seconds")
    print("  - Press 'q' to finish capturing and proceed with calibration")
    print(f"  - You need at least {min_pairs} valid pairs with detected checkerboards")
    print("=" * 60 + "\n")

    captured_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    pair_count = 0
    last_capture_time = time.time()
    capture_interval = 5.0  # seconds

    try:
        while True:
            left_frame = vision.left_camera.read_frame()
            right_frame = vision.right_camera.read_frame()

            if left_frame is None or right_frame is None:
                print("⚠️ Failed to grab one or both frames. Retrying...")
                continue

            preview_left = cv2.resize(left_frame, (800, 600))
            preview_right = cv2.resize(right_frame, (800, 600))

            # Display status without expensive corner detection
            current_time = time.time()
            time_until_next = max(0, capture_interval - (current_time - last_capture_time))
            status_text = f"Pairs captured: {len(captured_pairs)}/{min_pairs} | Next capture in: {time_until_next:.1f}s"
            cv2.putText(preview_left, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview_right, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Left Camera - Auto-capture every 5s, 'q' to finish", preview_left)
            cv2.imshow("Right Camera - Auto-capture every 5s, 'q' to finish", preview_right)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(f"\n✅ Finished capturing. Total pairs captured: {len(captured_pairs)}")
                break

            # Auto-capture every 5 seconds
            if current_time - last_capture_time >= capture_interval:
                gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                retL, cornersL = detect_corners(gray_left, CHECKERBOARD)
                retR, cornersR = detect_corners(gray_right, CHECKERBOARD)

                if retL and retR:
                    if not hasattr(cv2, "findChessboardCornersSB"):
                        cornersL = cv2.cornerSubPix(gray_left, cornersL, (11, 11), (-1, -1), criteria)
                        cornersR = cv2.cornerSubPix(gray_right, cornersR, (11, 11), (-1, -1), criteria)

                    captured_pairs.append((gray_left.copy(), gray_right.copy(), cornersL, cornersR))
                    print(f"✅ Captured pair {len(captured_pairs)}: Checkerboard detected in both images")
                else:
                    print(f"⚠️ Auto-capture: Checkerboard not detected in both images. Skipping...")
                
                last_capture_time = current_time
                pair_count += 1

    except KeyboardInterrupt:
        print("\n⚠️ Capture interrupted by user")
    finally:
        cv2.destroyAllWindows()

    if len(captured_pairs) < min_pairs:
        raise RuntimeError(f"Not enough valid pairs captured ({len(captured_pairs)}). Need at least {min_pairs}.")

    print(f"\n📸 Processing {len(captured_pairs)} captured stereo pairs...")

    vision.img_shape = captured_pairs[0][0].shape[::-1]

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpointsL, imgpointsR = [], [], []

    for i, (imgL, imgR, cornersL, cornersR) in enumerate(captured_pairs):
        objpoints.append(objp)
        imgpointsL.append(cornersL.reshape(1, -1, 2).astype(np.float64))
        imgpointsR.append(cornersR.reshape(1, -1, 2).astype(np.float64))
        print(f"  ✓ Processed pair {i + 1}/{len(captured_pairs)}")

    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs for calibration")

    if N_OK < min_pairs:
        raise RuntimeError(f"Not enough valid pairs ({N_OK}). Need at least {min_pairs}.")

    print("\n--- Single-Camera Calibration (Fisheye) ---")
    single_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    K1 = np.eye(3)
    D1 = np.zeros((4, 1))
    print("Calibrating left camera...")
    left_rms, K1, D1, left_rvecs, left_tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpointsL,
        vision.img_shape,
        K1,
        D1,
        None,
        None,
        flags=single_flags,
        criteria=criteria,
    )
    print(f"  Left RMS reprojection error: {left_rms:.4f}")

    K2 = np.eye(3)
    D2 = np.zeros((4, 1))
    print("Calibrating right camera...")
    right_rms, K2, D2, right_rvecs, right_tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpointsR,
        vision.img_shape,
        K2,
        D2,
        None,
        None,
        flags=single_flags,
        criteria=criteria,
    )
    print(f"  Right RMS reprojection error: {right_rms:.4f}")

    print("\n--- Stereo Calibration (Fisheye) ---")
    stereo_flags = cv2.fisheye.CALIB_FIX_INTRINSIC

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
        flags=stereo_flags,
    )

    print(f"RMS reprojection error: {rms:.4f}, baseline: {np.linalg.norm(T):.3f} units")

    R1, R2, P1, P2, vision.Q = cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        vision.img_shape,
        R,
        T,
        flags=cv2.fisheye.CALIB_ZERO_DISPARITY,
        balance=0.0,
        fov_scale=1.2,
    )

    newK1 = P1[:, :3].astype(np.float64, copy=True)
    newK2 = P2[:, :3].astype(np.float64, copy=True)

    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, newK1, vision.img_shape, cv2.CV_32FC1
    )
    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, newK2, vision.img_shape, cv2.CV_32FC1
    )

    check_maps(map1x, map1y, vision.img_shape, "left")
    check_maps(map2x, map2y, vision.img_shape, "right")

    try:  # Load in dill settings
        with open(config_path, "rb") as f:
            current_dill = dill.load(f)
    except Exception as e:
        raise RuntimeError(f"No config dill found: {e}")

    camera_settings = current_dill['camera'] #just camera

    # store rectification outputs and maps
    camera_settings.setdefault("left", {})
    camera_settings.setdefault("right", {})

    camera_settings["left"]["map_x"] = np.ascontiguousarray(map1x, dtype=np.float32)
    camera_settings["left"]["map_y"] = np.ascontiguousarray(map1y, dtype=np.float32)
    camera_settings["left"]["K"] = np.asarray(K1, dtype=np.float64)
    camera_settings["left"]["D"] = np.asarray(D1, dtype=np.float64)
    camera_settings["left"]["R"] = np.asarray(R1, dtype=np.float64)
    camera_settings["left"]["P"] = np.asarray(P1, dtype=np.float64)
    camera_settings["left"]["newK"] = newK1
    camera_settings["left"]["rms"] = float(left_rms)

    camera_settings["right"]["map_x"] = np.ascontiguousarray(map2x, dtype=np.float32)
    camera_settings["right"]["map_y"] = np.ascontiguousarray(map2y, dtype=np.float32)
    camera_settings["right"]["K"] = np.asarray(K2, dtype=np.float64)
    camera_settings["right"]["D"] = np.asarray(D2, dtype=np.float64)
    camera_settings["right"]["R"] = np.asarray(R2, dtype=np.float64)
    camera_settings["right"]["P"] = np.asarray(P2, dtype=np.float64)
    camera_settings["right"]["newK"] = newK2
    camera_settings["right"]["rms"] = float(right_rms)

    camera_settings["stereo"] = {
        "R": np.asarray(R, dtype=np.float64),
        "T": np.asarray(T, dtype=np.float64),
        "Q": np.asarray(vision.Q, dtype=np.float64),
        "flags": int(stereo_flags),
        "balance": 0.0,
        "fov_scale": 1.2,
    }

    # Persist calibration resolution for reference
    camera_settings["resolution"] = tuple(int(x) for x in vision.img_shape)
    camera_settings["left"]["map_size"] = tuple(map1x.shape[::-1])
    camera_settings["right"]["map_size"] = tuple(map2x.shape[::-1])
    camera_settings["Q"] = np.asarray(vision.Q, dtype=np.float64)

    print(f"\n💾 Calibration complete")
    print(f"   Maps shape: {camera_settings['left']['map_x'].shape}")
    print(f"   Image size: {vision.img_shape}")

    os.makedirs("calib_pairs", exist_ok=True)
    for i, (imgL, imgR, *_ ) in enumerate(captured_pairs):
        cv2.imwrite(f"calib_pairs/left_{i:03d}.png", imgL)
        cv2.imwrite(f"calib_pairs/right_{i:03d}.png", imgR)






    return camera_settings

if __name__ == "__main__":
    # Load config
    print("Loading Config...")
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        print("Loaded whole Dill")
        camera_config = config['camera']
        print("Loaded Camera Config")
    except Exception as e:
        raise KeyError(f"An unexpected error occurred Loading config.dill: {e}")

    # Instantiate vision system
    print("\n" + "="*60)
    print("Initializing Vision System...")
    print("="*60)
    
    # Import and load the vision class
    module_path, class_name = camera_config['who_to_run'].rsplit(".", 1)
    spec = importlib.util.spec_from_file_location(
        module_path, 
        os.path.join(os.path.dirname(__file__), *module_path.split(".")) + ".py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    VisionClass = getattr(module, class_name)
    
    # Create vision instance
    vision = VisionClass(name="camera", **camera_config)
    
    try:
        # Run calibration
        updated_camera_settings = run_calibration(vision, checkerboard=(7, 10), square_size=20.0, min_pairs=5)
        
        # Update the config with new calibration data
        config['camera'] = updated_camera_settings
        
        # Save updated config back to dill file
        print("\n" + "="*60)
        print("Saving updated configuration to config.dill...")
        with open(config_path, "wb") as f:
            dill.dump(config, f)
        print("✅ Configuration saved successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during calibration: {e}")
        raise
    finally:
        # Clean up
        if vision.connected:
            print("\nStopping vision system...")
            vision.stop()
