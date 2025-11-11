import cv2
import numpy as np
from typing import List, Tuple
import dill
import quaternion
import sys, os
import importlib.util
import Calibrate

config_path = r"config.dill"

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
    frame_count = 0  # Track frames for visualization frequency

    try:
        while True:
            left_frame = vision.left_camera.read_frame()
            right_frame = vision.right_camera.read_frame()

            if left_frame is None or right_frame is None:
                print("⚠️ Failed to grab one or both frames. Retrying...")
                continue

            preview_left = cv2.resize(left_frame, (800, 600))
            preview_right = cv2.resize(right_frame, (800, 600))

            # Every 5 frames, try to detect and draw checkerboard pattern
            if frame_count % 1 == 0:
                gray_left_viz = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                gray_right_viz = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                retL_viz, cornersL_viz = cv2.findChessboardCorners(gray_left_viz, CHECKERBOARD, None)
                retR_viz, cornersR_viz = cv2.findChessboardCorners(gray_right_viz, CHECKERBOARD, None)

                # Draw checkerboard corners on the preview frames
                if retL_viz:
                    # Scale corners to preview resolution
                    scale_x = 800 / left_frame.shape[1]
                    scale_y = 600 / left_frame.shape[0]
                    cornersL_scaled = cornersL_viz.copy()
                    cornersL_scaled[:, :, 0] *= scale_x
                    cornersL_scaled[:, :, 1] *= scale_y
                    cv2.drawChessboardCorners(preview_left, CHECKERBOARD, cornersL_scaled, retL_viz)

                if retR_viz:
                    scale_x = 800 / right_frame.shape[1]
                    scale_y = 600 / right_frame.shape[0]
                    cornersR_scaled = cornersR_viz.copy()
                    cornersR_scaled[:, :, 0] *= scale_x
                    cornersR_scaled[:, :, 1] *= scale_y
                    cv2.drawChessboardCorners(preview_right, CHECKERBOARD, cornersR_scaled, retR_viz)

            frame_count += 1

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

    try:  # Load in dill settings
        with open(config_path, "rb") as f:
            current_dill = dill.load(f)
    except Exception as e:
        raise RuntimeError(f"No config dill FOUND!: {e}")

    camera_settings = current_dill['camera'] #just camera


    #modify values we care abt to locations in dill (not dynamic cause Kyle_scher is evil)
    camera_settings["left"]["map_x"], camera_settings["left"]["map_y"] = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, vision.img_shape, cv2.CV_32FC1
    )
    camera_settings["right"]["map_x"], camera_settings["right"]["map_y"] = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, vision.img_shape, cv2.CV_32FC1
    )

    # Persist single-camera intrinsics and distortion
    camera_settings["left"]["K"] = K1
    camera_settings["left"]["D"] = D1
    camera_settings["left"]["rms"] = float(left_rms)
    camera_settings["right"]["K"] = K2
    camera_settings["right"]["D"] = D2
    camera_settings["right"]["rms"] = float(right_rms)

    #assign Q new value
    camera_settings["Q"] = vision.Q

    print(f"\n💾 Calibration complete")
    print(f"   Maps shape: {camera_settings['left']['map_x'].shape}")
    print(f"   Image size: {vision.img_shape}")






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
