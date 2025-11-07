import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import os
import glob
import dill
import copy

from ..cam.Camera import Camera, CAMERA_CONFIG


class VISION:
    BOOLEAN_FIELDS = {
        "useMorph",
        "useBilateral",
        "useWLS",
        "edgeEqualize",
        "edgeUseScharr",
    }
    """
    Vision system for stereo camera depth processing.
    Follows sensor pattern with start(), stop(), and read() methods.
    """
    
    def __init__(self, name="Unnamed VISION", **kwargs):
        """
        Initialize VISION instance.
        Supports nested config structure: camera.left and camera.right
        Also supports flat structure where left/right are at top level
        """
        self.name = name
        self.debug_mode = True
        
        # Load user-provided configuration (following MPU6250 pattern)
        for k, v in kwargs.items():
            setattr(self, k, v)
        #     print(f"self.{k} returns:{v}")
        # print(f"kyle is evil and am {vars(self)}")

        # Normalize boolean toggles (config may use 0/1 or True/False)
        for field in self.BOOLEAN_FIELDS:
            if hasattr(self, field):
                setattr(self, field, bool(getattr(self, field)))

        # Extract calibration maps from camera.left and camera.right
        # Maps are stored as map_x and map_y under each camera

        # Extract shared calibration data (imageSize and Q) from camera level
        # These are set via kwargs unpacking, but we can also get them from camera config
        if not hasattr(self, 'imageSize'):
            self.imageSize = kwargs.get('imageSize', None)
        if not hasattr(self, 'Q'):
            self.Q = kwargs.get('Q', None)
        
        # Initialize camera objects
        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        
        # Depth processing components (initialized in start())
        # Calibration maps are already set via kwargs (leftMapX, leftMapY, rightMapX, rightMapY, imageSize, Q)
        self.stereo = None
        
        # Runtime state
        self.connected = False
    
    def start(self):
        """Open cameras and initialize depth processor."""
        if self.connected:
            return
        
        print(f"{self.name}: Starting vision system...")
        
        # Initialize cameras
        self.left_camera = Camera(self.left['port'], CAMERA_CONFIG)
        self.right_camera = Camera(self.right['port'], CAMERA_CONFIG)
        
        # Open cameras
        try:
            self.left_camera.open()
            print(f"{self.name}: Left camera opened on port {self.left['port']}")
        except Exception as e:
            raise RuntimeError(f"{self.name}: Failed to open left camera: {e}")
        
        try:
            self.right_camera.open()
            print(f"{self.name}: Right camera opened on port {self.right['port']}")
        except Exception as e:
            self.left_camera.close()
            raise RuntimeError(f"{self.name}: Failed to open right camera: {e}")
        

        # Calibration maps are already set via kwargs unpacking
        # Use them directly (they're stored as leftMapX, leftMapY, etc.)
        print(f"{self.name}: Calibration data loaded from config")
        
        # Initialize stereo matcher using config settings
        block_size = self.blockSize
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        num_disparities = max(16, 16 * self.numDisparitiesK)
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.minDisparity,
            numDisparities=num_disparities,
            blockSize=max(3, block_size),
            preFilterCap=self.preFilterCap,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            disp12MaxDiff=self.disp12MaxDiff,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        
        self.connected = True
        print(f"{self.name}: Vision system started successfully")
    
    def stop(self):
        """Close cameras and cleanup resources."""
        if not self.connected:
            return
        
        print(f"{self.name}: Stopping vision system...")
        
        if self.left_camera:
            self.left_camera.close()
            self.left_camera = None
        
        if self.right_camera:
            self.right_camera.close()
            self.right_camera = None
        
        # Clear stereo matcher (calibration maps remain as they're from config)
        self.stereo = None
        
        self.connected = False
        print(f"{self.name}: Vision system stopped")
    
    def _rectify(self, left_frame: np.ndarray, right_frame: np.ndarray):
        """Rectify stereo pair using calibration maps."""

        rect_left = cv2.remap(left_frame, self.left['map_x'], self.left['map_y'], cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_frame, self.left['map_x'], self.left['map_y'], cv2.INTER_LINEAR)
        return rect_left, rect_right
    
    def _compute_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray):
        """Compute disparity map from rectified stereo pair."""
        if self.stereo is None:
            raise RuntimeError("Stereo matcher not initialized. Call start() first.")
        
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity < 0] = 0
        return disparity
    
    def _disparity_to_depth(self, disparity: np.ndarray):
        """Convert disparity map to depth map using Q matrix."""
        if not hasattr(self, 'Q') or self.Q is None:
            raise RuntimeError("Calibration Q matrix not loaded. Call start() first.")
        
        points = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points[:, :, 2]
        depth[~np.isfinite(depth)] = 0
        depth = np.maximum(depth, 0)
        return depth
    
    def read(self):
        """
        Capture frames, process depth, and return dictionary with depth_map and metadata.
        
        Returns:
            dict: {
                'depth_map': np.ndarray,
                'metadata': {
                    'timestamp': str,
                    'num_disparities': int
                }
            }
        """
        if not self.connected:
            raise RuntimeError(f"{self.name}: Not connected. Call start() first.")
        
        # Capture frames
        left_frame = self.left_camera.read_frame()
        right_frame = self.right_camera.read_frame()
        
        if left_frame is None or right_frame is None:
            return {
                'depth_map': np.array([]),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_disparities': 0,
                    'error': 'Failed to capture frames'
                }
            }
        
        # Process depth pipeline
        try:
            # Rectify
            rect_left, rect_right = self._rectify(left_frame, right_frame)
            
            # Compute disparity
            disparity = self._compute_disparity(rect_left, rect_right)
            
            # Convert to depth
            depth_map = self._disparity_to_depth(disparity)
            
            # Prepare metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'num_disparities': int(self.stereo.getNumDisparities()) if self.stereo else 0,
            }
            
            return {
                'depth_map': depth_map,
                'metadata': metadata
            }
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error processing depth: {e}")
            return {
                'depth_map': np.array([]),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_disparities': 0,
                    'error': str(e)
                }
            }
    
    def calibrate(self, checkerboard=(7, 10), square_size=20.0, min_pairs=5):
        """
        Perform stereo calibration by capturing images from cameras and return updated config dictionary.
        
        Args:
            checkerboard: Tuple of (cols, rows) for checkerboard pattern
            square_size: Size of checkerboard squares in mm
            min_pairs: Minimum number of valid stereo pairs required for calibration
        
        Returns:
            dict: Dictionary with updated calibration data in the format expected by Calibrate.py.
                  Structure: {self.name: {"left": {...}, "right": {...}, "imageSize": ..., "Q": ...}}
        """
        # Ensure cameras are started
        if not self.connected:
            print(f"{self.name}: Starting cameras for calibration...")
            self.start()
        
        # Import calibration constants
        CHECKERBOARD = checkerboard
        SQUARE_SIZE = square_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-6)
        
        print("\n" + "="*60)
        print("STEREO CALIBRATION - IMAGE CAPTURE")
        print("="*60)
        print("Instructions:")
        print("  - Press 's' to capture a stereo pair")
        print("  - Press 'q' to finish capturing and proceed with calibration")
        print(f"  - You need at least {min_pairs} valid pairs with detected checkerboards")
        print("="*60 + "\n")
        
        # Capture stereo pairs interactively
        captured_pairs = []
        pair_count = 0
        
        try:
            while True:
                # Capture frames
                left_frame = self.left_camera.read_frame()
                right_frame = self.right_camera.read_frame()
                
                if left_frame is None or right_frame is None:
                    print("⚠️ Failed to grab one or both frames. Retrying...")
                    continue
                
                # Resize for display (no live detection to avoid lag)
                preview_left = cv2.resize(left_frame, (800, 600))
                preview_right = cv2.resize(right_frame, (800, 600))
                
                # Add status text
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
                    # Only detect checkerboard when capturing (not on every frame)
                    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    
                    retL, cornersL = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
                    retR, cornersR = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)
                    
                    if retL and retR:
                        # Refine corners
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
        
        # Get image shape from first pair
        self.img_shape = captured_pairs[0][0].shape[::-1]
        
        # Prepare object points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE
        
        objpoints, imgpointsL, imgpointsR = [], [], []
        
        # Process captured pairs
        for i, (imgL, imgR, cornersL, cornersR) in enumerate(captured_pairs):
            objpoints.append(objp)
            imgpointsL.append(cornersL.reshape(1, -1, 2))
            imgpointsR.append(cornersR.reshape(1, -1, 2))
            print(f"  ✓ Processed pair {i+1}/{len(captured_pairs)}")
        
        N_OK = len(objpoints)
        print(f"✅ Using {N_OK} valid pairs for calibration")
        
        if N_OK < min_pairs:
            raise RuntimeError(f"Not enough valid pairs ({N_OK}). Need at least {min_pairs}.")
        
        # Initialize intrinsics
        K1 = np.eye(3)
        D1 = np.zeros((4, 1))
        K2 = np.eye(3)
        D2 = np.zeros((4, 1))
        
        print("\n--- Stereo Calibration (Fisheye) ---")
        
        # Fisheye stereo calibration
        # Returns: retval, K1, D1, K2, D2, R, T, rvecs, tvecs
        rms,K1, D1, K2, D2, R, T, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
            objpoints,
            imgpointsL,
            imgpointsR,
            K1, D1, K2, D2,
            self.img_shape,
            criteria=criteria,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        )
        
        print(f"RMS reprojection error: {rms:.4f}")
        
        # Stereo rectification (fisheye)
        R1, R2, P1,P2, self.Q = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2,
            self.img_shape, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0,
            fov_scale=1.2
        )
        
        # Generate rectification maps
        self.left['map_x'], self.left['map_y'] = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, P1, self.img_shape, cv2.CV_32FC1
        )
        self.right['map_x'], self.right['map_y'] = cv2.fisheye.initUndistortRectifyMap(
           K2, D2, R2, P2, self.img_shape, cv2.CV_32FC1
        )
        
        print(f"\n💾 Calibration complete")
        print(f"   Maps shape: {self.left['map_x'].shape}")
        print(f"   Image size: {self.img_shape}")

        # # Create a deep copy of the camera config to preserve all existing settings
        # # Save all calibration parameters for future use/debugging HOLD OVER FROM CURSOR LOOK AWAY
        # updated_camera_config["calibration"] = {
        #     "K1": K1,  # Left camera intrinsic matrix
        #     "D1": D1,  # Left camera distortion coefficients
        #     "K2": K2,  # Right camera intrinsic matrix
        #     "D2": D2,  # Right camera distortion coefficients
        #     "R": R,    # Rotation matrix between cameras
        #     "T": T,    # Translation vector between cameras
        #     "R1": R1,  # Left rectification rotation matrix
        #     "R2": R2,  # Right rectification rotation matrix
        #     "P1": P1,  # Left projection matrix
        #     "P2": P2,  # Right projection matrix
        #     "rms": rms,  # RMS reprojection error
        # }
        
        # Return dictionary in format expected by Calibrate.py
        # Note: dict.update() does shallow merge, so we return the complete config
        # to preserve all existing settings (minDisparity, numDisparitiesK, etc.)
        return {self.name: vars(self)}
    
    def __repr__(self):
        return f"<VISION name={self.name}, left_port={self.left['port']}, right_port={self.right['port']}, connected={self.connected}>"

