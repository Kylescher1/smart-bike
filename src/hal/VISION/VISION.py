import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from ..cam.Camera import Camera, CAMERA_CONFIG


class VISION:
    """
    Vision system for stereo camera depth processing.
    Follows sensor pattern with start(), stop(), and read() methods.
    """
    
    def __init__(self, name="Unnamed VISION", **kwargs):
        """
        Initialize VISION instance.
        Supports nested config structure: camera.left and camera.right
        """
        self.name = name
        self.debug_mode = True
        
        # Load user-provided configuration (following MPU6250 pattern)
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Validate required fields
        if not hasattr(self, 'camera') or not isinstance(self.camera, dict):
            raise KeyError(f"Camera configuration not found for {self.name}")
        
        if 'left' not in self.camera:
            raise KeyError(f"camera.left not specified for {self.name}")
        if 'right' not in self.camera:
            raise KeyError(f"camera.right not specified for {self.name}")
        
        left_cfg = self.camera['left']
        right_cfg = self.camera['right']
        
        if 'port' not in left_cfg:
            raise KeyError(f"camera.left.port not specified for {self.name}")
        if 'port' not in right_cfg:
            raise KeyError(f"camera.right.port not specified for {self.name}")
        
        # Extract camera indices from port values
        self.left_port = left_cfg['port']
        self.right_port = right_cfg['port']
        
        # Store position and z_direction if provided
        self.left_position = left_cfg.get('position', None)
        self.left_z_direction = left_cfg.get('z_direction', None)
        self.right_position = right_cfg.get('position', None)
        self.right_z_direction = right_cfg.get('z_direction', None)
        
        # Default calibration file path
        self.calibration_file = getattr(
            self, 
            'calibration_file', 
            Path(__file__).resolve().parent.parent / "cam" / "calibrate" / "data" / "stereo_calib.npz"
        )
        
        # Initialize camera objects
        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        
        # Depth processing components (initialized in start())
        self.left_map_x = None
        self.left_map_y = None
        self.right_map_x = None
        self.right_map_y = None
        self.image_size = None
        self.Q = None
        self.stereo = None
        
        # Runtime state
        self.connected = False
    
    def _load_calibration(self, filename=None):
        """
        Load calibration data from file.
        Copied from src/hal/cam/calibrate/calib.py
        """
        if filename is None:
            filename = self.calibration_file
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Calibration file not found: {filename}")
        
        data = np.load(filename, allow_pickle=True)
        return (
            data["leftMapX"],
            data["leftMapY"],
            data["rightMapX"],
            data["rightMapY"],
            tuple(data["imageSize"]),
            data["Q"],
        )
    
    def start(self):
        """Open cameras and initialize depth processor."""
        if self.connected:
            return
        
        print(f"{self.name}: Starting vision system...")
        
        # Initialize cameras
        self.left_camera = Camera(self.left_port, CAMERA_CONFIG)
        self.right_camera = Camera(self.right_port, CAMERA_CONFIG)
        
        # Open cameras
        try:
            self.left_camera.open()
            print(f"{self.name}: Left camera opened on port {self.left_port}")
        except Exception as e:
            raise RuntimeError(f"{self.name}: Failed to open left camera: {e}")
        
        try:
            self.right_camera.open()
            print(f"{self.name}: Right camera opened on port {self.right_port}")
        except Exception as e:
            self.left_camera.close()
            raise RuntimeError(f"{self.name}: Failed to open right camera: {e}")
        
        # Load calibration data
        try:
            (
                self.left_map_x,
                self.left_map_y,
                self.right_map_x,
                self.right_map_y,
                self.image_size,
                self.Q,
            ) = self._load_calibration()
            print(f"{self.name}: Calibration data loaded from {self.calibration_file}")
        except Exception as e:
            self.left_camera.close()
            self.right_camera.close()
            raise RuntimeError(f"{self.name}: Failed to load calibration: {e}")
        
        # Initialize stereo matcher
        block_size = 5
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max(16, 16 * 5),  # 80 disparities
            blockSize=max(3, block_size),
            preFilterCap=31,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1,
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
        
        # Clear calibration data
        self.left_map_x = None
        self.left_map_y = None
        self.right_map_x = None
        self.right_map_y = None
        self.image_size = None
        self.Q = None
        self.stereo = None
        
        self.connected = False
        print(f"{self.name}: Vision system stopped")
    
    def _rectify(self, left_frame: np.ndarray, right_frame: np.ndarray):
        """Rectify stereo pair using calibration maps."""
        if self.left_map_x is None or self.left_map_y is None:
            raise RuntimeError("Calibration data not loaded. Call start() first.")
        
        rect_left = cv2.remap(left_frame, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_frame, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
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
        if self.Q is None:
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
                    'calibration_file': str,
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
                    'calibration_file': str(self.calibration_file),
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
                'calibration_file': str(self.calibration_file),
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
                    'calibration_file': str(self.calibration_file),
                    'num_disparities': 0,
                    'error': str(e)
                }
            }
    
    def __repr__(self):
        return f"<VISION name={self.name}, left_port={self.left_port}, right_port={self.right_port}, connected={self.connected}>"

