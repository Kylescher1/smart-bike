import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import os
import glob
import dill
import copy
import time
from pathlib import Path

from ..cam.Camera import Camera, CAMERA_CONFIG
from .depth_processing import DepthProcessor
# from .calibration import run_calibration

# Import RKNNLite and YOLO processing functions for YOLO class
try:
    from yolo.rknn_inference import (
        RKNNLite,
        letterbox,
        process_output,
        draw_detections,
        ByteTrackerWrapper
    )
except ImportError:
    # Fallback: try direct import if yolo module not available
    try:
        from rknnlite.api import RKNNLite
        RKNNLite = RKNNLite
        letterbox = None
        process_output = None
        draw_detections = None
        ByteTrackerWrapper = None
    except ImportError:
        RKNNLite = None
        letterbox = None
        process_output = None
        draw_detections = None
        ByteTrackerWrapper = None


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

        if not hasattr(self, 'smoothingKernel'):
            self.smoothingKernel = 0
        if not hasattr(self, 'confidenceWindow'):
            self.confidenceWindow = 5
        if not hasattr(self, 'confidenceThreshold'):
            self.confidenceThreshold = 0.0

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
        self.depth_processor: Optional[DepthProcessor] = None
        
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
        
        # Get P1 and P2, with defaults if not set (for backward compatibility)
        P1 = getattr(self, 'P1', 8 * 1 * block_size * block_size)  # Default: 8 * channels * blockSize^2
        P2 = getattr(self, 'P2', 32 * 1 * block_size * block_size)  # Default: 32 * channels * blockSize^2
        
        # Map mode integer to OpenCV enum
        sgbm_mode = getattr(self, 'sgbmMode', 2)  # Default to SGBM_3WAY
        mode_map = {
            0: cv2.STEREO_SGBM_MODE_SGBM,
            1: cv2.STEREO_SGBM_MODE_HH,
            2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }
        mode = mode_map.get(sgbm_mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.minDisparity,
            numDisparities=num_disparities,
            blockSize=max(3, block_size),
            P1=P1,
            P2=P2,
            preFilterCap=self.preFilterCap,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            disp12MaxDiff=self.disp12MaxDiff,
            mode=mode,
        )
        
        self._refresh_depth_processor()
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
        self.depth_processor = None
        
        self.connected = False
        print(f"{self.name}: Vision system stopped")
    
    def read(self):
        """
        Capture frames, process depth, and return dictionary with depth_map and metadata.
        
        Returns:
            dict: {
                'depth_map': np.ndarray,
                'disparity_map': np.ndarray,
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
            empty = np.array([])
            return {
                'depth_map': empty,
                'disparity_map': empty,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_disparities': 0,
                    'error': 'Failed to capture frames'
                }
            }
        
        # Process depth pipeline
        try:
            if self.depth_processor is None:
                raise RuntimeError("Depth processor not initialized. Call start() first.")

            depth_map, disparity_map, metadata = self.depth_processor.process_frames(left_frame, right_frame)
            
            return {
                'depth_map': depth_map,
                'disparity_map': disparity_map,
                'metadata': metadata
            }
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error processing depth: {e}")
            empty = np.array([])
            return {
                'depth_map': empty,
                'disparity_map': empty,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_disparities': 0,
                    'error': str(e)
                }
            }
    
    def get_latest_frame(self):
        """
        Returns a 4×N numpy array:
            frame[0] = x  (meters)
            frame[1] = y  (meters)
            frame[2] = z  (depth in meters)
            frame[3] = q  (quality, 1 for valid points)

        Returns empty array if not connected or no valid points.
        """
        if not self.connected:
            return np.zeros((4, 0))
        
        if self.Q is None:
            return np.zeros((4, 0))
        
        # Get depth data
        result = self.read()
        disparity_map = result.get('disparity_map')
        depth_map = result.get('depth_map')
        
        if disparity_map is None or disparity_map.size == 0:
            return np.zeros((4, 0))
        
        # Convert disparity to 3D points
        points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) * 16.0, self.Q)
        
        # Filter out invalid points
        valid_mask = (depth_map > 0) & np.isfinite(points_3d[:, :, 2])
        valid_mask = valid_mask & (points_3d[:, :, 2] > 0)
        
        # Extract valid points
        points = points_3d[valid_mask]
        
        if len(points) == 0:
            return np.zeros((4, 0))
        
        # Extract x, y, z coordinates
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]  # depth
        
        # Quality: 1 for all valid points (can be enhanced with confidence metrics)
        qs = np.ones(len(points), dtype=np.float32)
        
        return np.vstack([xs, ys, zs, qs])
    
    def get_pointcloud(self, max_points=None, filters=None):
        """
        Generate 3D point cloud from current frame.
        
        Args:
            max_points: Maximum number of points to return (None = all points)
            filters: Optional dict with filter parameters:
                {
                    'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z': float or None,
                    'min_dist', 'max_dist': float or None  # distance from origin
                }
        
        Returns:
            dict: {
                'points': np.ndarray (Nx3),  # 3D coordinates
                'colors': np.ndarray (Nx3),   # BGR colors
                'num_points': int,
                'metadata': dict
            }
        """
        if not self.connected:
            raise RuntimeError(f"{self.name}: Not connected. Call start() first.")
        
        if self.Q is None:
            raise RuntimeError(f"{self.name}: Q matrix not available. Calibration required.")
        
        # Get depth data
        result = self.read()
        disparity_map = result.get('disparity_map')
        depth_map = result.get('depth_map')
        
        if disparity_map is None or disparity_map.size == 0:
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'num_points': 0,
                'metadata': result.get('metadata', {})
            }
        
        # Get left camera frame for colors
        left_frame = self.left_camera.read_frame()
        if left_frame is None:
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'num_points': 0,
                'metadata': result.get('metadata', {})
            }
        
        # Convert disparity to 3D points
        points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) * 16.0, self.Q)
        
        # Filter out invalid points
        valid_mask = (depth_map > 0) & np.isfinite(points_3d[:, :, 2])
        valid_mask = valid_mask & (points_3d[:, :, 2] > 0)
        
        # Get valid points and colors
        h, w = disparity_map.shape[:2]
        if left_frame.shape[:2] != (h, w):
            left_frame = cv2.resize(left_frame, (w, h))
        
        points = points_3d[valid_mask]
        colors = left_frame[valid_mask]
        
        if len(points) == 0:
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'num_points': 0,
                'metadata': result.get('metadata', {})
            }
        
        # Apply optional filters
        if filters is not None:
            filter_mask = np.ones(len(points), dtype=bool)
            
            if filters.get('min_x') is not None:
                filter_mask = filter_mask & (points[:, 0] >= filters['min_x'])
            if filters.get('max_x') is not None:
                filter_mask = filter_mask & (points[:, 0] <= filters['max_x'])
            if filters.get('min_y') is not None:
                filter_mask = filter_mask & (points[:, 1] >= filters['min_y'])
            if filters.get('max_y') is not None:
                filter_mask = filter_mask & (points[:, 1] <= filters['max_y'])
            if filters.get('min_z') is not None:
                filter_mask = filter_mask & (points[:, 2] >= filters['min_z'])
            if filters.get('max_z') is not None:
                filter_mask = filter_mask & (points[:, 2] <= filters['max_z'])
            
            # Distance filter
            if filters.get('min_dist') is not None or filters.get('max_dist') is not None:
                distances = np.linalg.norm(points, axis=1)
                if filters.get('min_dist') is not None:
                    filter_mask = filter_mask & (distances >= filters['min_dist'])
                if filters.get('max_dist') is not None:
                    filter_mask = filter_mask & (distances <= filters['max_dist'])
            
            points = points[filter_mask]
            colors = colors[filter_mask]
        
        # Subsample if requested
        if max_points is not None and len(points) > max_points:
            subsample = len(points) // max_points
            points = points[::subsample]
            colors = colors[::subsample]
        
        metadata = result.get('metadata', {})
        metadata['pointcloud_num_points'] = len(points)
        
        return {
            'points': points,
            'colors': colors,
            'num_points': len(points),
            'metadata': metadata
        }
    
    def calibrate(self, checkerboard=(7, 10), square_size=20.0, min_pairs=10):
        print("VISION/calibrate.py has been deprecated I hate kyle")
        return {}
        # """
        # Perform stereo calibration by capturing images from cameras and return updated config dictionary.
        #
        # Args:
        #     checkerboard: Tuple of (cols, rows) for checkerboard pattern
        #     square_size: Size of checkerboard squares in mm
        #     min_pairs: Minimum number of valid stereo pairs required for calibration
        #
        # Returns:
        #     dict: Dictionary with updated calibration data in the format expected by Calibrate.py.
        #           Structure: {self.name: {"left": {...}, "right": {...}, "imageSize": ..., "Q": ...}}
        # """
        # result = run_calibration(self, checkerboard=checkerboard, square_size=square_size, min_pairs=min_pairs)
        # self._refresh_depth_processor()
        # return result
    
    def debug(self):
        """
        Debug mode that continuously displays the depth map.
        Press 'o' to start recording, 'p' to stop recording, 'q' to quit.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start debug mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting debug mode - displaying depth map...")
        print(f"{self.name}: Press 'c' to output point cloud | 'o' to start recording | 'p' to stop | 'q' to exit")
        
        window_name = f"{self.name} - Depth Map Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Video recording state
        video_writer = None
        recording = False
        video_counter = 0
        frame_size = None
        
        try:
            while True:
                # Get depth data
                result = self.read()
                depth_map = result.get('depth_map')
                metadata = result.get('metadata', {})
                
                # Check for errors
                if 'error' in metadata:
                    print(f"{self.name}: Error in debug mode: {metadata['error']}")
                    continue
                
                if depth_map is None or depth_map.size == 0:
                    print(f"{self.name}: No depth data available")
                    continue
                
                # Normalize depth map for visualization (0-255 range)
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_normalized.astype(np.uint8)
                
                # Apply colormap for better visualization
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                # Store frame size for video writer initialization
                if frame_size is None:
                    frame_size = (depth_colored.shape[1], depth_colored.shape[0])
                
                # Add metadata text overlay
                timestamp = metadata.get('timestamp', 'N/A')
                num_disp = metadata.get('num_disparities', 'N/A')
                cv2.putText(depth_colored, f"Timestamp: {timestamp}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(depth_colored, f"Num Disparities: {num_disp}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add recording indicator
                if recording:
                    cv2.circle(depth_colored, (depth_colored.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(depth_colored, "REC", (depth_colored.shape[1] - 70, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Write frame to video if recording
                if recording and video_writer is not None:
                    video_writer.write(depth_colored)
                
                # Display the depth map
                cv2.imshow(window_name, depth_colored)
                
                # Check for keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):  # 'c' for point cloud
                    # Generate and output point cloud
                    try:
                        pc_result = self.get_pointcloud(max_points=50000)
                        if pc_result['num_points'] > 0:
                            points = pc_result['points']
                            colors = pc_result['colors']
                            print(f"\n{self.name}: Point Cloud Output")
                            print(f"  Total points: {pc_result['num_points']:,}")
                            print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] meters")
                            print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] meters")
                            print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] meters")
                            
                            # Calculate center point
                            center = np.mean(points, axis=0)
                            print(f"  Center point: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] meters")
                            
                            # Calculate distance statistics
                            distances = np.linalg.norm(points, axis=1)
                            print(f"  Distance from origin: min={distances.min():.3f}m, max={distances.max():.3f}m, mean={distances.mean():.3f}m")
                            
                            # Show first few points as example
                            print(f"  Sample points (first 5):")
                            for i in range(min(5, len(points))):
                                print(f"    Point {i}: [{points[i, 0]:.3f}, {points[i, 1]:.3f}, {points[i, 2]:.3f}] "
                                      f"Color: [{colors[i, 0]}, {colors[i, 1]}, {colors[i, 2]}]")
                        else:
                            print(f"{self.name}: No points in point cloud")
                    except Exception as e:
                        print(f"{self.name}: Error generating point cloud: {e}")
                elif key == ord('o'):
                    if not recording:
                        # Start recording
                        video_counter += 1
                        filename = f"disparity_recording_{video_counter}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
                        recording = True
                        print(f"🔴 Recording started: {filename}")
                    else:
                        print("⚠️ Already recording. Press 'p' to stop first.")
                elif key == ord('p'):
                    if recording:
                        # Stop recording
                        recording = False
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
                        print(f"⏹️ Recording stopped: disparity_recording_{video_counter}.avi")
                    else:
                        print("⚠️ Not currently recording.")
                    
        except KeyboardInterrupt:
            print(f"\n{self.name}: Debug mode interrupted by user")
        except Exception as e:
            print(f"{self.name}: Error in debug mode: {e}")
        finally:
            # Clean up video writer if still recording
            if video_writer is not None:
                video_writer.release()
                print(f"⏹️ Recording stopped (cleanup): disparity_recording_{video_counter}.avi")
            
            cv2.destroyWindow(window_name)
            print(f"{self.name}: Debug mode ended")
    
    def __repr__(self):
        return f"<VISION name={self.name}, left_port={self.left['port']}, right_port={self.right['port']}, connected={self.connected}>"

    def _refresh_depth_processor(self):

        #check if left and right are in the config
        if not hasattr(self, 'left') or not hasattr(self, 'right'):
            self.depth_processor = None
            return
        #check if stereo is in the config
        if self.stereo is None:
            self.depth_processor = None
            return
        
        # Recreate the depth processor so it always uses the latest stereo matcher and calibration maps.
        self.depth_processor = DepthProcessor(
            self.left,
            self.right,
            self.stereo,
            getattr(self, 'Q', None),
            debug=self.debug_mode,
            downSample=getattr(self, 'downSample', 0),
            crop=getattr(self, 'crop', 0),
            nearCutoff=getattr(self, 'nearCutoff', 0),
            farCutoff=getattr(self, 'farCutoff', 0),
            useMorph=getattr(self, 'useMorph', False),
            morphIter=getattr(self, 'morphIter', 5),
            useWLS=getattr(self, 'useWLS', False),
            wlsLambda=getattr(self, 'wlsLambda', 8000.0),
            wlsSigma=getattr(self, 'wlsSigma', 1.5),
            smoothingKernel=getattr(self, 'smoothingKernel', 0),
            confidenceWindow=getattr(self, 'confidenceWindow', 5),
            confidenceThreshold=getattr(self, 'confidenceThreshold', 0.0),
        )

class YOLO: 


    def __init__(self, name="Unnamed YOLO", **kwargs):
        """
        Initialize YOLO instance.
        Supports config via kwargs unpacking (following sensor pattern).
        
        Required kwargs:
            model_path: Path to RKNN model file (.rknn)
        
        Optional kwargs:
            conf_threshold: Confidence threshold (default: 0.25)
            imgsz: Input image size (default: 640)
            track_enabled: Enable object tracking (default: True)
            track_thresh: Tracking confidence threshold (default: 0.5)
            track_high_thresh: High confidence threshold for tracking (default: 0.6)
            track_match_thresh: IoU threshold for track matching (default: 0.8)
            frame_rate: Frame rate for tracking (default: 30)
            track_buffer: Number of frames to keep lost tracks (default: 30)
        """
        if RKNNLite is None or process_output is None:
            raise ImportError("YOLO dependencies not available. Please ensure yolo module is accessible.")
        
        # Load user-provided configuration (following MPU6250 pattern)
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.name = name
        self.debug_mode = False
        self.connected = False
        self.rknn = None
        self.tracker = None
        
        # Default values
        self.conf_threshold = getattr(self, 'conf_threshold', 0.25)
        self.imgsz = getattr(self, 'imgsz', 640)
        self.track_enabled = getattr(self, 'track_enabled', True)
        
        # Pre-allocated buffer for inference
        self.img_input_buffer = None

    def start(self):
        """Initialize and load RKNN model. Reads model_path from dill config."""
        if self.connected:
            return
        
        if RKNNLite is None:
            raise ImportError("RKNNLite not available. Please install rknnlite or ensure yolo module is accessible.")
        
        print(f"{self.name}: Starting YOLO...")
        
        # Get model path from config
        if not hasattr(self, 'model_path'):
            raise ValueError(f"{self.name}: model_path not provided in configuration")
        
        model_path = Path(self.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"{self.name}: Model file not found: {model_path}")
        
        print(f"📦 Loading RKNN model: {model_path}")
        
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            self.rknn = None
            raise RuntimeError(f"{self.name}: Failed to load RKNN model: {ret}")
        
        ret = self.rknn.init_runtime(target=None, core_mask=0)  # 0 = auto select core
        if ret != 0:
            self.rknn.release()
            self.rknn = None
            raise RuntimeError(f"{self.name}: Failed to initialize runtime: {ret}")
        
        # Initialize tracker if enabled
        if self.track_enabled and ByteTrackerWrapper is not None:
            self.tracker = ByteTrackerWrapper(
                track_thresh=getattr(self, 'track_thresh', 0.5),
                high_thresh=getattr(self, 'track_high_thresh', 0.6),
                match_thresh=getattr(self, 'track_match_thresh', 0.8),
                frame_rate=getattr(self, 'frame_rate', 30),
                track_buffer=getattr(self, 'track_buffer', 30)
            )
            print(f"✅ ByteTrack tracking enabled")
        else:
            self.tracker = None
        
        print("✅ RKNN model loaded successfully")
        self.connected = True
        print(f"{self.name}: YOLO started successfully")

    def stop(self):
        """Release RKNN resources and cleanup."""
        if not self.connected:
            return
        
        print(f"{self.name}: Stopping YOLO...")
        
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
        
        self.tracker = None
        self.img_input_buffer = None
        self.connected = False
        print(f"{self.name}: YOLO stopped successfully")

    def read(self, frame):
        """
        Process a frame through YOLO inference and return detections.
        
        Args:
            frame: Input frame (numpy array, BGR format)
        
        Returns:
            dict: {
                'detections': List of detection dicts with keys:
                    - 'bbox': [x1, y1, x2, y2]
                    - 'score': float
                    - 'class_id': int
                    - 'class_name': str
                    - 'track_id': int (if tracking enabled)
                'annotated_frame': Frame with detections drawn (optional, if draw_detections available)
                'num_detections': int
                'num_tracks': int (if tracking enabled)
                'inference_time_ms': float
                'metadata': dict with additional info
            }
        """
        if not self.connected:
            raise RuntimeError(f"{self.name}: Not connected. Call start() first.")
        
        if frame is None:
            return {
                'detections': [],
                'annotated_frame': None,
                'num_detections': 0,
                'num_tracks': 0,
                'inference_time_ms': 0.0,
                'metadata': {'error': 'No frame provided'}
            }
        
        try:
            h_orig, w_orig = frame.shape[:2]
            
            # Preprocess frame
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(self.imgsz, self.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Pre-allocate buffer
            if self.img_input_buffer is None or self.img_input_buffer.shape != (1, self.imgsz, self.imgsz, 3):
                self.img_input_buffer = np.zeros((1, self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.img_input_buffer[0] = img_rgb.astype(np.uint8)
            img_input = self.img_input_buffer
            
            # Run inference
            inference_start = time.time()
            try:
                outputs = self.rknn.inference([img_input])
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Inference failed: {e}")
                return {
                    'detections': [],
                    'annotated_frame': frame.copy(),
                    'num_detections': 0,
                    'num_tracks': 0,
                    'inference_time_ms': 0.0,
                    'metadata': {'error': str(e)}
                }
            
            inference_time_ms = (time.time() - inference_start) * 1000
            
            # Process output
            detections = []
            if outputs is not None:
                try:
                    detections = process_output(outputs, conf_threshold=self.conf_threshold, img_shape=(self.imgsz, self.imgsz))
                except Exception as e:
                    if self.debug_mode:
                        print(f"{self.name}: Failed to process output: {e}")
                    detections = []
            
            # Scale boxes back to original image size
            if detections:
                scale = min(self.imgsz / w_orig, self.imgsz / h_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                pad_x = (self.imgsz - new_w) / 2
                pad_y = (self.imgsz - new_h) / 2
                
                boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
                
                for i, det in enumerate(detections):
                    det['bbox'] = [int(float(boxes[i, 0])), int(float(boxes[i, 1])), 
                                  int(float(boxes[i, 2])), int(float(boxes[i, 3]))]
            
            # Update tracker if enabled
            num_tracks = 0
            if self.tracker is not None and self.track_enabled:
                tracked_detections = self.tracker.update(detections)
                detections = tracked_detections
                num_tracks = len(self.tracker.tracked_tracks) if self.tracker else 0
            
            # Draw detections if draw_detections function is available
            annotated_frame = None
            if draw_detections is not None:
                try:
                    annotated_frame = draw_detections(frame.copy(), detections, tracker=self.tracker if self.track_enabled else None)
                except Exception as e:
                    if self.debug_mode:
                        print(f"{self.name}: Failed to draw detections: {e}")
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()
            
            return {
                'detections': detections,
                'annotated_frame': annotated_frame,
                'num_detections': len(detections),
                'num_tracks': num_tracks,
                'inference_time_ms': inference_time_ms,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'frame_shape': (h_orig, w_orig),
                    'imgsz': self.imgsz,
                    'conf_threshold': self.conf_threshold,
                    'track_enabled': self.track_enabled
                }
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error processing frame: {e}")
                import traceback
                traceback.print_exc()
            return {
                'detections': [],
                'annotated_frame': frame.copy() if frame is not None else None,
                'num_detections': 0,
                'num_tracks': 0,
                'inference_time_ms': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def __getstate__(self):
        """
        Custom serialization for dill.
        Excludes non-serializable RKNN instance, only saves model_path and config.
        """
        state = self.__dict__.copy()
        # Remove non-serializable objects
        state.pop('rknn', None)
        state.pop('tracker', None)
        state.pop('img_input_buffer', None)
        # Convert Path objects to strings for serialization
        if 'model_path' in state and isinstance(state['model_path'], Path):
            state['model_path'] = str(state['model_path'])
        return state
    
    def __setstate__(self, state):
        """
        Custom deserialization for dill.
        Restores state and sets rknn to None (must call start() to reload model).
        """
        self.__dict__.update(state)
        # RKNN instance must be reloaded via start() after deserialization
        self.rknn = None
        self.tracker = None
        self.img_input_buffer = None
        self.connected = False
    
    def __repr__(self):
        model_path_str = getattr(self, 'model_path', 'Not set')
        return f"<YOLO name={self.name}, model_path={model_path_str}, connected={self.connected}>" 

class DEPTH:
    