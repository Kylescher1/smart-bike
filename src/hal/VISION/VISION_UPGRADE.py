"""
Vision System Upgrade - Complete Refactor

This module implements a new vision system architecture with:
- YOLO object detection
- ROI-based stereo depth estimation
- Temporal smoothing (EMA)
- Thread-safe buffering
- Structured object detection output

Architecture:
- Camera class: Handles stereo camera capture and rectification
- Yolo class: Runs YOLO inference on camera stream
- Depth class: Computes depth for detected objects using SGBM stereo
- VISION class: Main interface with start(), stop(), read(), debug()
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import deque
from pathlib import Path

from ..cam.Camera import Camera, CAMERA_CONFIG

# Import YOLO dependencies
try:
    # Try importing from yolo.rknn_inference (if it exists)
    from yolo.rknn_inference import (
        RKNNLite,
        letterbox,
        process_output,
        draw_detections,
        ByteTrackerWrapper
    )
    YOLO_AVAILABLE = True
except ImportError:
    try:
        # Try importing from yolo.yolo (where functions are actually defined)
        from yolo.yolo import (
            letterbox,
            process_output,
            draw_detections,
            ByteTrackerWrapper
        )
        # RKNNLite is imported in yolo.yolo but may not be exported, import directly
        from rknnlite.api import RKNNLite
        YOLO_AVAILABLE = True
    except ImportError:
        try:
            # Fallback: try direct import if yolo module not available
            from rknnlite.api import RKNNLite
            RKNNLite = RKNNLite
            letterbox = None
            process_output = None
            draw_detections = None
            ByteTrackerWrapper = None
            YOLO_AVAILABLE = False
        except ImportError:
            RKNNLite = None
            letterbox = None
            process_output = None
            draw_detections = None
            ByteTrackerWrapper = None
            YOLO_AVAILABLE = False


# ============================================================================
# Camera Class
# ============================================================================

class VisionCamera:
    """
    Handles capture from left and right cameras.
    Applies rectification and undistortion using calibration data.
    """
    
    def __init__(self, left_config: Dict, right_config: Dict):
        """
        Initialize camera system.
        
        Args:
            left_config: Configuration dict for left camera with keys:
                - port: Camera port/index
                - map_x: Rectification map X
                - map_y: Rectification map Y
            right_config: Configuration dict for right camera (same structure)
        """
        self.left_config = left_config
        self.right_config = right_config
        
        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        
        # Extract calibration maps
        self.left_map_x = left_config.get('map_x')
        self.left_map_y = left_config.get('map_y')
        self.right_map_x = right_config.get('map_x')
        self.right_map_y = right_config.get('map_y')
        
        # Validate maps
        if self.left_map_x is None or self.left_map_y is None:
            raise ValueError("Left camera calibration maps (map_x, map_y) not provided")
        if self.right_map_x is None or self.right_map_y is None:
            raise ValueError("Right camera calibration maps (map_x, map_y) not provided")
    
    def start(self):
        """Open cameras."""
        if self.left_camera is not None:
            return
        
        self.left_camera = Camera(self.left_config['port'], CAMERA_CONFIG)
        self.right_camera = Camera(self.right_config['port'], CAMERA_CONFIG)
        
        try:
            self.left_camera.open()
            print(f"✅ Left camera opened on port {self.left_config['port']}")
        except Exception as e:
            raise RuntimeError(f"Failed to open left camera: {e}")
        
        try:
            self.right_camera.open()
            print(f"✅ Right camera opened on port {self.right_config['port']}")
        except Exception as e:
            if self.left_camera:
                self.left_camera.close()
            raise RuntimeError(f"Failed to open right camera: {e}")
    
    def stop(self):
        """Close cameras."""
        if self.left_camera:
            self.left_camera.close()
            self.left_camera = None
        
        if self.right_camera:
            self.right_camera.close()
            self.right_camera = None
    
    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read and rectify frames from both cameras.
        
        Returns:
            Tuple of (left_rectified, right_rectified) frames, or (None, None) on error
        """
        if self.left_camera is None or self.right_camera is None:
            return None, None
        
        left_raw = self.left_camera.read_frame()
        right_raw = self.right_camera.read_frame()
        
        if left_raw is None or right_raw is None:
            return None, None
        
        # Apply rectification
        left_rect = cv2.remap(left_raw, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        
        return left_rect, right_rect


# ============================================================================
# Yolo Class
# ============================================================================

class VisionYolo:
    """
    Loads YOLO model and runs inference on camera stream.
    Outputs bounding boxes, segmentation masks (if available), class IDs, and confidence scores.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 imgsz: int = 640, track_enabled: bool = True, **track_kwargs):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to RKNN model file
            conf_threshold: Confidence threshold for detections
            imgsz: Input image size for YOLO
            track_enabled: Enable object tracking
            **track_kwargs: Additional tracking parameters
        """
        if not YOLO_AVAILABLE or RKNNLite is None or process_output is None:
            raise ImportError("YOLO dependencies not available. Please ensure yolo module is accessible.")
        
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.track_enabled = track_enabled
        self.track_kwargs = track_kwargs
        
        self.rknn: Optional[RKNNLite] = None
        self.tracker: Optional[ByteTrackerWrapper] = None
        self.img_input_buffer = None
        self.connected = False
    
    def start(self):
        """Load RKNN model and initialize tracker."""
        if self.connected:
            return
        
        print(f"📦 Loading RKNN model: {self.model_path}")
        
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            self.rknn = None
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        ret = self.rknn.init_runtime(target=None, core_mask=0)
        if ret != 0:
            self.rknn.release()
            self.rknn = None
            raise RuntimeError(f"Failed to initialize runtime: {ret}")
        
        # Initialize tracker if enabled
        if self.track_enabled and ByteTrackerWrapper is not None:
            self.tracker = ByteTrackerWrapper(
                track_thresh=self.track_kwargs.get('track_thresh', 0.5),
                high_thresh=self.track_kwargs.get('track_high_thresh', 0.6),
                match_thresh=self.track_kwargs.get('track_match_thresh', 0.8),
                frame_rate=self.track_kwargs.get('frame_rate', 30),
                track_buffer=self.track_kwargs.get('track_buffer', 30)
            )
            print("✅ ByteTrack tracking enabled")
        else:
            self.tracker = None
        
        print("✅ RKNN model loaded successfully")
        self.connected = True
    
    def stop(self):
        """Release RKNN resources."""
        if not self.connected:
            return
        
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
        
        self.tracker = None
        self.img_input_buffer = None
        self.connected = False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO inference on frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of detection dicts with keys: bbox, score, class_id, class_name, track_id (if tracking)
        """
        if not self.connected or frame is None:
            return []
        
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
            outputs = self.rknn.inference([img_input])
            
            # Process output
            detections = []
            if outputs is not None:
                detections = process_output(outputs, conf_threshold=self.conf_threshold, 
                                          img_shape=(self.imgsz, self.imgsz))
            
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
            if self.tracker is not None and self.track_enabled:
                detections = self.tracker.update(detections)
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []


# ============================================================================
# Depth Class
# ============================================================================

class VisionDepth:
    """
    Handles SGBM initialization, ROI disparity computation, depth calculation, and temporal smoothing (EMA).
    """
    
    def __init__(self, stereo_config: Dict, q_matrix: Optional[np.ndarray] = None,
                 baseline: float = 0.12, focal_length_px: float = 800.0,
                 ema_alpha: float = 0.3, roi_expansion: int = 10):
        """
        Initialize depth processor.
        
        Args:
            stereo_config: SGBM configuration dict with keys:
                - minDisparity, numDisparitiesK, blockSize, P1, P2, etc.
            q_matrix: Q matrix from stereo calibration (optional, for reprojection)
            baseline: Stereo baseline in meters
            focal_length_px: Focal length in pixels
            ema_alpha: EMA smoothing factor (0-1, higher = less smoothing)
            roi_expansion: Pixels to expand ROI around bounding box
        """
        self.stereo_config = stereo_config
        self.q_matrix = q_matrix
        self.baseline = baseline
        self.focal_length_px = focal_length_px
        self.ema_alpha = ema_alpha
        self.roi_expansion = roi_expansion
        
        # Initialize SGBM matcher
        self.stereo: Optional[cv2.StereoSGBM] = None
        
        # Temporal smoothing: track depth per object ID
        self.depth_history: Dict[int, float] = {}  # track_id -> smoothed_depth
    
    def start(self):
        """Initialize SGBM stereo matcher."""
        if self.stereo is not None:
            return
        
        block_size = self.stereo_config.get('blockSize', 11)
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        num_disparities = max(16, 16 * self.stereo_config.get('numDisparitiesK', 2))
        
        P1 = self.stereo_config.get('P1', 8 * 1 * block_size * block_size)
        P2 = self.stereo_config.get('P2', 32 * 1 * block_size * block_size)
        
        sgbm_mode = self.stereo_config.get('sgbmMode', 2)
        mode_map = {
            0: cv2.STEREO_SGBM_MODE_SGBM,
            1: cv2.STEREO_SGBM_MODE_HH,
            2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }
        mode = mode_map.get(sgbm_mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.stereo_config.get('minDisparity', 0),
            numDisparities=num_disparities,
            blockSize=max(3, block_size),
            P1=P1,
            P2=P2,
            preFilterCap=self.stereo_config.get('preFilterCap', 43),
            uniquenessRatio=self.stereo_config.get('uniquenessRatio', 1),
            speckleWindowSize=self.stereo_config.get('speckleWindowSize', 196),
            speckleRange=self.stereo_config.get('speckleRange', 34),
            disp12MaxDiff=self.stereo_config.get('disp12MaxDiff', 18),
            mode=mode,
        )
        
        print("✅ SGBM stereo matcher initialized")
    
    def stop(self):
        """Cleanup."""
        self.stereo = None
        self.depth_history.clear()
    
    def compute_roi_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray,
                              bbox: List[int]) -> Optional[np.ndarray]:
        """
        Compute disparity only in ROI around bounding box.
        
        Args:
            left_rect: Rectified left frame
            right_rect: Rectified right frame
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Disparity map for ROI, or None on error
        """
        if self.stereo is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = left_rect.shape[:2]
        
        # Expand ROI
        x1 = max(0, x1 - self.roi_expansion)
        y1 = max(0, y1 - self.roi_expansion)
        x2 = min(w, x2 + self.roi_expansion)
        y2 = min(h, y2 + self.roi_expansion)
        
        # Extract ROI
        left_roi = left_rect[y1:y2, x1:x2]
        right_roi = right_rect[y1:y2, x1:x2]
        
        if left_roi.size == 0 or right_roi.size == 0:
            return None
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY) if len(left_roi.shape) == 3 else left_roi
        gray_right = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY) if len(right_roi.shape) == 3 else right_roi
        
        # Compute disparity
        disparity = self.stereo.compute(gray_left, gray_right)
        disparity = disparity.astype(np.float32) / 16.0
        disparity[disparity < 0] = 0
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray, method: str = 'median') -> Optional[float]:
        """
        Convert disparity to depth using specified method.
        
        Args:
            disparity: Disparity map
            method: 'median', 'mean', or 'q_matrix' (if Q matrix available)
        
        Returns:
            Depth in meters, or None if invalid
        """
        if disparity is None or disparity.size == 0:
            return None
        
        valid_disparity = disparity[disparity > 0]
        if valid_disparity.size == 0:
            return None
        
        # Remove outliers using z-score thresholding
        if len(valid_disparity) > 3:
            mean_disp = np.mean(valid_disparity)
            std_disp = np.std(valid_disparity)
            if std_disp > 0:
                z_scores = np.abs((valid_disparity - mean_disp) / std_disp)
                valid_disparity = valid_disparity[z_scores < 2.0]  # 2 sigma threshold
        
        if valid_disparity.size == 0:
            return None
        
        # Compute median disparity
        median_disp = np.median(valid_disparity)
        
        if median_disp <= 0:
            return None
        
        # Convert to depth: Z = (f * B) / d
        depth = (self.focal_length_px * self.baseline) / median_disp
        
        return float(depth)
    
    def estimate_depth(self, left_rect: np.ndarray, right_rect: np.ndarray,
                      bbox: List[int], track_id: Optional[int] = None) -> Optional[float]:
        """
        Estimate depth for a detected object.
        
        Args:
            left_rect: Rectified left frame
            right_rect: Rectified right frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Optional track ID for temporal smoothing
        
        Returns:
            Smoothed depth in meters, or None if invalid
        """
        # Compute ROI disparity
        disparity = self.compute_roi_disparity(left_rect, right_rect, bbox)
        if disparity is None:
            return None
        
        # Convert to depth
        depth = self.disparity_to_depth(disparity)
        if depth is None:
            return None
        
        # Apply temporal smoothing (EMA) if track_id provided
        if track_id is not None:
            if track_id in self.depth_history:
                # EMA: new_depth = alpha * current + (1 - alpha) * previous
                depth = self.ema_alpha * depth + (1 - self.ema_alpha) * self.depth_history[track_id]
            
            self.depth_history[track_id] = depth
        
        return depth


# ============================================================================
# Main VISION Class
# ============================================================================

class VISION:
    """
    Main vision system interface.
    Provides start(), stop(), read(), and debug() methods.
    """
    
    def __init__(self, name: str = "Unnamed VISION", **kwargs):
        """
        Initialize VISION system.
        
        Args:
            name: System name
            **kwargs: Configuration dict with keys:
                - camera.left: Left camera config
                - camera.right: Right camera config
                - camera.yolo: YOLO config (model_path, conf_threshold, etc.)
                - SGBM parameters (minDisparity, blockSize, etc.)
                - Q: Q matrix from calibration
                - baseline: Stereo baseline (meters)
                - focal_length_px: Focal length (pixels)
                - ema_alpha: EMA smoothing factor
                - roi_expansion: ROI expansion pixels
                - buffer_size: Circular buffer size
        """
        self.name = name
        self.debug_mode = True
        
        # Load configuration
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Extract camera configs
        camera_config = kwargs.get('camera', {})
        self.left_config = camera_config.get('left', {})
        self.right_config = camera_config.get('right', {})
        self.yolo_config = camera_config.get('yolo', {})
        
        # Extract stereo config (all SGBM params)
        self.stereo_config = {}
        stereo_keys = ['minDisparity', 'numDisparitiesK', 'blockSize', 'P1', 'P2',
                      'preFilterCap', 'uniquenessRatio', 'speckleWindowSize',
                      'speckleRange', 'disp12MaxDiff', 'sgbmMode']
        for key in stereo_keys:
            if hasattr(self, key):
                self.stereo_config[key] = getattr(self, key)
        
        # Depth parameters
        self.baseline = getattr(self, 'baseline', 0.12)  # meters
        self.focal_length_px = getattr(self, 'focal_length_px', 800.0)  # pixels
        self.ema_alpha = getattr(self, 'ema_alpha', 0.3)
        self.roi_expansion = getattr(self, 'roi_expansion', 10)
        
        # Q matrix
        self.Q = getattr(self, 'Q', None)
        
        # Buffer configuration
        self.buffer_size = getattr(self, 'buffer_size', 10)
        
        # Initialize components
        self.camera: Optional[VisionCamera] = None
        self.yolo: Optional[VisionYolo] = None
        self.depth: Optional[VisionDepth] = None
        
        # Runtime state
        self.connected = False
        self.data_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Debug state
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
        self.last_disparity: Optional[np.ndarray] = None
        self.last_yolo_frame: Optional[np.ndarray] = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
    
    def start(self):
        """Initialize all components and start background processing thread."""
        if self.connected:
            return
        
        print(f"{self.name}: Starting vision system...")
        
        # Initialize components
        self.camera = VisionCamera(self.left_config, self.right_config)
        self.camera.start()
        
        # Initialize YOLO
        if not self.yolo_config or 'model_path' not in self.yolo_config:
            raise ValueError("YOLO model_path not provided in camera.yolo config")
        
        self.yolo = VisionYolo(
            model_path=self.yolo_config['model_path'],
            conf_threshold=self.yolo_config.get('conf_threshold', 0.25),
            imgsz=self.yolo_config.get('imgsz', 640),
            track_enabled=self.yolo_config.get('track_enabled', True),
            track_thresh=self.yolo_config.get('track_thresh', 0.5),
            track_high_thresh=self.yolo_config.get('track_high_thresh', 0.6),
            track_match_thresh=self.yolo_config.get('track_match_thresh', 0.8),
            frame_rate=self.yolo_config.get('frame_rate', 30),
            track_buffer=self.yolo_config.get('track_buffer', 30)
        )
        self.yolo.start()
        
        # Initialize depth processor
        self.depth = VisionDepth(
            stereo_config=self.stereo_config,
            q_matrix=self.Q,
            baseline=self.baseline,
            focal_length_px=self.focal_length_px,
            ema_alpha=self.ema_alpha,
            roi_expansion=self.roi_expansion
        )
        self.depth.start()
        
        # Start background processing thread
        self.stop_event.clear()
        self.data_thread = threading.Thread(target=self._data_collector, daemon=True)
        self.data_thread.start()
        
        self.connected = True
        print(f"{self.name}: Vision system started successfully")
    
    def stop(self):
        """Stop thread, release cameras, and cleanup."""
        if not self.connected:
            return
        
        print(f"{self.name}: Stopping vision system...")
        
        self.connected = False
        self.stop_event.set()
        
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=2)
        
        if self.camera:
            self.camera.stop()
            self.camera = None
        
        if self.yolo:
            self.yolo.stop()
            self.yolo = None
        
        if self.depth:
            self.depth.stop()
            self.depth = None
        
        print(f"{self.name}: Vision system stopped")
    
    def _data_collector(self):
        """Background thread to continuously process frames."""
        print(f"{self.name}: Data collector started.")
        
        object_id_counter = 0  # Incremental ID for objects
        
        while not self.stop_event.is_set():
            try:
                # Read frames
                left_rect, right_rect = self.camera.read_frames()
                if left_rect is None or right_rect is None:
                    time.sleep(0.01)
                    continue
                
                # Store for debug
                self.last_left_frame = left_rect.copy()
                self.last_right_frame = right_rect.copy()
                
                # Run YOLO detection on left frame
                detections = self.yolo.detect(left_rect)
                
                # Process each detection: compute depth and angles
                objects = []
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Compute depth
                    track_id = det.get('track_id', None)
                    depth = self.depth.estimate_depth(left_rect, right_rect, bbox, track_id)
                    
                    if depth is None:
                        continue  # Skip if depth estimation failed
                    
                    # Compute angles (theta = horizontal, alpha = vertical)
                    h, w = left_rect.shape[:2]
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    
                    # Convert pixel coordinates to angles
                    # Assuming camera FOV (can be configured)
                    fov_h = getattr(self, 'fov_horizontal', 60.0)  # degrees
                    fov_v = getattr(self, 'fov_vertical', 45.0)  # degrees
                    
                    # Pixel to angle conversion
                    theta = ((center_x - w / 2.0) / w) * fov_h  # horizontal angle
                    alpha = ((center_y - h / 2.0) / h) * fov_v  # vertical angle
                    
                    # Use track_id as object ID, or assign new ID
                    obj_id = track_id if track_id is not None else object_id_counter
                    if track_id is None:
                        object_id_counter += 1
                    
                    # Create object dict
                    obj = {
                        'theta': float(theta),
                        'alpha': float(alpha),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'confidence': float(det['score']),
                        'id': int(obj_id),
                        'type': str(det['class_name']),
                        'depth': float(depth)
                    }
                    objects.append(obj)
                
                # Create buffer entry
                buffer_entry = {
                    'timestamp': time.time(),
                    'objects': objects,
                    'raw_yolo': detections,  # Optional debug info
                }
                
                # Update buffer (thread-safe)
                with self.buffer_lock:
                    self.data_buffer.append(buffer_entry)
                
                # Update FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Error in data collector: {e}")
                time.sleep(0.01)
        
        print(f"{self.name}: Data collector stopped.")
    
    def read(self) -> Dict:
        """
        Return the latest buffer frame (non-blocking).
        
        Returns:
            dict: {
                'timestamp': float,
                'objects': [
                    {
                        'theta': float,      # horizontal angle from center
                        'alpha': float,      # vertical angle from center
                        'width': int,        # bounding box width
                        'height': int,       # bounding box height
                        'confidence': float,  # YOLO confidence
                        'id': int,           # unique object ID
                        'type': str,         # class label
                        'depth': float       # depth in meters
                    },
                    ...
                ]
            }
        """
        with self.buffer_lock:
            if len(self.data_buffer) > 0:
                return self.data_buffer[-1].copy()
            else:
                return {
                    'timestamp': time.time(),
                    'objects': []
                }
    
    def debug(self) -> Dict:
        """
        Return internal diagnostics.
        
        Returns:
            dict: {
                'last_left_image': np.ndarray,
                'last_right_image': np.ndarray,
                'disparity_map': np.ndarray,
                'yolo_visualization': np.ndarray,
                'fps': float,
                'errors': List[str],
                'buffer_size': int,
                'num_objects': int
            }
        """
        errors = []
        
        # Get latest data
        latest = self.read()
        num_objects = len(latest.get('objects', []))
        
        # Create YOLO visualization if available
        yolo_viz = None
        if self.last_left_frame is not None and latest.get('raw_yolo'):
            try:
                yolo_viz = self.last_left_frame.copy()
                if draw_detections is not None:
                    yolo_viz = draw_detections(yolo_viz, latest['raw_yolo'], 
                                             tracker=self.yolo.tracker if self.yolo else None)
            except Exception as e:
                errors.append(f"YOLO visualization error: {e}")
                yolo_viz = self.last_left_frame.copy() if self.last_left_frame is not None else None
        
        return {
            'last_left_image': self.last_left_frame.copy() if self.last_left_frame is not None else None,
            'last_right_image': self.last_right_frame.copy() if self.last_right_frame is not None else None,
            'disparity_map': self.last_disparity.copy() if self.last_disparity is not None else None,
            'yolo_visualization': yolo_viz,
            'fps': self.current_fps,
            'errors': errors,
            'buffer_size': len(self.data_buffer),
            'num_objects': num_objects
        }
    
    def __repr__(self):
        return f"<VISION name={self.name}, connected={self.connected}>"

