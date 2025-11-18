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
import dill
import sys
import gc

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
        
        # Extract calibration maps (optional - needed for rectification, not for calibration)
        self.left_map_x = left_config.get('map_x')
        self.left_map_y = left_config.get('map_y')
        self.right_map_x = right_config.get('map_x')
        self.right_map_y = right_config.get('map_y')
        
        # Check if maps are available (for calibration mode, maps may not exist yet)
        self.has_maps = (
            self.left_map_x is not None and self.left_map_y is not None and
            self.right_map_x is not None and self.right_map_y is not None
        )
        
        if not self.has_maps:
            print("⚠️ Calibration maps not provided - will return raw frames (calibration mode)")
    
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
            Tuple of (left_rectified, right_rectified) frames, or raw frames if maps not available, or (None, None) on error
        """
        if self.left_camera is None or self.right_camera is None:
            return None, None
        
        left_raw = self.left_camera.read_frame()
        right_raw = self.right_camera.read_frame()
        
        if left_raw is None or right_raw is None:
            return None, None
        
        # Apply rectification only if maps are available
        if self.has_maps:
            left_rect = cv2.remap(left_raw, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
            return left_rect, right_rect
        else:
            # Return raw frames for calibration
            return left_raw, right_raw


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
        
        block_size = int(self.stereo_config.get('blockSize', 11))
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        
        # Calculate numDisparities - must be integer and divisible by 16
        num_disp_k = int(self.stereo_config.get('numDisparitiesK', 2))
        num_disparities = max(16, 16 * num_disp_k)
        # Ensure it's divisible by 16 (round down if needed)
        num_disparities = (num_disparities // 16) * 16
        
        P1 = int(self.stereo_config.get('P1', 8 * 1 * block_size * block_size))
        P2 = int(self.stereo_config.get('P2', 32 * 1 * block_size * block_size))
        
        sgbm_mode = int(self.stereo_config.get('sgbmMode', 2))
        mode_map = {
            0: cv2.STEREO_SGBM_MODE_SGBM,
            1: cv2.STEREO_SGBM_MODE_HH,
            2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }
        mode = mode_map.get(sgbm_mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=int(self.stereo_config.get('minDisparity', 0)),
            numDisparities=int(num_disparities),
            blockSize=max(3, block_size),
            P1=P1,
            P2=P2,
            preFilterCap=int(self.stereo_config.get('preFilterCap', 43)),
            uniquenessRatio=int(self.stereo_config.get('uniquenessRatio', 1)),
            speckleWindowSize=int(self.stereo_config.get('speckleWindowSize', 196)),
            speckleRange=int(self.stereo_config.get('speckleRange', 34)),
            disp12MaxDiff=int(self.stereo_config.get('disp12MaxDiff', 18)),
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
        This method crops the images to the object's ROI before passing to SGBM,
        making it much more efficient than processing the full frame.
        
        Args:
            left_rect: Rectified left frame (full resolution)
            right_rect: Rectified right frame (full resolution)
            bbox: Bounding box [x1, y1, x2, y2] in full frame coordinates
        
        Returns:
            Disparity map for ROI only, or None on error
        """
        if self.stereo is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = left_rect.shape[:2]
        
        # Expand ROI and convert to integers for array slicing
        x1 = int(max(0, x1 - self.roi_expansion))
        y1 = int(max(0, y1 - self.roi_expansion))
        x2 = int(min(w, x2 + self.roi_expansion))
        y2 = int(min(h, y2 + self.roi_expansion))
        
        # Extract ROI (cropped region around object)
        left_roi = left_rect[y1:y2, x1:x2]
        right_roi = right_rect[y1:y2, x1:x2]
        
        if left_roi.size == 0 or right_roi.size == 0:
            return None
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY) if len(left_roi.shape) == 3 else left_roi
        gray_right = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY) if len(right_roi.shape) == 3 else right_roi
        
        # Compute disparity on cropped ROI only (much faster than full frame)
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
        
        # Extract camera configs (config is passed directly, not nested under 'camera')
        self.left_config = kwargs.get('left', {})
        self.right_config = kwargs.get('right', {})
        self.yolo_config = kwargs.get('yolo', {})
        
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
        
        # Buffer configuration (reduced for memory efficiency)
        # Force smaller buffer even if config says otherwise
        config_buffer_size = getattr(self, 'buffer_size', 2)
        self.buffer_size = min(config_buffer_size, 2)  # Max 2 entries
        
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
        self.frame_lock = threading.Lock()  # Lock for frame access (prevents segfaults)
        
        # Debug state
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
        self.last_disparity: Optional[np.ndarray] = None
        self.last_yolo_frame: Optional[np.ndarray] = None
        self.last_detections_cache: List[Dict] = []  # Cache last detections to avoid double YOLO call
        self.last_detections_time: float = 0.0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Parameter tuning state
        self.config_path = getattr(self, 'config_path', 'config.dill')
        self.original_config = None  # Store original config for saving
        self.sgbm_needs_reinit = False  # Flag to reinitialize SGBM when params change
        
        # Console logging state
        self.last_print_time = 0.0
        self.print_interval = 2.0  # Print every 2 seconds (reduced frequency)
        self.console_logging = True  # Can be disabled to save memory
        
        # Memory debugging state
        self.memory_debug = True
        self.last_memory_check = 0.0
        self.memory_check_interval = 5.0  # Check memory every 5 seconds
        self.initial_memory = None
        self.last_data_collector_memory_check = 0.0
        self.data_collector_memory_interval = 10.0  # Check memory in data collector every 10 seconds
    
    @property
    def left_camera(self) -> Optional[Camera]:
        """Expose left camera for calibration purposes."""
        if self.camera and self.camera.left_camera:
            return self.camera.left_camera
        return None
    
    @property
    def right_camera(self) -> Optional[Camera]:
        """Expose right camera for calibration purposes."""
        if self.camera and self.camera.right_camera:
            return self.camera.right_camera
        return None
    
    def start(self):
        """Initialize all components and start background processing thread."""
        if self.connected:
            return
        
        print(f"{self.name}: Starting vision system...")
        
        # Initialize components
        self.camera = VisionCamera(self.left_config, self.right_config)
        self.camera.start()
        
        # Initialize YOLO (optional - not needed for calibration)
        if self.yolo_config and 'model_path' in self.yolo_config:
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
        else:
            print("⚠️ YOLO config not provided - skipping YOLO initialization (calibration mode)")
            self.yolo = None
        
        # Initialize depth processor (optional - not needed for calibration)
        if self.stereo_config:
            self.depth = VisionDepth(
                stereo_config=self.stereo_config,
                q_matrix=self.Q,
                baseline=self.baseline,
                focal_length_px=self.focal_length_px,
                ema_alpha=self.ema_alpha,
                roi_expansion=self.roi_expansion
            )
            self.depth.start()
        else:
            self.depth = None
        
        # Start background processing thread only if YOLO is available
        # (calibration doesn't need background processing)
        if self.yolo is not None:
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
                
                # Store for debug (limit frame size to prevent memory issues)
                # Thread-safe frame storage with lock
                with self.frame_lock:
                    if left_rect.shape[0] * left_rect.shape[1] <= 1920 * 1200:
                        self.last_left_frame = left_rect.copy()
                        self.last_right_frame = right_rect.copy()
                    else:
                        # Resize before storing
                        scale = min(1920 / left_rect.shape[1], 1200 / left_rect.shape[0])
                        new_w, new_h = int(left_rect.shape[1] * scale), int(left_rect.shape[0] * scale)
                        self.last_left_frame = cv2.resize(left_rect, (new_w, new_h))
                        self.last_right_frame = cv2.resize(right_rect, (new_w, new_h))
                
                # Run YOLO detection on left frame
                detections = self.yolo.detect(left_rect)
                
                # Cache detections for debug_visual to avoid double YOLO call (thread-safe)
                with self.frame_lock:  # Reuse frame_lock for detection cache
                    self.last_detections_cache = detections.copy() if detections else []
                    self.last_detections_time = time.time()
                
                # Only compute depth if there are detections (ROI-based, not full frame)
                # Process each detection: compute depth and angles
                objects = []
                if detections and self.depth is not None:
                    for det in detections:
                        bbox = det['bbox']
                        # Ensure bbox coordinates are integers
                        bbox = [int(coord) for coord in bbox]
                        x1, y1, x2, y2 = bbox
                        
                        # Compute depth using ROI-based SGBM (only processes cropped region around object)
                        # This crops the images to the object's ROI before passing to SGBM for efficiency
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
                
                # Create buffer entry (minimal data to save memory)
                buffer_entry = {
                    'timestamp': time.time(),
                    'objects': objects,
                    # Don't store raw_yolo in buffer to save memory - only available in debug mode
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
                
                # Periodic memory check in data collector
                current_time = time.time()
                if self.memory_debug and (current_time - self.last_data_collector_memory_check) >= self.data_collector_memory_interval:
                    self.last_data_collector_memory_check = current_time
                    self._print_memory_stats("Data collector")
                    gc.collect()
                
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
        
        # Create YOLO visualization if available (thread-safe)
        yolo_viz = None
        with self.frame_lock:
            last_left = self.last_left_frame.copy() if self.last_left_frame is not None else None
            last_right = self.last_right_frame.copy() if self.last_right_frame is not None else None
        
        if last_left is not None and latest.get('raw_yolo'):
            try:
                yolo_viz = last_left.copy()
                if draw_detections is not None:
                    yolo_viz = draw_detections(yolo_viz, latest['raw_yolo'], 
                                             tracker=self.yolo.tracker if self.yolo else None)
            except Exception as e:
                errors.append(f"YOLO visualization error: {e}")
                yolo_viz = last_left.copy() if last_left is not None else None
        
        return {
            'last_left_image': last_left,
            'last_right_image': last_right,
            'disparity_map': self.last_disparity.copy() if self.last_disparity is not None else None,
            'yolo_visualization': yolo_viz,
            'fps': self.current_fps,
            'errors': errors,
            'buffer_size': len(self.data_buffer),
            'num_objects': num_objects
        }
    
    def _load_config_for_saving(self):
        """Load config.dill to prepare for saving."""
        try:
            with open(self.config_path, "rb") as f:
                self.original_config = dill.load(f)
        except Exception as e:
            print(f"{self.name}: Warning: Could not load config.dill for saving: {e}")
            self.original_config = None
    
    def _save_config(self):
        """Save current parameters to config.dill."""
        if self.original_config is None:
            self._load_config_for_saving()
        
        if self.original_config is None:
            print(f"{self.name}: ❌ Cannot save - config.dill not loaded")
            return False
        
        try:
            # Find camera config in the loaded config
            camera_config = None
            for key, value in self.original_config.items():
                if isinstance(value, dict) and 'who_to_run' in value:
                    if 'VISION' in str(value.get('who_to_run', '')):
                        camera_config = value
                        break
            
            if camera_config is None:
                print(f"{self.name}: ❌ Cannot find camera config in config.dill")
                return False
            
            # Update YOLO parameters
            if 'yolo' in camera_config:
                camera_config['yolo']['conf_threshold'] = self.yolo_config.get('conf_threshold', 0.25)
            
            # Update SGBM parameters
            stereo_keys = ['minDisparity', 'numDisparitiesK', 'blockSize', 'P1', 'P2',
                          'preFilterCap', 'uniquenessRatio', 'speckleWindowSize',
                          'speckleRange', 'disp12MaxDiff', 'sgbmMode']
            for key in stereo_keys:
                if hasattr(self, key):
                    camera_config[key] = getattr(self, key)
            
            # Update depth parameters
            camera_config['baseline'] = self.baseline
            camera_config['focal_length_px'] = self.focal_length_px
            camera_config['ema_alpha'] = self.ema_alpha
            camera_config['roi_expansion'] = self.roi_expansion
            
            # Save updated config
            with open(self.config_path, "wb") as f:
                dill.dump(self.original_config, f)
            
            print(f"{self.name}: ✅ Parameters saved to {self.config_path}")
            return True
            
        except Exception as e:
            print(f"{self.name}: ❌ Error saving config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_trackbar_change(self, val):
        """Callback for trackbar changes - triggers SGBM reinitialization if needed."""
        self.sgbm_needs_reinit = True
    
    def _update_yolo_threshold(self, val):
        """Update YOLO confidence threshold."""
        threshold = val / 100.0  # Convert from 0-100 to 0.0-1.0
        if self.yolo_config:
            self.yolo_config['conf_threshold'] = threshold
        if self.yolo:
            self.yolo.conf_threshold = threshold
    
    def _update_stereo_param(self, param_name, val, scale=1.0):
        """Update stereo parameter."""
        value = val * scale
        # Ensure integer parameters are integers
        int_params = ['minDisparity', 'numDisparitiesK', 'blockSize', 'preFilterCap', 
                     'uniquenessRatio', 'speckleWindowSize', 'speckleRange', 
                     'disp12MaxDiff', 'sgbmMode']
        if param_name in int_params:
            value = int(value)
        
        # Ensure blockSize is odd (SGBM requirement)
        if param_name == 'blockSize':
            if value % 2 == 0:
                value += 1
        
        # Ensure P1 and P2 are integers
        if param_name in ['P1', 'P2']:
            value = int(value)
        
        setattr(self, param_name, value)
        if param_name in self.stereo_config:
            self.stereo_config[param_name] = value
        self.sgbm_needs_reinit = True
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return {
                'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                'percent': process.memory_percent(),
            }
        except ImportError:
            # Fallback: estimate from numpy arrays and objects
            total_size = 0
            
        # Estimate frame sizes (thread-safe)
        with self.frame_lock:
            if self.last_left_frame is not None:
                total_size += self.last_left_frame.nbytes / 1024 / 1024
            if self.last_right_frame is not None:
                total_size += self.last_right_frame.nbytes / 1024 / 1024
        
        # Estimate buffer size (separate lock to avoid nested locks)
        with self.buffer_lock:
            buffer_size_mb = sum(
                sys.getsizeof(entry) + sum(sys.getsizeof(obj) for obj in entry.get('objects', []))
                for entry in self.data_buffer
            ) / 1024 / 1024
        
        return {
            'estimated_mb': total_size + buffer_size_mb,
            'frames_mb': total_size,
            'buffer_mb': buffer_size_mb,
        }
    
    def _print_memory_stats(self, context: str = ""):
        """Print memory statistics."""
        if not self.memory_debug:
            return
        
        mem_info = self._get_memory_usage()
        if 'rss_mb' in mem_info:
            print(f"{self.name}: MEM[{context}] RSS: {mem_info['rss_mb']:.1f}MB, "
                  f"VMS: {mem_info['vms_mb']:.1f}MB, Percent: {mem_info['percent']:.1f}%")
        else:
            print(f"{self.name}: MEM[{context}] Est: {mem_info.get('estimated_mb', 0):.1f}MB, "
                  f"Frames: {mem_info.get('frames_mb', 0):.1f}MB, "
                  f"Buffer: {mem_info.get('buffer_mb', 0):.1f}MB")
        
        # Print numpy array sizes (thread-safe)
        frame_sizes = []
        with self.frame_lock:
            if self.last_left_frame is not None:
                frame_sizes.append(f"left: {self.last_left_frame.shape} {self.last_left_frame.nbytes/1024/1024:.1f}MB")
            if self.last_right_frame is not None:
                frame_sizes.append(f"right: {self.last_right_frame.shape} {self.last_right_frame.nbytes/1024/1024:.1f}MB")
        if frame_sizes:
            print(f"{self.name}: MEM[{context}] Frames: {', '.join(frame_sizes)}")
        
        # Print buffer info
        with self.buffer_lock:
            buffer_len = len(self.data_buffer)
            total_objects = sum(len(entry.get('objects', [])) for entry in self.data_buffer)
            print(f"{self.name}: MEM[{context}] Buffer: {buffer_len} entries, {total_objects} objects")
    
    def _reinit_sgbm(self):
        """Reinitialize SGBM matcher with current parameters."""
        if self.depth is None:
            return
        
        try:
            if self.memory_debug:
                self._print_memory_stats("Before SGBM reinit")
            
            # Stop old SGBM
            self.depth.stop()
            
            # Force garbage collection
            gc.collect()
            
            # Update stereo_config with current values
            stereo_keys = ['minDisparity', 'numDisparitiesK', 'blockSize', 'P1', 'P2',
                          'preFilterCap', 'uniquenessRatio', 'speckleWindowSize',
                          'speckleRange', 'disp12MaxDiff', 'sgbmMode']
            for key in stereo_keys:
                if hasattr(self, key):
                    self.stereo_config[key] = getattr(self, key)
            
            # Update depth processor attributes
            self.depth.stereo_config = self.stereo_config
            self.depth.baseline = self.baseline
            self.depth.focal_length_px = self.focal_length_px
            self.depth.ema_alpha = self.ema_alpha
            self.depth.roi_expansion = self.roi_expansion
            
            # Restart SGBM
            self.depth.start()
            self.sgbm_needs_reinit = False
            
            if self.memory_debug:
                self._print_memory_stats("After SGBM reinit")
            
        except Exception as e:
            print(f"{self.name}: Error reinitializing SGBM: {e}")
            import traceback
            traceback.print_exc()
    
    def debug_visual(self):
        """
        Visual debug mode with interactive parameter tuning via trackbars.
        Press 'q' to quit, 's' to save parameters to config.dill.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start visual debug mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting visual debug mode with parameter tuning...")
        print(f"{self.name}: Press 'q' to exit, 's' to save parameters")
        
        # Initialize memory tracking
        if self.memory_debug:
            self.initial_memory = self._get_memory_usage()
            self._print_memory_stats("Debug start")
        
        # Load config for saving
        self._load_config_for_saving()
        
        # Fix Qt/Wayland issues by forcing X11 backend
        import os
        if 'WAYLAND_DISPLAY' in os.environ:
            # If running on Wayland, try to use X11 instead
            os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11 backend
            print(f"{self.name}: Detected Wayland, forcing X11 backend")
        
        window_name = f"{self.name} - Visual Debug"
        trackbar_window = f"{self.name} - Parameters"
        radar_window = f"{self.name} - Radar Map"
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(trackbar_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(radar_window, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"{self.name}: Warning: Window creation issue: {e}")
            print(f"{self.name}: Attempting to continue...")
            # Try alternative window flags
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow(trackbar_window, cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow(radar_window, cv2.WINDOW_AUTOSIZE)
            except Exception as e2:
                print(f"{self.name}: Failed to create window: {e2}")
                print(f"{self.name}: Visual debug mode may not work properly")
                return
        
        # Create trackbars for YOLO parameters
        if self.yolo_config:
            yolo_conf = int(self.yolo_config.get('conf_threshold', 0.25) * 100)
            cv2.createTrackbar('YOLO Conf', trackbar_window, yolo_conf, 100, self._update_yolo_threshold)
        
        # Create trackbars for SGBM parameters
        min_disparity = getattr(self, 'minDisparity', 0)
        num_disp_k = getattr(self, 'numDisparitiesK', 2)
        block_size = getattr(self, 'blockSize', 11)
        p1_val = getattr(self, 'P1', 968)
        p2_val = getattr(self, 'P2', 3872)
        prefilter_cap = getattr(self, 'preFilterCap', 43)
        uniqueness = getattr(self, 'uniquenessRatio', 1)
        speckle_win = getattr(self, 'speckleWindowSize', 196)
        speckle_range = getattr(self, 'speckleRange', 34)
        disp12_max_diff = getattr(self, 'disp12MaxDiff', 18)
        sgbm_mode = getattr(self, 'sgbmMode', 2)
        
        cv2.createTrackbar('MinDisparity', trackbar_window, min_disparity, 50, 
                          lambda v: self._update_stereo_param('minDisparity', v))
        cv2.createTrackbar('NumDispK', trackbar_window, num_disp_k, 10, 
                          lambda v: self._update_stereo_param('numDisparitiesK', v))
        cv2.createTrackbar('BlockSize', trackbar_window, block_size, 25, 
                          lambda v: self._update_stereo_param('blockSize', v))
        cv2.createTrackbar('P1 (x100)', trackbar_window, p1_val // 100, 200, 
                          lambda v: self._update_stereo_param('P1', v, 100.0))
        cv2.createTrackbar('P2 (x100)', trackbar_window, p2_val // 100, 500, 
                          lambda v: self._update_stereo_param('P2', v, 100.0))
        cv2.createTrackbar('PreFilterCap', trackbar_window, prefilter_cap, 100, 
                          lambda v: self._update_stereo_param('preFilterCap', v))
        cv2.createTrackbar('Uniqueness', trackbar_window, uniqueness, 20, 
                          lambda v: self._update_stereo_param('uniquenessRatio', v))
        cv2.createTrackbar('SpeckleWin', trackbar_window, speckle_win, 300, 
                          lambda v: self._update_stereo_param('speckleWindowSize', v))
        cv2.createTrackbar('SpeckleRange', trackbar_window, speckle_range, 100, 
                          lambda v: self._update_stereo_param('speckleRange', v))
        cv2.createTrackbar('Disp12MaxDiff', trackbar_window, disp12_max_diff, 50, 
                          lambda v: self._update_stereo_param('disp12MaxDiff', v))
        cv2.createTrackbar('SGBMMode', trackbar_window, sgbm_mode, 2, 
                          lambda v: self._update_stereo_param('sgbmMode', v))
        
        # Create trackbars for depth parameters
        baseline_val = int(self.baseline * 1000)  # Convert to mm for trackbar
        focal_val = int(self.focal_length_px)
        ema_val = int(self.ema_alpha * 100)
        roi_exp_val = self.roi_expansion
        
        def update_baseline(v):
            self.baseline = v / 1000.0
            if self.depth:
                self.depth.baseline = v / 1000.0
        
        def update_focal(v):
            self.focal_length_px = float(v)
            if self.depth:
                self.depth.focal_length_px = float(v)
        
        def update_ema(v):
            self.ema_alpha = v / 100.0
            if self.depth:
                self.depth.ema_alpha = v / 100.0
        
        def update_roi(v):
            self.roi_expansion = v
            if self.depth:
                self.depth.roi_expansion = v
        
        cv2.createTrackbar('Baseline (mm)', trackbar_window, baseline_val, 500, update_baseline)
        cv2.createTrackbar('Focal (px)', trackbar_window, focal_val, 2000, update_focal)
        cv2.createTrackbar('EMA Alpha (x100)', trackbar_window, ema_val, 100, update_ema)
        cv2.createTrackbar('ROI Expand', trackbar_window, roi_exp_val, 50, update_roi)
        
        # Pre-allocate reusable buffers to avoid memory allocation every frame
        radar_size = 400
        radar_img = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
        trackbar_img = np.zeros((400, 300, 3), dtype=np.uint8)
        depth_overlay_buffer = None  # Will be allocated when needed
        
        try:
            while True:
                # Get latest frames and data (thread-safe copy)
                with self.frame_lock:
                    if self.last_left_frame is not None:
                        left_frame = self.last_left_frame.copy()  # Copy immediately to avoid race condition
                    else:
                        left_frame = None
                    if self.last_right_frame is not None:
                        right_frame = self.last_right_frame.copy()  # Copy immediately to avoid race condition
                    else:
                        right_frame = None
                
                latest = self.read()
                
                if left_frame is None:
                    time.sleep(0.01)
                    continue
                
                # Validate frame shape to prevent segfaults
                if not hasattr(left_frame, 'shape') or len(left_frame.shape) < 2:
                    time.sleep(0.01)
                    continue
                
                # Get detections and objects
                # Reuse cached detections from data_collector to avoid double YOLO inference
                objects = latest.get('objects', [])
                detections = []
                
                # Use cached detections if recent (within 1.0 seconds), otherwise skip to save memory
                # Increased timeout to prevent flickering when data collector is slow
                # Thread-safe access to detection cache
                current_time = time.time()
                with self.frame_lock:  # Reuse frame_lock for detection cache
                    if hasattr(self, 'last_detections_cache') and \
                       (current_time - self.last_detections_time) < 1.0:
                        detections = self.last_detections_cache.copy() if self.last_detections_cache else []
                    else:
                        detections = []
                # Don't call YOLO again - saves memory and CPU
                
                # Memory check (periodic)
                if self.memory_debug:
                    current_time = time.time()
                    if current_time - self.last_memory_check >= self.memory_check_interval:
                        self.last_memory_check = current_time
                        self._print_memory_stats("Periodic check")
                        # Force garbage collection periodically
                        gc.collect()
                
                # Print to console (rate limited, simplified output)
                if self.console_logging:
                    current_time = time.time()
                    if current_time - self.last_print_time >= self.print_interval:
                        self.last_print_time = current_time
                        if detections:
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Detections: {len(detections)}")
                            for i, det in enumerate(detections[:5]):  # Limit to 5 detections
                                class_name = det.get('class_name', 'unknown')
                                score = det.get('score', 0.0)
                                track_id = det.get('track_id')
                                print(f"  [{i}] {class_name} conf:{score:.2f} ID:{track_id}")
                        
                        if objects:
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Objects: {len(objects)}")
                            for obj in objects[:5]:  # Limit to 5 objects
                                obj_id = obj.get('id', -1)
                                obj_type = obj.get('type', 'unknown')
                                depth = obj.get('depth', 0.0)
                                theta = obj.get('theta', 0.0)
                                print(f"  ID:{obj_id} {obj_type} depth:{depth:.2f}m θ:{theta:.1f}°")
                
                # Start with left frame - use reference, only copy when needed for drawing
                h, w = left_frame.shape[:2]
                # Limit frame processing size to prevent memory issues
                # Resize frames if too large (downscale for display)
                display_scale = 1.0
                if h * w > 1280 * 720:  # Downscale if larger than 720p for display
                    display_scale = min(1280 / w, 720 / h)
                    new_w, new_h = int(w * display_scale), int(h * display_scale)
                    # Resize for display only
                    display_frame = cv2.resize(left_frame, (new_w, new_h))
                    # Scale detection boxes accordingly
                    scale_x = new_w / w
                    scale_y = new_h / h
                else:
                    display_frame = left_frame.copy()  # Need copy for drawing operations
                    scale_x = 1.0
                    scale_y = 1.0
                    new_h, new_w = h, w
                
                # Initialize depth overlay (reuse buffer if size matches)
                # Use display size, not original frame size
                depth_overlay = None
                if depth_overlay_buffer is None or depth_overlay_buffer.shape != (new_h, new_w):
                    depth_overlay_buffer = np.zeros((new_h, new_w), dtype=np.float32)
                
                # Only compute ROI-based disparity for detected objects (not full frame)
                # This matches production behavior - SGBM only processes cropped regions
                # Limit to max 3 detections to prevent memory issues
                max_detections_for_depth = 3
                if detections and right_frame is not None and self.depth is not None and self.depth.stereo is not None:
                    # Reuse depth overlay buffer (clear it first only if we have new detections)
                    # This prevents flickering when detections temporarily disappear
                    depth_overlay_buffer.fill(0)
                    depth_overlay = depth_overlay_buffer
                    
                    # Use original frame size for depth computation (not resized display)
                    orig_h, orig_w = left_frame.shape[:2] if hasattr(left_frame, 'shape') else (h, w)
                    if display_scale != 1.0:
                        # Need to use original frames for depth, not resized
                        # Thread-safe access with lock
                        with self.frame_lock:
                            if self.last_left_frame is not None and self.last_right_frame is not None:
                                orig_left = self.last_left_frame.copy()  # Copy to avoid race condition
                                orig_right = self.last_right_frame.copy()
                            else:
                                orig_left = None
                                orig_right = None
                        if orig_left is None or orig_right is None:
                            continue  # Skip if frames not available
                    else:
                        orig_left = left_frame
                        orig_right = right_frame
                    
                    # Compute ROI-based disparity for each detected object (limited)
                    for det in detections[:max_detections_for_depth]:
                        bbox = det.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                        
                        # Use original bbox coordinates (not scaled)
                        bbox_int = [int(coord) for coord in bbox]
                        x1, y1, x2, y2 = bbox_int
                        
                        # Compute disparity only for this object's ROI (cropped region)
                        try:
                            # Validate frames before processing
                            if orig_left is None or orig_right is None:
                                continue
                            if not hasattr(orig_left, 'shape') or not hasattr(orig_right, 'shape'):
                                continue
                            if len(orig_left.shape) < 2 or len(orig_right.shape) < 2:
                                continue
                            
                            # Validate bbox coordinates (allow some margin for roi_expansion)
                            h_orig, w_orig = orig_left.shape[:2]
                            margin = self.depth.roi_expansion + 10  # Allow some margin
                            if x1 < -margin or y1 < -margin or x2 > w_orig + margin or y2 > h_orig + margin:
                                continue
                            if x2 <= x1 or y2 <= y1:
                                continue
                            # Ensure bbox has reasonable size
                            if (x2 - x1) < 5 or (y2 - y1) < 5:
                                continue
                            
                            roi_disparity = self.depth.compute_roi_disparity(orig_left, orig_right, bbox_int)
                            if roi_disparity is not None and roi_disparity.size > 0:
                                # Expand ROI coordinates (accounting for roi_expansion)
                                # Use original frame dimensions
                                roi_x1 = max(0, x1 - self.depth.roi_expansion)
                                roi_y1 = max(0, y1 - self.depth.roi_expansion)
                                roi_x2 = min(orig_w, x2 + self.depth.roi_expansion)
                                roi_y2 = min(orig_h, y2 + self.depth.roi_expansion)
                                
                                # Scale ROI coordinates for display overlay
                                disp_roi_x1 = int(roi_x1 * scale_x)
                                disp_roi_y1 = int(roi_y1 * scale_y)
                                disp_roi_x2 = int(roi_x2 * scale_x)
                                disp_roi_y2 = int(roi_y2 * scale_y)
                                
                                # Convert ROI disparity to depth map
                                roi_h, roi_w = roi_disparity.shape[:2]
                                # Reuse buffer if possible, otherwise create new one
                                if not hasattr(self, '_roi_depth_map_buffer') or \
                                   self._roi_depth_map_buffer.shape != (roi_h, roi_w):
                                    self._roi_depth_map_buffer = np.zeros((roi_h, roi_w), dtype=np.float32)
                                roi_depth_map = self._roi_depth_map_buffer
                                roi_depth_map.fill(0)  # Clear buffer
                                
                                # Convert disparity to depth pixel by pixel
                                valid_mask = roi_disparity > 0
                                if np.any(valid_mask) and self.focal_length_px > 0 and self.baseline > 0:
                                    roi_depth_map[valid_mask] = (self.focal_length_px * self.baseline) / roi_disparity[valid_mask]
                                
                                # Place ROI depth map into display overlay (scaled)
                                if disp_roi_y2 > disp_roi_y1 and disp_roi_x2 > disp_roi_x1:
                                    # Resize depth map to display size
                                    roi_depth_display = cv2.resize(roi_depth_map, (disp_roi_x2 - disp_roi_x1, disp_roi_y2 - disp_roi_y1))
                                    depth_overlay[disp_roi_y1:disp_roi_y2, disp_roi_x1:disp_roi_x2] = roi_depth_display
                                    del roi_depth_display  # Cleanup immediately
                        except Exception as e:
                            if self.debug_mode:
                                print(f"{self.name}: Error computing ROI disparity: {e}")
                elif depth_overlay_buffer is not None:
                    # Keep previous overlay if no detections (prevents flickering)
                    depth_overlay = depth_overlay_buffer
                else:
                    depth_overlay = None
                
                # Overlay depth visualization only in ROI regions
                # Only update overlay if we have new detections (prevents flickering)
                if detections and np.any(depth_overlay > 0):
                    # Normalize depth map for visualization
                    valid_mask = depth_overlay > 0
                    if np.any(valid_mask):
                        # Reuse buffers
                        if not hasattr(self, '_depth_normalized_buffer') or \
                           self._depth_normalized_buffer.shape != depth_overlay.shape:
                            self._depth_normalized_buffer = np.zeros_like(depth_overlay, dtype=np.uint8)
                        depth_normalized = self._depth_normalized_buffer
                        depth_normalized.fill(0)
                        
                        min_depth = depth_overlay[valid_mask].min()
                        max_depth = depth_overlay[valid_mask].max()
                        if max_depth > min_depth:
                            depth_normalized[valid_mask] = ((depth_overlay[valid_mask] - min_depth) / 
                                                             (max_depth - min_depth) * 255).astype(np.uint8)
                        
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        
                        # Blend only in ROI regions (50/50 blend)
                        # Reuse mask buffer
                        if not hasattr(self, '_mask_3d_buffer') or \
                           self._mask_3d_buffer.shape[:2] != valid_mask.shape:
                            self._mask_3d_buffer = np.stack([valid_mask] * 3, axis=2)
                        else:
                            # Update mask buffer
                            for i in range(3):
                                self._mask_3d_buffer[:, :, i] = valid_mask
                        mask_3d = self._mask_3d_buffer
                        
                        display_frame[mask_3d] = (display_frame[mask_3d] * 0.5 + 
                                                  depth_colored[mask_3d] * 0.5).astype(np.uint8)
                elif depth_overlay is not None and np.any(depth_overlay > 0):
                    # Keep previous overlay visible even if no new detections (smooth transition)
                    valid_mask = depth_overlay > 0
                    if np.any(valid_mask):
                        # Reuse buffers
                        if not hasattr(self, '_depth_normalized_buffer') or \
                           self._depth_normalized_buffer.shape != depth_overlay.shape:
                            self._depth_normalized_buffer = np.zeros_like(depth_overlay, dtype=np.uint8)
                        depth_normalized = self._depth_normalized_buffer
                        
                        # Only update if buffer exists and is valid
                        if depth_normalized.shape == depth_overlay.shape:
                            min_depth = depth_overlay[valid_mask].min()
                            max_depth = depth_overlay[valid_mask].max()
                            if max_depth > min_depth:
                                depth_normalized[valid_mask] = ((depth_overlay[valid_mask] - min_depth) / 
                                                                 (max_depth - min_depth) * 255).astype(np.uint8)
                                
                                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                                
                                # Blend with reduced opacity for stale overlays (fade effect)
                                if not hasattr(self, '_mask_3d_buffer') or \
                                   self._mask_3d_buffer.shape[:2] != valid_mask.shape:
                                    self._mask_3d_buffer = np.stack([valid_mask] * 3, axis=2)
                                else:
                                    for i in range(3):
                                        self._mask_3d_buffer[:, :, i] = valid_mask
                                mask_3d = self._mask_3d_buffer
                                
                                # Reduced opacity (30% instead of 50%) for stale overlays
                                display_frame[mask_3d] = (display_frame[mask_3d] * 0.7 + 
                                                          depth_colored[mask_3d] * 0.3).astype(np.uint8)
                
                # Create mapping from detection index to object depth
                depth_by_detection = {}
                if detections and objects:
                    # Match detections with objects by bbox or track_id
                    for obj in objects:
                        obj_id = obj.get('id')
                        obj_depth = obj.get('depth', 0)
                        # Try to match with detection
                        for i, det in enumerate(detections):
                            det_track_id = det.get('track_id')
                            if det_track_id == obj_id or (det_track_id is None and i < len(objects)):
                                depth_by_detection[i] = obj_depth
                                break
                
                # Draw detections (scale bbox if frame was resized)
                for i, det in enumerate(detections[:10]):  # Limit to 10 detections for display
                    bbox = det.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    # Scale bbox coordinates if display was resized
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)
                    
                    # Get depth for this detection (from objects list which uses ROI-based computation)
                    depth_value = depth_by_detection.get(i, 0)
                    if depth_value == 0 and depth_overlay is not None:
                        # Fallback: extract depth from ROI overlay at bbox center
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        if 0 <= center_y < depth_overlay.shape[0] and 0 <= center_x < depth_overlay.shape[1]:
                            depth_value = depth_overlay[center_y, center_x]
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    class_name = det.get('class_name', 'object')
                    score = det.get('score', 0.0)
                    track_id = det.get('track_id')
                    
                    label = f"{class_name} {score:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"
                    if depth_value > 0:
                        label += f" {depth_value:.2f}m"
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x1, y1 - label_h - 5), 
                                (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add metadata overlay
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(display_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if detections:
                    det_text = f"Detections: {len(detections)}"
                    cv2.putText(display_frame, det_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if objects:
                    obj_text = f"Objects: {len(objects)}"
                    cv2.putText(display_frame, obj_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check if SGBM needs reinitialization
                if self.sgbm_needs_reinit and self.depth is not None:
                    self._reinit_sgbm()
                
                # Create radar map visualization (reuse buffer)
                radar_img.fill(0)  # Clear previous frame
                center_x, center_y = radar_size // 2, radar_size // 2
                max_range = 10.0  # Maximum depth in meters to display
                
                # Draw radar grid (concentric circles and angle lines)
                for r in range(1, 6):  # 5 range circles
                    radius = int((r / 5.0) * (radar_size // 2 - 20))
                    cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
                    # Draw range labels
                    range_text = f"{r * max_range / 5:.1f}m"
                    cv2.putText(radar_img, range_text, (center_x + radius - 30, center_y - radius + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                
                # Draw angle lines (0°, 90°, 180°, 270°)
                for angle_deg in [0, 90, 180, 270]:
                    angle_rad = np.radians(angle_deg)
                    end_x = int(center_x + (radar_size // 2 - 10) * np.cos(angle_rad))
                    end_y = int(center_y + (radar_size // 2 - 10) * np.sin(angle_rad))
                    cv2.line(radar_img, (center_x, center_y), (end_x, end_y), (50, 50, 50), 1)
                    # Draw angle labels
                    label_x = int(center_x + (radar_size // 2 - 5) * np.cos(angle_rad))
                    label_y = int(center_y + (radar_size // 2 - 5) * np.sin(angle_rad))
                    cv2.putText(radar_img, f"{angle_deg}°", (label_x - 10, label_y + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                
                # Draw center point (camera position)
                cv2.circle(radar_img, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Plot objects on radar map
                if objects:
                    for obj in objects:
                        depth = obj.get('depth', 0.0)
                        theta = obj.get('theta', 0.0)  # Horizontal angle in degrees
                        obj_type = obj.get('type', 'unknown')
                        obj_id = obj.get('id', -1)
                        confidence = obj.get('confidence', 0.0)
                        
                        if depth > 0 and depth <= max_range:
                            # Convert polar coordinates (theta, depth) to cartesian (x, y)
                            # Note: theta is horizontal angle, positive = right, negative = left
                            # In radar view: 0° = right, 90° = down, 180° = left, 270° = up
                            # Camera view: theta=0 is center, positive = right, negative = left
                            # Convert camera theta to radar angle (camera right = radar 0°)
                            radar_angle_rad = np.radians(theta)
                            
                            # Scale depth to radar size
                            radius = int((depth / max_range) * (radar_size // 2 - 20))
                            
                            # Calculate position
                            obj_x = int(center_x + radius * np.cos(radar_angle_rad))
                            obj_y = int(center_y + radius * np.sin(radar_angle_rad))
                            
                            # Color based on object type (simple hash)
                            type_hash = hash(obj_type) % 6
                            colors = [
                                (0, 255, 0),    # Green
                                (255, 0, 0),    # Blue
                                (0, 0, 255),    # Red
                                (255, 255, 0),  # Cyan
                                (255, 0, 255),  # Magenta
                                (0, 255, 255),  # Yellow
                            ]
                            color = colors[abs(type_hash)]
                            
                            # Draw object as circle
                            cv2.circle(radar_img, (obj_x, obj_y), 6, color, -1)
                            cv2.circle(radar_img, (obj_x, obj_y), 6, (255, 255, 255), 1)
                            
                            # Draw line from center to object
                            cv2.line(radar_img, (center_x, center_y), (obj_x, obj_y), color, 1)
                            
                            # Draw label
                            label = f"{obj_type[:3]}"
                            if obj_id >= 0:
                                label += f":{obj_id}"
                            cv2.putText(radar_img, label, (obj_x + 8, obj_y - 8),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
                            # Draw depth text below
                            depth_text = f"{depth:.1f}m"
                            cv2.putText(radar_img, depth_text, (obj_x - 15, obj_y + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                
                # Add title
                cv2.putText(radar_img, "Radar Map (Top View)", (10, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(radar_img, f"Objects: {len(objects)}", (10, radar_size - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display the frame (limit update rate to reduce memory pressure)
                try:
                    cv2.imshow(window_name, display_frame)
                except Exception as e:
                    if self.debug_mode:
                        print(f"{self.name}: imshow error: {e}")
                
                # Show trackbar window (reuse buffer)
                trackbar_img.fill(0)  # Clear previous frame
                cv2.putText(trackbar_img, "Parameter Tuning", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(trackbar_img, "Press 's' to save", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(trackbar_img, "Press 'q' to quit", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                try:
                    cv2.imshow(trackbar_window, trackbar_img)
                except:
                    pass
                
                # Show radar map
                try:
                    cv2.imshow(radar_window, radar_img)
                except:
                    pass
                
                # Check for quit key or save key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self._save_config()
                
                # Cleanup temporary variables
                del display_frame
                if depth_overlay is not None:
                    del depth_overlay
                
                # Periodic garbage collection
                frame_count = getattr(self, '_debug_frame_count', 0)
                frame_count += 1
                self._debug_frame_count = frame_count
                if frame_count % 30 == 0:  # Every 30 frames
                    gc.collect()
                
                # Adaptive sleep: shorter delay for smoother display
                # Only sleep if we're rendering faster than the data collector updates
                time.sleep(0.01)  # Reduced delay for smoother frame rate
                
        except KeyboardInterrupt:
            print(f"\n{self.name}: Visual debug mode interrupted by user")
        except Exception as e:
            print(f"{self.name}: Error in visual debug mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(trackbar_window)
            cv2.destroyWindow(radar_window)
            
            # Cleanup buffers
            if hasattr(self, '_roi_depth_map_buffer'):
                del self._roi_depth_map_buffer
            if hasattr(self, '_roi_resize_buffer'):
                del self._roi_resize_buffer
            if hasattr(self, '_depth_normalized_buffer'):
                del self._depth_normalized_buffer
            if hasattr(self, '_mask_3d_buffer'):
                del self._mask_3d_buffer
            
            gc.collect()
            
            if self.memory_debug:
                self._print_memory_stats("Debug end")
            
            print(f"{self.name}: Visual debug mode ended")
    
    def __repr__(self):
        return f"<VISION name={self.name}, connected={self.connected}>"

