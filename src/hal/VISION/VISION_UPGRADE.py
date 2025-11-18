"""
Vision System Upgrade - Complete Refactor

This module implements a new vision system architecture with:
- YOLO object detection
- Custom block matching for depth estimation (in debug mode)
- Thread-safe buffering
- Structured object detection output

Architecture:
- Camera class: Handles stereo camera capture and rectification
- Yolo class: Runs YOLO inference on camera stream
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
import os

# Disable OpenCV threading to prevent segfaults in multi-threaded environments
# OpenCV's threading can cause crashes when used from multiple Python threads
os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
cv2.setNumThreads(1)  # Force single-threaded OpenCV

from ..cam.Camera import Camera, CAMERA_CONFIG

# ============================================================================
# Safe OpenCV Wrappers (prevent segfaults)
# ============================================================================

def safe_cv2_operation(operation_name: str):
    """Decorator to safely wrap OpenCV operations with validation"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Validate all numpy array arguments
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        if not arg.flags['C_CONTIGUOUS']:
                            arg = np.ascontiguousarray(arg)
                        if arg.size == 0:
                            return None
                        if not hasattr(arg, 'shape') or len(arg.shape) < 2:
                            return None
                return func(*args, **kwargs)
            except Exception as e:
                print(f"OpenCV {operation_name} error: {e}")
                return None
        return wrapper
    return decorator

def validate_cv2_array(arr, min_dims=2, allow_none=False):
    """Validate array before passing to OpenCV
    
    Returns:
        True if valid, False if invalid, None if None and allow_none=True
    """
    if arr is None:
        return None if allow_none else False
    if not isinstance(arr, np.ndarray):
        return False
    if arr.size == 0:
        return False
    if not hasattr(arr, 'shape'):
        return False
    if len(arr.shape) < min_dims:
        return False
    # Note: Caller should ensure contiguous with np.ascontiguousarray() if needed
    return True

# Import YOLO dependencies
# Prioritize RKNN NPU backend (much faster than CPU/GPU)
# Add system dist-packages to path for rknnlite (system package)
import site
system_packages = '/usr/lib/python3/dist-packages'
if system_packages not in sys.path:
    sys.path.insert(0, system_packages)

# Try RKNN imports first (preferred - NPU backend)
try:
    from rknnlite.api import RKNNLite
    from yolo.rknn_inference import (
        letterbox,
        process_output,
        draw_detections,
        COCO_CLASSES,
    )
    RKNN_AVAILABLE = True
    RKNNLite_available = True
except ImportError:
    try:
        # Try alternative import path
        from yolo.yolo import (
            RKNNLite,
            letterbox,
            process_output,
            draw_detections,
        )
        # COCO_CLASSES might not be in yolo.yolo, try to get it from rknn_inference
        try:
            from yolo.rknn_inference import COCO_CLASSES
        except ImportError:
            # Fallback: define COCO_CLASSES if not available
            COCO_CLASSES = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        RKNN_AVAILABLE = True
        RKNNLite_available = True
    except ImportError:
        RKNNLite = None
        letterbox = None
        process_output = None
        draw_detections = None
        COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        RKNN_AVAILABLE = False
        RKNNLite_available = False

# Fallback to Ultralytics YOLO (CPU/GPU - slower)
# Lazy import: Only import when actually needed to avoid logging conflicts
# Check availability without importing (to avoid torch logging issues)
ULTRALYTICS_AVAILABLE = False
try:
    import importlib.util
    spec = importlib.util.find_spec("ultralytics")
    ULTRALYTICS_AVAILABLE = spec is not None
except Exception:
    ULTRALYTICS_AVAILABLE = False

# Don't import YOLO here - import it lazily when needed
YOLO = None

# Import tracking wrapper (still needed for ByteTrack)
try:
    from yolo.yolo import ByteTrackerWrapper
    TRACKER_AVAILABLE = True
except ImportError:
    try:
        from yolo.rknn_inference import ByteTrackerWrapper
        TRACKER_AVAILABLE = True
    except ImportError:
        ByteTrackerWrapper = None
        TRACKER_AVAILABLE = False

YOLO_AVAILABLE = RKNN_AVAILABLE or ULTRALYTICS_AVAILABLE


# ============================================================================
# Segmentation Processing Function
# ============================================================================

def process_seg_output(output, detections, img_shape=(640, 640)):
    """
    Process segmentation model output to extract masks.
    Segmentation models output additional mask data after detection outputs.
    
    Args:
        output: Raw RKNN model output (list of arrays)
        detections: List of detection dicts from process_output
        img_shape: Image shape (h, w)
    
    Returns:
        Updated detections with 'mask' key added if masks are available
    """
    if not detections or len(output) < 4:
        return detections
    
    h, w = img_shape
    
    # YOLOv8-seg typically outputs: [boxes, scores, classes, proto] or [boxes, scores, classes, coeffs, proto]
    proto_mask = None
    mask_coeffs = None
    
    if len(output) == 4:
        # Format: [boxes, scores, classes, proto]
        proto_mask = output[3]
    elif len(output) >= 5:
        # Format: [boxes, scores, classes, coeffs, proto]
        mask_coeffs = output[3]
        proto_mask = output[4]
    
    if proto_mask is None:
        return detections
    
    # Get proto dimensions
    if len(proto_mask.shape) == 4:
        # Shape: [1, 32, proto_h, proto_w]
        _, num_proto, proto_h, proto_w = proto_mask.shape
        proto = proto_mask[0]  # [32, proto_h, proto_w]
    elif len(proto_mask.shape) == 3:
        # Shape: [32, proto_h, proto_w]
        num_proto, proto_h, proto_w = proto_mask.shape
        proto = proto_mask
    else:
        return detections  # Unknown format
    
    # Process masks if we have coefficients
    if mask_coeffs is not None and len(mask_coeffs.shape) >= 2:
        # mask_coeffs shape: [num_detections, 32] or [batch, num_detections, 32]
        if len(mask_coeffs.shape) == 3:
            mask_coeffs = mask_coeffs[0]  # Remove batch dimension
        
        for i, det in enumerate(detections):
            if i < mask_coeffs.shape[0]:
                coeffs = mask_coeffs[i]  # [32]
                
                # Combine: mask = sigmoid(coeffs @ proto)
                # Reshape coeffs to [32, 1, 1] and proto to [32, proto_h, proto_w]
                mask = np.sum(coeffs[:, None, None] * proto, axis=0)
                mask = 1 / (1 + np.exp(-mask))  # sigmoid
                
                # Resize to image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                det['mask'] = (mask_resized > 0.5).astype(np.uint8)  # Binary mask
    
    return detections


# ============================================================================
# Streaming Camera Class
# ============================================================================

class VisionCamera:
    """
    Streaming camera that continuously captures frames in background thread.
    Feeds video stream directly to YOLO without blocking.
    Only uses left camera for YOLO (right camera optional for stereo/depth).
    """
    
    def __init__(self, left_config: Dict, right_config: Optional[Dict] = None):
        """
        Initialize streaming camera system.
        
        Args:
            left_config: Configuration dict for left camera with keys:
                - port: Camera port/index
                - map_x: Rectification map X (optional)
                - map_y: Rectification map Y (optional)
            right_config: Optional right camera config (for stereo/depth, not used for YOLO)
        """
        self.left_config = left_config
        self.right_config = right_config
        
        # Camera objects
        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        
        # Extract calibration maps (optional)
        self.left_map_x = left_config.get('map_x')
        self.left_map_y = left_config.get('map_y')
        self.right_map_x = right_config.get('map_x') if right_config else None
        self.right_map_y = right_config.get('map_y') if right_config else None
        
        # Check if maps are available
        self.has_maps = (
            self.left_map_x is not None and self.left_map_y is not None
        )
        
        # Streaming state
        self.streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Latest frame buffers (thread-safe)
        self.latest_left_frame: Optional[np.ndarray] = None
        self.latest_right_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.frame_ready_event = threading.Event()
        
        # Pre-allocated remap buffers (reused for performance)
        self.left_map_x_scaled = None
        self.left_map_y_scaled = None
        self.right_map_x_scaled = None
        self.right_map_y_scaled = None
        self.map_lock = threading.Lock()
    
    def start(self):
        """Start streaming cameras in background thread."""
        if self.streaming:
            return
        
        # Open left camera (required for YOLO)
        self.left_camera = Camera(self.left_config['port'], CAMERA_CONFIG)
        try:
            self.left_camera.open()
            print(f"✅ Left camera opened on port {self.left_config['port']}")
        except Exception as e:
            raise RuntimeError(f"Failed to open left camera: {e}")
        
        # Open right camera (optional, for stereo/depth)
        if self.right_config:
            self.right_camera = Camera(self.right_config['port'], CAMERA_CONFIG)
            try:
                self.right_camera.open()
                print(f"✅ Right camera opened on port {self.right_config['port']}")
            except Exception as e:
                print(f"⚠️ Failed to open right camera: {e} (continuing with left only)")
                self.right_camera = None
        
        # Pre-scale maps if available (one-time setup for performance)
        if self.has_maps and self.left_camera:
            # Get frame size from camera
            test_frame = self.left_camera.read_frame()
            if test_frame is not None:
                h_raw, w_raw = test_frame.shape[:2]
                h_map, w_map = self.left_map_x.shape[:2]
                
                if (h_raw, w_raw) != (h_map, w_map):
                    # Scale maps once, reuse for all frames
                    scale_x = w_raw / w_map
                    scale_y = h_raw / h_map
                    self.left_map_x_scaled = cv2.resize(self.left_map_x, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR) * scale_x
                    self.left_map_y_scaled = cv2.resize(self.left_map_y, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR) * scale_y
                else:
                    self.left_map_x_scaled = self.left_map_x
                    self.left_map_y_scaled = self.left_map_y
                
                # Scale right maps if right camera available
                if self.right_camera and self.right_map_x is not None:
                    if (h_raw, w_raw) != (h_map, w_map):
                        self.right_map_x_scaled = cv2.resize(self.right_map_x, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR) * scale_x
                        self.right_map_y_scaled = cv2.resize(self.right_map_y, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR) * scale_y
                    else:
                        self.right_map_x_scaled = self.right_map_x
                        self.right_map_y_scaled = self.right_map_y
        
        # Start streaming thread
        self.stop_event.clear()
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print("✅ Camera streaming started")
    
    def stop(self):
        """Stop streaming and close cameras."""
        if not self.streaming:
            return
        
        self.streaming = False
        self.stop_event.set()
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        if self.left_camera:
            self.left_camera.close()
            self.left_camera = None
        
        if self.right_camera:
            self.right_camera.close()
            self.right_camera = None
        
        with self.frame_lock:
            self.latest_left_frame = None
            self.latest_right_frame = None
    
    def _stream_loop(self):
        """Background thread that continuously captures frames."""
        while not self.stop_event.is_set():
            try:
                # Read left frame (for YOLO)
                if self.left_camera:
                    left_raw = self.left_camera.read_frame()
                    if left_raw is not None:
                        # Apply rectification if maps available
                        if self.has_maps and self.left_map_x_scaled is not None:
                            with self.map_lock:
                                left_rect = cv2.remap(left_raw, self.left_map_x_scaled, self.left_map_y_scaled, cv2.INTER_LINEAR)
                        else:
                            left_rect = left_raw
                        
                        # Update latest frame (thread-safe)
                        with self.frame_lock:
                            self.latest_left_frame = left_rect.copy()
                        self.frame_ready_event.set()
                
                # Read right frame (optional, for stereo/depth)
                if self.right_camera:
                    right_raw = self.right_camera.read_frame()
                    if right_raw is not None:
                        # Apply rectification if maps available
                        if self.has_maps and self.right_map_x_scaled is not None:
                            with self.map_lock:
                                right_rect = cv2.remap(right_raw, self.right_map_x_scaled, self.right_map_y_scaled, cv2.INTER_LINEAR)
                        else:
                            right_rect = right_raw
                        
                        # Update latest frame (thread-safe)
                        with self.frame_lock:
                            self.latest_right_frame = right_rect.copy()
                
                # Small sleep to prevent CPU spinning (camera buffer handles timing)
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                if self.streaming:  # Only print if still supposed to be streaming
                    print(f"⚠️ Camera stream error: {e}")
                time.sleep(0.01)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get latest left frame (for YOLO).
        Non-blocking - returns immediately with latest frame or None.
        
        Returns:
            Latest left frame (rectified if maps available), or None if not ready
        """
        with self.frame_lock:
            return self.latest_left_frame.copy() if self.latest_left_frame is not None else None
    
    def get_latest_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get latest frames from both cameras.
        Non-blocking - returns immediately with latest frames or None.
        
        Returns:
            Tuple of (left_frame, right_frame), or (None, None) if not ready
        """
        with self.frame_lock:
            left = self.latest_left_frame.copy() if self.latest_left_frame is not None else None
            right = self.latest_right_frame.copy() if self.latest_right_frame is not None else None
            return left, right
    
    def wait_for_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Wait for next frame (blocking).
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            Latest left frame, or None if timeout
        """
        if self.frame_ready_event.wait(timeout=timeout):
            self.frame_ready_event.clear()
            return self.get_latest_frame()
        return None


# ============================================================================
# Yolo Class
# ============================================================================

class VisionYolo:
    """
    Loads YOLO model and runs inference on camera stream using RKNN NPU backend (fast) or Ultralytics YOLO (fallback).
    Outputs bounding boxes, segmentation masks (if available), class IDs, and confidence scores.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 imgsz: int = 640, track_enabled: bool = True, device: Optional[str] = None, 
                 target: Optional[str] = None, core: int = 0, **track_kwargs):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model file (.rknn preferred, .pt fallback)
            conf_threshold: Confidence threshold for detections
            imgsz: Input image size for YOLO
            track_enabled: Enable object tracking
            device: Computation device (for Ultralytics fallback: None for auto, 'cpu', '0' for GPU, etc.)
            target: RKNN target platform (None for on-device NPU, or 'RK3562'/'RK3566'/'RK3568'/'RK3588')
            core: NPU core mask (0=auto, 1=core0, 2=core1, 4=core2, 3=core0+1, 7=all)
            **track_kwargs: Additional tracking parameters
        """
        if not RKNN_AVAILABLE and not ULTRALYTICS_AVAILABLE:
            raise ImportError("Neither RKNN nor Ultralytics YOLO available. Please install rknnlite or ultralytics")
        
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.track_enabled = track_enabled
        self.device = device
        self.target = target
        self.core = core
        self.track_kwargs = track_kwargs
        
        # Determine backend based on model file extension
        self.use_rknn = self.model_path.suffix.lower() == '.rknn' and RKNN_AVAILABLE
        self.use_ultralytics = not self.use_rknn and ULTRALYTICS_AVAILABLE
        
        if self.use_rknn:
            self.rknn: Optional[RKNNLite] = None
            self.model = None
        else:
            self.rknn = None
            self.model = None  # Will be Ultralytics YOLO instance when loaded
        
        self.tracker: Optional[ByteTrackerWrapper] = None
        self.connected = False
        self.is_seg_model = False
        
        # Pre-allocated buffers for RKNN preprocessing (performance optimization)
        self.img_input_buffer: Optional[np.ndarray] = None
        self.last_letterbox_params = None  # Cache letterbox params for box scaling
    
    def start(self):
        """Load YOLO model (RKNN NPU preferred, Ultralytics fallback) and initialize tracker."""
        if self.connected:
            return
        
        if self.use_rknn:
            # Use RKNN NPU backend (much faster)
            if not RKNN_AVAILABLE or RKNNLite is None:
                raise RuntimeError("RKNN not available but .rknn model specified")
            
            print(f"📦 Loading RKNN NPU model: {self.model_path}")
            
            try:
                self.rknn = RKNNLite(verbose=False)
                
                # Load model
                ret = self.rknn.load_rknn(str(self.model_path))
                if ret != 0:
                    raise RuntimeError(f"Failed to load RKNN model: {ret}")
                
                # Initialize runtime (None = on-device NPU)
                target = None if (self.target is None or self.target.lower() == 'none' or self.target == '') else self.target
                ret = self.rknn.init_runtime(target=target, core_mask=self.core)
                if ret != 0:
                    raise RuntimeError(f"Failed to initialize RKNN runtime: {ret}")
                
                print(f"✅ RKNN NPU model loaded successfully from {self.model_path}")
                print(f"   Using NPU backend (target={target}, core={self.core})")
                
            except Exception as e:
                self.rknn = None
                raise RuntimeError(f"Failed to load RKNN model: {e}")
        
        elif self.use_ultralytics:
            # Fallback to Ultralytics YOLO (CPU/GPU - slower)
            # Lazy import to avoid logging conflicts at module load time
            if not ULTRALYTICS_AVAILABLE:
                raise RuntimeError("Ultralytics YOLO not available. Please install: pip install ultralytics")
            
            # Import here (lazy) to avoid torch logging issues at module load
            try:
                from ultralytics import YOLO as UltralyticsYOLO
            except Exception as e:
                raise RuntimeError(f"Failed to import Ultralytics YOLO: {e}. This may be due to logging conflicts.")
            
            print(f"📦 Loading Ultralytics YOLO model: {self.model_path}")
            print(f"⚠️  Using CPU/GPU backend (slower than NPU)")
            
            try:
                self.model = UltralyticsYOLO(str(self.model_path))
                print(f"✅ Model loaded from {self.model_path}")
            except Exception as e:
                self.model = None
                raise RuntimeError(f"Failed to load YOLO model: {e}")
            
            # Detect if this is a segmentation model
            model_name = str(self.model_path).lower()
            self.is_seg_model = 'seg' in model_name or 'segmentation' in model_name
            
            # Also check model task type
            if hasattr(self.model, 'task'):
                if self.model.task == 'segment':
                    self.is_seg_model = True
            
            if self.is_seg_model:
                print("✅ Segmentation model detected - masks will be processed")
        else:
            raise RuntimeError("No suitable backend available for model")
        
        # Initialize tracker if enabled
        if self.track_enabled and TRACKER_AVAILABLE and ByteTrackerWrapper is not None:
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
        
        self.connected = True
    
    def stop(self):
        """Release model resources."""
        if not self.connected:
            return
        
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
        
        if self.model is not None:
            self.model = None
        
        self.tracker = None
        self.img_input_buffer = None
        self.last_letterbox_params = None
        self.connected = False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO inference on frame using RKNN NPU (preferred) or Ultralytics YOLO (fallback).
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of detection dicts with keys: bbox, score, class_id, class_name, track_id (if tracking), mask (if seg model)
        """
        if not self.connected or frame is None:
            return []
        
        if self.rknn is not None:
            # RKNN NPU backend (fast path)
            return self._detect_rknn(frame)
        elif self.model is not None:
            # Ultralytics YOLO backend (fallback - slower)
            return self._detect_ultralytics(frame)
        else:
            return []
    
    def _detect_rknn(self, frame: np.ndarray) -> List[Dict]:
        """Run inference using RKNN NPU backend (optimized)."""
        try:
            # Initialize timing stats if not exists
            if not hasattr(self, '_timing_stats'):
                self._timing_stats = {
                    'preprocess': [],
                    'inference': [],
                    'postprocess': [],
                    'box_scale': [],
                    'tracking': [],
                    'total': []
                }
                self._timing_frame_count = 0
            
            total_start = time.time()
            h_orig, w_orig = frame.shape[:2]
            
            # Preprocess (optimized: reuse buffers, same as rknn_inference.py)
            preprocess_start = time.time()
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(self.imgsz, self.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # RKNN expects 4D input: (batch, height, width, channels) for NHWC format
            # Pre-allocate buffer if not exists, or reuse if same size (performance optimization)
            if self.img_input_buffer is None or self.img_input_buffer.shape != (1, self.imgsz, self.imgsz, 3):
                self.img_input_buffer = np.zeros((1, self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.img_input_buffer[0] = img_rgb.astype(np.uint8)
            img_input = self.img_input_buffer
            
            # Cache letterbox params for box scaling
            self.last_letterbox_params = {
                'ratio': ratio,
                'dw': dw,
                'dh': dh,
                'h_orig': h_orig,
                'w_orig': w_orig
            }
            preprocess_time = (time.time() - preprocess_start) * 1000  # ms
            
            # Run inference on NPU
            inference_start = time.time()
            try:
                outputs = self.rknn.inference([img_input])
            except Exception as e:
                print(f"RKNN inference error: {e}")
                return []
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            if outputs is None:
                return []
            
            # Process output (same as rknn_inference.py)
            postprocess_start = time.time()
            try:
                detections = process_output(outputs, conf_threshold=self.conf_threshold, img_shape=(self.imgsz, self.imgsz))
            except Exception as e:
                print(f"RKNN post-processing error: {e}")
                return []
            postprocess_time = (time.time() - postprocess_start) * 1000  # ms
            
            # Scale boxes back to original image size (vectorized for speed)
            box_scale_start = time.time()
            if detections and self.last_letterbox_params:
                params = self.last_letterbox_params
                scale = params['ratio']
                pad_x = params['dw']
                pad_y = params['dh']
                
                # Vectorized box scaling (much faster than loop)
                boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
                boxes = boxes.astype(np.int32)
                
                # Clip to image bounds
                boxes[:, 0] = np.clip(boxes[:, 0], 0, params['w_orig'])
                boxes[:, 1] = np.clip(boxes[:, 1], 0, params['h_orig'])
                boxes[:, 2] = np.clip(boxes[:, 2], 0, params['w_orig'])
                boxes[:, 3] = np.clip(boxes[:, 3], 0, params['h_orig'])
                
                for i, det in enumerate(detections):
                    det['bbox'] = boxes[i].tolist()
            box_scale_time = (time.time() - box_scale_start) * 1000  # ms
            
            # Update tracker if enabled
            tracking_start = time.time()
            if self.tracker is not None and self.track_enabled:
                try:
                    detections = self.tracker.update(detections)
                except Exception as e:
                    print(f"Tracker update error: {e}")
                    # Return detections without tracking on error
            tracking_time = (time.time() - tracking_start) * 1000  # ms
            
            total_time = (time.time() - total_start) * 1000  # ms
            
            # Store timing stats
            self._timing_stats['preprocess'].append(preprocess_time)
            self._timing_stats['inference'].append(inference_time)
            self._timing_stats['postprocess'].append(postprocess_time)
            self._timing_stats['box_scale'].append(box_scale_time)
            self._timing_stats['tracking'].append(tracking_time)
            self._timing_stats['total'].append(total_time)
            self._timing_frame_count += 1
            
            # Print timing summary every 60 frames
            if self._timing_frame_count % 60 == 0:
                avg_preprocess = np.mean(self._timing_stats['preprocess'][-60:])
                avg_inference = np.mean(self._timing_stats['inference'][-60:])
                avg_postprocess = np.mean(self._timing_stats['postprocess'][-60:])
                avg_box_scale = np.mean(self._timing_stats['box_scale'][-60:])
                avg_tracking = np.mean(self._timing_stats['tracking'][-60:])
                avg_total = np.mean(self._timing_stats['total'][-60:])
                
                print(f"[RKNN TIMING] Frame {self._timing_frame_count}: "
                      f"Preprocess={avg_preprocess:.1f}ms | "
                      f"Inference={avg_inference:.1f}ms | "
                      f"Postprocess={avg_postprocess:.1f}ms | "
                      f"BoxScale={avg_box_scale:.1f}ms | "
                      f"Tracking={avg_tracking:.1f}ms | "
                      f"Total={avg_total:.1f}ms ({1000/avg_total:.1f} FPS)")
            
            return detections
            
        except Exception as e:
            print(f"RKNN detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_ultralytics(self, frame: np.ndarray) -> List[Dict]:
        """Run inference using Ultralytics YOLO backend (fallback - slower)."""
        try:
            h_orig, w_orig = frame.shape[:2]
            
            # Run inference with Ultralytics YOLO
            # Note: Ultralytics expects BGR frames and handles preprocessing internally
            try:
                results = self.model.predict(
                    source=[frame],
                    imgsz=self.imgsz,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False,
                    stream=False  # Return list of results
                )
                result = results[0]  # Get first (and only) result
            except Exception as e:
                print(f"Ultralytics YOLO inference error: {e}")
                return []
            
            # Extract detections from Ultralytics result
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Get boxes, scores, and classes
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                # Get class names
                names = result.names if hasattr(result, 'names') else None
                if names is None and hasattr(result, 'model') and hasattr(result.model, 'names'):
                    names = result.model.names
                
                # Process each detection
                for idx in range(len(boxes_xyxy)):
                    bbox = boxes_xyxy[idx]
                    conf = float(confs[idx])
                    class_id = int(classes[idx])
                    
                    # Get class name
                    class_name = str(class_id)
                    if isinstance(names, dict):
                        class_name = names.get(class_id, class_name)
                    elif isinstance(names, (list, tuple)) and class_id < len(names):
                        class_name = names[class_id]
                    elif class_id < len(COCO_CLASSES):
                        class_name = COCO_CLASSES[class_id]
                    
                    # Convert bbox to list format [x1, y1, x2, y2]
                    bbox_list = [int(float(bbox[0])), int(float(bbox[1])), 
                                 int(float(bbox[2])), int(float(bbox[3]))]
                    
                    det = {
                        'bbox': bbox_list,
                        'score': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    # Extract segmentation mask if available
                    if self.is_seg_model and result.masks is not None:
                        try:
                            # Get mask for this detection
                            if idx < len(result.masks.data):
                                mask_data = result.masks.data[idx].cpu().numpy()  # [H, W] mask
                                
                                # Resize mask to original image size
                                if mask_data.shape != (h_orig, w_orig):
                                    mask_resized = cv2.resize(
                                        mask_data.astype(np.float32), 
                                        (w_orig, h_orig), 
                                        interpolation=cv2.INTER_LINEAR
                                    )
                                else:
                                    mask_resized = mask_data
                                
                                # Convert to binary mask
                                det['mask'] = (mask_resized > 0.5).astype(np.uint8)
                        except Exception as e:
                            # Mask extraction failed, continue without mask
                            pass  # Silently skip mask if extraction fails
                    
                    detections.append(det)
            
            # Update tracker if enabled
            if self.tracker is not None and self.track_enabled:
                try:
                    detections = self.tracker.update(detections)
                except Exception as e:
                    print(f"Tracker update error: {e}")
                    # Return detections without tracking on error
            
            return detections
            
        except Exception as e:
            print(f"Ultralytics YOLO detection error: {e}")
            import traceback
            traceback.print_exc()
            return []


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
                - baseline: Stereo baseline (meters)
                - focal_length_px: Focal length (pixels)
                - buffer_size: Circular buffer size
                - safe_mode: If True, disables features that might cause segfaults
        """
        self.name = name
        self.debug_mode = True
        self.safe_mode = kwargs.get('safe_mode', False)  # Safe mode disables risky operations
        
        # Load configuration
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Extract camera configs (config is passed directly, not nested under 'camera')
        self.left_config = kwargs.get('left', {})
        self.right_config = kwargs.get('right', {})
        self.yolo_config = kwargs.get('yolo', {})
        
        # Depth parameters (for custom block matching)
        self.baseline = getattr(self, 'baseline', 0.12)  # meters
        self.focal_length_px = getattr(self, 'focal_length_px', 800.0)  # pixels
        
        # Buffer configuration (reduced for memory efficiency)
        # Force smaller buffer even if config says otherwise
        config_buffer_size = getattr(self, 'buffer_size', 2)
        self.buffer_size = min(config_buffer_size, 2)  # Max 2 entries
        
        # Initialize components
        self.camera: Optional[VisionCamera] = None
        self.yolo: Optional[VisionYolo] = None
        
        # Runtime state
        self.connected = False
        self.data_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        self.frame_lock = threading.Lock()  # Lock for frame access (prevents segfaults)
        self.opencv_lock = threading.Lock()  # Lock for ALL OpenCV operations (prevents segfaults)
        
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
        model_path = self.yolo_config.get('model_path') if self.yolo_config else None
        
        # Prefer .rknn models (NPU backend - much faster), fallback to .pt if needed
        if model_path is None:
            # Try to find default .rknn model first (NPU - fast)
            models_dir = Path('yolo/models')
            if models_dir.exists():
                rknn_models = list(models_dir.glob('yolo11n*.rknn'))
                if rknn_models:
                    model_path = str(rknn_models[0])
                    print(f"✅ Found default RKNN model: {model_path}")
                else:
                    # Fallback to .pt model (CPU/GPU - slower)
                    pt_models = list(models_dir.glob('yolo11n*.pt'))
                    if pt_models:
                        model_path = str(pt_models[0])
                        print(f"⚠️  No .rknn model found, using .pt model (slower): {model_path}")
                    else:
                        # Last resort: use default path
                        default_rknn = 'yolo/models/yolo11n.rknn'
                        default_pt = 'yolo/models/yolo11n.pt'
                        if Path(default_rknn).exists():
                            model_path = default_rknn
                        elif Path(default_pt).exists():
                            model_path = default_pt
                            print(f"⚠️  Using default .pt model (slower): {model_path}")
                        else:
                            model_path = default_rknn  # Will error if not found
        
        if model_path:
            # Extract RKNN-specific config if available
            target = self.yolo_config.get('target', None)
            core = self.yolo_config.get('core', 0)
            
            self.yolo = VisionYolo(
                model_path=model_path,
                conf_threshold=self.yolo_config.get('conf_threshold', 0.25),
                imgsz=self.yolo_config.get('imgsz', 640),
                device=self.yolo_config.get('device', None),  # For Ultralytics fallback
                target=target,  # RKNN target platform
                core=core,  # NPU core mask
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
        
        print(f"{self.name}: Vision system stopped")
    
    def _data_collector(self):
        """Background thread to continuously process frames from streaming camera."""
        print(f"{self.name}: Data collector started.")
        
        object_id_counter = 0  # Incremental ID for objects
        
        # Initialize timing stats
        timing_stats = {
            'frame_get': [],
            'yolo_detect': [],
            'angle_compute': [],
            'buffer_update': [],
            'total_loop': []
        }
        frame_count = 0
        
        while not self.stop_event.is_set():
            try:
                loop_start = time.time()
                
                # Get latest frame from streaming camera (non-blocking, already captured in background)
                frame_get_start = time.time()
                left_rect = self.camera.get_latest_frame()
                frame_get_time = (time.time() - frame_get_start) * 1000  # ms
                
                if left_rect is None:
                    time.sleep(0.001)  # Very short sleep if no frame ready
                    continue
                
                # Store for debug (thread-safe frame storage)
                with self.frame_lock:
                    self.last_left_frame = left_rect.copy()
                    # Get right frame if available (for debug visualization)
                    _, right_rect = self.camera.get_latest_frames()
                    self.last_right_frame = right_rect.copy() if right_rect is not None else None
                
                # Run YOLO detection on left frame (streaming - no blocking)
                yolo_start = time.time()
                detections = self.yolo.detect(left_rect)
                yolo_time = (time.time() - yolo_start) * 1000  # ms
                
                # Cache detections for debug_visual to avoid double YOLO call (thread-safe)
                with self.frame_lock:  # Reuse frame_lock for detection cache
                    self.last_detections_cache = detections.copy() if detections else []
                    self.last_detections_time = time.time()
                
                # Process each detection: compute angles (depth computation removed - SGBM outdated)
                objects = []
                angle_total_time = 0
                
                if detections:
                    for det in detections:
                        bbox = det['bbox']
                        # Ensure bbox coordinates are integers
                        bbox = [int(coord) for coord in bbox]
                        x1, y1, x2, y2 = bbox
                        
                        # Compute angles (theta = horizontal, alpha = vertical)
                        angle_start = time.time()
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
                        angle_total_time += (time.time() - angle_start) * 1000  # ms
                        
                        # Use track_id as object ID, or assign new ID
                        track_id = det.get('track_id', None)
                        obj_id = track_id if track_id is not None else object_id_counter
                        if track_id is None:
                            object_id_counter += 1
                        
                        # Create object dict (depth set to 0.0 since SGBM removed)
                        obj = {
                            'theta': float(theta),
                            'alpha': float(alpha),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(det['score']),
                            'id': int(obj_id),
                            'type': str(det['class_name']),
                            'depth': 0.0  # Depth computation removed (SGBM outdated)
                        }
                        objects.append(obj)
                
                # Create buffer entry (minimal data to save memory)
                buffer_start = time.time()
                buffer_entry = {
                    'timestamp': time.time(),
                    'objects': objects,
                    # Don't store raw_yolo in buffer to save memory - only available in debug mode
                }
                
                # Update buffer (thread-safe)
                with self.buffer_lock:
                    self.data_buffer.append(buffer_entry)
                buffer_time = (time.time() - buffer_start) * 1000  # ms
                
                total_loop_time = (time.time() - loop_start) * 1000  # ms
                
                # Store timing stats
                timing_stats['frame_get'].append(frame_get_time)
                timing_stats['yolo_detect'].append(yolo_time)
                timing_stats['angle_compute'].append(angle_total_time)
                timing_stats['buffer_update'].append(buffer_time)
                timing_stats['total_loop'].append(total_loop_time)
                frame_count += 1
                
                # Print timing summary every 60 frames
                if frame_count % 60 == 0:
                    avg_frame_get = np.mean(timing_stats['frame_get'][-60:])
                    avg_yolo = np.mean(timing_stats['yolo_detect'][-60:])
                    avg_angle = np.mean(timing_stats['angle_compute'][-60:])
                    avg_buffer = np.mean(timing_stats['buffer_update'][-60:])
                    avg_total = np.mean(timing_stats['total_loop'][-60:])
                    
                    print(f"[VISION TIMING] Frame {frame_count}: "
                          f"FrameGet={avg_frame_get:.1f}ms | "
                          f"YOLO={avg_yolo:.1f}ms | "
                          f"Angle={avg_angle:.1f}ms | "
                          f"Buffer={avg_buffer:.1f}ms | "
                          f"Total={avg_total:.1f}ms ({1000/avg_total:.1f} FPS)")
                
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
            
            # Update depth parameters (for custom block matching)
            camera_config['baseline'] = self.baseline
            camera_config['focal_length_px'] = self.focal_length_px
            
            # Update FOV parameters if they exist
            if hasattr(self, 'fov_horizontal'):
                camera_config['fov_horizontal'] = self.fov_horizontal
            if hasattr(self, 'fov_vertical'):
                camera_config['fov_vertical'] = self.fov_vertical
            
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
    
    def _update_yolo_threshold(self, val):
        """Update YOLO confidence threshold."""
        threshold = val / 100.0  # Convert from 0-100 to 0.0-1.0
        if self.yolo_config:
            self.yolo_config['conf_threshold'] = threshold
        if self.yolo:
            self.yolo.conf_threshold = threshold
    
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
    
    def debug_visual(self):
        """
        Visual debug mode with interactive parameter tuning via trackbars.
        Press 'q' to quit, 's' to save parameters to config.dill.
        """
        
        def custom_block_matching(left_img: np.ndarray, right_img: np.ndarray, 
                                  bbox: List[int], circle_radius: int = 15,
                                  vis_data: Optional[Dict] = None) -> Tuple[Optional[float], Optional[Dict]]:
            """
            Custom block matching algorithm for depth estimation.
            
            Extracts a center circle from the object in left image and searches for it
            in the right image by finding minimum intensity difference.
            
            Args:
                left_img: Left rectified image
                right_img: Right rectified image  
                bbox: Bounding box [x1, y1, x2, y2]
                circle_radius: Radius of center circle to extract (default 15 = ~30 pixels wide)
            
            Returns:
                Tuple of (disparity value, visualization dict) or (None, None) if not found
                Visualization dict contains: 'left_circle', 'right_match', 'center_x', 'center_y', 'best_offset'
            """
            if left_img is None or right_img is None:
                return None, None
            
            # Initialize vis_data dict if provided, otherwise use None
            if vis_data is None:
                vis_data = {}
            
            x1, y1, x2, y2 = bbox
            
            # Calculate center of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Ensure circle fits within bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            actual_radius = min(circle_radius, bbox_w // 2 - 2, bbox_h // 2 - 2)
            if actual_radius < 5:
                return None, None  # Too small
            
            if vis_data is not None:
                vis_data['center_x'] = center_x
                vis_data['center_y'] = center_y
                vis_data['radius'] = actual_radius
            
            # Validate bounds
            h, w = left_img.shape[:2]
            if center_x < actual_radius or center_x >= w - actual_radius:
                return None, None
            if center_y < actual_radius or center_y >= h - actual_radius:
                return None, None
            
            # Extract circular region from left image (bounding box around circle)
            patch_x1 = max(0, center_x - actual_radius)
            patch_y1 = max(0, center_y - actual_radius)
            patch_x2 = min(w, center_x + actual_radius)
            patch_y2 = min(h, center_y + actual_radius)
            
            # Extract patch and convert to grayscale if needed
            if len(left_img.shape) == 3:
                left_patch = cv2.cvtColor(left_img[patch_y1:patch_y2, patch_x1:patch_x2], cv2.COLOR_BGR2GRAY)
            else:
                left_patch = left_img[patch_y1:patch_y2, patch_x1:patch_x2]
            
            # Create circular mask for the patch
            patch_h, patch_w = left_patch.shape[:2]
            patch_center_x = patch_w // 2
            patch_center_y = patch_h // 2
            y_coords, x_coords = np.ogrid[:patch_h, :patch_w]
            circle_mask = (x_coords - patch_center_x)**2 + (y_coords - patch_center_y)**2 <= actual_radius**2
            
            # Apply mask to extract circular region
            left_circle = left_patch.copy()
            left_circle[~circle_mask] = 0
            
            if left_circle.size == 0:
                return None, None
            
            # Store visualization data (only if vis_data dict provided)
            if vis_data is not None:
                vis_data['left_circle'] = left_circle.copy()
                vis_data['circle_mask'] = circle_mask.copy()
                vis_data['patch_coords'] = (patch_x1, patch_y1, patch_x2, patch_y2)
            
            # Convert right image to grayscale if needed
            if len(right_img.shape) == 3:
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_img
            
            # Search range: objects in right image appear shifted left (negative disparity)
            # Search from center_x to left (negative direction)
            search_range = min(200, center_x - actual_radius)  # Limit search range
            if search_range < 10:
                return None, None
            
            # Performance optimization: Use step size to reduce search iterations
            # Step by 2 pixels for faster search (can be adjusted)
            step_size = 2
            search_offsets = range(-search_range, 0, step_size)
            
            min_diff = float('inf')
            best_disparity = 0
            best_right_patch = None
            best_offset = 0
            
            # Pre-compute left circle masked for faster comparison
            left_circle_masked = left_circle[circle_mask].astype(np.float32)
            
            # Search horizontally for best match (optimized with step size)
            for offset in search_offsets:
                search_x1 = center_x + offset - actual_radius
                search_x2 = center_x + offset + actual_radius
                
                if search_x1 < 0 or search_x2 >= w:
                    continue
                
                # Extract corresponding region from right image
                right_patch = right_gray[patch_y1:patch_y2, search_x1:search_x2]
                
                if right_patch.shape != left_circle.shape:
                    continue
                
                # Apply same circular mask to right patch (vectorized)
                right_patch_masked = right_patch[circle_mask].astype(np.float32)
                
                # Calculate intensity difference (vectorized SAD)
                diff = np.sum(np.abs(left_circle_masked - right_patch_masked))
                
                if diff < min_diff:
                    min_diff = diff
                    best_disparity = abs(offset)  # Disparity is positive value
                    best_offset = offset
                    # Only copy patch if we need visualization (lazy evaluation)
                    if vis_data is not None:
                        best_right_patch = right_patch.copy()
                        best_right_patch[~circle_mask] = 0
            
            # Return disparity if we found a reasonable match
            if min_diff < float('inf') and best_disparity > 0:
                if vis_data is not None:
                    vis_data['best_offset'] = best_offset
                    vis_data['min_diff'] = min_diff
                    if best_right_patch is not None:
                        vis_data['right_match'] = best_right_patch
                return float(best_disparity), vis_data
            
            return None, None
        
        if not self.connected:
            print(f"{self.name}: Cannot start visual debug mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting visual debug mode with parameter tuning...")
        print(f"{self.name}: Press 'q' to exit, 's' to save parameters, 'b' to toggle block matching visualization")
        if self.safe_mode:
            print(f"{self.name}: ⚠️ SAFE MODE ENABLED - Some features disabled to prevent crashes")
        
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
        blockmatch_window = f"{self.name} - Block Matching"
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(trackbar_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(radar_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(blockmatch_window, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"{self.name}: Warning: Window creation issue: {e}")
            print(f"{self.name}: Attempting to continue...")
            # Try alternative window flags
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow(trackbar_window, cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow(radar_window, cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow(blockmatch_window, cv2.WINDOW_AUTOSIZE)
            except Exception as e2:
                print(f"{self.name}: Failed to create window: {e2}")
                print(f"{self.name}: Visual debug mode may not work properly")
                return
        
        # Create trackbars for YOLO parameters
        if self.yolo_config:
            yolo_conf = int(self.yolo_config.get('conf_threshold', 0.25) * 100)
            cv2.createTrackbar('YOLO Conf', trackbar_window, yolo_conf, 100, self._update_yolo_threshold)
        
        # Create trackbars for depth parameters (for custom block matching)
        baseline_val = int(self.baseline * 1000)  # Convert to mm for trackbar
        focal_val = int(self.focal_length_px)
        
        def update_baseline(v):
            self.baseline = v / 1000.0
        
        def update_focal(v):
            self.focal_length_px = float(v)
        
        cv2.createTrackbar('Baseline (mm)', trackbar_window, baseline_val, 500, update_baseline)
        cv2.createTrackbar('Focal (px)', trackbar_window, focal_val, 2000, update_focal)
        
        # Create trackbars for FOV parameters
        fov_h_val = int(getattr(self, 'fov_horizontal', 60.0))
        fov_v_val = int(getattr(self, 'fov_vertical', 45.0))
        
        def update_fov_h(v):
            self.fov_horizontal = float(v)
        
        def update_fov_v(v):
            self.fov_vertical = float(v)
        
        cv2.createTrackbar('FOV H (deg)', trackbar_window, fov_h_val, 180, update_fov_h)
        cv2.createTrackbar('FOV V (deg)', trackbar_window, fov_v_val, 180, update_fov_v)
        
        # Pre-allocate reusable buffers to avoid memory allocation every frame
        radar_size = 400
        radar_img = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
        trackbar_img = np.zeros((200, 300, 3), dtype=np.uint8)  # Increased height for save button
        depth_overlay_buffer = None  # Will be allocated when needed
        
        # Performance optimization: Cache grayscale right image
        right_gray_cache = None
        right_gray_cache_frame_id = None
        
        # Performance optimization: Reduce radar redraw frequency
        radar_update_counter = 0
        radar_update_interval = 2  # Update radar every N frames
        
        # Performance optimization: Skip block matching visualization unless needed
        show_blockmatch = False  # Can be toggled with 'b' key
        
        # Save button coordinates (for visual feedback)
        button_x, button_y = 10, 120
        button_w, button_h = 150, 40
        
        try:
            while True:
                # Get latest frames and data (thread-safe copy)
                # Optimize: Minimize lock time by copying only what's needed
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
                
                # Only compute ROI-based disparity for detected objects using custom block matching
                # Limit to max 3 detections to prevent memory issues
                # In safe mode, skip depth computation entirely to isolate segfault source
                max_detections_for_depth = 3
                if (not self.safe_mode and detections and right_frame is not None):
                    # Reuse depth overlay buffer (clear it first only if we have new detections)
                    # This prevents flickering when detections temporarily disappear
                    depth_overlay_buffer.fill(0)
                    depth_overlay = depth_overlay_buffer
                    
                    # Use original frame size for depth computation (not resized display)
                    orig_h, orig_w = left_frame.shape[:2] if hasattr(left_frame, 'shape') else (h, w)
                    if display_scale != 1.0:
                        # Need to use original frames for depth, not resized
                        # Optimize: Reuse frames we already copied above
                        orig_left = left_frame
                        orig_right = right_frame
                    else:
                        orig_left = left_frame
                        orig_right = right_frame
                    
                    # Performance optimization: Cache grayscale right image
                    # Only convert if frame changed or cache is invalid
                    current_frame_id = id(orig_right) if orig_right is not None else None
                    if right_gray_cache is None or right_gray_cache_frame_id != current_frame_id:
                        if orig_right is not None:
                            if len(orig_right.shape) == 3:
                                right_gray_cache = cv2.cvtColor(orig_right, cv2.COLOR_BGR2GRAY)
                            else:
                                right_gray_cache = orig_right
                            right_gray_cache_frame_id = current_frame_id
                    
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
                            
                            # Validate bbox coordinates
                            h_orig, w_orig = orig_left.shape[:2]
                            margin = 10  # Allow some margin
                            if x1 < -margin or y1 < -margin or x2 > w_orig + margin or y2 > h_orig + margin:
                                continue
                            if x2 <= x1 or y2 <= y1:
                                continue
                            # Ensure bbox has reasonable size
                            if (x2 - x1) < 5 or (y2 - y1) < 5:
                                continue
                            
                            # Use custom block matching instead of SGBM
                            # Performance: Only compute visualization if blockmatch window is shown
                            vis_data_dict = {} if show_blockmatch else None
                            disparity_value, vis_data = custom_block_matching(
                                orig_left, 
                                right_gray_cache if right_gray_cache is not None else orig_right, 
                                bbox_int, 
                                circle_radius=15,
                                vis_data=vis_data_dict
                            )
                            
                            if disparity_value is not None and disparity_value > 0:
                                # Calculate depth from disparity using power law
                                # Higher disparity = closer objects (inverse relationship)
                                # Using power law: depth = k / (disparity ^ power)
                                # Power of ~1.2-1.5 is typical for stereo depth (rough approximation)
                                if self.focal_length_px > 0 and self.baseline > 0:
                                    # Base linear calculation
                                    k = self.focal_length_px * self.baseline
                                    # Apply power law (power ~1.3 as rough guess)
                                    power = 1.3
                                    depth_value = k / (disparity_value ** power)
                                else:
                                    depth_value = 0.0
                                
                                # Store visualization data for block matching window (only if enabled)
                                if show_blockmatch and vis_data and 'left_circle' in vis_data:
                                    # Create visualization showing search algorithm on full right image
                                    left_circle_vis = vis_data['left_circle']
                                    vis_center_x = vis_data.get('center_x', (x1 + x2) // 2)
                                    vis_center_y = vis_data.get('center_y', (y1 + y2) // 2)
                                    radius = vis_data.get('radius', 15)
                                    best_offset = vis_data.get('best_offset', 0)
                                    
                                    # Convert right image to BGR for visualization
                                    if len(orig_right.shape) == 2:
                                        right_img_vis = cv2.cvtColor(orig_right.copy(), cv2.COLOR_GRAY2BGR)
                                    else:
                                        right_img_vis = orig_right.copy()
                                    
                                    # Draw search area (where we searched)
                                    search_start_x = max(0, vis_center_x - 200)  # Approximate search range
                                    search_end_x = vis_center_x
                                    cv2.rectangle(right_img_vis, 
                                                 (search_start_x, max(0, vis_center_y - radius - 50)),
                                                 (search_end_x, min(orig_h, vis_center_y + radius + 50)),
                                                 (255, 255, 0), 2)  # Yellow rectangle for search area
                                    
                                    # Draw the best match location
                                    match_x = vis_center_x + best_offset
                                    match_x1 = match_x - radius
                                    match_y1 = vis_center_y - radius
                                    match_x2 = match_x + radius
                                    match_y2 = vis_center_y + radius
                                    
                                    # Draw circle at best match location
                                    cv2.circle(right_img_vis, (match_x, vis_center_y), radius, (0, 255, 0), 2)  # Green circle
                                    cv2.rectangle(right_img_vis, (match_x1, match_y1), (match_x2, match_y2), 
                                                 (0, 255, 0), 2)  # Green rectangle
                                    
                                    # Draw line from center to match location
                                    cv2.line(right_img_vis, (vis_center_x, vis_center_y), (match_x, vis_center_y), 
                                            (0, 0, 255), 2)  # Red line showing offset
                                    
                                    # Draw arrow pointing to match
                                    arrow_length = abs(best_offset)
                                    if arrow_length > 5:
                                        cv2.arrowedLine(right_img_vis, 
                                                        (vis_center_x, vis_center_y - radius - 20),
                                                        (match_x, vis_center_y - radius - 20),
                                                        (0, 0, 255), 2, tipLength=0.3)
                                    
                                    # Draw reference point (where we started searching from)
                                    cv2.circle(right_img_vis, (vis_center_x, vis_center_y), 3, (255, 0, 0), -1)  # Blue dot
                                    
                                    # Create side-by-side visualization
                                    # Resize right image if too large for display
                                    max_display_h = 400
                                    if right_img_vis.shape[0] > max_display_h:
                                        scale = max_display_h / right_img_vis.shape[0]
                                        new_w = int(right_img_vis.shape[1] * scale)
                                        right_img_vis = cv2.resize(right_img_vis, (new_w, max_display_h))
                                    
                                    # Convert left circle to BGR
                                    if len(left_circle_vis.shape) == 2:
                                        left_circle_bgr = cv2.cvtColor(left_circle_vis, cv2.COLOR_GRAY2BGR)
                                    else:
                                        left_circle_bgr = left_circle_vis
                                    
                                    # Resize left circle to match height
                                    lh, lw = left_circle_bgr.shape[:2]
                                    rh, rw = right_img_vis.shape[:2]
                                    
                                    if lh != rh:
                                        left_circle_bgr = cv2.resize(left_circle_bgr, 
                                                                    (int(lw * rh / lh), rh))
                                        lh, lw = left_circle_bgr.shape[:2]
                                    
                                    # Create combined visualization
                                    vis_h = rh
                                    vis_w = lw + rw + 30
                                    blockmatch_vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
                                    
                                    # Place left circle on left
                                    blockmatch_vis[:lh, :lw] = left_circle_bgr
                                    
                                    # Place right image with search visualization on right
                                    blockmatch_vis[:rh, lw+30:lw+30+rw] = right_img_vis
                                    
                                    # Add text labels
                                    cv2.putText(blockmatch_vis, "Left Circle", (5, 20), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    cv2.putText(blockmatch_vis, "Right Image (Search Area)", (lw+35, 20), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    
                                    # Add legend
                                    legend_y = 50
                                    cv2.circle(blockmatch_vis, (lw+35, legend_y), 5, (255, 0, 0), -1)
                                    cv2.putText(blockmatch_vis, "Search Start", (lw+50, legend_y+5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
                                    legend_y += 25
                                    cv2.circle(blockmatch_vis, (lw+35, legend_y), 5, (0, 255, 0), -1)
                                    cv2.putText(blockmatch_vis, "Best Match", (lw+50, legend_y+5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
                                    legend_y += 25
                                    cv2.line(blockmatch_vis, (lw+30, legend_y), (lw+50, legend_y), (0, 0, 255), 2)
                                    cv2.putText(blockmatch_vis, "Disparity", (lw+55, legend_y+5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
                                    # Add disparity and depth values at bottom
                                    info_y = vis_h - 50
                                    info_text = f"Disparity: {disparity_value:.2f} px"
                                    cv2.putText(blockmatch_vis, info_text, (5, info_y), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    depth_text = f"Depth: {depth_value:.3f} m"
                                    cv2.putText(blockmatch_vis, depth_text, (5, info_y + 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    # Display in block matching window (only if enabled)
                                    if show_blockmatch:
                                        with self.opencv_lock:
                                            cv2.imshow(blockmatch_window, blockmatch_vis)
                                
                                # Create a small depth overlay for visualization
                                # Use bbox region for overlay
                                roi_x1 = max(0, x1)
                                roi_y1 = max(0, y1)
                                roi_x2 = min(orig_w, x2)
                                roi_y2 = min(orig_h, y2)
                                
                                # Scale ROI coordinates for display overlay
                                disp_roi_x1 = int(roi_x1 * scale_x)
                                disp_roi_y1 = int(roi_y1 * scale_y)
                                disp_roi_x2 = int(roi_x2 * scale_x)
                                disp_roi_y2 = int(roi_y2 * scale_y)
                                
                                # Fill ROI region with depth value for visualization
                                if (disp_roi_y1 >= 0 and disp_roi_y2 <= depth_overlay.shape[0] and
                                    disp_roi_x1 >= 0 and disp_roi_x2 <= depth_overlay.shape[1] and
                                    disp_roi_y2 > disp_roi_y1 and disp_roi_x2 > disp_roi_x1):
                                    depth_overlay[disp_roi_y1:disp_roi_y2, disp_roi_x1:disp_roi_x2] = depth_value
                                
                                # Also draw disparity and depth on main display frame
                                if display_frame is not None:
                                    text_x = int(x1 * scale_x)
                                    text_y = int(y1 * scale_y - 10) if y1 * scale_y > 30 else int(y2 * scale_y + 20)
                                    text_y = max(20, min(text_y, new_h - 40))
                                    
                                    disp_text = f"D:{disparity_value:.1f}px"
                                    depth_text = f"Z:{depth_value:.2f}m"
                                    
                                    cv2.putText(display_frame, disp_text, (text_x, text_y), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    cv2.putText(display_frame, depth_text, (text_x, text_y + 20), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
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
                # Validate depth_overlay before use to prevent segfaults
                if (detections and depth_overlay is not None and 
                    hasattr(depth_overlay, 'shape') and len(depth_overlay.shape) >= 2 and
                    np.any(depth_overlay > 0)):
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
                        
                        # Validate before applyColorMap (common segfault source)
                        if not validate_cv2_array(depth_normalized, min_dims=2):
                            continue
                        if depth_normalized.dtype != np.uint8:
                            depth_normalized = depth_normalized.astype(np.uint8)
                        if not depth_normalized.flags['C_CONTIGUOUS']:
                            depth_normalized = np.ascontiguousarray(depth_normalized)
                        # Use lock for thread-safe OpenCV applyColorMap
                        with self.opencv_lock:
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        if depth_colored is None:
                            continue
                        
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
                elif (depth_overlay is not None and hasattr(depth_overlay, 'shape') and 
                      len(depth_overlay.shape) >= 2 and np.any(depth_overlay > 0)):
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
                                
                                # Validate before applyColorMap
                                if not validate_cv2_array(depth_normalized, min_dims=2):
                                    continue
                                if depth_normalized.dtype != np.uint8:
                                    depth_normalized = depth_normalized.astype(np.uint8)
                                if not depth_normalized.flags['C_CONTIGUOUS']:
                                    depth_normalized = np.ascontiguousarray(depth_normalized)
                                # Use lock for thread-safe OpenCV applyColorMap
                                with self.opencv_lock:
                                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                                if depth_colored is None:
                                    continue
                                
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
                
                
                # Create radar map visualization (reuse buffer)
                # Performance optimization: Only redraw grid periodically, update objects every frame
                radar_update_counter += 1
                should_redraw_grid = (radar_update_counter % radar_update_interval == 0)
                
                if should_redraw_grid:
                    radar_img.fill(0)  # Clear previous frame
                    center_x, center_y = radar_size // 2, radar_size // 2
                    max_range = 2.0  # Maximum depth in meters to display (reduced for better resolution)
                    
                    # Draw radar grid (concentric circles and angle lines)
                    for r in range(1, 6):  # 5 range circles
                        radius = int((r / 5.0) * (radar_size // 2 - 20))
                        cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
                        # Draw range labels
                        range_text = f"{r * max_range / 5:.1f}m"
                        cv2.putText(radar_img, range_text, (center_x + radius - 30, center_y - radius + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    
                    # Draw angle lines (rotated 90° left: 0° = up, 90° = right, 180° = down, 270° = left)
                    # After rotation: original 0° (right) becomes -90° (up), so we show -90°, 0°, 90°, 180°
                    for angle_deg in [-90, 0, 90, 180]:
                        angle_rad = np.radians(angle_deg)
                        end_x = int(center_x + (radar_size // 2 - 10) * np.cos(angle_rad))
                        end_y = int(center_y + (radar_size // 2 - 10) * np.sin(angle_rad))
                        cv2.line(radar_img, (center_x, center_y), (end_x, end_y), (50, 50, 50), 1)
                        # Draw angle labels
                        label_x = int(center_x + (radar_size // 2 - 5) * np.cos(angle_rad))
                        label_y = int(center_y + (radar_size // 2 - 5) * np.sin(angle_rad))
                        # Display angle as positive (0-360 range)
                        display_angle = angle_deg if angle_deg >= 0 else angle_deg + 360
                        cv2.putText(radar_img, f"{display_angle}°", (label_x - 10, label_y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    
                    # Draw center point (camera position)
                    cv2.circle(radar_img, (center_x, center_y), 5, (0, 255, 255), -1)
                else:
                    # Just clear objects area (faster than full redraw)
                    center_x, center_y = radar_size // 2, radar_size // 2
                    max_range = 2.0
                    # Clear only the area where objects are drawn
                    cv2.rectangle(radar_img, (0, 0), (radar_size, radar_size), (0, 0, 0), -1)
                    # Redraw grid quickly (just circles and lines, no text)
                    for r in range(1, 6):
                        radius = int((r / 5.0) * (radar_size // 2 - 20))
                        cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
                    for angle_deg in [-90, 0, 90, 180]:
                        angle_rad = np.radians(angle_deg)
                        end_x = int(center_x + (radar_size // 2 - 10) * np.cos(angle_rad))
                        end_y = int(center_y + (radar_size // 2 - 10) * np.sin(angle_rad))
                        cv2.line(radar_img, (center_x, center_y), (end_x, end_y), (50, 50, 50), 1)
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
                            # Rotate radar 90° to the left:
                            # - Original: 0° = right, 90° = down, 180° = left, 270° = up
                            # - Rotated left: 0° = up, 90° = right, 180° = down, 270° = left
                            # Camera view: theta=0 is center, positive = right, negative = left
                            # Rotate left by subtracting 90 degrees
                            radar_angle_rad = np.radians(theta - 90)
                            
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
                # Validate frame before display to prevent segfaults
                try:
                    if display_frame is not None and hasattr(display_frame, 'shape') and len(display_frame.shape) >= 2:
                        if display_frame.size > 0:
                            # Use lock for thread-safe OpenCV imshow
                            with self.opencv_lock:
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
                
                # Draw save button (visual indicator)
                cv2.rectangle(trackbar_img, (button_x, button_y), 
                            (button_x + button_w, button_y + button_h), (0, 255, 0), 2)
                cv2.putText(trackbar_img, "SAVE (S)", (button_x + 20, button_y + 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                try:
                    if trackbar_img is not None and hasattr(trackbar_img, 'shape') and trackbar_img.size > 0:
                        # Use lock for thread-safe OpenCV imshow
                        with self.opencv_lock:
                            cv2.imshow(trackbar_window, trackbar_img)
                except Exception as e:
                    if self.debug_mode:
                        print(f"{self.name}: trackbar imshow error: {e}")
                
                # Show radar map
                try:
                    if radar_img is not None and hasattr(radar_img, 'shape') and radar_img.size > 0:
                        # Use lock for thread-safe OpenCV imshow
                        with self.opencv_lock:
                            cv2.imshow(radar_window, radar_img)
                except Exception as e:
                    if self.debug_mode:
                        print(f"{self.name}: radar imshow error: {e}")
                
                # Check for quit key or save key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    if self._save_config():
                        # Visual feedback - flash the button green
                        cv2.rectangle(trackbar_img, (button_x, button_y), 
                                    (button_x + button_w, button_y + button_h), (0, 255, 0), -1)
                        cv2.putText(trackbar_img, "SAVED!", (button_x + 15, button_y + 28), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        with self.opencv_lock:
                            cv2.imshow(trackbar_window, trackbar_img)
                        time.sleep(0.5)  # Show feedback for 0.5 seconds
                elif key == ord('b') or key == ord('B'):
                    # Toggle block matching visualization
                    show_blockmatch = not show_blockmatch
                    if not show_blockmatch:
                        # Hide window if disabling
                        try:
                            cv2.destroyWindow(blockmatch_window)
                        except:
                            pass
                    print(f"{self.name}: Block matching visualization: {'ON' if show_blockmatch else 'OFF'}")
                
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

