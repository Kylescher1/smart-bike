"""
Vision System Upgrade - Complete Refactor

This module implements a new vision system architecture with:
- Dual camera YOLO object detection
- Thread-safe buffering
- Structured object detection output

Architecture:
- Camera class: Handles dual camera capture and rectification
- Yolo class: Runs YOLO inference on camera streams (shared model)
- VISION class: Main interface with start(), stop(), read(), debug()
"""

import cv2
import numpy as np
import threading
import time
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
# Streaming Camera Class
# ============================================================================

class VisionCamera:
    """
    Streaming camera that continuously captures frames from both cameras in background thread.
    Feeds video streams directly to YOLO without blocking.
    Both cameras are processed independently with the same YOLO model.
    """
    
    def __init__(self, left_config: Dict, right_config: Dict):
        """
        Initialize streaming camera system.
        
        Args:
            left_config: Configuration dict for left camera with keys:
                - port: Camera port/index
                - map_x: Rectification map X (optional)
                - map_y: Rectification map Y (optional)
            right_config: Right camera config (required) with same structure as left_config
        """
        self.left_config = left_config
        self.right_config = right_config
        
        # Camera objects
        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        
        # Extract calibration maps (optional)
        self.left_map_x = left_config.get('map_x')
        self.left_map_y = left_config.get('map_y')
        self.right_map_x = right_config.get('map_x')
        self.right_map_y = right_config.get('map_y')
        
        # Check if maps are available
        self.has_maps = (
            self.left_map_x is not None and self.left_map_y is not None and
            self.right_map_x is not None and self.right_map_y is not None
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
        
        # Open left camera (required)
        self.left_camera = Camera(self.left_config['port'], CAMERA_CONFIG)
        try:
            self.left_camera.open()
            print(f"✅ Left camera opened on port {self.left_config['port']}")
        except Exception as e:
            raise RuntimeError(f"Failed to open left camera: {e}")
        
        # Open right camera (required)
        self.right_camera = Camera(self.right_config['port'], CAMERA_CONFIG)
        try:
            self.right_camera.open()
            print(f"✅ Right camera opened on port {self.right_config['port']}")
        except Exception as e:
            raise RuntimeError(f"Failed to open right camera: {e}")
        
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
                
                # Scale right maps
                if self.right_map_x is not None:
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
                
                # Read right frame (required)
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
            
            # Print detailed timing summary every 60 frames
            if self._timing_frame_count % 60 == 0:
                window_size = min(60, len(self._timing_stats['total']))
                window_slice = slice(-window_size, None)
                
                avg_preprocess = np.mean(self._timing_stats['preprocess'][window_slice])
                avg_inference = np.mean(self._timing_stats['inference'][window_slice])
                avg_postprocess = np.mean(self._timing_stats['postprocess'][window_slice])
                avg_box_scale = np.mean(self._timing_stats['box_scale'][window_slice])
                avg_tracking = np.mean(self._timing_stats['tracking'][window_slice])
                avg_total = np.mean(self._timing_stats['total'][window_slice])
                
                # Calculate percentages
                pct_preprocess = (avg_preprocess / avg_total) * 100 if avg_total > 0 else 0
                pct_inference = (avg_inference / avg_total) * 100 if avg_total > 0 else 0
                pct_postprocess = (avg_postprocess / avg_total) * 100 if avg_total > 0 else 0
                pct_box_scale = (avg_box_scale / avg_total) * 100 if avg_total > 0 else 0
                pct_tracking = (avg_tracking / avg_total) * 100 if avg_total > 0 else 0
                
                print(f"\n[RKNN YOLO TIMING] Frame {self._timing_frame_count} (avg over last {window_size}):")
                print(f"  Preprocess:   {avg_preprocess:6.2f}ms ({pct_preprocess:5.1f}%)")
                print(f"  Inference:    {avg_inference:6.2f}ms ({pct_inference:5.1f}%) ⚡")
                print(f"  Postprocess:  {avg_postprocess:6.2f}ms ({pct_postprocess:5.1f}%)")
                print(f"  Box Scale:    {avg_box_scale:6.2f}ms ({pct_box_scale:5.1f}%)")
                print(f"  Tracking:     {avg_tracking:6.2f}ms ({pct_tracking:5.1f}%)")
                print(f"  TOTAL:        {avg_total:6.2f}ms ({1000/avg_total:.1f} FPS)")
            
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
    
    # ========================================================================
    # EASY MODEL SWITCHING - Change this to switch between .rknn and .pt
    # ========================================================================
    # Set to None to use config file or auto-detect, or set a specific path:
    # Examples:
    #   DEFAULT_MODEL_PATH = 'yolo/models/yolo11n.pt'      # Use .pt model (Windows/CPU/GPU)
    #   DEFAULT_MODEL_PATH = 'yolo/models/yolo11n.rknn'    # Use .rknn model (NPU)
    #   DEFAULT_MODEL_PATH = None                          # Auto-detect (default)
    DEFAULT_MODEL_PATH = 'yolo/models/yolo11n.pt' 
    # ========================================================================
    
    def __init__(self, name: str = "Unnamed VISION", **kwargs):
        """
        Initialize VISION system.
        
        Args:
            name: System name
            **kwargs: Configuration dict with keys:
                - left: Left camera config (required)
                - right: Right camera config (required)
                - yolo: YOLO config (model_path, conf_threshold, etc.)
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
        
        # Validate that both camera configs are provided
        if not self.left_config:
            raise ValueError("Left camera config is required")
        if not self.right_config:
            raise ValueError("Right camera config is required")
        
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
        
        # Debug state
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
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
        # Priority: 1) DEFAULT_MODEL_PATH class constant (overrides config), 2) config file, 3) auto-detect
        model_path = None
        
        # Use hardcoded DEFAULT_MODEL_PATH first (takes priority over config)
        if VISION.DEFAULT_MODEL_PATH is not None:
            model_path = VISION.DEFAULT_MODEL_PATH
            if Path(model_path).exists():
                print(f"✅ Using hardcoded model path (overrides config): {model_path}")
            else:
                print(f"⚠️  Hardcoded model path not found: {model_path}, falling back to config/auto-detect")
                model_path = None
        
        # If no hardcoded path, use config file
        if model_path is None:
            model_path = self.yolo_config.get('model_path') if self.yolo_config else None
            
            # If config specifies .rknn but RKNN is not available, try to use .pt version instead
            if model_path and model_path.endswith('.rknn') and not RKNN_AVAILABLE:
                # Try to find corresponding .pt file
                pt_path = model_path.replace('.rknn', '.pt')
                if Path(pt_path).exists():
                    print(f"⚠️  Config specifies .rknn model but RKNN not available on Windows")
                    print(f"✅ Using .pt version instead: {pt_path}")
                    model_path = pt_path
                elif VISION.DEFAULT_MODEL_PATH is not None and Path(VISION.DEFAULT_MODEL_PATH).exists():
                    print(f"⚠️  Config specifies .rknn model but RKNN not available on Windows")
                    print(f"✅ Using DEFAULT_MODEL_PATH instead: {VISION.DEFAULT_MODEL_PATH}")
                    model_path = VISION.DEFAULT_MODEL_PATH
                else:
                    print(f"⚠️  Config specifies .rknn model but RKNN not available, and no .pt fallback found")
                    model_path = None
        
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
        """Background thread to continuously process frames from both cameras with shared YOLO model."""
        print(f"{self.name}: Data collector started.")
        print(f"{self.name}: Timing reports will be printed every 30 frames or every 2 seconds.")
        print(f"{self.name}: Detailed performance analysis will show bottlenecks and operation breakdowns.\n")
        
        object_id_counter = 0  # Incremental ID for objects
        
        # Initialize timing stats
        timing_stats = {
            'frame_get': [],
            'frame_copy': [],
            'yolo_left': [],
            'yolo_right': [],
            'detection_tag': [],
            'angle_compute': [],
            'buffer_update': [],
            'total_loop': []
        }
        frame_count = 0
        last_timing_report = time.time()
        
        while not self.stop_event.is_set():
            try:
                loop_start = time.time()
                
                # Get latest frames from both cameras (non-blocking, already captured in background)
                frame_get_start = time.time()
                left_rect, right_rect = self.camera.get_latest_frames()
                frame_get_time = (time.time() - frame_get_start) * 1000  # ms
                
                if left_rect is None or right_rect is None:
                    time.sleep(0.001)  # Very short sleep if frames not ready
                    continue
                
                # Store for debug (thread-safe frame storage)
                frame_copy_start = time.time()
                with self.frame_lock:
                    self.last_left_frame = left_rect.copy()
                    self.last_right_frame = right_rect.copy()
                frame_copy_time = (time.time() - frame_copy_start) * 1000  # ms
                
                # Run YOLO detection on left frame (shared model instance)
                yolo_left_start = time.time()
                detections_left = self.yolo.detect(left_rect)
                yolo_left_time = (time.time() - yolo_left_start) * 1000  # ms
                
                # Run YOLO detection on right frame (same shared model instance)
                yolo_right_start = time.time()
                detections_right = self.yolo.detect(right_rect)
                yolo_right_time = (time.time() - yolo_right_start) * 1000  # ms
                
                # Tag detections with camera source and combine
                detection_tag_start = time.time()
                for det in detections_left:
                    det['camera'] = 'left'
                for det in detections_right:
                    det['camera'] = 'right'
                all_detections = detections_left + detections_right
                detection_tag_time = (time.time() - detection_tag_start) * 1000  # ms
                
                # Cache detections to avoid double YOLO call (thread-safe)
                with self.frame_lock:  # Reuse frame_lock for detection cache
                    self.last_detections_cache = all_detections.copy() if all_detections else []
                    self.last_detections_time = time.time()
                
                # Process each detection: compute angles
                objects = []
                angle_total_time = 0
                
                # Process left camera detections
                if detections_left:
                    for det in detections_left:
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
                        
                        # Create object dict
                        obj = {
                            'theta': float(theta),
                            'alpha': float(alpha),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(det['score']),
                            'id': int(obj_id),
                            'type': str(det['class_name']),
                            'camera': 'left'
                        }
                        objects.append(obj)
                
                # Process right camera detections
                if detections_right:
                    for det in detections_right:
                        bbox = det['bbox']
                        # Ensure bbox coordinates are integers
                        bbox = [int(coord) for coord in bbox]
                        x1, y1, x2, y2 = bbox
                        
                        # Compute angles (theta = horizontal, alpha = vertical)
                        angle_start = time.time()
                        h, w = right_rect.shape[:2]
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
                        
                        # Create object dict
                        obj = {
                            'theta': float(theta),
                            'alpha': float(alpha),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(det['score']),
                            'id': int(obj_id),
                            'type': str(det['class_name']),
                            'camera': 'right'
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
                timing_stats['frame_copy'].append(frame_copy_time)
                timing_stats['yolo_left'].append(yolo_left_time)
                timing_stats['yolo_right'].append(yolo_right_time)
                timing_stats['detection_tag'].append(detection_tag_time)
                timing_stats['angle_compute'].append(angle_total_time)
                timing_stats['buffer_update'].append(buffer_time)
                timing_stats['total_loop'].append(total_loop_time)
                frame_count += 1
                
                # Print detailed timing summary every 30 frames or every 2 seconds
                current_time = time.time()
                should_report = (frame_count % 30 == 0) or (current_time - last_timing_report >= 2.0)
                
                if should_report and len(timing_stats['total_loop']) >= 10:
                    # Calculate averages over last 30 frames (or all if less than 30)
                    window_size = min(30, len(timing_stats['total_loop']))
                    window_slice = slice(-window_size, None)
                    
                    avg_frame_get = np.mean(timing_stats['frame_get'][window_slice])
                    avg_frame_copy = np.mean(timing_stats['frame_copy'][window_slice])
                    avg_yolo_left = np.mean(timing_stats['yolo_left'][window_slice])
                    avg_yolo_right = np.mean(timing_stats['yolo_right'][window_slice])
                    avg_detection_tag = np.mean(timing_stats['detection_tag'][window_slice])
                    avg_angle = np.mean(timing_stats['angle_compute'][window_slice])
                    avg_buffer = np.mean(timing_stats['buffer_update'][window_slice])
                    avg_total = np.mean(timing_stats['total_loop'][window_slice])
                    
                    # Calculate percentages
                    yolo_total = avg_yolo_left + avg_yolo_right
                    pct_frame_get = (avg_frame_get / avg_total) * 100 if avg_total > 0 else 0
                    pct_frame_copy = (avg_frame_copy / avg_total) * 100 if avg_total > 0 else 0
                    pct_yolo_total = (yolo_total / avg_total) * 100 if avg_total > 0 else 0
                    pct_yolo_left = (avg_yolo_left / avg_total) * 100 if avg_total > 0 else 0
                    pct_yolo_right = (avg_yolo_right / avg_total) * 100 if avg_total > 0 else 0
                    pct_detection_tag = (avg_detection_tag / avg_total) * 100 if avg_total > 0 else 0
                    pct_angle = (avg_angle / avg_total) * 100 if avg_total > 0 else 0
                    pct_buffer = (avg_buffer / avg_total) * 100 if avg_total > 0 else 0
                    
                    # Find max times to identify outliers
                    max_yolo_left = np.max(timing_stats['yolo_left'][window_slice])
                    max_yolo_right = np.max(timing_stats['yolo_right'][window_slice])
                    max_total = np.max(timing_stats['total_loop'][window_slice])
                    
                    # Count detections
                    num_detections_left = len(detections_left) if detections_left else 0
                    num_detections_right = len(detections_right) if detections_right else 0
                    total_detections = num_detections_left + num_detections_right
                    
                    print(f"\n{'='*80}")
                    print(f"[VISION TIMING REPORT] Frame {frame_count} | FPS: {1000/avg_total:.1f} | Detections: L={num_detections_left}, R={num_detections_right}, Total={total_detections}")
                    print(f"{'='*80}")
                    print(f"Operation Breakdown (avg over last {window_size} frames):")
                    print(f"  Frame Get:        {avg_frame_get:6.2f}ms ({pct_frame_get:5.1f}%)")
                    print(f"  Frame Copy:       {avg_frame_copy:6.2f}ms ({pct_frame_copy:5.1f}%)")
                    print(f"  YOLO Left:        {avg_yolo_left:6.2f}ms ({pct_yolo_left:5.1f}%) [max: {max_yolo_left:.1f}ms]")
                    print(f"  YOLO Right:       {avg_yolo_right:6.2f}ms ({pct_yolo_right:5.1f}%) [max: {max_yolo_right:.1f}ms]")
                    print(f"  YOLO Total:       {yolo_total:6.2f}ms ({pct_yolo_total:5.1f}%)")
                    print(f"  Detection Tag:    {avg_detection_tag:6.2f}ms ({pct_detection_tag:5.1f}%)")
                    print(f"  Angle Compute:    {avg_angle:6.2f}ms ({pct_angle:5.1f}%)")
                    print(f"  Buffer Update:    {avg_buffer:6.2f}ms ({pct_buffer:5.1f}%)")
                    print(f"  {'-'*76}")
                    print(f"  TOTAL LOOP:       {avg_total:6.2f}ms (100.0%) [max: {max_total:.1f}ms]")
                    
                    # Identify bottleneck
                    bottlenecks = []
                    if pct_yolo_total > 70:
                        bottlenecks.append(f"YOLO inference ({pct_yolo_total:.1f}%)")
                    if pct_frame_copy > 10:
                        bottlenecks.append(f"Frame copying ({pct_frame_copy:.1f}%)")
                    if pct_angle > 10:
                        bottlenecks.append(f"Angle computation ({pct_angle:.1f}%)")
                    if pct_buffer > 5:
                        bottlenecks.append(f"Buffer operations ({pct_buffer:.1f}%)")
                    
                    if bottlenecks:
                        print(f"\n⚠️  Potential Bottlenecks: {', '.join(bottlenecks)}")
                    else:
                        print(f"\n✅ Processing balanced across operations")
                    print(f"{'='*80}\n")
                    
                    last_timing_report = current_time
                
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
        Contains combined detections from both left and right cameras.
        
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
                        'camera': str        # 'left' or 'right'
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
        Return internal diagnostics with visualizations for both camera streams.
        
        Returns:
            dict: {
                'last_left_image': np.ndarray,
                'last_right_image': np.ndarray,
                'yolo_visualization_left': np.ndarray,
                'yolo_visualization_right': np.ndarray,
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
        
        # Get cached detections (thread-safe)
        with self.frame_lock:
            last_left = self.last_left_frame.copy() if self.last_left_frame is not None else None
            last_right = self.last_right_frame.copy() if self.last_right_frame is not None else None
            detections = self.last_detections_cache.copy() if self.last_detections_cache else []
        
        # Separate detections by camera
        detections_left = [det for det in detections if det.get('camera') == 'left']
        detections_right = [det for det in detections if det.get('camera') == 'right']
        
        # Create YOLO visualization for left camera
        yolo_viz_left = None
        if last_left is not None:
            try:
                yolo_viz_left = last_left.copy()
                if draw_detections is not None and detections_left:
                    yolo_viz_left = draw_detections(yolo_viz_left, detections_left, 
                                                  tracker=self.yolo.tracker if self.yolo else None)
            except Exception as e:
                errors.append(f"YOLO visualization error (left): {e}")
                yolo_viz_left = last_left.copy() if last_left is not None else None
        
        # Create YOLO visualization for right camera
        yolo_viz_right = None
        if last_right is not None:
            try:
                yolo_viz_right = last_right.copy()
                if draw_detections is not None and detections_right:
                    yolo_viz_right = draw_detections(yolo_viz_right, detections_right, 
                                                    tracker=self.yolo.tracker if self.yolo else None)
            except Exception as e:
                errors.append(f"YOLO visualization error (right): {e}")
                yolo_viz_right = last_right.copy() if last_right is not None else None
        
        return {
            'last_left_image': last_left,
            'last_right_image': last_right,
            'yolo_visualization_left': yolo_viz_left,
            'yolo_visualization_right': yolo_viz_right,
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
        Visual debug mode showing YOLO detections for both camera streams.
        Press 'q' to quit.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start visual debug mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting dual camera YOLO visualization mode...")
        print(f"{self.name}: Press 'q' to exit")
        
        window_name_left = f"{self.name} - Left Camera"
        window_name_right = f"{self.name} - Right Camera"
        
        try:
            cv2.namedWindow(window_name_left, cv2.WINDOW_NORMAL)
            cv2.namedWindow(window_name_right, cv2.WINDOW_NORMAL)
            # Position windows side by side
            cv2.moveWindow(window_name_left, 100, 100)
            cv2.moveWindow(window_name_right, 750, 100)
        except Exception as e:
            print(f"{self.name}: Warning: Window creation issue: {e}")
            print(f"{self.name}: Visual debug mode may not work properly")
            return
        
        try:
            while True:
                # Get latest frames and detections (thread-safe copy)
                with self.frame_lock:
                    left_frame = self.last_left_frame.copy() if self.last_left_frame is not None else None
                    right_frame = self.last_right_frame.copy() if self.last_right_frame is not None else None
                    
                    # Get cached detections if recent
                    current_time = time.time()
                    if hasattr(self, 'last_detections_cache') and \
                       (current_time - self.last_detections_time) < 1.0:
                        all_detections = self.last_detections_cache.copy() if self.last_detections_cache else []
                    else:
                        all_detections = []
                
                # Separate detections by camera
                detections_left = [det for det in all_detections if det.get('camera') == 'left']
                detections_right = [det for det in all_detections if det.get('camera') == 'right']
                
                # Process left camera frame
                if left_frame is not None:
                    # Validate frame shape
                    if hasattr(left_frame, 'shape') and len(left_frame.shape) >= 2 and left_frame.size > 0:
                        # Create display frame (copy for drawing)
                        display_left = left_frame.copy()
                        h, w = display_left.shape[:2]
                        
                        # Draw left camera detections
                        for det in detections_left:
                            bbox = det.get('bbox', [])
                            if len(bbox) != 4:
                                continue
                            
                            x1, y1, x2, y2 = [int(coord) for coord in bbox]
                            
                            # Draw bounding box (green for left)
                            color = (0, 255, 0)  # Green
                            cv2.rectangle(display_left, (x1, y1), (x2, y2), color, 2)
                            
                            # Prepare label
                            class_name = det.get('class_name', 'object')
                            score = det.get('score', 0.0)
                            track_id = det.get('track_id')
                            
                            label = f"{class_name} {score:.2f}"
                            if track_id is not None:
                                label += f" ID:{track_id}"
                            
                            # Draw label background
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(display_left, (x1, y1 - label_h - 5), 
                                        (x1 + label_w, y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(display_left, label, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Add metadata overlay
                        fps_text = f"FPS: {self.current_fps:.1f}"
                        cv2.putText(display_left, fps_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        det_text = f"Left Detections: {len(detections_left)}"
                        cv2.putText(display_left, det_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Display left frame
                        try:
                            cv2.imshow(window_name_left, display_left)
                        except Exception as e:
                            if self.debug_mode:
                                print(f"{self.name}: imshow error (left): {e}")
                
                # Process right camera frame
                if right_frame is not None:
                    # Validate frame shape
                    if hasattr(right_frame, 'shape') and len(right_frame.shape) >= 2 and right_frame.size > 0:
                        # Create display frame (copy for drawing)
                        display_right = right_frame.copy()
                        h, w = display_right.shape[:2]
                        
                        # Draw right camera detections
                        for det in detections_right:
                            bbox = det.get('bbox', [])
                            if len(bbox) != 4:
                                continue
                            
                            x1, y1, x2, y2 = [int(coord) for coord in bbox]
                            
                            # Draw bounding box (blue for right)
                            color = (255, 0, 0)  # Blue
                            cv2.rectangle(display_right, (x1, y1), (x2, y2), color, 2)
                            
                            # Prepare label
                            class_name = det.get('class_name', 'object')
                            score = det.get('score', 0.0)
                            track_id = det.get('track_id')
                            
                            label = f"{class_name} {score:.2f}"
                            if track_id is not None:
                                label += f" ID:{track_id}"
                            
                            # Draw label background
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(display_right, (x1, y1 - label_h - 5), 
                                        (x1 + label_w, y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(display_right, label, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Add metadata overlay
                        fps_text = f"FPS: {self.current_fps:.1f}"
                        cv2.putText(display_right, fps_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        det_text = f"Right Detections: {len(detections_right)}"
                        cv2.putText(display_right, det_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Display right frame
                        try:
                            cv2.imshow(window_name_right, display_right)
                        except Exception as e:
                            if self.debug_mode:
                                print(f"{self.name}: imshow error (right): {e}")
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.01)  # Small delay for smoother display
                
        except KeyboardInterrupt:
            print(f"\n{self.name}: Visual debug mode interrupted by user")
        except Exception as e:
            print(f"{self.name}: Error in visual debug mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cv2.destroyWindow(window_name_left)
                cv2.destroyWindow(window_name_right)
            except:
                pass
            print(f"{self.name}: Visual debug mode ended")
    
    def debug_radar(self):
        """
        Show radar visualization of detected objects.
        Displays two windows:
        1. Top-down view showing horizontal angles (theta)
        2. Front view showing both horizontal (theta) and vertical (alpha) angles
        Press 'q' to quit.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start radar view. Vision system not connected. Call start() first.")
            return
        
        try:
            from radar_view import RadarView
        except ImportError:
            print(f"{self.name}: Error: radar_view module not found. Make sure radar_view.py is in the project root.")
            return
        
        print(f"{self.name}: Starting radar visualization...")
        radar = RadarView(self, canvas_size=600, max_range=90.0)
        radar.run()
    
    def __repr__(self):
        return f"<VISION name={self.name}, connected={self.connected}>"

