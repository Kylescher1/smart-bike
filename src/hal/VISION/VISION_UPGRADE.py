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
        
        # Detection smoothing parameters (modifiable)
        self.smoothing_enabled = kwargs.get('smoothing_enabled', True)  # Enable/disable smoothing
        self.smoothing_box_alpha = kwargs.get('smoothing_box_alpha', 0.3)  # EMA alpha for bounding boxes (0.0-1.0, lower = more smoothing)
        self.smoothing_angle_alpha = kwargs.get('smoothing_angle_alpha', 0.4)  # EMA alpha for angles (0.0-1.0, lower = more smoothing)
        self.smoothing_confidence_alpha = kwargs.get('smoothing_confidence_alpha', 0.5)  # EMA alpha for confidence (0.0-1.0, lower = more smoothing)
        self.smoothing_min_frames = kwargs.get('smoothing_min_frames', 3)  # Minimum frames before smoothing kicks in
        self.smoothing_timeout = kwargs.get('smoothing_timeout', 1.0)  # Seconds before removing stale smoothed values
        
        # Smoothing state: track smoothed values per object ID
        # Format: {obj_id: {'bbox': [x1, y1, x2, y2], 'theta': float, 'alpha': float, 'confidence': float, 'frame_count': int, 'last_seen': float}}
        self.smoothing_state: Dict[int, Dict] = {}
        self.smoothing_lock = threading.Lock()  # Lock for smoothing state access
    
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
        # HARDCODED: Always use yolo11n-seg.pt first (overrides config)
        hardcoded_seg_model = Path('yolo/models/yolo11n-seg.pt')
        if hardcoded_seg_model.exists():
            model_path = str(hardcoded_seg_model)
            print(f"✅ Using hardcoded segmentation model: {model_path} (overrides config)")
        else:
            # Fallback to config if hardcoded model doesn't exist
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
                
                # Cache detections to avoid double YOLO call (thread-safe)
                with self.frame_lock:  # Reuse frame_lock for detection cache
                    self.last_detections_cache = detections.copy() if detections else []
                    self.last_detections_time = time.time()
                
                # Process each detection: compute angles (depth computation removed - SGBM outdated)
                objects = []
                angle_total_time = 0
                current_time = time.time()
                
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
                        
                        # Apply smoothing if enabled
                        if self.smoothing_enabled:
                            with self.smoothing_lock:
                                # Get or initialize smoothing state for this object
                                if obj_id not in self.smoothing_state:
                                    self.smoothing_state[obj_id] = {
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'theta': float(theta),
                                        'alpha': float(alpha),
                                        'confidence': float(det['score']),
                                        'frame_count': 0,
                                        'last_seen': current_time
                                    }
                                
                                smooth_state = self.smoothing_state[obj_id]
                                smooth_state['frame_count'] += 1
                                smooth_state['last_seen'] = current_time
                                
                                # Apply exponential moving average smoothing
                                # Only smooth after minimum frames threshold
                                if smooth_state['frame_count'] >= self.smoothing_min_frames:
                                    # Smooth bounding box
                                    alpha_box = self.smoothing_box_alpha
                                    smooth_state['bbox'][0] = alpha_box * float(x1) + (1 - alpha_box) * smooth_state['bbox'][0]  # x1
                                    smooth_state['bbox'][1] = alpha_box * float(y1) + (1 - alpha_box) * smooth_state['bbox'][1]  # y1
                                    smooth_state['bbox'][2] = alpha_box * float(x2) + (1 - alpha_box) * smooth_state['bbox'][2]  # x2
                                    smooth_state['bbox'][3] = alpha_box * float(y2) + (1 - alpha_box) * smooth_state['bbox'][3]  # y2
                                    
                                    # Smooth angles
                                    alpha_angle = self.smoothing_angle_alpha
                                    smooth_state['theta'] = alpha_angle * float(theta) + (1 - alpha_angle) * smooth_state['theta']
                                    smooth_state['alpha'] = alpha_angle * float(alpha) + (1 - alpha_angle) * smooth_state['alpha']
                                    
                                    # Smooth confidence
                                    alpha_conf = self.smoothing_confidence_alpha
                                    smooth_state['confidence'] = alpha_conf * float(det['score']) + (1 - alpha_conf) * smooth_state['confidence']
                                    
                                    # Use smoothed values
                                    x1_smooth, y1_smooth, x2_smooth, y2_smooth = [int(coord) for coord in smooth_state['bbox']]
                                    theta_smooth = smooth_state['theta']
                                    alpha_smooth = smooth_state['alpha']
                                    confidence_smooth = smooth_state['confidence']
                                else:
                                    # Not enough frames yet, use raw values but update state
                                    smooth_state['bbox'] = [float(x1), float(y1), float(x2), float(y2)]
                                    smooth_state['theta'] = float(theta)
                                    smooth_state['alpha'] = float(alpha)
                                    smooth_state['confidence'] = float(det['score'])
                                    x1_smooth, y1_smooth, x2_smooth, y2_smooth = x1, y1, x2, y2
                                    theta_smooth = theta
                                    alpha_smooth = alpha
                                    confidence_smooth = det['score']
                        else:
                            # Smoothing disabled, use raw values
                            x1_smooth, y1_smooth, x2_smooth, y2_smooth = x1, y1, x2, y2
                            theta_smooth = theta
                            alpha_smooth = alpha
                            confidence_smooth = det['score']
                        
                        # Convert angles from degrees to radians
                        theta_rad = np.deg2rad(theta_smooth)
                        alpha_rad = np.deg2rad(alpha_smooth)
                        
                        # Map angles to unit circle coordinates (spherical to Cartesian)
                        x = np.sin(theta_rad) * np.cos(alpha_rad)
                        y = np.sin(theta_rad) * np.sin(alpha_rad)
                        z = np.cos(theta_rad)
                        
                        # Create object dict with unit circle coordinates
                        obj = {
                            'x': float(x),
                            'y': float(y),
                            'z': float(z),
                            'width': int(x2_smooth - x1_smooth),
                            'height': int(y2_smooth - y1_smooth),
                            'confidence': float(confidence_smooth),
                            'id': int(obj_id),
                            'type': str(det['class_name']),
                            'depth': 0.0  # Depth computation removed (SGBM outdated)
                        }
                        objects.append(obj)
                    
                    # Clean up stale smoothing state (objects that haven't been seen recently)
                    if self.smoothing_enabled:
                        with self.smoothing_lock:
                            stale_ids = [
                                obj_id for obj_id, state in self.smoothing_state.items()
                                if (current_time - state['last_seen']) > self.smoothing_timeout
                            ]
                            for obj_id in stale_ids:
                                del self.smoothing_state[obj_id]
                
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
                        'x': float,          # x coordinate on unit circle (sin(theta) * cos(alpha))
                        'y': float,          # y coordinate on unit circle (sin(theta) * sin(alpha))
                        'z': float,          # z coordinate on unit circle (cos(theta))
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
                # Update tracking parameters
                camera_config['yolo']['track_thresh'] = self.yolo_config.get('track_thresh', 0.5)
                camera_config['yolo']['track_high_thresh'] = self.yolo_config.get('track_high_thresh', 0.6)
                camera_config['yolo']['track_match_thresh'] = self.yolo_config.get('track_match_thresh', 0.8)
                camera_config['yolo']['track_buffer'] = self.yolo_config.get('track_buffer', 30)
                camera_config['yolo']['frame_rate'] = self.yolo_config.get('frame_rate', 30)
            
            # Update depth parameters (for custom block matching)
            camera_config['baseline'] = self.baseline
            camera_config['focal_length_px'] = self.focal_length_px
            
            # Update FOV parameters if they exist
            if hasattr(self, 'fov_horizontal'):
                camera_config['fov_horizontal'] = self.fov_horizontal
            if hasattr(self, 'fov_vertical'):
                camera_config['fov_vertical'] = self.fov_vertical
            
            # Update smoothing parameters
            camera_config['smoothing_enabled'] = self.smoothing_enabled
            camera_config['smoothing_box_alpha'] = self.smoothing_box_alpha
            camera_config['smoothing_angle_alpha'] = self.smoothing_angle_alpha
            camera_config['smoothing_confidence_alpha'] = self.smoothing_confidence_alpha
            camera_config['smoothing_min_frames'] = self.smoothing_min_frames
            camera_config['smoothing_timeout'] = self.smoothing_timeout
            
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
        Simple visual debug mode showing YOLO detections.
        Press 'q' to quit.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start visual debug mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting YOLO visualization mode...")
        print(f"{self.name}: Press 'q' to exit, 'm' to toggle masks")
        
        window_name = f"{self.name} - YOLO Visualization"
        
        # Mask display toggle state
        show_masks = False
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"{self.name}: Warning: Window creation issue: {e}")
            print(f"{self.name}: Visual debug mode may not work properly")
            return
        
        try:
            while True:
                # Get latest frame and detections (thread-safe copy)
                with self.frame_lock:
                    if self.last_left_frame is not None:
                        frame = self.last_left_frame.copy()
                    else:
                        frame = None
                    
                    # Get cached detections if recent
                    current_time = time.time()
                    if hasattr(self, 'last_detections_cache') and \
                       (current_time - self.last_detections_time) < 1.0:
                        detections = self.last_detections_cache.copy() if self.last_detections_cache else []
                    else:
                        detections = []
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Validate frame shape
                if not hasattr(frame, 'shape') or len(frame.shape) < 2:
                    time.sleep(0.01)
                    continue
                
                # Create display frame (copy for drawing)
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # Draw detections
                for det in detections:
                    bbox = det.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Draw segmentation mask if available and enabled
                    if show_masks and 'mask' in det and det['mask'] is not None:
                        try:
                            mask = det['mask']
                            # Ensure mask is the right size
                            if mask.shape[:2] != (h, w):
                                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            else:
                                mask_resized = mask.astype(np.uint8)
                            
                            # Create colored mask overlay (semi-transparent)
                            mask_color = (0, 255, 255)  # Cyan for masks
                            mask_overlay = display_frame.copy()
                            
                            # Apply mask with color
                            mask_bool = mask_resized > 0
                            mask_overlay[mask_bool] = (
                                display_frame[mask_bool] * 0.5 + 
                                np.array(mask_color) * 0.5
                            ).astype(np.uint8)
                            
                            # Blend overlay onto display frame
                            display_frame = cv2.addWeighted(display_frame, 0.7, mask_overlay, 0.3, 0)
                            
                            # Draw mask contour for better visibility
                            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display_frame, contours, -1, mask_color, 2)
                        except Exception as e:
                            if self.debug_mode:
                                print(f"{self.name}: Mask drawing error: {e}")
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center point (circle with crosshair)
                    center_color = (255, 0, 255)  # Magenta
                    center_radius = 5
                    cv2.circle(display_frame, (center_x, center_y), center_radius, center_color, -1)  # Filled circle
                    cv2.circle(display_frame, (center_x, center_y), center_radius + 2, center_color, 1)  # Outer circle
                    # Draw crosshair lines
                    crosshair_size = 8
                    cv2.line(display_frame, (center_x - crosshair_size, center_y), 
                            (center_x + crosshair_size, center_y), center_color, 1)
                    cv2.line(display_frame, (center_x, center_y - crosshair_size), 
                            (center_x, center_y + crosshair_size), center_color, 1)
                    
                    # Prepare label with ID
                    class_name = det.get('class_name', 'object')
                    score = det.get('score', 0.0)
                    track_id = det.get('track_id')
                    
                    # Always show ID if available, otherwise show index
                    if track_id is not None:
                        label = f"ID:{track_id} {class_name} {score:.2f}"
                    else:
                        # Try to get ID from detection if available
                        det_id = det.get('id')
                        if det_id is not None:
                            label = f"ID:{det_id} {class_name} {score:.2f}"
                        else:
                            label = f"{class_name} {score:.2f}"
                    
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
                
                # Show mask toggle state
                mask_status = "Masks: ON" if show_masks else "Masks: OFF"
                mask_color = (0, 255, 0) if show_masks else (0, 0, 255)
                cv2.putText(display_frame, mask_status, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, mask_color, 2)
                
                # Count masks available
                masks_count = sum(1 for det in detections if 'mask' in det and det['mask'] is not None)
                if masks_count > 0:
                    mask_info = f"Masks Available: {masks_count}"
                    cv2.putText(display_frame, mask_info, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display the frame
                try:
                    if display_frame is not None and hasattr(display_frame, 'shape') and len(display_frame.shape) >= 2:
                        if display_frame.size > 0:
                            cv2.imshow(window_name, display_frame)
                except Exception as e:
                    if self.debug_mode:
                        print(f"{self.name}: imshow error: {e}")
                
                # Check for quit key and mask toggle
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    show_masks = not show_masks
                    print(f"{self.name}: Masks {'enabled' if show_masks else 'disabled'}")
                
                time.sleep(0.01)  # Small delay for smoother display
                
        except KeyboardInterrupt:
            print(f"\n{self.name}: Visual debug mode interrupted by user")
        except Exception as e:
            print(f"{self.name}: Error in visual debug mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
            print(f"{self.name}: Visual debug mode ended")
    
    def debug_tuner(self):
        """
        Open a tuner window with trackbars to adjust smoothing parameters in real-time.
        Press 'q' to quit, 's' to save parameters to config.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start tuner mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting parameter tuner window...")
        print(f"{self.name}: Press 'q' to exit, 's' to save parameters")
        
        tuner_window_name = f"{self.name} - Parameter Tuner"
        
        # Callback functions for trackbars (OpenCV requires these to be simple functions)
        # We'll use a closure to access self
        def nothing(x):
            pass
        
        try:
            # Create tuner window (larger to fit all trackbars)
            cv2.namedWindow(tuner_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(tuner_window_name, 500, 800)
            
            # Create trackbars (values are stored as integers 0-100 for precision)
            # Smoothing enabled (0 = disabled, 1 = enabled)
            cv2.createTrackbar('Smoothing Enabled', tuner_window_name, 
                             1 if self.smoothing_enabled else 0, 1, nothing)
            
            # Box alpha (0-100, representing 0.0-1.0)
            cv2.createTrackbar('Box Alpha (x100)', tuner_window_name, 
                             int(self.smoothing_box_alpha * 100), 100, nothing)
            
            # Angle alpha (0-100, representing 0.0-1.0)
            cv2.createTrackbar('Angle Alpha (x100)', tuner_window_name, 
                             int(self.smoothing_angle_alpha * 100), 100, nothing)
            
            # Confidence alpha (0-100, representing 0.0-1.0)
            cv2.createTrackbar('Conf Alpha (x100)', tuner_window_name, 
                             int(self.smoothing_confidence_alpha * 100), 100, nothing)
            
            # Min frames (0-20)
            cv2.createTrackbar('Min Frames', tuner_window_name, 
                             self.smoothing_min_frames, 20, nothing)
            
            # Timeout (0-500, representing 0.0-5.0 seconds, in 0.01s increments)
            cv2.createTrackbar('Timeout (x100ms)', tuner_window_name, 
                             int(self.smoothing_timeout * 100), 500, nothing)
            
            # YOLO confidence threshold (0-100, representing 0.0-1.0)
            yolo_conf = self.yolo_config.get('conf_threshold', 0.25) if self.yolo_config else 0.25
            cv2.createTrackbar('YOLO Conf (x100)', tuner_window_name, 
                             int(yolo_conf * 100), 100, nothing)
            
            # Tracking parameters (ByteTrack)
            track_thresh = self.yolo_config.get('track_thresh', 0.5) if self.yolo_config else 0.5
            high_thresh = self.yolo_config.get('track_high_thresh', 0.6) if self.yolo_config else 0.6
            match_thresh = self.yolo_config.get('track_match_thresh', 0.8) if self.yolo_config else 0.8
            track_buffer = self.yolo_config.get('track_buffer', 30) if self.yolo_config else 30
            frame_rate = self.yolo_config.get('frame_rate', 30) if self.yolo_config else 30
            
            cv2.createTrackbar('Track Thresh (x100)', tuner_window_name, 
                             int(track_thresh * 100), 100, nothing)
            cv2.createTrackbar('High Thresh (x100)', tuner_window_name, 
                             int(high_thresh * 100), 100, nothing)
            cv2.createTrackbar('Match Thresh (x100)', tuner_window_name, 
                             int(match_thresh * 100), 100, nothing)
            cv2.createTrackbar('Track Buffer', tuner_window_name, 
                             track_buffer, 100, nothing)
            cv2.createTrackbar('Frame Rate', tuner_window_name, 
                             frame_rate, 60, nothing)
            
            print(f"{self.name}: Tuner window ready. Adjust parameters and see changes in real-time.")
            
            while True:
                # Read trackbar values
                smoothing_enabled_val = cv2.getTrackbarPos('Smoothing Enabled', tuner_window_name)
                box_alpha_val = cv2.getTrackbarPos('Box Alpha (x100)', tuner_window_name)
                angle_alpha_val = cv2.getTrackbarPos('Angle Alpha (x100)', tuner_window_name)
                conf_alpha_val = cv2.getTrackbarPos('Conf Alpha (x100)', tuner_window_name)
                min_frames_val = cv2.getTrackbarPos('Min Frames', tuner_window_name)
                timeout_val = cv2.getTrackbarPos('Timeout (x100ms)', tuner_window_name)
                yolo_conf_val = cv2.getTrackbarPos('YOLO Conf (x100)', tuner_window_name)
                
                # Read tracking parameter values
                track_thresh_val = cv2.getTrackbarPos('Track Thresh (x100)', tuner_window_name)
                high_thresh_val = cv2.getTrackbarPos('High Thresh (x100)', tuner_window_name)
                match_thresh_val = cv2.getTrackbarPos('Match Thresh (x100)', tuner_window_name)
                track_buffer_val = cv2.getTrackbarPos('Track Buffer', tuner_window_name)
                frame_rate_val = cv2.getTrackbarPos('Frame Rate', tuner_window_name)
                
                # Update parameters (thread-safe)
                self.smoothing_enabled = bool(smoothing_enabled_val)
                self.smoothing_box_alpha = box_alpha_val / 100.0
                self.smoothing_angle_alpha = angle_alpha_val / 100.0
                self.smoothing_confidence_alpha = conf_alpha_val / 100.0
                self.smoothing_min_frames = max(1, min_frames_val)  # Ensure at least 1
                self.smoothing_timeout = timeout_val / 100.0
                
                # Update YOLO confidence threshold
                yolo_threshold = yolo_conf_val / 100.0
                if self.yolo_config:
                    self.yolo_config['conf_threshold'] = yolo_threshold
                if self.yolo:
                    self.yolo.conf_threshold = yolo_threshold
                
                # Update tracking parameters
                track_thresh = track_thresh_val / 100.0
                high_thresh = high_thresh_val / 100.0
                match_thresh = match_thresh_val / 100.0
                track_buffer = max(1, track_buffer_val)  # Ensure at least 1
                frame_rate = max(1, frame_rate_val)  # Ensure at least 1
                
                # Update config
                if self.yolo_config:
                    self.yolo_config['track_thresh'] = track_thresh
                    self.yolo_config['track_high_thresh'] = high_thresh
                    self.yolo_config['track_match_thresh'] = match_thresh
                    self.yolo_config['track_buffer'] = track_buffer
                    self.yolo_config['frame_rate'] = frame_rate
                
                # Update tracker dynamically if it exists
                if self.yolo and self.yolo.tracker is not None:
                    try:
                        # Access the underlying ByteTracker instance
                        if hasattr(self.yolo.tracker, 'tracker'):
                            tracker_instance = self.yolo.tracker.tracker
                            tracker_instance.track_thresh = track_thresh
                            tracker_instance.high_thresh = high_thresh
                            tracker_instance.match_thresh = match_thresh
                            tracker_instance.track_buffer = track_buffer
                            tracker_instance.frame_rate = frame_rate
                    except Exception as e:
                        if self.debug_mode:
                            print(f"{self.name}: Warning: Could not update tracker parameters: {e}")
                
                # Create info display image (larger to fit all parameters)
                info_img = np.zeros((600, 500, 3), dtype=np.uint8)
                
                # Display current parameter values
                y_offset = 25
                line_height = 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                color = (255, 255, 255)
                thickness = 1
                
                # Smoothing section
                cv2.putText(info_img, "--- Smoothing ---", 
                           (10, y_offset), font, font_scale, (0, 255, 255), thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Smoothing: {'ON' if self.smoothing_enabled else 'OFF'}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Box Alpha: {self.smoothing_box_alpha:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Angle Alpha: {self.smoothing_angle_alpha:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Conf Alpha: {self.smoothing_confidence_alpha:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Min Frames: {self.smoothing_min_frames}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Timeout: {self.smoothing_timeout:.2f}s", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                # YOLO section
                y_offset += 5
                cv2.putText(info_img, "--- YOLO ---", 
                           (10, y_offset), font, font_scale, (0, 255, 255), thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"YOLO Conf: {yolo_threshold:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                # Tracking section
                y_offset += 5
                cv2.putText(info_img, "--- ByteTrack ---", 
                           (10, y_offset), font, font_scale, (0, 255, 255), thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Track Thresh: {track_thresh:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"High Thresh: {high_thresh:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Match Thresh: {match_thresh:.2f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Track Buffer: {track_buffer}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                cv2.putText(info_img, f"Frame Rate: {frame_rate}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                # Status section
                y_offset += 5
                cv2.putText(info_img, "--- Status ---", 
                           (10, y_offset), font, font_scale, (0, 255, 255), thickness)
                y_offset += line_height
                
                # Display smoothing state info
                with self.smoothing_lock:
                    num_tracked = len(self.smoothing_state)
                cv2.putText(info_img, f"Tracked Objects: {num_tracked}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                # Display FPS
                cv2.putText(info_img, f"FPS: {self.current_fps:.1f}", 
                           (10, y_offset), font, font_scale, color, thickness)
                y_offset += line_height
                
                # Instructions
                cv2.putText(info_img, "Press 'q' to quit", 
                           (10, y_offset), font, font_scale, (0, 255, 0), thickness)
                y_offset += line_height
                cv2.putText(info_img, "Press 's' to save", 
                           (10, y_offset), font, font_scale, (0, 255, 0), thickness)
                
                # Show info window
                cv2.imshow(tuner_window_name, info_img)
                
                # Check for keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save parameters to config
                    if self._save_config():
                        print(f"{self.name}: ✅ Parameters saved!")
                    else:
                        print(f"{self.name}: ❌ Failed to save parameters")
                
                time.sleep(0.05)  # Small delay to reduce CPU usage
                
        except KeyboardInterrupt:
            print(f"\n{self.name}: Tuner mode interrupted by user")
        except Exception as e:
            print(f"{self.name}: Error in tuner mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cv2.destroyWindow(tuner_window_name)
            except:
                pass
            print(f"{self.name}: Tuner mode ended")
    
    def __repr__(self):
        return f"<VISION name={self.name}, connected={self.connected}>"

