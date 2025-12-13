#!/usr/bin/env python3
"""
RKNN YOLO Detector

YOLO detector using RKNN NPU backend for fast inference on Rockchip devices.
Follows the same interface as YOLODetector for compatibility with MultiCameraYOLO.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from collections import deque

# Add system dist-packages to path for rknnlite
import site
system_packages = '/usr/lib/python3/dist-packages'
if system_packages not in sys.path:
    sys.path.insert(0, system_packages)

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    print("[WARN] rknnlite not available. RKNN detector will not work.")

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.cam.Camera import Camera

ROOT = Path(__file__).resolve().parent

# COCO class names
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


@dataclass
class Detection:
    """Single detection (compatible with YOLODetector format)"""
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    class_id: int = -1


@dataclass
class DetectionResult:
    """Detection result (compatible with YOLODetector format)"""
    frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Detection]
    fps: float
    inference_time_ms: float
    timestamp: float


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to new_shape while maintaining aspect ratio and padding."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.array([_compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def post_process_yolov8(input_data, conf_threshold=0.25, iou_threshold=0.45, img_size=(640, 640)):
    """Post-process YOLOv8/v11 outputs."""
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    
    if len(input_data) == 9:
        filtered_outputs = []
        for i in range(3):
            filtered_outputs.append(input_data[i * 3])
            filtered_outputs.append(input_data[i * 3 + 1])
        input_data = filtered_outputs
    elif len(input_data) != 6:
        raise ValueError(f"Unexpected number of outputs: {len(input_data)}. Expected 6 or 9.")
    
    pair_per_branch = len(input_data) // default_branch
    
    for i in range(default_branch):
        boxes.append(box_process_yolov8(input_data[pair_per_branch * i], img_size))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    
    def sp_flatten(_in):
        """Flatten spatial dimensions while preserving channels."""
        if len(_in.shape) == 4:
            # Shape: (batch, channels, height, width)
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)  # (batch, height, width, channels)
            return _in.reshape(-1, ch)
        elif len(_in.shape) == 3:
            # Already flattened or different format
            # Try to reshape to (batch, channels, spatial)
            if _in.shape[0] == 1:
                # (batch, channels, spatial) -> flatten spatial
                ch = _in.shape[1]
                return _in.reshape(-1, ch)
            else:
                # Assume (spatial, channels) or similar
                return _in.reshape(-1, _in.shape[-1])
        else:
            # Already 2D or unexpected shape
            return _in.reshape(-1, _in.shape[-1])
    
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    classes_conf = classes_conf.astype(np.float32)
    scores = scores.astype(np.float32)
    
    box_confidences = scores.reshape(-1)
    candidate, class_num = classes_conf.shape
    
    class_max_score = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    
    _class_pos = np.where(class_max_score * box_confidences >= conf_threshold)
    scores = (class_max_score * box_confidences)[_class_pos]
    
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    
    if len(boxes) == 0:
        return [], [], []
    
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_vals = classes[inds]
        s = scores[inds]
        keep = nms(b, s, iou_threshold)
        
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_vals[keep])
            nscores.append(s[keep])
    
    if not nclasses and not nscores:
        return [], [], []
    
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    
    return boxes, classes, scores


def dfl(position):
    """Distribution Focal Loss (DFL) for YOLOv8/v11 box decoding."""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    x = position.reshape(n, p_num, mc, h, w)
    # Softmax
    exp_x = np.exp(x - np.max(x, axis=2, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, axis=2, keepdims=True)
    # Weighted sum
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    y = np.sum(softmax_x * acc_metrix, axis=2)
    return y


def box_process_yolov8(position, img_size=(640, 640)):
    """Process YOLOv8/v11 box outputs."""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w]).reshape(1, 2, 1, 1)
    
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    
    return xyxy


class RKNNYOLODetector:
    """
    RKNN-based YOLO detector following the same interface as YOLODetector.
    Uses NPU for fast inference on Rockchip devices.
    """
    
    def __init__(self, name="RKNNYOLODetector", camera=None, weights=None, conf=0.25, imgsz=640):
        """
        Initialize RKNN YOLO detector.
        
        Args:
            name: Detector name
            camera: Camera object
            weights: Path to .rknn model file
            conf: Confidence threshold
            imgsz: Input image size
        """
        if not RKNN_AVAILABLE:
            raise ImportError("rknnlite not available. Cannot use RKNN detector.")
        
        self.name = name
        self.camera = camera
        self.conf = conf
        self.imgsz = imgsz
        
        # Convert weights path
        if weights is None:
            weights = ROOT / "models" / "yolo11n.rknn"
        if isinstance(weights, str):
            weights = Path(weights)
        if not weights.is_absolute():
            weights = ROOT / weights
        
        self.weights = weights
        
        if not self.weights.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.weights}")
        
        # RKNN instance
        self.rknn = None
        
        # Detection loop
        self.running = False
        self.thread = None
        self.data_buffer = deque(maxlen=2)
        self.connected = False
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Pre-allocated buffers
        self.img_input_buffer = None
        
    def start(self):
        """Initialize RKNN and start detection thread."""
        if not RKNN_AVAILABLE:
            raise RuntimeError("RKNN not available")
        
        print(f"[{self.name}] Loading RKNN model: {self.weights}")
        
        # Initialize RKNN
        self.rknn = RKNNLite(verbose=False)
        
        # Load model
        ret = self.rknn.load_rknn(str(self.weights))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        # Initialize runtime (None = on-device NPU)
        ret = self.rknn.init_runtime(target=None, core_mask=0)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKNN runtime: {ret}")
        
        # Query model input size and adjust if needed
        try:
            input_attrs = self.rknn.query_input_attrs(0)
            model_input_size = input_attrs['dims'][2]  # Height/width (assuming square)
            if model_input_size != self.imgsz:
                print(f"[{self.name}] Model expects {model_input_size}x{model_input_size} input, adjusting from {self.imgsz}")
                self.imgsz = model_input_size
        except Exception as e:
            print(f"[{self.name}] Could not query model input size, using provided size {self.imgsz}: {e}")
        
        print(f"[{self.name}] RKNN model loaded successfully (input size: {self.imgsz}x{self.imgsz})")
        
        # Start detection thread
        self.running = True
        self.connected = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        
        # Wait for first result
        timeout = time.time() + 5.0
        while len(self.data_buffer) == 0 and time.time() < timeout:
            time.sleep(0.1)
        
        if len(self.data_buffer) == 0:
            raise RuntimeError("Failed to get first detection result")
    
    def _detection_loop(self):
        """Background thread for continuous detection."""
        while self.running:
            try:
                if self.camera is None:
                    time.sleep(0.1)
                    continue
                
                # Read frame
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Preprocess
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
                    inference_time = (time.time() - inference_start) * 1000  # ms
                except Exception as e:
                    print(f"[{self.name}] RKNN inference error: {e}")
                    print(f"[{self.name}] Input shape: {img_input.shape}, Expected: model input size")
                    inference_time = 0
                    outputs = None
                
                if outputs is None:
                    time.sleep(0.01)
                    continue
                
                # Debug: print output shapes on first run
                if not hasattr(self, '_debug_printed'):
                    print(f"[{self.name}] RKNN output shapes: {[out.shape for out in outputs]}")
                    print(f"[{self.name}] Number of outputs: {len(outputs)}")
                    self._debug_printed = True
                
                # Post-process
                try:
                    boxes, classes, scores = post_process_yolov8(outputs, self.conf, 0.45, (self.imgsz, self.imgsz))
                except Exception as e:
                    print(f"[{self.name}] Post-processing error: {e}")
                    if outputs:
                        print(f"[{self.name}] Output shapes: {[out.shape for out in outputs]}")
                    time.sleep(0.01)
                    continue
                
                # Convert to Detection objects
                detections = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    # Scale back to original frame size
                    x1 = int((x1 - dw) / ratio)
                    y1 = int((y1 - dh) / ratio)
                    x2 = int((x2 - dw) / ratio)
                    y2 = int((y2 - dh) / ratio)
                    
                    # Clip to frame bounds
                    h, w = frame.shape[:2]
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    
                    class_id = int(classes[i])
                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f'class_{class_id}'
                    
                    detections.append(Detection(
                        label=class_name,
                        confidence=float(scores[i]),
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id
                    ))
                
                # Create annotated frame
                annotated_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.label} {det.confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Update FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Store result
                result = DetectionResult(
                    frame=frame,
                    annotated_frame=annotated_frame,
                    detections=detections,
                    fps=self.current_fps,
                    inference_time_ms=inference_time,
                    timestamp=time.time()
                )
                
                self.data_buffer.append(result)
                
            except Exception as e:
                print(f"[{self.name}] Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def read(self) -> Optional[DetectionResult]:
        """Return the most recent detection result."""
        if not self.connected or len(self.data_buffer) == 0:
            return None
        return self.data_buffer[-1]
    
    def stop(self):
        """Stop detector and cleanup."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.rknn:
            self.rknn.release()
        self.connected = False

