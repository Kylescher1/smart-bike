"""
RKNN Model Inference Script

This script loads and runs RKNN models using rknnlite for inference on the NPU.
It supports YOLO models and can process camera feeds or image files.

Usage:
    python yolo_web_stream.py --model yolo/models/yolo11n.rknn --source 1

    python yolo/rknn_inference.py --model yolo/models/yolov8n.rknn --source /path/to/image.jpg
"""

import argparse
import sys
import time
import logging
import threading
from pathlib import Path
from collections import defaultdict

# Initialize logging before any imports that use it (especially torch)
# This prevents the "Unknown level: 'WARNING'" error
# Ensure logging module is fully loaded and initialized
import logging.handlers
# Force re-import of logging._checkLevel to ensure it's properly initialized
# The issue is that torch's matcher_utils tries to setLevel('WARNING') which
# requires logging._checkLevel to recognize 'WARNING' as a valid level string
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
# Ensure _checkLevel function works with string levels
if hasattr(logging, '_checkLevel'):
    # Monkey-patch _checkLevel to handle string levels if it doesn't already
    original_checkLevel = logging._checkLevel
    def patched_checkLevel(level):
        if isinstance(level, str):
            # Map string to numeric level
            level_mapping = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'WARN': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
            }
            if level.upper() in level_mapping:
                return level_mapping[level.upper()]
        return original_checkLevel(level)
    logging._checkLevel = patched_checkLevel

# Add system dist-packages to path for rknnlite (system package)
import site
system_packages = '/usr/lib/python3/dist-packages'
if system_packages not in sys.path:
    sys.path.insert(0, system_packages)

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.cam.Camera import Camera, CAMERA_CONFIG


class FastCamera:
    """
    Optimized camera wrapper with threaded async frame capture for minimal latency.
    Continuously grabs frames in background thread so latest frame is always ready.
    This eliminates blocking on frame capture, reducing overall latency.
    """
    def __init__(self, index: int, config: dict):
        self.camera = Camera(index=index, config=config)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.capture_thread = None
        
    def open(self):
        self.camera.open()
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        # Wait for first frame
        timeout = time.time() + 2.0
        while self.latest_frame is None and time.time() < timeout:
            time.sleep(0.01)
        if self.latest_frame is None:
            self.close()
            raise RuntimeError("Failed to capture initial frame")
    
    def _capture_loop(self):
        """Background thread that continuously grabs the latest frame."""
        while self.running:
            try:
                # Multiple grabs to ensure we get the absolute latest frame
                # This flushes any buffered frames and minimizes latency
                for _ in range(2):  # Reduced from 3 to 2 for better balance
                    self.camera.cap.grab()
                
                ret, frame = self.camera.cap.retrieve()
                if ret and frame is not None:
                    with self.frame_lock:
                        # Copy to avoid race conditions with main thread
                        self.latest_frame = frame.copy()
            except Exception:
                # If capture fails, continue trying (camera might be temporarily unavailable)
                time.sleep(0.001)
    
    def read_frame(self):
        """Get the latest captured frame (non-blocking, returns immediately)."""
        with self.frame_lock:
            # Return a copy to ensure thread safety
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def close(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        self.camera.close()
    
    @property
    def width(self):
        """Expose camera width for compatibility."""
        return self.camera.width
    
    @property
    def height(self):
        """Expose camera height for compatibility."""
        return self.camera.height

# Try to import Ultralytics ByteTracker, fallback to custom implementation
try:
    from ultralytics.trackers import BYTETracker as UltralyticsByteTracker
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolo11n.rknn"

# COCO class names (YOLO models typically use COCO dataset)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run RKNN model inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to RKNN model file. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (e.g., 0), image file, or video file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (width/height). Default: 640",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0). Default: 0.25",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target platform (RK3562/RK3566/RK3568/RK3588). Use 'None' or leave empty for on-device NPU. Default: None (on-device)",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=0,
        help="NPU core mask (0=auto, 1=core0, 2=core1, 4=core2, 3=core0+1, 7=all). Default: 0 (auto)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between inferences (0=process all frames). Default: 0",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window for maximum speed",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable ByteTrack multi-object tracking",
    )
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.5,
        help="Tracking confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=0.6,
        help="High confidence threshold for tracking. Default: 0.6",
    )
    parser.add_argument(
        "--track-match-thresh",
        type=float,
        default=0.8,
        help="IoU threshold for track matching. Default: 0.8",
    )
    parser.add_argument(
        "--fast-capture",
        action="store_true",
        default=True,
        help="Use threaded async frame capture for faster frame retrieval (default: True)",
    )
    parser.add_argument(
        "--no-fast-capture",
        dest="fast_capture",
        action="store_false",
        help="Disable threaded async capture (use regular Camera class)",
    )
    return parser.parse_args()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image to new_shape while maintaining aspect ratio and padding with color.
    Returns: (resized_image, ratio, (dw, dh))
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    # Sort by score
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        # Calculate IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + \
              (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


def iou_batch(boxes1, boxes2):
    """Calculate IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) array of boxes in [x1, y1, x2, y2] format
        boxes2: (M, 4) array of boxes in [x1, y1, x2, y2] format
    
    Returns:
        (N, M) array of IoU values
    """
    # Calculate intersection
    xx1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    yy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    xx2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    yy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    
    # Calculate union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    
    # Avoid division by zero
    union = np.maximum(union, 1e-6)
    iou = inter / union
    
    return iou


class Track:
    """Represents a single tracked object."""
    def __init__(self, track_id, bbox, score, class_id, class_name):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.score = score
        self.class_id = class_id
        self.class_name = class_name
        self.state = 'tracked'  # 'tracked', 'lost', 'removed'
        self.time_since_update = 0
        self.history = []  # List of (x_center, y_center) tuples
        
    def update(self, bbox, score):
        """Update track with new detection."""
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = score
        self.state = 'tracked'
        self.time_since_update = 0
        
        # Update history with center point
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        self.history.append((float(x_center), float(y_center)))
        
        # Keep only last 30 points
        if len(self.history) > 30:
            self.history.pop(0)
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = 'lost'
        self.time_since_update += 1
    
    def mark_removed(self):
        """Mark track as removed."""
        self.state = 'removed'


class ByteTrackerWrapper:
    """
    Wrapper for ByteTracker that uses Ultralytics ByteTracker when available,
    otherwise falls back to custom implementation.
    
    Note: Ultralytics ByteTracker is tightly coupled with YOLO Results objects,
    so we use the custom implementation which works directly with detection dicts.
    """
    def __init__(self, track_thresh=0.5, high_thresh=0.6, match_thresh=0.8, 
                 frame_rate=30, track_buffer=30):
        """
        Args:
            track_thresh: Detection confidence threshold for tracking
            high_thresh: High confidence threshold (for first matching stage)
            match_thresh: IoU threshold for matching
            frame_rate: Video frame rate
            track_buffer: Number of frames to keep lost tracks
        """
        # Use custom implementation (Ultralytics tracker requires Results objects)
        # The custom implementation follows the same ByteTrack algorithm
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
            track_buffer=track_buffer
        )
        self.use_ultralytics = False
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys: 'bbox', 'score', 'class_id', 'class_name'
        
        Returns:
            List of detections with added 'track_id' key
        """
        return self.tracker.update(detections)
    
    def get_track_history(self, track_id):
        """Get track history for a given track ID."""
        return self.tracker.get_track_history(track_id)
    
    @property
    def tracked_tracks(self):
        """Get tracked tracks (for compatibility)."""
        return self.tracker.tracked_tracks


class ByteTracker:
    """Custom ByteTrack multi-object tracker (fallback when Ultralytics not available)."""
    def __init__(self, track_thresh=0.5, high_thresh=0.6, match_thresh=0.8, 
                 frame_rate=30, track_buffer=30):
        """
        Args:
            track_thresh: Detection confidence threshold for tracking
            high_thresh: High confidence threshold (for first matching stage)
            match_thresh: IoU threshold for matching
            frame_rate: Video frame rate
            track_buffer: Number of frames to keep lost tracks
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        
        self.track_id_count = 0
        self.tracked_tracks = []  # List of Track objects
        self.lost_tracks = []     # List of Track objects
        self.removed_tracks = []  # List of Track objects
        
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys: 'bbox', 'score', 'class_id', 'class_name'
        
        Returns:
            List of detections with added 'track_id' key
        """
        self.frame_count += 1
        
        # Convert detections to numpy arrays
        if len(detections) == 0:
            # No detections - mark all tracks as lost
            for track in self.tracked_tracks:
                track.mark_lost()
            self.lost_tracks.extend(self.tracked_tracks)
            self.tracked_tracks = []
            return []
        
        # Separate detections by confidence
        det_high = []
        det_low = []
        for det in detections:
            if det['score'] >= self.high_thresh:
                det_high.append(det)
            elif det['score'] >= self.track_thresh:
                det_low.append(det)
        
        # Get boxes for matching
        det_high_boxes = np.array([det['bbox'] for det in det_high], dtype=np.float32)
        det_low_boxes = np.array([det['bbox'] for det in det_low], dtype=np.float32)
        tracked_boxes = np.array([track.bbox for track in self.tracked_tracks], dtype=np.float32)
        lost_boxes = np.array([track.bbox for track in self.lost_tracks], dtype=np.float32)
        
        # Stage 1: Match high-confidence detections with tracked tracks
        matched_pairs_high = []
        unmatched_dets_high = []
        unmatched_track_objects = []
        
        if len(det_high_boxes) > 0 and len(tracked_boxes) > 0:
            iou_matrix = iou_batch(det_high_boxes, tracked_boxes)
            matched_indices = self._linear_assignment(-iou_matrix)  # Negative for maximization
            
            matched_track_indices_set = set()
            for det_idx, track_idx in matched_indices:
                if iou_matrix[det_idx, track_idx] >= self.match_thresh:
                    matched_pairs_high.append((det_idx, track_idx))
                    matched_track_indices_set.add(track_idx)
            
            # Find unmatched detections and tracks
            all_det_indices = set(range(len(det_high)))
            matched_det_indices = set([d for d, _ in matched_pairs_high])
            unmatched_dets_high = list(all_det_indices - matched_det_indices)
            
            all_track_indices = set(range(len(self.tracked_tracks)))
            unmatched_track_indices = all_track_indices - matched_track_indices_set
            unmatched_track_objects = [self.tracked_tracks[i] for i in unmatched_track_indices]
        else:
            unmatched_dets_high = list(range(len(det_high)))
            unmatched_track_objects = list(self.tracked_tracks)
        
        # Update matched tracks
        for det_idx, track_idx in matched_pairs_high:
            det = det_high[det_idx]
            track = self.tracked_tracks[track_idx]
            track.update(det['bbox'], det['score'])
        
        # Stage 2: Match remaining high-confidence detections with lost tracks
        matched_pairs_lost = []
        unmatched_dets_high_remaining = []
        reactivated_tracks = []  # Initialize here
        
        if len(unmatched_dets_high) > 0 and len(lost_boxes) > 0:
            unmatched_det_boxes = det_high_boxes[unmatched_dets_high]
            iou_matrix = iou_batch(unmatched_det_boxes, lost_boxes)
            matched_indices = self._linear_assignment(-iou_matrix)
            
            for i, (det_idx_local, lost_idx) in enumerate(matched_indices):
                det_idx_global = unmatched_dets_high[det_idx_local]
                if iou_matrix[det_idx_local, lost_idx] >= self.match_thresh:
                    matched_pairs_lost.append((det_idx_global, lost_idx))
                else:
                    unmatched_dets_high_remaining.append(det_idx_global)
        else:
            unmatched_dets_high_remaining = unmatched_dets_high
        
        # Reactivate lost tracks (process in reverse order to avoid index issues)
        matched_pairs_lost_sorted = sorted(matched_pairs_lost, key=lambda x: x[1], reverse=True)
        for det_idx, lost_idx in matched_pairs_lost_sorted:
            det = det_high[det_idx]
            track = self.lost_tracks[lost_idx]
            track.update(det['bbox'], det['score'])
            reactivated_tracks.append(track)
            self.tracked_tracks.append(track)
            self.lost_tracks.pop(lost_idx)
        
        # Stage 3: Match low-confidence detections with remaining unmatched tracks
        matched_pairs_low = []
        unmatched_dets_low = []
        
        if len(det_low_boxes) > 0 and len(unmatched_track_objects) > 0:
            unmatched_track_boxes = np.array([track.bbox for track in unmatched_track_objects], dtype=np.float32)
            iou_matrix = iou_batch(det_low_boxes, unmatched_track_boxes)
            matched_indices = self._linear_assignment(-iou_matrix)
            
            matched_track_objects_set = set()
            for det_idx, track_idx_local in matched_indices:
                if iou_matrix[det_idx, track_idx_local] >= self.match_thresh:
                    matched_track = unmatched_track_objects[track_idx_local]
                    matched_pairs_low.append((det_idx, matched_track))
                    matched_track_objects_set.add(matched_track)
            
            unmatched_dets_low = [i for i in range(len(det_low)) 
                                 if i not in [d for d, _ in matched_pairs_low]]
        else:
            unmatched_dets_low = list(range(len(det_low)))
        
        # Update matched tracks with low-confidence detections
        for det_idx, track in matched_pairs_low:
            det = det_low[det_idx]
            track.update(det['bbox'], det['score'])
        
        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets_high_remaining:
            det = det_high[det_idx]
            new_track = Track(
                track_id=self.track_id_count,
                bbox=det['bbox'],
                score=det['score'],
                class_id=det['class_id'],
                class_name=det['class_name']
            )
            self.track_id_count += 1
            self.tracked_tracks.append(new_track)
        
        # Mark unmatched tracks as lost
        matched_track_objects = set()
        for _, track_idx in matched_pairs_high:
            matched_track_objects.add(self.tracked_tracks[track_idx])
        for _, track in matched_pairs_low:
            matched_track_objects.add(track)
        # Include reactivated tracks from Stage 2
        for track in reactivated_tracks:
            matched_track_objects.add(track)
        
        tracks_to_remove = []
        for track in self.tracked_tracks:
            if track not in matched_track_objects:
                track.mark_lost()
                self.lost_tracks.append(track)
                tracks_to_remove.append(track)
        
        # Remove lost tracks from tracked list
        self.tracked_tracks = [t for t in self.tracked_tracks if t not in tracks_to_remove]
        
        # Update lost tracks and remove old ones
        for track in self.lost_tracks:
            track.time_since_update += 1
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_tracks.append(track)
        
        self.lost_tracks = [t for t in self.lost_tracks if t.state != 'removed']
        
        # Prepare output detections with track IDs
        output_detections = []
        
        # Add tracked detections
        for track in self.tracked_tracks:
            if track.state == 'tracked':
                output_detections.append({
                    'bbox': track.bbox.tolist(),
                    'score': track.score,
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'track_id': track.track_id
                })
        
        return output_detections
    
    def _linear_assignment(self, cost_matrix):
        """
        Simple linear assignment using greedy matching.
        For better performance, could use Hungarian algorithm (scipy.optimize.linear_sum_assignment).
        """
        if cost_matrix.size == 0:
            return []
        
        # Greedy matching: sort by cost and match greedily
        matches = []
        used_rows = set()
        used_cols = set()
        
        # Flatten and sort
        flat_indices = np.argsort(cost_matrix.flatten())
        rows, cols = np.unravel_index(flat_indices, cost_matrix.shape)
        
        for row, col in zip(rows, cols):
            if row not in used_rows and col not in used_cols:
                matches.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        
        return matches
    
    def get_track_history(self, track_id):
        """Get track history for a given track ID."""
        for track in self.tracked_tracks + self.lost_tracks:
            if track.track_id == track_id:
                return track.history
        return []


def dfl(position):
    """Distribution Focal Loss (DFL) for YOLOv8/v11 box decoding."""
    # Use numpy implementation (matches official Radxa script)
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


def post_process_yolov8(input_data, conf_threshold=0.25, iou_threshold=0.45, img_size=(640, 640)):
    """Post-process YOLOv8/v11 outputs (multi-branch format).
    
    Matches official Radxa script - expects 6 outputs (3 scales × 2: boxes + classes).
    """
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    
    # Official Radxa script expects 6 outputs (3 branches × 2 outputs each)
    # If we have 9 outputs, we need to filter out the objectness outputs
    # The pattern is: [boxes_0, classes_0, obj_0, boxes_1, classes_1, obj_1, boxes_2, classes_2, obj_2]
    # We only use: [boxes_0, classes_0, boxes_1, classes_1, boxes_2, classes_2]
    if len(input_data) == 9:
        # Filter to get only boxes and classes (skip objectness outputs)
        filtered_outputs = []
        for i in range(3):
            filtered_outputs.append(input_data[i * 3])      # boxes
            filtered_outputs.append(input_data[i * 3 + 1])  # classes
        input_data = filtered_outputs
        # print(f"[INFO] Filtered 9 outputs to 6 (removed objectness outputs)")
    elif len(input_data) != 6:
        raise ValueError(f"Unexpected number of outputs: {len(input_data)}. Expected 6 or 9.")
    
    pair_per_branch = len(input_data) // default_branch
    
    # Process each branch (scale) - matches official Radxa script
    for i in range(default_branch):
        boxes.append(box_process_yolov8(input_data[pair_per_branch * i], img_size))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        # Use ones as placeholder (matches official script)
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    # Convert to float32 if needed (RKNN outputs might be int8)
    classes_conf = classes_conf.astype(np.float32)
    scores = scores.astype(np.float32)
    
    # DO NOT apply sigmoid to class scores - they're already probabilities or properly scaled
    # The official Radxa script uses them directly
    
    # Filter by confidence (matches official script)
    box_confidences = scores.reshape(-1)
    candidate, class_num = classes_conf.shape
    
    class_max_score = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    
    # Combined confidence = objectness (ones) * class_confidence
    _class_pos = np.where(class_max_score * box_confidences >= conf_threshold)
    scores = (class_max_score * box_confidences)[_class_pos]
    
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    
    if len(boxes) == 0:
        return [], [], []
    
    # NMS per class
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


def process_output(output, conf_threshold=0.25, iou_threshold=0.45, img_shape=(640, 640)):
    """
    Process RKNN model output to get detections.
    Supports YOLOv8/v11 (6 or 9 outputs) and older YOLO (single output) formats.
    """
    if not isinstance(output, list):
        output = [output]
    
    # Detect output format: YOLOv8/v11 has 6 or 9 outputs, older YOLO has 1 output
    if len(output) == 6 or len(output) == 9:
        # YOLOv8/v11 format (6 outputs = boxes+classes, 9 outputs = boxes+classes+objectness)
        # post_process_yolov8 will handle both cases
        boxes, classes, scores = post_process_yolov8(output, conf_threshold, iou_threshold, img_shape)
    elif len(output) == 1:
        # Older YOLO format: [batch, num_boxes, 85] where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        output_data = output[0]
        if len(output_data.shape) == 3:
            output_data = output_data.reshape(-1, output_data.shape[-1])
        
        # Extract boxes, scores, classes
        boxes_norm = output_data[:, :4]  # [x_center, y_center, w, h] (normalized)
        objectness = output_data[:, 4:5]  # objectness score
        class_scores = output_data[:, 5:]  # class scores
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = np.max(class_scores, axis=1)
        
        # Combined confidence = objectness * class_confidence
        scores = objectness.flatten() * class_conf
        
        # Filter by confidence threshold
        valid = scores > conf_threshold
        boxes_norm = boxes_norm[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        
        if len(boxes_norm) == 0:
            return []
        
        # Convert normalized [x_center, y_center, w, h] to pixel [x1, y1, x2, y2]
        h, w = img_shape
        boxes = np.zeros_like(boxes_norm)
        boxes[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * w  # x1
        boxes[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * h  # y1
        boxes[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * w  # x2
        boxes[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * h  # y2
        
        # Clip to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        
        # Apply NMS
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = class_ids[keep]
    else:
        # print(f"[WARN] Unexpected number of outputs: {len(output)}. Trying to process as single output...")
        # Try to process first output
        output_data = output[0]
        if len(output_data.shape) == 3:
            output_data = output_data.reshape(-1, output_data.shape[-1])
        # Assume same format as older YOLO
        boxes_norm = output_data[:, :4]
        objectness = output_data[:, 4:5] if output_data.shape[1] > 4 else np.ones((len(output_data), 1))
        class_scores = output_data[:, 5:] if output_data.shape[1] > 5 else output_data[:, 4:]
        
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = np.max(class_scores, axis=1)
        scores = objectness.flatten() * class_conf
        
        valid = scores > conf_threshold
        boxes_norm = boxes_norm[valid]
        scores = scores[valid]
        classes = class_ids[valid]
        
        if len(boxes_norm) == 0:
            return []
        
        h, w = img_shape
        boxes = np.zeros_like(boxes_norm)
        boxes[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * w
        boxes[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * h
        boxes[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * w
        boxes[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * h
        
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
    
    if len(boxes) == 0:
        return []
    
    # Format as list of detections
    detections = []
    for i in range(len(boxes)):
        detections.append({
            'bbox': boxes[i].astype(int),
            'score': float(scores[i]),
            'class_id': int(classes[i]),
            'class_name': COCO_CLASSES[classes[i]] if classes[i] < len(COCO_CLASSES) else f'class_{classes[i]}'
        })
    
    return detections


def draw_detections(img, detections, tracker=None):
    """Draw bounding boxes, labels, and tracking trails on image."""
    for det in detections:
        bbox = det['bbox']
        # Ensure bbox is a list/array with 4 elements and convert to ints
        if isinstance(bbox, np.ndarray):
            bbox = bbox.flatten()
        if len(bbox) != 4:
            print(f"[WARN] Invalid bbox format: {bbox}, skipping detection", file=sys.stderr)
            continue
        
        # Convert to Python ints (OpenCV requires native ints, not numpy ints)
        x1, y1, x2, y2 = [int(float(c)) for c in bbox[:4]]
        
        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            print(f"[WARN] Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2}), skipping", file=sys.stderr)
            continue
        
        score = det['score']
        class_name = det['class_name']
        track_id = det.get('track_id', None)
        
        # Generate color based on track_id for consistency
        if track_id is not None:
            # Use hash of track_id to generate consistent color
            color_hash = hash(track_id) % 360
            color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[color_hash, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with track ID if available
        if track_id is not None:
            label = f"{class_name} #{track_id} {score:.2f}"
        else:
            label = f"{class_name} {score:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw tracking trail if tracker is provided and track_id exists
        if tracker is not None and track_id is not None:
            track_history = tracker.get_track_history(track_id)
            if len(track_history) > 1:
                # Convert history to numpy array for drawing
                points = np.array(track_history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points], isClosed=False, color=color, thickness=2)
    
    return img


def main():
    args = parse_args()
    
    # Validate model file
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1
    
    # print(f"[INFO] Loading RKNN model: {model_path}")
    
    # Initialize RKNN
    rknn = RKNNLite(verbose=False)
    
    # Load model
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        print(f"[ERROR] Failed to load RKNN model: {ret}", file=sys.stderr)
        rknn.release()
        return 1
    
    # Initialize runtime
    # When running on-device, target should be None (uses internal NPU)
    # target is only needed when connecting via ADB to a remote device
    if args.target is None or (isinstance(args.target, str) and (args.target.lower() == 'none' or args.target == '')):
        target = None
        # print("[INFO] Initializing runtime for on-device NPU...")
    else:
        target = args.target
        # print(f"[INFO] Initializing runtime for {target}...")
    
    ret = rknn.init_runtime(target=target, core_mask=args.core)
    if ret != 0:
        print(f"[ERROR] Failed to initialize runtime: {ret}", file=sys.stderr)
        rknn.release()
        return 1
    
    # print("[INFO] Model loaded and runtime initialized successfully!")
    
    # Determine input source
    source = args.source if args.source is not None else "0"
    camera = None
    cap = None  # Keep for video files
    
    if source.isdigit():
        # Camera - use Camera class
        source = int(source)
        
        # Optimized camera config: use lower resolution for faster capture
        # Since we resize to 640x640 anyway, lower capture resolution reduces capture time significantly
        camera_config = CAMERA_CONFIG.copy()
        camera_config.update({
            "width": 800,    # Lower resolution for faster capture (was 1280)
            "height": 600,   # Lower resolution for faster capture (was 720)
            "fps": 30,       # Lower FPS target for stability
            "fourcc": "MJPG"  # MJPEG is usually faster than raw formats
        })
        
        try:
            # Use FastCamera for threaded async capture (faster than regular Camera)
            if args.fast_capture:
                camera = FastCamera(index=source, config=camera_config)
            else:
                camera = Camera(index=source, config=camera_config)
            camera.open()
            
            # Test if we can actually read a frame
            test_frame = camera.read_frame()
            if test_frame is None:
                print(f"[ERROR] Camera {source} opened but cannot read frames. Check camera connection.", file=sys.stderr)
                camera.close()
                rknn.release()
                return 1
            
            # print(f"[INFO] Using camera {source} (resolution: {camera.width}x{camera.height})")
        except RuntimeError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            rknn.release()
            return 1
        
        is_video = True
    else:
        # Image or video file
        source_path = Path(source)
        if not source_path.exists():
            print(f"[ERROR] Source file not found: {source}", file=sys.stderr)
            rknn.release()
            return 1
        
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Single image
            cap = None
            img = cv2.imread(str(source_path))
            if img is None:
                print(f"[ERROR] Failed to load image: {source}", file=sys.stderr)
                rknn.release()
                return 1
            # print(f"[INFO] Processing image: {source}")
            is_video = False
        else:
            # Video file
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                print(f"[ERROR] Failed to open video: {source}", file=sys.stderr)
                rknn.release()
                return 1
            # print(f"[INFO] Processing video: {source}")
            is_video = True
    
    window_name = "RKNN Inference"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Initialize ByteTracker if tracking is enabled
    tracker = None
    if args.track:
        tracker = ByteTrackerWrapper(
            track_thresh=args.track_thresh,
            high_thresh=args.track_high_thresh,
            match_thresh=args.track_match_thresh,
            frame_rate=30,
            track_buffer=30
        )
        print("[INFO] ByteTrack tracking enabled", file=sys.stderr)
    
    frame_count = 0
    prev_time = time.time()
    
    # Pre-allocate buffers for better performance
    img_input_buffer = None
    
    # Timing diagnostics
    total_time = 0
    capture_time = 0
    preprocess_time = 0
    inference_time_total = 0
    postprocess_time = 0
    display_time = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Get frame
            capture_start = time.time()
            if is_video:
                if camera is not None:
                    # Use Camera class
                    frame = camera.read_frame()
                    if frame is None:
                        # print("[WARN] Failed to read frame from camera. Retrying...")
                        time.sleep(0.1)
                        continue
                else:
                    # Fallback for video files
                    ret, frame = cap.read()
                    if not ret:
                        # print("[WARN] Failed to read frame from camera. Retrying...")
                        time.sleep(0.1)
                        continue
                    if frame is None:
                        # print("[WARN] Received empty frame. Retrying...")
                        time.sleep(0.1)
                        continue
                
                # Skip frames if requested (for faster processing)
                if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                    frame_count += 1
                    continue
            else:
                frame = img.copy()
            capture_time += (time.time() - capture_start) * 1000
            
            # Preprocess (optimized: avoid unnecessary copies)
            preprocess_start = time.time()
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(args.imgsz, args.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # RKNN expects 4D input: (batch, height, width, channels) for NHWC format
            # Pre-allocate buffer if not exists, or reuse if same size
            if img_input_buffer is None or img_input_buffer.shape != (1, args.imgsz, args.imgsz, 3):
                img_input_buffer = np.zeros((1, args.imgsz, args.imgsz, 3), dtype=np.uint8)
            img_input_buffer[0] = img_rgb.astype(np.uint8)
            img_input = img_input_buffer
            preprocess_time += (time.time() - preprocess_start) * 1000
            
            # Run inference
            inference_start = time.time()
            try:
                outputs = rknn.inference([img_input])
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
                continue
            inference_time_ms = (time.time() - inference_start) * 1000  # ms
            inference_time_total += inference_time_ms
            
            # Process output
            if outputs is None:
                # print("[WARN] Inference returned None, skipping frame")
                continue
            
            postprocess_start = time.time()
            detections = []
            try:
                detections = process_output(outputs, conf_threshold=args.conf, img_shape=(args.imgsz, args.imgsz))
                # Debug: print detection stats occasionally
                # if frame_count % 30 == 0 and len(detections) == 0:
                #     # Check raw output stats
                #     if isinstance(outputs, list) and len(outputs) > 0:
                #         print(f"[DEBUG] Output shapes: {[o.shape for o in outputs]}")
                #         if len(outputs) >= 3:
                #             # Check objectness score range
                #             obj_scores = outputs[2]  # First objectness output
                #             print(f"[DEBUG] Objectness range: [{obj_scores.min():.3f}, {obj_scores.max():.3f}]")
            except Exception as e:
                print(f"[ERROR] Failed to process output: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                # Continue to display frame even if processing failed
                detections = []
            
            # Scale boxes back to original image size (vectorized for speed)
            if detections:
                h_orig, w_orig = frame.shape[:2]
                scale = min(args.imgsz / w_orig, args.imgsz / h_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                pad_x = (args.imgsz - new_w) / 2
                pad_y = (args.imgsz - new_h) / 2
                
                # Vectorized box scaling (much faster than loop)
                boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
                # Convert to Python ints (not numpy ints) for OpenCV compatibility
                boxes = boxes.astype(np.float32)  # Keep as float first
                
                for i, det in enumerate(detections):
                    # Convert to list of Python ints
                    det['bbox'] = [int(float(boxes[i, 0])), int(float(boxes[i, 1])), 
                                   int(float(boxes[i, 2])), int(float(boxes[i, 3]))]
            
            # Update tracker if enabled
            if tracker is not None:
                num_detections_before = len(detections)
                tracked_detections = tracker.update(detections)
                num_detections_after = len(tracked_detections)
                detections = tracked_detections
                # Debug: log if tracking filtered out many detections
                if num_detections_before > 0 and num_detections_after == 0 and frame_count % 30 == 0:
                    print(f"[DEBUG] Tracker filtered out all {num_detections_before} detections (frame {frame_count})", file=sys.stderr)
            
            postprocess_time += (time.time() - postprocess_start) * 1000
            
            # Draw detections and display (skip if no-display mode)
            display_start = time.time()
            if not args.no_display:
                annotated = draw_detections(frame.copy(), detections, tracker=tracker)
                
                # Calculate FPS
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                
                # Add info text with detailed timing
                track_info = f" | Tracks: {len(tracker.tracked_tracks)}" if tracker else ""
                info_text = f"FPS: {fps:.1f} | Inf: {inference_time_ms:.1f}ms | Det: {len(detections)}{track_info}"
                cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display
                cv2.imshow(window_name, annotated)
                display_time += (time.time() - display_start) * 1000
                
                # Exit on 'q' or Esc (only if display is enabled)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    # print("[INFO] Exiting...")
                    break
            else:
                # Still calculate FPS for monitoring
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                display_time += (time.time() - display_start) * 1000
                # In no-display mode, check for keyboard interrupt
                time.sleep(0.001)  # Small sleep to allow interrupt
            
            loop_time = (time.time() - loop_start) * 1000
            total_time += loop_time
            frame_count += 1
            
            # Print timing breakdown every 60 frames
            if frame_count % 60 == 0:
                avg_capture = capture_time / frame_count
                avg_preprocess = preprocess_time / frame_count
                avg_inference = inference_time_total / frame_count
                avg_postprocess = postprocess_time / frame_count
                avg_display = display_time / frame_count
                avg_total = total_time / frame_count
                print(f"[TIMING] Frame {frame_count}: Capture={avg_capture:.1f}ms | "
                      f"Preprocess={avg_preprocess:.1f}ms | Inference={avg_inference:.1f}ms | "
                      f"Postprocess={avg_postprocess:.1f}ms | Display={avg_display:.1f}ms | "
                      f"Total={avg_total:.1f}ms ({1000/avg_total:.1f} FPS)", file=sys.stderr)
            
            # For single image, wait for key press
            if not is_video:
                # print(f"[INFO] Processed image. Found {len(detections)} detections")
                # print("[INFO] Press any key to exit...")
                cv2.waitKey(0)
                break
    
    except KeyboardInterrupt:
        # print("\n[INFO] Interrupted by user")
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Print final timing summary
        if frame_count > 0:
            avg_capture = capture_time / frame_count
            avg_preprocess = preprocess_time / frame_count
            avg_inference = inference_time_total / frame_count
            avg_postprocess = postprocess_time / frame_count
            avg_display = display_time / frame_count
            avg_total = total_time / frame_count
            print(f"\n[TIMING SUMMARY] Processed {frame_count} frames:", file=sys.stderr)
            print(f"  Capture:     {avg_capture:.2f}ms ({100*avg_capture/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Preprocess:  {avg_preprocess:.2f}ms ({100*avg_preprocess/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Inference:   {avg_inference:.2f}ms ({100*avg_inference/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Postprocess: {avg_postprocess:.2f}ms ({100*avg_postprocess/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Display:     {avg_display:.2f}ms ({100*avg_display/avg_total:.1f}%)", file=sys.stderr)
            print(f"  Total:       {avg_total:.2f}ms ({1000/avg_total:.2f} FPS)", file=sys.stderr)
        
        # Clean up camera resources
        if camera is not None:
            camera.close()
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        rknn.release()
        # print("[INFO] Released RKNN resources")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

