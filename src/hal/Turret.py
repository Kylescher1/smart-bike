import logging
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

import serial
import sys
import time
import argparse
import threading
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from collections import deque

import cv2
import numpy as np

#God I hate this import kyle
# Try to import RKNN for Radxa Rock Pi hardware acceleration
try:
    import site
    system_packages = '/usr/lib/python3/dist-packages'
    if system_packages not in sys.path:
        sys.path.insert(0, system_packages)
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# YOLODetector import is conditional - only imported when not using RKNN
# to avoid loading PyTorch unnecessarily
from src.hal.cam.Camera import Camera, CAMERA_CONFIG

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


class RKNNDetector:
    """RKNN-accelerated YOLO detector for Rock Pi 5B"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, imgsz: int = 640):
        if not RKNN_AVAILABLE:
            raise ImportError("RKNN not available. Please install rknnlite on Rock Pi 5B")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.rknn = None

    def load(self):
        """Load and initialize RKNN model"""
        self.rknn = RKNNLite(verbose=False)

        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.rknn.init_runtime(target=None, core_mask=0)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKNN runtime: {ret}")

        print("RKNN model loaded successfully")

    def letterbox(self, img, new_shape=(640, 640)):
        """Resize with padding"""
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
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def dfl(self, position):
        """Distribution Focal Loss for YOLOv8/v11"""
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        x = position.reshape(n, p_num, mc, h, w)
        exp_x = np.exp(x - np.max(x, axis=2, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=2, keepdims=True)
        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
        y = np.sum(softmax_x * acc_metrix, axis=2)
        return y

    def box_process(self, position, img_size=(640, 640)):
        """Process box outputs"""
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def nms(self, boxes, scores, iou_threshold=0.45):
        """Non-maximum suppression"""
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) +
                           (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                                       boxes[order[1:], 3] - boxes[order[1:], 1]) - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return np.array(keep)

    def post_process(self, output):
        """Post-process RKNN output"""
        boxes, scores, classes_conf = [], [], []
        default_branch = 3

        # Filter outputs if needed
        if len(output) == 9:
            filtered_outputs = []
            for i in range(3):
                filtered_outputs.append(output[i * 3])
                filtered_outputs.append(output[i * 3 + 1])
            output = filtered_outputs

        pair_per_branch = len(output) // default_branch

        for i in range(default_branch):
            boxes.append(self.box_process(output[pair_per_branch * i], (self.imgsz, self.imgsz)))
            classes_conf.append(output[pair_per_branch * i + 1])
            scores.append(np.ones_like(output[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf).astype(np.float32)
        scores = np.concatenate(scores).astype(np.float32)

        box_confidences = scores.reshape(-1)
        class_max_score = np.max(classes_conf, axis=-1)
        classes = np.argmax(classes_conf, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.conf_threshold)
        scores = (class_max_score * box_confidences)[_class_pos]
        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        if len(boxes) == 0:
            return []

        # NMS per class
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c_vals = classes[inds]
            s = scores[inds]
            keep = self.nms(b, s, 0.45)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c_vals[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return []

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        # Convert to detection format
        detections = []
        for i in range(len(boxes)):
            det = type('obj', (object,), {
                'bbox': boxes[i].astype(int).tolist(),
                'confidence': float(scores[i]),
                'class_id': int(classes[i]),
                'label': COCO_CLASSES[classes[i]] if classes[i] < len(COCO_CLASSES) else f'class_{classes[i]}'
            })()
            detections.append(det)

        return detections

    def detect(self, frame):
        """Run detection on frame"""
        # Preprocess
        img_resized, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb.astype(np.uint8), axis=0)

        # Inference
        outputs = self.rknn.inference([img_input])
        if outputs is None:
            return []

        # Post-process
        detections = self.post_process(outputs)

        # Scale boxes back to original image
        if detections:
            h_orig, w_orig = frame.shape[:2]
            scale = min(self.imgsz / w_orig, self.imgsz / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            pad_x = (self.imgsz - new_w) / 2
            pad_y = (self.imgsz - new_h) / 2

            for det in detections:
                bbox = det.bbox
                bbox[0] = int((bbox[0] - pad_x) / scale)
                bbox[1] = int((bbox[1] - pad_y) / scale)
                bbox[2] = int((bbox[2] - pad_x) / scale)
                bbox[3] = int((bbox[3] - pad_y) / scale)
                det.bbox = bbox

        return detections

    def release(self):
        """Release RKNN resources"""
        if self.rknn:
            self.rknn.release()


class Turret3DVisualizer:
    """Real-time 3D visualization of turret orientation and tracking"""

    def __init__(self, history_length: int = 100):
        self.history_length = history_length

        # State data
        self.pan_angle = 90.0  # Bottom servo (horizontal)
        self.tilt_angle = 90.0  # Top servo (vertical)
        self.target_x = 0.0  # Target position in image coordinates
        self.target_y = 0.0
        self.has_target = False
        self.error_x = 0.0
        self.error_y = 0.0

        # History tracking
        self.pan_history = deque(maxlen=history_length)
        self.tilt_history = deque(maxlen=history_length)
        self.target_history = deque(maxlen=history_length)

        # Threading
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Matplotlib figure and axes
        self.fig = None
        self.ax = None

        # Lazy-loaded matplotlib modules (only when 3D viz is actually used)
        self.plt = None
        self.FuncAnimation = None
        self.Axes3D = None
        self.Poly3DCollection = None

    def _import_matplotlib(self):
        """Lazy import matplotlib modules only when needed"""
        if self.plt is None:
            try:
                import matplotlib
                matplotlib.use('TkAgg')  # Use TkAgg backend for threading compatibility
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                self.plt = plt
                self.FuncAnimation = FuncAnimation
                self.Axes3D = Axes3D
                self.Poly3DCollection = Poly3DCollection
            except ImportError as e:
                raise ImportError(
                    f"Failed to import matplotlib 3D modules. "
                    f"This is required for 3D visualization. Error: {e}\n"
                    f"Try: pip install matplotlib"
                ) from e

    def start(self):
        """Start the visualization in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_visualization, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the visualization"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.fig and self.plt:
            self.plt.close(self.fig)

    def update(self, pan: float, tilt: float, target_x: float = 0, target_y: float = 0,
               has_target: bool = False, error_x: float = 0, error_y: float = 0):
        """Update the visualization data"""
        with self.lock:
            self.pan_angle = pan
            self.tilt_angle = tilt
            self.target_x = target_x
            self.target_y = target_y
            self.has_target = has_target
            self.error_x = error_x
            self.error_y = error_y

            # Update history
            self.pan_history.append(pan)
            self.tilt_history.append(tilt)
            if has_target:
                self.target_history.append((target_x, target_y))

    def _rotation_matrix_z(self, angle_deg: float) -> np.ndarray:
        """Create rotation matrix around Z axis (pan)"""
        angle_rad = np.radians(angle_deg - 90)  # 90° is forward
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _rotation_matrix_y(self, angle_deg: float) -> np.ndarray:
        """Create rotation matrix around Y axis (tilt)"""
        angle_rad = np.radians(angle_deg - 90)  # 90° is horizontal
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _create_camera_cone(self, fov_h: float = 60, fov_v: float = 45, length: float = 2.0):
        """Create vertices for camera field of view cone"""
        # Convert FOV to radians
        fov_h_rad = np.radians(fov_h / 2)
        fov_v_rad = np.radians(fov_v / 2)

        # Cone tip at origin
        tip = np.array([0, 0, 0])

        # Four corners of the cone base
        corners = [
            np.array([length, length * np.tan(fov_h_rad), length * np.tan(fov_v_rad)]),
            np.array([length, -length * np.tan(fov_h_rad), length * np.tan(fov_v_rad)]),
            np.array([length, -length * np.tan(fov_h_rad), -length * np.tan(fov_v_rad)]),
            np.array([length, length * np.tan(fov_h_rad), -length * np.tan(fov_v_rad)]),
        ]

        return tip, corners

    def _draw_gimbal(self, ax, pan: float, tilt: float):
        """Draw the turret gimbal structure"""
        # Base
        base_height = 0.3
        base_radius = 0.2
        theta = np.linspace(0, 2 * np.pi, 20)
        x_base = base_radius * np.cos(theta)
        y_base = base_radius * np.sin(theta)
        z_base_bottom = np.zeros_like(theta) - 0.5
        z_base_top = np.zeros_like(theta) - 0.5 + base_height

        ax.plot(x_base, y_base, z_base_bottom, 'k-', linewidth=2, alpha=0.5)
        ax.plot(x_base, y_base, z_base_top, 'k-', linewidth=2, alpha=0.5)

        # Pan axis (vertical post)
        ax.plot([0, 0], [0, 0], [-0.5, 0], 'b-', linewidth=3, label='Pan Axis')

        # Tilt arm (rotates with pan)
        R_pan = self._rotation_matrix_z(pan)
        tilt_arm = R_pan @ np.array([0.3, 0, 0])
        ax.plot([0, tilt_arm[0]], [0, tilt_arm[1]], [0, tilt_arm[2]],
                'g-', linewidth=3, label='Tilt Arm')

        # Camera position (at end of tilt arm)
        return tilt_arm

    def _draw_camera_view(self, ax, pan: float, tilt: float, camera_pos: np.ndarray):
        """Draw the camera field of view cone"""
        tip, corners = self._create_camera_cone()

        # Apply rotations: first tilt, then pan
        R_pan = self._rotation_matrix_z(pan)
        R_tilt = self._rotation_matrix_y(tilt)
        R_combined = R_pan @ R_tilt

        # Transform cone
        tip_world = camera_pos + R_combined @ tip
        corners_world = [camera_pos + R_combined @ c for c in corners]

        # Draw cone edges
        for corner in corners_world:
            ax.plot([tip_world[0], corner[0]],
                    [tip_world[1], corner[1]],
                    [tip_world[2], corner[2]],
                    'c-', alpha=0.3, linewidth=1)

        # Draw base rectangle
        base_points = corners_world + [corners_world[0]]
        base_x = [p[0] for p in base_points]
        base_y = [p[1] for p in base_points]
        base_z = [p[2] for p in base_points]
        ax.plot(base_x, base_y, base_z, 'c-', alpha=0.5, linewidth=1)

        # Fill cone faces with transparency
        faces = [
            [tip_world, corners_world[0], corners_world[1]],
            [tip_world, corners_world[1], corners_world[2]],
            [tip_world, corners_world[2], corners_world[3]],
            [tip_world, corners_world[3], corners_world[0]],
        ]

        poly = self.Poly3DCollection(faces, alpha=0.1, facecolor='cyan', edgecolor='none')
        ax.add_collection3d(poly)

        return tip_world, corners_world

    def _draw_target(self, ax, tip_world: np.ndarray, corners_world: List[np.ndarray]):
        """Draw target position in 3D space"""
        if not self.has_target:
            return

        # Map 2D image coordinates to 3D position on the FOV plane
        # Assume normalized coordinates (-0.5 to 0.5)
        norm_x = (self.target_x / 640.0) - 0.5  # Assuming 640 width
        norm_y = (self.target_y / 480.0) - 0.5  # Assuming 480 height

        # Calculate target position on the FOV base
        # Use bilinear interpolation on the cone base
        center = np.mean(corners_world, axis=0)

        # Simple projection: map to cone base
        right_vec = (corners_world[0] + corners_world[3]) / 2 - center
        down_vec = (corners_world[2] + corners_world[3]) / 2 - center

        target_pos = center + norm_x * right_vec + norm_y * down_vec

        # Draw target marker
        ax.scatter(*target_pos, c='red', s=200, marker='*',
                   edgecolors='yellow', linewidths=2, label='Target')

        # Draw line from camera to target
        ax.plot([tip_world[0], target_pos[0]],
                [tip_world[1], target_pos[1]],
                [tip_world[2], target_pos[2]],
                'r--', linewidth=2, alpha=0.7, label='To Target')

        # Draw error vector in 3D
        if abs(self.error_x) > 10 or abs(self.error_y) > 10:
            error_scale = 0.5
            error_vec = np.array([0, self.error_x * error_scale / 100,
                                  self.error_y * error_scale / 100])
            error_end = target_pos + error_vec
            ax.quiver(target_pos[0], target_pos[1], target_pos[2],
                      error_vec[0], error_vec[1], error_vec[2],
                      color='orange', arrow_length_ratio=0.3, linewidth=2,
                      label='Error Vector')

    def _draw_trajectory(self, ax):
        """Draw historical trajectory of pan/tilt angles"""
        if len(self.pan_history) < 2:
            return

        # Convert angle history to 3D positions
        positions = []
        for pan, tilt in zip(self.pan_history, self.tilt_history):
            R_pan = self._rotation_matrix_z(pan)
            R_tilt = self._rotation_matrix_y(tilt)
            # Camera pointing direction
            direction = R_pan @ R_tilt @ np.array([1.5, 0, 0])
            positions.append(direction)

        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'y-', alpha=0.5, linewidth=1, label='Trajectory')

    def _update_plot(self, frame):
        """Update function for animation"""
        with self.lock:
            pan = self.pan_angle
            tilt = self.tilt_angle

        self.ax.clear()

        # Set labels and title
        self.ax.set_xlabel('X (Forward)', fontsize=10)
        self.ax.set_ylabel('Y (Right)', fontsize=10)
        self.ax.set_zlabel('Z (Up)', fontsize=10)
        self.ax.set_title(f'Turret 3D View | Pan: {pan:.1f}° Tilt: {tilt:.1f}°',
                          fontsize=12, fontweight='bold')

        # Set axis limits
        limit = 3
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-1, limit])

        # Draw grid
        self.ax.grid(True, alpha=0.3)

        # Draw coordinate system origin
        origin = np.array([0, 0, 0])
        axis_length = 0.5
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.2, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.2, linewidth=2)

        # Draw gimbal
        camera_pos = self._draw_gimbal(self.ax, pan, tilt)

        # Draw camera view
        tip_world, corners_world = self._draw_camera_view(self.ax, pan, tilt, camera_pos)

        # Draw target
        self._draw_target(self.ax, tip_world, corners_world)

        # Draw trajectory
        self._draw_trajectory(self.ax)

        # Set view angle
        self.ax.view_init(elev=20, azim=45)

        # Add legend
        self.ax.legend(loc='upper left', fontsize=8)

        return self.ax,

    def _run_visualization(self):
        """Run the visualization loop"""
        try:
            # Import matplotlib modules (lazy loading)
            self._import_matplotlib()

            # Create figure
            self.fig = self.plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')

            # Set up animation
            ani = self.FuncAnimation(self.fig, self._update_plot, interval=50,
                                     blit=False, cache_frame_data=False)

            self.plt.tight_layout()
            self.plt.show()

        except Exception as e:
            print(f"3D Visualization error: {e}")
        finally:
            self.running = False


class ErrorPlotter:
    """Real-time plotting of position errors, PID outputs, and servo positions for debugging"""

    def __init__(self, history_length: int = 500):
        self.history_length = history_length

        # Data storage
        self.time_history = deque(maxlen=history_length)
        self.error_x_history = deque(maxlen=history_length)
        self.error_y_history = deque(maxlen=history_length)
        self.pid_output_x_history = deque(maxlen=history_length)
        self.pid_output_y_history = deque(maxlen=history_length)
        self.move_x_history = deque(maxlen=history_length)
        self.move_y_history = deque(maxlen=history_length)
        self.bottom_pos_history = deque(maxlen=history_length)
        self.top_pos_history = deque(maxlen=history_length)
        self.target_x_history = deque(maxlen=history_length)
        self.target_y_history = deque(maxlen=history_length)
        self.has_target_history = deque(maxlen=history_length)

        # Threading
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Matplotlib figure and axes
        self.fig = None
        self.axes = None

        # Lazy-loaded matplotlib modules
        self.plt = None
        self.FuncAnimation = None

        # Start time for relative time tracking
        self.start_time = None

    def _import_matplotlib(self):
        """Lazy import matplotlib modules only when needed"""
        if self.plt is None:
            try:
                import matplotlib
                # Use TkAgg backend for threading compatibility and non-blocking operation
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt

                # Enable interactive mode for non-blocking operation
                plt.ion()

                self.plt = plt
            except ImportError as e:
                raise ImportError(
                    f"Failed to import matplotlib modules. "
                    f"This is required for error plotting. Error: {e}\n"
                    f"Try: pip install matplotlib"
                ) from e

    def start(self):
        """Start the plotting in a separate thread"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run_plotting, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the plotting"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.fig and self.plt:
            try:
                self.plt.close(self.fig)
                self.plt.ioff()  # Turn off interactive mode
            except:
                pass

    def update(self, error_x_px: float, error_y_px: float,
               pid_output_x: float, pid_output_y: float,
               move_x: float, move_y: float,
               bottom_pos: float, top_pos: float,
               target_x: float = 0, target_y: float = 0,
               has_target: bool = False):
        """Update the plotting data"""
        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time() - self.start_time

        with self.lock:
            self.time_history.append(current_time)
            self.error_x_history.append(error_x_px)
            self.error_y_history.append(error_y_px)
            self.pid_output_x_history.append(pid_output_x)
            self.pid_output_y_history.append(pid_output_y)
            self.move_x_history.append(move_x)
            self.move_y_history.append(move_y)
            self.bottom_pos_history.append(bottom_pos)
            self.top_pos_history.append(top_pos)
            self.target_x_history.append(target_x if has_target else np.nan)
            self.target_y_history.append(target_y if has_target else np.nan)
            self.has_target_history.append(has_target)

    def _update_plot(self, frame=None):
        """Update function for async plotting"""
        if self.fig is None or self.axes is None:
            return

        with self.lock:
            if len(self.time_history) < 1:
                return

            times = np.array(self.time_history)
            error_x = np.array(self.error_x_history)
            error_y = np.array(self.error_y_history)
            pid_x = np.array(self.pid_output_x_history)
            pid_y = np.array(self.pid_output_y_history)
            move_x = np.array(self.move_x_history)
            move_y = np.array(self.move_y_history)
            bottom_pos = np.array(self.bottom_pos_history)
            top_pos = np.array(self.top_pos_history)
            target_x = np.array(self.target_x_history)
            target_y = np.array(self.target_y_history)
            has_target = np.array(self.has_target_history)

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Plot 1: Position Errors (pixels)
        ax = self.axes[0]
        ax.plot(times, error_x, 'r-', label='Error X (px)', linewidth=1.5, alpha=0.7)
        ax.plot(times, error_y, 'b-', label='Error Y (px)', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (pixels)')
        ax.set_title('Position Error (X=Horizontal, Y=Vertical)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Highlight large jumps (>50px change)
        if len(error_x) > 1:
            error_x_diff = np.abs(np.diff(error_x))
            error_y_diff = np.abs(np.diff(error_y))
            jump_threshold = 50
            x_jumps = np.where(error_x_diff > jump_threshold)[0]
            y_jumps = np.where(error_y_diff > jump_threshold)[0]

            for idx in x_jumps:
                if idx < len(times) - 1:
                    ax.axvline(x=times[idx], color='orange', linestyle=':', alpha=0.5, linewidth=1)
            for idx in y_jumps:
                if idx < len(times) - 1:
                    ax.axvline(x=times[idx], color='purple', linestyle=':', alpha=0.5, linewidth=1)

        # Plot 2: PID Outputs
        ax = self.axes[1]
        ax.plot(times, pid_x, 'r-', label='PID Output X', linewidth=1.5, alpha=0.7)
        ax.plot(times, pid_y, 'b-', label='PID Output Y', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('PID Output (normalized)')
        ax.set_title('PID Controller Outputs')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 3: Servo Movements (degrees)
        ax = self.axes[2]
        ax.plot(times, move_x, 'r-', label='Move X (deg)', linewidth=1.5, alpha=0.7)
        ax.plot(times, move_y, 'b-', label='Move Y (deg)', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Movement (degrees)')
        ax.set_title('Servo Movement Commands')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Highlight large movements (>5 degrees)
        if len(move_x) > 0:
            large_moves_x = np.where(np.abs(move_x) > 5)[0]
            large_moves_y = np.where(np.abs(move_y) > 5)[0]
            for idx in large_moves_x:
                ax.axvline(x=times[idx], color='red', linestyle=':', alpha=0.3, linewidth=1)
            for idx in large_moves_y:
                ax.axvline(x=times[idx], color='blue', linestyle=':', alpha=0.3, linewidth=1)

        # Plot 4: Servo Positions
        ax = self.axes[3]
        ax.plot(times, bottom_pos, 'r-', label='Bottom Servo (pan)', linewidth=1.5, alpha=0.7)
        ax.plot(times, top_pos, 'b-', label='Top Servo (tilt)', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (degrees)')
        ax.set_title('Servo Positions')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 5: Target Position (if available)
        ax = self.axes[4]
        if np.any(has_target):
            valid_mask = ~np.isnan(target_x)
            if np.any(valid_mask):
                ax.plot(times[valid_mask], target_x[valid_mask], 'r-', label='Target X', linewidth=1.5, alpha=0.7,
                        marker='o', markersize=2)
            valid_mask = ~np.isnan(target_y)
            if np.any(valid_mask):
                ax.plot(times[valid_mask], target_y[valid_mask], 'b-', label='Target Y', linewidth=1.5, alpha=0.7,
                        marker='o', markersize=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (pixels)')
        ax.set_title('Target Position in Frame')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 6: Error Magnitude
        ax = self.axes[5]
        error_magnitude = np.sqrt(error_x ** 2 + error_y ** 2)
        ax.plot(times, error_magnitude, 'g-', label='Error Magnitude', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error Magnitude (pixels)')
        ax.set_title('Total Error Magnitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Adjust layout
        self.plt.tight_layout()

        # Draw the figure (non-blocking)
        self.fig.canvas.draw_idle()

    def _run_plotting(self):
        """Run the plotting loop in async thread"""
        try:
            # Import matplotlib modules (lazy loading)
            self._import_matplotlib()

            # Create figure with subplots
            self.fig, self.axes = self.plt.subplots(3, 2, figsize=(14, 10))
            self.axes = self.axes.flatten()

            self.plt.tight_layout()
            self.fig.show()

            # Async update loop - runs independently without blocking
            update_interval = 0.1  # Update every 100ms
            while self.running:
                try:
                    # Update plots
                    self._update_plot(None)

                    # Non-blocking pause to allow GUI to process events
                    self.plt.pause(update_interval)

                    # Check if window is still open
                    if not self.plt.get_fignums():
                        break

                except Exception as e:
                    # Continue running even if update fails
                    if self.running:
                        print(f"Error plotter update error: {e}")
                    time.sleep(update_interval)

        except Exception as e:
            print(f"Error Plotting error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            if self.fig and self.plt:
                try:
                    self.plt.close(self.fig)
                except:
                    pass


class TrackingState:
    """Enumeration of turret tracking states"""
    TRACKING = "TRACKING"  # Actively tracking a target
    LOST = "LOST"  # Just lost track, waiting
    RETURN_TO_LAST = "RETURN_TO_LAST"  # Returning to last known position
    SEARCHING = "SEARCHING"  # At last position, looking for target
    SWEEPING = "SWEEPING"  # Executing sweep pattern
    GOING_HOME = "GOING_HOME"  # Moving to home before sweep


class DetectionSmoother:
    """Smooths detection positions over multiple frames to reduce jitter

    Uses a moving average filter on target center positions.
    """

    def __init__(self, window_size: int = 3, max_jump_px: float = 100.0):
        """
        Args:
            window_size: Number of frames to average over
            max_jump_px: Maximum allowed jump between frames (pixels).
                        Larger jumps are treated as new targets.
        """
        self.window_size = window_size
        self.max_jump_px = max_jump_px

        # History buffers
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.width_history = deque(maxlen=window_size)
        self.height_history = deque(maxlen=window_size)

        # Last raw detection for jump detection
        self.last_x = None
        self.last_y = None

    def update(self, x: float, y: float, width: float, height: float) -> Tuple[float, float, float, float]:
        """Update with new detection and return smoothed position

        Args:
            x, y: Center of detection (pixels)
            width, height: Size of detection (pixels)

        Returns:
            Smoothed (x, y, width, height)
        """
        # Check for large jump (possible new target or bad detection)
        if self.last_x is not None and self.last_y is not None:
            jump = np.sqrt((x - self.last_x) ** 2 + (y - self.last_y) ** 2)
            if jump > self.max_jump_px:
                # Large jump detected - could be new target, reset history
                self.clear()

        # Store raw values for next jump detection
        self.last_x = x
        self.last_y = y

        # Add to history
        self.x_history.append(x)
        self.y_history.append(y)
        self.width_history.append(width)
        self.height_history.append(height)

        # Calculate smoothed values (simple moving average)
        # If history is empty (shouldn't happen with window_size > 0), return raw values
        if len(self.x_history) == 0:
            return x, y, width, height

        smoothed_x = np.mean(self.x_history)
        smoothed_y = np.mean(self.y_history)
        smoothed_width = np.mean(self.width_history)
        smoothed_height = np.mean(self.height_history)

        # Check for NaN (can happen if window_size=0 or empty history)
        if np.isnan(smoothed_x) or np.isnan(smoothed_y):
            return x, y, width, height

        return smoothed_x, smoothed_y, smoothed_width, smoothed_height

    def clear(self):
        """Clear history (call when target is lost)"""
        self.x_history.clear()
        self.y_history.clear()
        self.width_history.clear()
        self.height_history.clear()
        self.last_x = None
        self.last_y = None

    def get_sample_count(self) -> int:
        """Get number of samples currently in buffer"""
        return len(self.x_history)

    def is_stable(self) -> bool:
        """Check if we have enough samples for stable smoothing"""
        return len(self.x_history) >= self.window_size


class SweepController:
    """Controls sweep pattern when target is lost

    Sweep pattern: A → X → B → X → C → X → D → X → A (repeat)
    Where:
        A = top-left corner
        B = top-right corner
        C = bottom-left corner
        D = bottom-right corner
        X = center (home)
    """

    def __init__(self, pan_min: float = 45, pan_max: float = 135,
                 tilt_min: float = 70, tilt_max: float = 110,
                 home_pan: float = 90, home_tilt: float = 90,
                 move_speed: float = 2.0):
        """
        Args:
            pan_min/max: Sweep area horizontal limits (degrees)
            tilt_min/max: Sweep area vertical limits (degrees)
            home_pan/tilt: Center position
            move_speed: Degrees per step when sweeping
        """
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.home_pan = home_pan
        self.home_tilt = home_tilt
        self.move_speed = move_speed

        # Define waypoints: A(top-left), X(center), B(top-right), X, C(bottom-left), X, D(bottom-right), X
        self.waypoints = [
            ('A', pan_min, tilt_max),  # Top-left
            ('X', home_pan, home_tilt),  # Center
            ('B', pan_max, tilt_max),  # Top-right
            ('X', home_pan, home_tilt),  # Center
            ('C', pan_min, tilt_min),  # Bottom-left
            ('X', home_pan, home_tilt),  # Center
            ('D', pan_max, tilt_min),  # Bottom-right
            ('X', home_pan, home_tilt),  # Center
        ]

        self.current_waypoint_idx = 0
        self.is_moving = False

    def reset(self):
        """Reset sweep to beginning"""
        self.current_waypoint_idx = 0
        self.is_moving = False

    def get_current_waypoint(self) -> Tuple[str, float, float]:
        """Get current waypoint (name, pan, tilt)"""
        return self.waypoints[self.current_waypoint_idx]

    def advance_waypoint(self):
        """Move to next waypoint in pattern"""
        self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)

    def calculate_move(self, current_pan: float, current_tilt: float) -> Tuple[float, float, bool]:
        """Calculate movement towards current waypoint

        Returns:
            (pan_move, tilt_move, reached_waypoint)
        """
        name, target_pan, target_tilt = self.get_current_waypoint()

        # Calculate error to waypoint
        pan_error = target_pan - current_pan
        tilt_error = target_tilt - current_tilt

        # Check if we've reached the waypoint (within tolerance)
        tolerance = 2.0  # degrees
        if abs(pan_error) < tolerance and abs(tilt_error) < tolerance:
            return 0, 0, True

        # Calculate movement (limited by move_speed)
        pan_move = np.clip(pan_error, -self.move_speed, self.move_speed)
        tilt_move = np.clip(tilt_error, -self.move_speed, self.move_speed)

        return pan_move, tilt_move, False

    def update_limits(self, pan_min: float, pan_max: float, tilt_min: float, tilt_max: float,
                      home_pan: float = 90, home_tilt: float = 90):
        """Update sweep area limits based on servo configuration"""
        # Add some margin from absolute limits
        margin = 10
        self.pan_min = pan_min + margin
        self.pan_max = pan_max - margin
        self.tilt_min = tilt_min + margin
        self.tilt_max = tilt_max - margin
        self.home_pan = home_pan
        self.home_tilt = home_tilt

        # Rebuild waypoints with new limits
        self.waypoints = [
            ('A', self.pan_min, self.tilt_max),
            ('X', self.home_pan, self.home_tilt),
            ('B', self.pan_max, self.tilt_max),
            ('X', self.home_pan, self.home_tilt),
            ('C', self.pan_min, self.tilt_min),
            ('X', self.home_pan, self.home_tilt),
            ('D', self.pan_max, self.tilt_min),
            ('X', self.home_pan, self.home_tilt),
        ]


class PIDController:
    """PID controller for servo positioning"""

    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1, max_output: float = 10.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.max_output = max_output  # Maximum output per cycle (degrees)

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0

        self.output_x = 0.0
        self.output_y = 0.0

    def update(self, error_x: float, error_y: float, dt: float = 0.033):
        """Update PID controller with new error values"""
        # X-axis (horizontal - bottom servo)
        self.integral_x += error_x * dt
        self.integral_x = np.clip(self.integral_x, -50, 50)  # Limit integral windup
        derivative_x = (error_x - self.last_error_x) / dt

        self.output_x = (self.kp * error_x +
                         self.ki * self.integral_x +
                         self.kd * derivative_x)

        # Y-axis (vertical - top servo)
        self.integral_y += error_y * dt
        self.integral_y = np.clip(self.integral_y, -50, 50)  # Limit integral windup
        derivative_y = (error_y - self.last_error_y) / dt

        self.output_y = (self.kp * error_y +
                         self.ki * self.integral_y +
                         self.kd * derivative_y)

        # PID OUTPUT CLAMP DISABLED - let PID move freely
        # self.output_x = np.clip(self.output_x, -self.max_output, self.max_output)
        # self.output_y = np.clip(self.output_y, -self.max_output, self.max_output)

        self.last_error_x = error_x
        self.last_error_y = error_y

        return self.output_x, self.output_y

    def reset(self):
        """Reset PID state"""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.output_x = 0.0
        self.output_y = 0.0


class TurretController:
    """Simple turret controller for serial communication"""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None

        # Use float positions internally (FIX #1)
        self.top_pos = 90.0
        self.bottom_pos = 90.0
        self.top_min = 0      # Full range - no artificial limits
        self.top_max = 180
        self.bottom_min = 0
        self.bottom_max = 180

        # Rate limiting (FIX #9)
        self.last_command_time = 0.0
        self.command_interval = 0.05  # 20 Hz max command rate
        self.min_angle_change = 0.5  # Don't send if change < 0.5 degrees

        # TF03 LiDAR distance tracking
        self.distance_cm = None  # Current distance in cm
        self.distance_available = None  # Whether LiDAR is available (None=unknown, True=yes, False=no)
        self.last_distance_time = 0.0
        self.distance_update_interval = 0.5  # Update distance every 500ms (2 Hz - less aggressive)
        self.last_distance_print_time = 0.0  # For periodic console output
        self.distance_print_interval = 1.0  # Print distance to console every 1 second

        # Limits will be updated from Arduino on connect

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.5,
                write_timeout=0.5
            )
            time.sleep(2)  # Wait for Arduino reset
            # Get initial status
            self.update_status()
            return True
        except Exception as e:
            print(f"Error connecting to turret: {e}")
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_command(self, command: str, read_response: bool = False) -> Optional[str]:
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            self.ser.write((command + '\n').encode())
            self.ser.flush()
            
            if not read_response:
                return None

            response = ""
            start_time = time.time()
            while time.time() - start_time < 0.5:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response += line + "\n"
                        if line.startswith("OK:") or line.startswith("ERROR:"):
                            break
                time.sleep(0.01)
            return response.strip() if response else None
        except:
            return None

    def update_status(self):
        """Update internal status from Arduino (FIX #8)"""
        resp = self.send_command("STATUS", read_response=True)
        if resp:
            for line in resp.split('\n'):
                if 'Top servo position:' in line:
                    try:
                        self.top_pos = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Bottom servo position:' in line:
                    try:
                        self.bottom_pos = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Top limits' in line:
                    try:
                        parts = line.split('MIN:')[1].split(',')
                        self.top_min = int(parts[0].strip())
                        self.top_max = int(parts[1].split('MAX:')[1].strip())
                    except:
                        pass
                elif 'Bottom limits' in line:
                    try:
                        parts = line.split('MIN:')[1].split(',')
                        self.bottom_min = int(parts[0].strip())
                        self.bottom_max = int(parts[1].split('MAX:')[1].strip())
                    except:
                        pass

    def read_distance(self) -> Optional[float]:
        """Read distance from TF03 LiDAR sensor

        Returns:
            Distance in cm, or None if not available
        """
        # If previously determined to be unavailable, skip immediately
        if self.distance_available == False:
            return self.distance_cm

        # Rate limit distance reads
        current_time = time.time()
        if current_time - self.last_distance_time < self.distance_update_interval:
            return self.distance_cm

        self.last_distance_time = current_time

        # Try to read distance from Arduino (with short timeout to avoid blocking)
        if not self.ser or not self.ser.is_open:
            return self.distance_cm

        try:
            self.ser.reset_input_buffer()
            self.ser.write(b'GET_RANGE\n')  # Use GET_RANGE command (matches turret_debug.ino)
            self.ser.flush()

            # Quick read with 50ms timeout (won't block control loop)
            response = ""
            start_time = time.time()
            while time.time() - start_time < 0.05:  # 50ms timeout instead of 500ms!
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response += line + "\n"
                        if line.startswith("OK:") or line.startswith("ERROR:"):
                            break
                time.sleep(0.001)  # 1ms sleep

            if response:
                # Check if command not supported or sensor error - disable permanently
                if 'ERROR' in response:
                    if 'not available' in response.lower():
                        if self.distance_available is None:  # First check
                            print("Distance sensor not available (ToF sensor not connected)")
                        self.distance_available = False
                    return self.distance_cm

                # Try multiple parsing formats
                for line in response.split('\n'):
                    # Format 1: "OK: Range: X.XX in" (from turret_debug.ino GET_RANGE command)
                    if 'Range:' in line and 'in' in line:
                        try:
                            # Extract the number between "Range:" and "in"
                            dist_str = line.split('Range:')[1].split('in')[0].strip()
                            distance_inches = float(dist_str)
                            distance_cm = distance_inches * 2.54  # Convert inches to cm
                            self.distance_cm = distance_cm
                            if self.distance_available is None:  # First successful read
                                print(f"TF03 LiDAR available (read {distance_cm:.1f} cm / {distance_inches:.1f} in)")
                            self.distance_available = True
                            return distance_cm
                        except:
                            pass

                    # Format 2: "Dist: XXX cm" (legacy format)
                    elif 'Dist:' in line and 'cm' in line:
                        try:
                            dist_str = line.split('Dist:')[1].split('cm')[0].strip()
                            distance = float(dist_str)
                            self.distance_cm = distance
                            if self.distance_available is None:
                                print(f"Distance sensor available (read {distance:.1f} cm)")
                            self.distance_available = True
                            return distance
                        except:
                            pass
        except:
            pass  # Silently fail to avoid breaking control loop

        return self.distance_cm

    def move_to(self, target_bottom: float, target_top: float, force: bool = False):
        """Move servos to absolute positions (FIX #7)

        Args:
            target_bottom: Target angle for bottom servo (float)
            target_top: Target angle for top servo (float)
            force: If True, bypass rate limiting
        """
        # Rate limiting (FIX #9)
        current_time = time.time()
        if not force and (current_time - self.last_command_time) < self.command_interval:
            return  # Too soon, skip this command
        
        # Clamp to limits (keep as float until sending)
        target_bottom = max(self.bottom_min, min(self.bottom_max, target_bottom))
        target_top = max(self.top_min, min(self.top_max, target_top))
        
        # Check if change is significant enough (FIX #9)
        bottom_change = abs(target_bottom - self.bottom_pos)
        top_change = abs(target_top - self.top_pos)
        
        commands_sent = False
        
        if bottom_change >= self.min_angle_change or force:
            # Round to integer for sending (FIX #1)
            bottom_int = round(target_bottom)
            self.send_command(f"BOTTOM:{bottom_int}", read_response=False)
            self.bottom_pos = target_bottom  # Keep float internally
            commands_sent = True
        
        if top_change >= self.min_angle_change or force:
            # Round to integer for sending (FIX #1)
            top_int = round(target_top)
            self.send_command(f"TOP:{top_int}", read_response=False)
            self.top_pos = target_top  # Keep float internally
            commands_sent = True
        
        if commands_sent:
            self.last_command_time = current_time


class YOLOGimbal:
    """YOLO-based automatic gimbal tracking system"""

    def __init__(self, camera_index: int, turret_port: str,
                 target_class: Optional[str] = None, conf_threshold: float = 0.5,
                 kp: float = 1.0, ki: float = 0.2, kd: float = 0.1,
                 deadzone: float = 15.0, deadzone_degrees: float = 0.5,
                 movement_scale: float = 15.0, min_step: float = 0.5,
                 max_movement: float = 10.0,
                 control_rate: float = 60.0,
                 camera_fps: float = 60.0, display_fps: Optional[float] = None,
                 disable_display: bool = True,
                 invert_x: bool = False, invert_y: bool = False,
                 swap_servos: bool = False, enable_3d_viz: bool = False,
                 use_rknn: bool = False, rknn_model: Optional[str] = None,
                 enable_error_plot: bool = False, enable_timing: bool = False,
                 detection_imgsz: int = 640, enable_distance: bool = True,
                 fov_horizontal: float = 126.0, fov_vertical: Optional[float] = None,
                 frame_width: int = 1920, frame_height: int = 1080,
                 pid_max_output: float = 0.75,
                 lost_timeout: float = 1.0, search_timeout: float = 2.0,
                 sweep_speed: float = 2.0, sweep_dwell: float = 0.5,
                 smooth_window: int = 0, smooth_max_jump: float = 100.0,
                 yolo_device: Optional[str] = None, yolo_half: bool = False):
        self.camera_index = camera_index
        self.turret_port = turret_port
        self.target_class = target_class
        self.conf_threshold = conf_threshold
        self.deadzone = deadzone  # Pixels - don't move if error is smaller (FIX #12)
        self.deadzone_degrees = deadzone_degrees  # Degrees - minimum movement (FIX #12)
        self.movement_scale = movement_scale  # Scale for normalized error to degrees (FIX #3) - kept for backwards compatibility
        self.min_step = min_step  # Minimum step to overcome deadband (FIX #2)
        self.max_movement = max_movement  # Maximum movement per command (degrees) - safety limit
        self.control_rate = control_rate  # Control loop rate in Hz (FIX #11)
        self.control_dt = 1.0 / control_rate  # Fixed dt for PID
        self.camera_fps = camera_fps  # Camera FPS setting
        self.display_fps = display_fps if display_fps is not None else min(30.0,
                                                                           control_rate)  # Display FPS (default: min of 30 or control_rate)
        self.display_dt = 1.0 / self.display_fps if self.display_fps > 0 else 0
        self.disable_display = disable_display  # Disable display for maximum FPS
        self.invert_x = invert_x  # Invert horizontal movement
        self.invert_y = invert_y  # Invert vertical movement
        self.swap_servos = swap_servos  # Swap top and bottom servos
        self.enable_3d_viz = enable_3d_viz  # Enable 3D visualization
        self.use_rknn = use_rknn  # Use RKNN acceleration
        self.rknn_model = rknn_model  # Path to RKNN model
        self.enable_error_plot = enable_error_plot  # Enable error plotting
        self.enable_timing = enable_timing  # Enable timing profiling
        self.detection_imgsz = detection_imgsz  # Detection input size (lower = faster)
        self.enable_distance = enable_distance  # Enable TF03 LiDAR distance reading
        self.yolo_device = yolo_device  # YOLO device ('cpu', 'cuda', '0', etc.)
        self.yolo_half = yolo_half  # Use FP16 half precision (faster on GPU)

        # Camera FOV-based error mapping (physically accurate pixel-to-degree conversion)
        self.fov_horizontal = fov_horizontal  # Horizontal field of view in degrees
        self.fov_vertical = fov_vertical  # Vertical FOV (calculated from aspect ratio if None)
        self.requested_frame_width = frame_width  # Requested frame width
        self.requested_frame_height = frame_height  # Requested frame height

        # These will be calculated in initialize() after getting actual frame dimensions
        self.degrees_per_pixel_x = None
        self.degrees_per_pixel_y = None

        # Initialize components
        self.camera = None
        self.yolo = None
        self.rknn_detector = None
        self.turret = TurretController(turret_port)
        self.pid_max_output = pid_max_output
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, max_output=pid_max_output)  # Max output now in degrees
        self.visualizer = None
        self.error_plotter = None
        self.sweep_controller = SweepController(move_speed=sweep_speed)  # For sweep pattern when target lost
        self.smooth_window = smooth_window
        self.detection_smoother = DetectionSmoother(window_size=smooth_window,
                                                    max_jump_px=smooth_max_jump)  # Smooth YOLO jitter

        # Tracking state machine
        self.tracking_state = TrackingState.TRACKING
        self.last_target_time = 0.0  # When we last saw a target
        self.last_known_pan = 90.0  # Last known target position (servo angles)
        self.last_known_tilt = 90.0
        self.lost_timeout = lost_timeout  # Seconds before returning to last position
        self.search_timeout = search_timeout  # Seconds to search at last position before sweeping
        self.state_start_time = 0.0  # When current state started
        self.sweep_dwell_time = sweep_dwell  # Seconds to pause at each waypoint
        self.sweep_speed = sweep_speed  # Degrees per step during sweep

        # State
        self.running = False
        self.frame_width = self.requested_frame_width
        self.frame_height = self.requested_frame_height
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        # Timing (FIX #4)
        self.last_control_time = 0.0
        self.last_fps_time = 0.0
        self.frame_count = 0
        self.current_fps = 0.0

        # Timing statistics (for --timing flag)
        self.timing_stats = {
            'frame_read': [],
            'detection': [],
            'pid_calc': [],
            'servo_cmd': [],
            'display': [],
            'total_loop': [],
        }
        self.last_timing_print = 0.0
        self.timing_print_interval = 1.0  # Print timing stats every 1 second

        # Calibration mode (FIX #6)
        self.calibration_mode = False

    def initialize(self):
        """Initialize camera, YOLO/RKNN detector, and turret"""
        print("Initializing camera...")
        camera_config = CAMERA_CONFIG.copy()
        camera_config.update({
            "width": self.requested_frame_width,
            "height": self.requested_frame_height,
            "fps": int(self.camera_fps),
            "fourcc": "MJPG"
        })

        self.camera = Camera(index=self.camera_index, config=camera_config)
        self.camera.open()

        # Get actual frame size
        test_frame = self.camera.read_frame()
        if test_frame is None:
            raise RuntimeError("Failed to read frame from camera")
        self.frame_height, self.frame_width = test_frame.shape[:2]
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")

        # Calculate FOV-based degrees per pixel for accurate error mapping
        # This maps pixel error directly to servo degrees based on camera geometry
        self.degrees_per_pixel_x = self.fov_horizontal / self.frame_width

        # Calculate vertical FOV from aspect ratio if not provided
        if self.fov_vertical is None:
            # Using proper FOV calculation: vertical_fov = 2 * atan(tan(h_fov/2) * height/width)
            aspect_ratio = self.frame_height / self.frame_width
            self.fov_vertical = 2 * np.degrees(np.arctan(
                np.tan(np.radians(self.fov_horizontal / 2)) * aspect_ratio
            ))

        self.degrees_per_pixel_y = self.fov_vertical / self.frame_height

        print(f"FOV mapping: {self.fov_horizontal:.1f}° horizontal, {self.fov_vertical:.1f}° vertical")
        print(f"Degrees per pixel: X={self.degrees_per_pixel_x:.4f}°/px, Y={self.degrees_per_pixel_y:.4f}°/px")
        print(
            f"Max error (at edge): X={self.frame_width / 2 * self.degrees_per_pixel_x:.1f}°, Y={self.frame_height / 2 * self.degrees_per_pixel_y:.1f}°")

        # Initialize detector (RKNN or YOLO)
        if self.use_rknn:
            if not RKNN_AVAILABLE:
                print("WARNING: RKNN not available, falling back to YOLO")
                self.use_rknn = False
            else:
                print("Initializing RKNN detector (hardware accelerated)...")
                # Determine RKNN model path
                if self.rknn_model is None:
                    self.rknn_model = str(Path(PROJECT_ROOT) / "yolo" / "models" / "yolo11n.rknn")

                if not Path(self.rknn_model).exists():
                    print(f"WARNING: RKNN model not found at {self.rknn_model}, falling back to YOLO")
                    self.use_rknn = False
                else:
                    self.rknn_detector = RKNNDetector(
                        model_path=self.rknn_model,
                        conf_threshold=self.conf_threshold,
                        imgsz=640
                    )
                    self.rknn_detector.load()
                    print("RKNN detector initialized")

        if not self.use_rknn:
            print("Initializing YOLO...")
            # Import YOLODetector only when needed (lazy import to avoid PyTorch dependency)
            from yolo.live_demo import YOLODetector

            yolo_weights = Path(PROJECT_ROOT) / "yolo" / "models" / "yolo11n.pt"
            self.yolo = YOLODetector(
                name="GimbalTracker",
                camera=self.camera,
                weights=str(yolo_weights),
                conf=self.conf_threshold,
                imgsz=self.detection_imgsz,  # Use configurable image size
                device=self.yolo_device,  # GPU acceleration if available
                half=self.yolo_half  # FP16 half precision (GPU only)
            )
            self.yolo.start()
            if self.yolo_half:
                print("YOLO: Using FP16 half precision for faster inference")
            print("YOLO initialized")

        print("Connecting to turret...")
        if not self.turret.connect():
            raise RuntimeError(f"Failed to connect to turret on {self.turret_port}")
        print("Turret connected")

        # Reset servo limits to full range (0-180) to prevent false limit detection
        print("Setting servo limits to full range (0-180)...")
        self.turret.send_command("SET_BOTTOM_MIN:0", read_response=False)
        self.turret.send_command("SET_BOTTOM_MAX:180", read_response=False)
        self.turret.send_command("SET_TOP_MIN:0", read_response=False)
        self.turret.send_command("SET_TOP_MAX:180", read_response=False)
        time.sleep(0.1)

        # Update status to get current positions and limits
        self.turret.update_status()

        # Print current limits
        print(f"\nServo Limits:")
        top_range = self.turret.top_max - self.turret.top_min
        bottom_range = self.turret.bottom_max - self.turret.bottom_min
        print(f"  Top (tilt):   {self.turret.top_min}° - {self.turret.top_max}° (range: {top_range}°)")
        print(f"  Bottom (pan): {self.turret.bottom_min}° - {self.turret.bottom_max}° (range: {bottom_range}°)")
        print(f"  Current positions: Top={self.turret.top_pos}°, Bottom={self.turret.bottom_pos}°")

        # Check if servo limits are too restrictive
        warnings = []
        if self.turret.bottom_max <= 90:
            warnings.append(f"Bottom servo max is {self.turret.bottom_max}° (should be ~180°) - can't pan right!")
        if self.turret.bottom_min >= 90:
            warnings.append(f"Bottom servo min is {self.turret.bottom_min}° (should be ~0°) - can't pan left!")
        if bottom_range < 90:
            warnings.append(f"Bottom servo range is only {bottom_range}° (recommend 120°+ for good pan)")
        if top_range < 30:
            warnings.append(f"Top servo range is only {top_range}° (recommend 40°+ for good tilt)")
        if self.turret.top_min >= 90:
            warnings.append(f"Top servo min is {self.turret.top_min}° (should be ~60°) - can't tilt up!")
        if self.turret.top_max <= 90:
            warnings.append(f"Top servo max is {self.turret.top_max}° (should be ~120°) - can't tilt down!")

        if warnings:
            print(f"\n⚠️  LIMIT WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  • {w}")
            print("  Fix with: SET_TOP_MIN:60, SET_TOP_MAX:120, SET_BOTTOM_MIN:0, SET_BOTTOM_MAX:180")
            print("  Or press 'l' during tracking to reset limits to full range")

        # Move to home position on startup
        print("\nHoming turret...")
        home_response = self.turret.send_command("HOME", read_response=True)
        if home_response:
            print(f"  HOME response: {home_response.split(chr(10))[0]}")  # First line only
        time.sleep(2)  # Give servos time to reach home position
        self.turret.update_status()
        print(f"  Turret homed to: Pan={self.turret.bottom_pos:.1f}°, Tilt={self.turret.top_pos:.1f}°")

        # Configure sweep controller with actual servo limits
        home_pan = (self.turret.bottom_min + self.turret.bottom_max) / 2
        home_tilt = (self.turret.top_min + self.turret.top_max) / 2
        self.sweep_controller.update_limits(
            pan_min=self.turret.bottom_min,
            pan_max=self.turret.bottom_max,
            tilt_min=self.turret.top_min,
            tilt_max=self.turret.top_max,
            home_pan=home_pan,
            home_tilt=home_tilt
        )
        print(
            f"Sweep pattern configured: Pan {self.sweep_controller.pan_min:.0f}°-{self.sweep_controller.pan_max:.0f}°, "
            f"Tilt {self.sweep_controller.tilt_min:.0f}°-{self.sweep_controller.tilt_max:.0f}°")

        # Initialize 3D visualizer if enabled
        if self.enable_3d_viz:
            print("Starting 3D visualization...")
            self.visualizer = Turret3DVisualizer(history_length=200)
            self.visualizer.start()
            print("3D visualization started")

        # Initialize error plotter if enabled
        if self.enable_error_plot:
            print("Starting error plotting...")
            self.error_plotter = ErrorPlotter(history_length=500)
            self.error_plotter.start()
            print("Error plotting started")

    def find_target_detection(self, detections) -> Optional[Tuple[float, float, float, float]]:
        """Find the best target detection (largest, highest confidence)"""
        if not detections:
            return None

        # Filter by class if specified
        filtered = []
        for det in detections:
            if self.target_class is None:
                filtered.append(det)
            else:
                # Check if class name or ID matches
                class_name = getattr(det, 'label', '')
                class_id = getattr(det, 'class_id', -1)
                if (isinstance(self.target_class, str) and
                        (self.target_class.lower() in class_name.lower() or
                         self.target_class == str(class_id))):
                    filtered.append(det)
                elif isinstance(self.target_class, int) and class_id == self.target_class:
                    filtered.append(det)

        if not filtered:
            return None

        # Find largest detection (by area)
        best = max(filtered, key=lambda d:
        (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

        # Return center coordinates and size
        x1, y1, x2, y2 = best.bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        return center_x, center_y, width, height

    def read(self) -> Optional[Dict]:
        """
        Read current detections and return them in XYZ coordinates.

        Returns:
            Dictionary with detection data including XYZ coordinates in meters:
            {
                'detections': [
                    {
                        'label': str,
                        'class_id': int,
                        'confidence': float,
                        'bbox': [x1, y1, x2, y2],  # pixels
                        'center_px': [x, y],  # pixel coordinates
                        'xyz': [x, y, z],  # meters (relative to turret)
                        'distance_m': float,  # meters
                        'pan_deg': float,  # degrees
                        'tilt_deg': float,  # degrees
                    },
                    ...
                ],
                'timestamp': float,
                'pan_deg': float,  # Current turret pan angle
                'tilt_deg': float,  # Current turret tilt angle
                'distance_m': float or None,  # LiDAR distance if available
            }
            Returns None if no detections or not running.
        """
        if not self.running or self.camera is None:
            return None

        # Get current frame
        frame = self.camera.read_frame()
        if frame is None:
            return None

        # Run detection
        if self.use_rknn:
            detections = self.rknn_detector.detect(frame)
        else:
            if self.yolo is None:
                return None
            result = self.yolo.read()
            if result is None:
                return None
            detections = result.detections

        if not detections:
            return None

        # Get current turret state
        self.turret.update_status()
        pan_deg = self.turret.bottom_pos  # Bottom servo = pan (horizontal)
        tilt_deg = self.turret.top_pos  # Top servo = tilt (vertical)

        # Get LiDAR distance (in cm, convert to meters)
        distance_cm = self.turret.read_distance() if self.enable_distance else None
        distance_m = distance_cm / 100.0 if distance_cm is not None else None

        # Convert angles to radians
        pan_rad = np.radians(pan_deg)
        tilt_rad = np.radians(tilt_deg)

        # Build detection list with XYZ coordinates
        detection_list = []
        for det in detections:
            # Extract bbox and center
            x1, y1, x2, y2 = det.bbox
            center_x_px = (x1 + x2) / 2.0
            center_y_px = (y1 + y2) / 2.0

            # Calculate XYZ coordinates using provided formulas
            # x = sin(bottom_angle) * cos(top_angle) * depth
            # y = sin(bottom_angle) * sin(top_angle) * depth
            # z = cos(bottom_angle) * depth
            if distance_m is not None and distance_m > 0:
                x = np.sin(pan_rad) * np.cos(tilt_rad) * distance_m
                y = np.sin(pan_rad) * np.sin(tilt_rad) * distance_m
                z = np.cos(pan_rad) * distance_m
            else:
                # No depth available - set to None
                x = y = z = None

            detection_dict = {
                'label': getattr(det, 'label', 'unknown'),
                'class_id': getattr(det, 'class_id', -1),
                'confidence': getattr(det, 'confidence', 0.0),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],  # pixels
                'center_px': [float(center_x_px), float(center_y_px)],  # pixels
                'xyz': [x, y, z] if x is not None else None,  # meters (or None if no depth)
                'distance_m': distance_m,  # meters (or None)
                'pan_deg': float(pan_deg),  # degrees
                'tilt_deg': float(tilt_deg),  # degrees
            }
            detection_list.append(detection_dict)

        return {
            'detections': detection_list,
            'timestamp': time.time(),
            'pan_deg': float(pan_deg),
            'tilt_deg': float(tilt_deg),
            'distance_m': distance_m,
            'num_detections': len(detection_list),
        }

    def calculate_error(self, target_x: float, target_y: float) -> Tuple[float, float, float, float, float, float]:
        """Calculate error from center in pixels, normalized, and degrees (FOV-based)

        Returns:
            error_x_px: Horizontal error in pixels
            error_y_px: Vertical error in pixels
            error_x_norm: Normalized horizontal error (-0.5 to 0.5)
            error_y_norm: Normalized vertical error (-0.5 to 0.5)
            error_x_deg: Horizontal error in degrees (based on camera FOV)
            error_y_deg: Vertical error in degrees (based on camera FOV)
        """
        # Error: how far target is from center
        # Positive error_x = target is RIGHT of center, need to move turret RIGHT (increase bottom servo)
        # Positive error_y = target is BELOW center, need to move turret DOWN (increase top servo)
        error_x_px = target_x - self.center_x
        error_y_px = target_y - self.center_y

        # Normalize error (legacy, kept for backwards compatibility)
        error_x_norm = error_x_px / self.frame_width  # -0.5 to 0.5
        error_y_norm = error_y_px / self.frame_height  # -0.5 to 0.5

        # FOV-based degree error (physically accurate!)
        # This directly tells us how many degrees the servo needs to move
        # to center the target, based on actual camera geometry
        error_x_deg = error_x_px * self.degrees_per_pixel_x
        error_y_deg = error_y_px * self.degrees_per_pixel_y

        return error_x_px, error_y_px, error_x_norm, error_y_norm, error_x_deg, error_y_deg

    def _print_timing_stats(self):
        """Print timing statistics to console"""
        if not self.enable_timing:
            return

        print("\n" + "=" * 70)
        print("TIMING STATISTICS (last 1 second)")
        print("=" * 70)

        def print_stat(name, times_ms):
            if not times_ms:
                print(f"  {name:20s}: No data")
                return
            avg = np.mean(times_ms)
            min_t = np.min(times_ms)
            max_t = np.max(times_ms)
            p50 = np.percentile(times_ms, 50)
            p95 = np.percentile(times_ms, 95)
            p99 = np.percentile(times_ms, 99)
            count = len(times_ms)
            print(f"  {name:20s}: avg={avg:6.2f}ms  min={min_t:6.2f}ms  max={max_t:6.2f}ms  "
                  f"p50={p50:6.2f}ms  p95={p95:6.2f}ms  p99={p99:6.2f}ms  (n={count})")

        print_stat("Frame Read", self.timing_stats['frame_read'])
        print_stat("Detection", self.timing_stats['detection'])
        print_stat("PID Calc", self.timing_stats['pid_calc'])
        print_stat("Servo Cmd", self.timing_stats['servo_cmd'])
        if self.timing_stats['display']:
            print_stat("Display", self.timing_stats['display'])
        print_stat("Total Loop", self.timing_stats['total_loop'])

        # Calculate effective FPS from loop time
        if self.timing_stats['total_loop']:
            avg_loop_ms = np.mean(self.timing_stats['total_loop'])
            effective_fps = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0
            print(f"\n  Effective Loop FPS: {effective_fps:.1f} Hz (target: {self.control_rate:.1f} Hz)")

        # Calculate detection FPS
        if self.timing_stats['detection']:
            avg_detection_ms = np.mean(self.timing_stats['detection'])
            detection_fps = 1000.0 / avg_detection_ms if avg_detection_ms > 0 else 0
            print(f"  Detection FPS: {detection_fps:.1f} Hz")

        print("=" * 70 + "\n")

    def run(self):
        """Main tracking loop with all fixes applied"""
        self.running = True
        print("\n=== YOLO Gimbal Tracking Started ===")
        print("Press 'q' to quit, 'r' to reset PID, 'h' to home, 'l' to reset limits")
        print("Press 'c' to enter calibration mode, 's' to force status update")
        print(f"Tracking: {self.target_class or 'all classes'}")
        print(f"PID gains: Kp={self.pid.kp}, Ki={self.pid.ki}, Kd={self.pid.kd}, max_out={self.pid_max_output}°")
        print(f"Deadzone: {self.deadzone}px (gradient), {self.deadzone_degrees}° min move")
        print(
            f"Detection smoothing: {self.smooth_window}-frame moving average" if self.smooth_window > 1 else "Detection smoothing: DISABLED")
        print(f"FOV-based mapping: {self.degrees_per_pixel_x:.4f}°/px (H), {self.degrees_per_pixel_y:.4f}°/px (V)")
        print(f"Max error at edge: ±{self.fov_horizontal / 2:.1f}° horizontal, ±{self.fov_vertical / 2:.1f}° vertical")
        print(f"Max step: {self.max_movement}° per command (hard clamp)")
        print(
            f"Lost target behavior: Wait {self.lost_timeout}s → Return to last → Search {self.search_timeout}s → Sweep")
        print(f"Sweep pattern: A(top-left)→X(center)→B(top-right)→X→C(bottom-left)→X→D(bottom-right)→X")
        print(f"Control rate: {self.control_rate} Hz")
        print(f"Camera FPS: {self.camera_fps} Hz")
        if not self.disable_display:
            print(f"Display FPS: {self.display_fps} Hz")
        else:
            print("Display: DISABLED (maximum performance)")
        print()

        self.last_control_time = time.time()
        self.last_fps_time = time.time()
        self.last_display_time = time.time()
        self.last_target_time = time.time()  # Initialize to now so we don't immediately think target is lost
        self.tracking_state = TrackingState.TRACKING  # Start in tracking state
        no_target_count = 0
        status_update_counter = 0

        # Initialize timing
        if self.enable_timing:
            self.last_timing_print = time.time()
            print("Timing profiling ENABLED - performance stats will be printed every 1 second")

        try:
            while self.running:
                loop_start_time = time.time()

                # Fixed control rate (FIX #11)
                current_time = time.time()
                dt_since_last = current_time - self.last_control_time

                # Sleep to maintain control rate
                if dt_since_last < self.control_dt:
                    time.sleep(self.control_dt - dt_since_last)
                    current_time = time.time()

                # Read frame (FIX #5 - read fresh frame)
                frame_read_start = time.time()
                frame = self.camera.read_frame()
                if frame is None:
                    continue
                if self.enable_timing:
                    self.timing_stats['frame_read'].append((time.time() - frame_read_start) * 1000)  # ms

                # Run detection (RKNN or YOLO)
                detection_start = time.time()
                if self.use_rknn:
                    # RKNN: direct synchronous inference on frame
                    detections = self.rknn_detector.detect(frame)
                    target = self.find_target_detection(detections)
                else:
                    # YOLO: read from threaded detector
                    result = self.yolo.read()
                    if result is None:
                        continue
                    target = self.find_target_detection(result.detections)
                if self.enable_timing:
                    self.timing_stats['detection'].append((time.time() - detection_start) * 1000)  # ms

                # Periodic status re-sync (FIX #8) - adaptive based on control rate
                status_update_counter += 1
                status_update_interval = max(10, int(self.control_rate))  # Update every ~1 second
                if status_update_counter >= status_update_interval:
                    self.turret.update_status()
                    status_update_counter = 0

                # Read distance from TF03 LiDAR (non-blocking with rate limiting)
                distance_cm = None
                if self.enable_distance:
                    distance_cm = self.turret.read_distance()

                    # Periodic console output of distance (every 1 second)
                    if distance_cm is not None and self.turret.distance_available:
                        if current_time - self.turret.last_distance_print_time >= self.turret.distance_print_interval:
                            distance_m = distance_cm / 100.0
                            distance_in = distance_cm / 2.54
                            print(f"[Range] {distance_cm:.1f} cm ({distance_m:.2f} m / {distance_in:.1f} in)")
                            self.turret.last_distance_print_time = current_time

                # Check if we should display this frame (for display FPS limiting)
                dt_display = current_time - self.last_display_time
                should_display_frame = self.disable_display == False and dt_display >= self.display_dt

                # Draw center crosshair (only if displaying)
                if should_display_frame:
                    cv2.line(frame, (self.center_x - 20, self.center_y),
                             (self.center_x + 20, self.center_y), (0, 255, 0), 2)
                    cv2.line(frame, (self.center_x, self.center_y - 20),
                             (self.center_x, self.center_y + 20), (0, 255, 0), 2)

                if target is not None:
                    raw_x, raw_y, raw_width, raw_height = target
                    no_target_count = 0

                    # SMOOTH DETECTION (FIX #17) - average over multiple frames to reduce YOLO jitter
                    # Skip smoothing if disabled (window <= 1) to avoid NaN from empty deque
                    if self.smooth_window > 1:
                        target_x, target_y, width, height = self.detection_smoother.update(
                            raw_x, raw_y, raw_width, raw_height
                        )
                    else:
                        # Smoothing disabled - use raw values directly
                        target_x, target_y, width, height = raw_x, raw_y, raw_width, raw_height

                    # TARGET FOUND - update state machine
                    self.last_target_time = current_time
                    # Save last known position (servo angles when we see target)
                    self.last_known_pan = self.turret.bottom_pos
                    self.last_known_tilt = self.turret.top_pos

                    # Reset to tracking state if we were searching/sweeping
                    if self.tracking_state != TrackingState.TRACKING:
                        print(f"Target ACQUIRED - switching from {self.tracking_state} to TRACKING")
                        self.tracking_state = TrackingState.TRACKING
                        self.sweep_controller.reset()
                        self.detection_smoother.clear()  # Start fresh when re-acquiring

                    # Draw target bounding box (only if displaying)
                    if should_display_frame:
                        # Draw raw detection in dark red (before smoothing)
                        if self.smooth_window > 1:
                            rx1 = int(raw_x - raw_width / 2)
                            ry1 = int(raw_y - raw_height / 2)
                            rx2 = int(raw_x + raw_width / 2)
                            ry2 = int(raw_y + raw_height / 2)
                            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 128), 1)  # Dark red = raw

                        # Draw smoothed detection in bright red
                        x1 = int(target_x - width / 2)
                        y1 = int(target_y - height / 2)
                        x2 = int(target_x + width / 2)
                        y2 = int(target_y + height / 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Bright red = smoothed
                        cv2.circle(frame, (int(target_x), int(target_y)), 5, (0, 0, 255), -1)

                        # Draw line from center to smoothed target
                        cv2.line(frame, (self.center_x, self.center_y),
                                 (int(target_x), int(target_y)), (255, 0, 0), 2)

                    # Calculate error (FIX #3 - now returns pixels, normalized, AND degrees)
                    error_x_px, error_y_px, error_x_norm, error_y_norm, error_x_deg, error_y_deg = self.calculate_error(
                        target_x, target_y)

                    # Update PID with degree-based error (FIX #14 - FOV-based mapping)
                    # Using degree error makes PID output directly in degrees
                    # This is physically accurate: if target is 5° off, PID operates on 5°
                    # Max error is limited to half FOV (e.g., 25° for 50° FOV)
                    pid_start = time.time()
                    output_x, output_y = self.pid.update(error_x_deg, error_y_deg, self.control_dt)
                    if self.enable_timing:
                        self.timing_stats['pid_calc'].append((time.time() - pid_start) * 1000)  # ms

                    # PID output is now already in degrees (no arbitrary scaling needed)
                    move_x = output_x
                    move_y = output_y

                    # Apply direction inversions
                    if self.invert_x:
                        move_x = -move_x
                    if self.invert_y:
                        move_y = -move_y

                    # ALL DEADZONES DISABLED - full PID movement
                    # # GRADIENT DEADZONE (FIX #15) - smooth ramp instead of hard cutoff
                    # if self.deadzone > 0:
                    #     gain_x = min(1.0, (abs(error_x_px) / self.deadzone) ** 2)
                    #     gain_y = min(1.0, (abs(error_y_px) / self.deadzone) ** 2)
                    # else:
                    #     gain_x = 1.0
                    #     gain_y = 1.0
                    # move_x *= gain_x
                    # move_y *= gain_y
                    #
                    # # Apply minimum step
                    # if abs(error_x_px) > self.deadzone * 0.5:
                    #     if 0 < abs(move_x) < self.min_step:
                    #         move_x = self.min_step if move_x > 0 else -self.min_step
                    # if abs(error_y_px) > self.deadzone * 0.5:
                    #     if 0 < abs(move_y) < self.min_step:
                    #         move_y = self.min_step if move_y > 0 else -self.min_step
                    #
                    # # Degree deadzone
                    # if abs(move_x) < self.deadzone_degrees:
                    #     move_x = 0
                    # if abs(move_y) < self.deadzone_degrees:
                    #     move_y = 0

                    # MAX MOVEMENT CLAMP DISABLED - let PID move freely
                    # original_move_x = move_x
                    # original_move_y = move_y
                    # move_x = np.clip(move_x, -self.max_movement, self.max_movement)
                    # move_y = np.clip(move_y, -self.max_movement, self.max_movement)
                    # if abs(original_move_x) > self.max_movement or abs(original_move_y) > self.max_movement:
                    #     print(f"CLAMPED: Requested X={original_move_x:.2f}° Y={original_move_y:.2f}° → "
                    #           f"Limited to X={move_x:.2f}° Y={move_y:.2f}° (max={self.max_movement}°)")

                    # Only move if there's actual movement to do
                    if abs(move_x) > 0 or abs(move_y) > 0:
                        
                        # BRUTE FORCE: If position is at/near limits, reset to center so we never think we're stuck
                        limit_margin = 5.0
                        if self.turret.bottom_pos <= self.turret.bottom_min + limit_margin:
                            self.turret.bottom_pos = 90.0  # Reset to center
                        if self.turret.bottom_pos >= self.turret.bottom_max - limit_margin:
                            self.turret.bottom_pos = 90.0  # Reset to center
                        if self.turret.top_pos <= self.turret.top_min + limit_margin:
                            self.turret.top_pos = 90.0  # Reset to center
                        if self.turret.top_pos >= self.turret.top_max - limit_margin:
                            self.turret.top_pos = 90.0  # Reset to center

                        # Calculate target absolute positions (FIX #7)
                        target_bottom = self.turret.bottom_pos + move_x
                        target_top = self.turret.top_pos + move_y

                        # Disable limit checking - just always say we're not at limits
                        at_limit_x = False
                        at_limit_y = False

                        # Apply servo swap if needed and move (FIX #7 - absolute positioning)
                        servo_start = time.time()
                        if self.swap_servos:
                            self.turret.move_to(target_top, target_bottom)
                        else:
                            self.turret.move_to(target_bottom, target_top)
                        if self.enable_timing:
                            self.timing_stats['servo_cmd'].append((time.time() - servo_start) * 1000)  # ms

                        # Update 3D visualizer
                        if self.visualizer:
                            self.visualizer.update(
                                pan=self.turret.bottom_pos,
                                tilt=self.turret.top_pos,
                                target_x=target_x,
                                target_y=target_y,
                                has_target=True,
                                error_x=error_x_px,
                                error_y=error_y_px
                            )

                        # Update error plotter
                        if self.error_plotter:
                            self.error_plotter.update(
                                error_x_px=error_x_px,
                                error_y_px=error_y_px,
                                pid_output_x=output_x,
                                pid_output_y=output_y,
                                move_x=move_x,
                                move_y=move_y,
                                bottom_pos=self.turret.bottom_pos,
                                top_pos=self.turret.top_pos,
                                target_x=target_x,
                                target_y=target_y,
                                has_target=True
                            )

                        # Better logging (FIX #13) - now with degree error
                        if abs(error_x_deg) > 2.0:  # Significant horizontal error (>2 degrees)
                            direction = "RIGHT" if error_x_px > 0 else "LEFT"
                            move_dir = "RIGHT" if move_x > 0 else "LEFT"
                            dist_info = f", Range={distance_cm:.1f}cm" if distance_cm is not None else ""
                            print(f"Target {direction} (err={error_x_deg:.2f}°/{error_x_px:.0f}px), "
                                  f"PID={output_x:.2f}°, move {move_dir} {move_x:.2f}°, "
                                  f"pos={self.turret.bottom_pos:.1f}°{dist_info}")

                        # Display info (only if displaying)
                        if should_display_frame:
                            limit_text = ""
                            if at_limit_x:
                                limit_text += " [X LIM]"
                            if at_limit_y:
                                limit_text += " [Y LIM]"

                            # Show state, smoothing status (gain disabled - always 100%)
                            gain_text = " G:100%/100%"
                            smooth_text = f" S:{self.detection_smoother.get_sample_count()}/{self.smooth_window}" if self.smooth_window > 1 else ""

                            cv2.putText(frame,
                                        f"TRACKING{smooth_text} | Err: X={error_x_deg:.1f} Y={error_y_deg:.1f}{gain_text}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, f"Move: X={move_x:.2f}deg Y={move_y:.2f}deg{limit_text}",
                                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(frame,
                                        f"Pos: Pan={self.turret.bottom_pos:.1f}deg Tilt={self.turret.top_pos:.1f}deg",
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        if should_display_frame:
                            cv2.putText(frame, "LOCKED ON TARGET", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame,
                                        f"Pos: Pan={self.turret.bottom_pos:.1f}deg Tilt={self.turret.top_pos:.1f}deg",
                                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # Show distance even when locked
                            if distance_cm is not None:
                                cv2.putText(frame, f"Range: {distance_cm:.1f}cm", (10, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Update visualizer even when locked
                        if self.visualizer:
                            self.visualizer.update(
                                pan=self.turret.bottom_pos,
                                tilt=self.turret.top_pos,
                                target_x=target_x,
                                target_y=target_y,
                                has_target=True,
                                error_x=error_x_px,
                                error_y=error_y_px
                            )

                        # Update error plotter even when locked
                        if self.error_plotter:
                            self.error_plotter.update(
                                error_x_px=error_x_px,
                                error_y_px=error_y_px,
                                pid_output_x=0.0,  # No movement when locked
                                pid_output_y=0.0,
                                move_x=0.0,
                                move_y=0.0,
                                bottom_pos=self.turret.bottom_pos,
                                top_pos=self.turret.top_pos,
                                target_x=target_x,
                                target_y=target_y,
                                has_target=True
                            )

                    self.last_control_time = current_time
                else:
                    # NO TARGET - run state machine for search/sweep behavior
                    no_target_count += 1
                    time_since_target = current_time - self.last_target_time

                    # STATE MACHINE for lost target behavior
                    if self.tracking_state == TrackingState.TRACKING:
                        # Just lost track - transition to LOST state
                        if time_since_target > 0.1:  # Small delay to avoid flicker
                            self.tracking_state = TrackingState.LOST
                            self.state_start_time = current_time
                            self.pid.reset()
                            self.detection_smoother.clear()  # Clear smoother when target lost
                            print(f"Target LOST - waiting {self.lost_timeout}s before returning to last position")

                    elif self.tracking_state == TrackingState.LOST:
                        # Waiting before returning to last known position
                        time_in_state = current_time - self.state_start_time
                        if time_in_state >= self.lost_timeout:
                            self.tracking_state = TrackingState.RETURN_TO_LAST
                            self.state_start_time = current_time
                            print(
                                f"Returning to last known position: Pan={self.last_known_pan:.1f}°, Tilt={self.last_known_tilt:.1f}°")

                    elif self.tracking_state == TrackingState.RETURN_TO_LAST:
                        # Moving back to last known position
                        pan_error = self.last_known_pan - self.turret.bottom_pos
                        tilt_error = self.last_known_tilt - self.turret.top_pos

                        # Move towards last position (with speed limit)
                        move_speed = 3.0  # degrees per step
                        move_x = np.clip(pan_error, -move_speed, move_speed)
                        move_y = np.clip(tilt_error, -move_speed, move_speed)

                        if abs(pan_error) > 1.0 or abs(tilt_error) > 1.0:
                            # Still moving to position
                            target_bottom = self.turret.bottom_pos + move_x
                            target_top = self.turret.top_pos + move_y
                            if self.swap_servos:
                                self.turret.move_to(target_top, target_bottom)
                            else:
                                self.turret.move_to(target_bottom, target_top)
                        else:
                            # Reached last position - start searching
                            self.tracking_state = TrackingState.SEARCHING
                            self.state_start_time = current_time
                            print(f"At last position - searching for {self.search_timeout}s")

                    elif self.tracking_state == TrackingState.SEARCHING:
                        # At last position, looking for target
                        time_in_state = current_time - self.state_start_time
                        if time_in_state >= self.search_timeout:
                            # Nothing found - go home and start sweep
                            self.tracking_state = TrackingState.GOING_HOME
                            self.state_start_time = current_time
                            print("Target not found - going HOME to start sweep pattern")

                    elif self.tracking_state == TrackingState.GOING_HOME:
                        # Moving to home position before sweep
                        home_pan = self.sweep_controller.home_pan
                        home_tilt = self.sweep_controller.home_tilt
                        pan_error = home_pan - self.turret.bottom_pos
                        tilt_error = home_tilt - self.turret.top_pos

                        move_speed = 3.0
                        move_x = np.clip(pan_error, -move_speed, move_speed)
                        move_y = np.clip(tilt_error, -move_speed, move_speed)

                        if abs(pan_error) > 1.0 or abs(tilt_error) > 1.0:
                            target_bottom = self.turret.bottom_pos + move_x
                            target_top = self.turret.top_pos + move_y
                            if self.swap_servos:
                                self.turret.move_to(target_top, target_bottom)
                            else:
                                self.turret.move_to(target_bottom, target_top)
                        else:
                            # At home - start sweep
                            self.tracking_state = TrackingState.SWEEPING
                            self.state_start_time = current_time
                            self.sweep_controller.reset()
                            wp_name, _, _ = self.sweep_controller.get_current_waypoint()
                            print(f"Starting SWEEP pattern: A→X→B→X→C→X→D→X (at waypoint {wp_name})")

                    elif self.tracking_state == TrackingState.SWEEPING:
                        # Executing sweep pattern: A → X → B → X → C → X → D → X → repeat
                        pan_move, tilt_move, reached = self.sweep_controller.calculate_move(
                            self.turret.bottom_pos, self.turret.top_pos
                        )

                        if reached:
                            # Dwell at waypoint briefly
                            time_at_waypoint = current_time - self.state_start_time
                            if time_at_waypoint >= self.sweep_dwell_time:
                                # Move to next waypoint
                                old_wp, _, _ = self.sweep_controller.get_current_waypoint()
                                self.sweep_controller.advance_waypoint()
                                new_wp, _, _ = self.sweep_controller.get_current_waypoint()
                                self.state_start_time = current_time
                                print(f"Sweep: {old_wp} → {new_wp}")
                        else:
                            # Move towards current waypoint
                            target_bottom = self.turret.bottom_pos + pan_move
                            target_top = self.turret.top_pos + tilt_move
                            if self.swap_servos:
                                self.turret.move_to(target_top, target_bottom)
                            else:
                                self.turret.move_to(target_bottom, target_top)
                            self.state_start_time = current_time  # Reset dwell timer while moving

                    # Display current state
                    if should_display_frame:
                        state_colors = {
                            TrackingState.TRACKING: (0, 255, 0),  # Green
                            TrackingState.LOST: (0, 165, 255),  # Orange
                            TrackingState.RETURN_TO_LAST: (0, 255, 255),  # Yellow
                            TrackingState.SEARCHING: (255, 255, 0),  # Cyan
                            TrackingState.GOING_HOME: (255, 0, 255),  # Magenta
                            TrackingState.SWEEPING: (255, 0, 0),  # Blue
                        }
                        color = state_colors.get(self.tracking_state, (0, 0, 255))

                        cv2.putText(frame, f"State: {self.tracking_state}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Show additional info based on state
                        if self.tracking_state == TrackingState.LOST:
                            time_left = max(0, self.lost_timeout - (current_time - self.state_start_time))
                            cv2.putText(frame, f"Returning in {time_left:.1f}s", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        elif self.tracking_state == TrackingState.SEARCHING:
                            time_left = max(0, self.search_timeout - (current_time - self.state_start_time))
                            cv2.putText(frame, f"Sweep in {time_left:.1f}s", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        elif self.tracking_state == TrackingState.SWEEPING:
                            wp_name, wp_pan, wp_tilt = self.sweep_controller.get_current_waypoint()
                            cv2.putText(frame, f"Waypoint: {wp_name} ({wp_pan:.0f}, {wp_tilt:.0f})", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        elif self.tracking_state == TrackingState.RETURN_TO_LAST:
                            cv2.putText(frame, f"Last: ({self.last_known_pan:.0f}, {self.last_known_tilt:.0f})",
                                        (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Update visualizer with no target
                    if self.visualizer:
                        self.visualizer.update(
                            pan=self.turret.bottom_pos,
                            tilt=self.turret.top_pos,
                            has_target=False
                        )

                    # Update error plotter with no target
                    if self.error_plotter:
                        self.error_plotter.update(
                            error_x_px=0.0,
                            error_y_px=0.0,
                            pid_output_x=0.0,
                            pid_output_y=0.0,
                            move_x=0.0,
                            move_y=0.0,
                            bottom_pos=self.turret.bottom_pos,
                            top_pos=self.turret.top_pos,
                            target_x=0.0,
                            target_y=0.0,
                            has_target=False
                        )

                # Record total loop time and print timing statistics
                if self.enable_timing:
                    self.timing_stats['total_loop'].append((time.time() - loop_start_time) * 1000)  # ms

                    # Print timing statistics periodically
                    if current_time - self.last_timing_print >= self.timing_print_interval:
                        self._print_timing_stats()
                        self.last_timing_print = current_time
                        # Clear old stats (keep last 100 samples)
                        for key in self.timing_stats:
                            if len(self.timing_stats[key]) > 100:
                                self.timing_stats[key] = self.timing_stats[key][-100:]

                # Calculate FPS properly (FIX #4)
                self.frame_count += 1
                fps_elapsed = current_time - self.last_fps_time
                if fps_elapsed >= 1.0:  # Update FPS every second
                    self.current_fps = self.frame_count / fps_elapsed
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # Display frame if enabled and enough time has passed for display FPS
                if should_display_frame:
                    display_start = time.time()
                    cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Display distance if available
                    if distance_cm is not None and self.turret.distance_available:
                        distance_text = f"Range: {distance_cm:.1f} cm ({distance_cm / 100:.2f} m)"
                        cv2.putText(frame, distance_text, (10, frame.shape[0] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Show frame
                    cv2.imshow("YOLO Gimbal Tracking", frame)
                    self.last_display_time = current_time
                    if self.enable_timing:
                        self.timing_stats['display'].append((time.time() - display_start) * 1000)  # ms

                # Handle keys (always check, even if display is disabled)
                key = cv2.waitKey(1) & 0xFF if not self.disable_display else -1
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.pid.reset()
                    print("PID reset")
                elif key == ord('h'):
                    self.turret.send_command("HOME", read_response=False)
                    self.turret.update_status()
                    self.pid.reset()
                    print("Moved to home")
                elif key == ord('l'):
                    # Reset limits to full range for BOTH servos
                    print("Resetting ALL servo limits to 0-180...")
                    self.turret.send_command("SET_BOTTOM_MIN:0", read_response=False)
                    self.turret.send_command("SET_BOTTOM_MAX:180", read_response=False)
                    self.turret.send_command("SET_TOP_MIN:0", read_response=False)
                    self.turret.send_command("SET_TOP_MAX:180", read_response=False)
                    time.sleep(0.1)
                    self.turret.update_status()
                    print(f"Limits reset: Bottom={self.turret.bottom_min}-{self.turret.bottom_max}°, Top={self.turret.top_min}-{self.turret.top_max}°")
                elif key == ord('c'):
                    # Calibration mode (FIX #6)
                    self.run_calibration()
                elif key == ord('s'):
                    # Force status update
                    self.turret.update_status()
                    print(f"Status: Bottom={self.turret.bottom_pos:.1f}° Top={self.turret.top_pos:.1f}°")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def _get_detections(self, frame):
        """Helper method to get detections from either RKNN or YOLO"""
        if self.use_rknn:
            return self.rknn_detector.detect(frame)
        else:
            result = self.yolo.read()
            return result.detections if result else []

    def run_calibration(self):
        """Calibration mode to determine correct axis directions (FIX #6)"""
        print("\n=== CALIBRATION MODE ===")
        print("This will help determine correct servo directions")
        print("Place a target in view and press any key to continue...")

        while True:
            frame = self.camera.read_frame()
            if frame is None:
                continue
            cv2.putText(frame, "CALIBRATION: Place target in view", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("YOLO Gimbal Tracking", frame)
            if cv2.waitKey(1) != -1:
                break

        # Test horizontal movement
        print("\nTesting HORIZONTAL (bottom servo)...")
        print("Moving servo +10° - observe if camera moves RIGHT")

        # Get initial target position
        frame = self.camera.read_frame()
        if frame is not None:
            detections = self._get_detections(frame)
            target = self.find_target_detection(detections)
            if target:
                initial_x, _, _, _ = target
                print(f"Initial target X: {initial_x:.1f}")

                # Move +10 degrees
                original_bottom = self.turret.bottom_pos
                self.turret.move_to(original_bottom + 10, self.turret.top_pos, force=True)
                time.sleep(1.5)

                # Check new position
                frame = self.camera.read_frame()
                if frame is not None:
                    detections = self._get_detections(frame)
                    target = self.find_target_detection(detections)
                    if target:
                        new_x, _, _, _ = target
                        print(f"New target X: {new_x:.1f}")
                        delta_x = new_x - initial_x

                        if abs(delta_x) > 20:
                            if delta_x > 0:
                                print("✓ Camera moved RIGHT (target moved right in frame)")
                                print("  -> X axis is CORRECT (no inversion needed)")
                            else:
                                print("✗ Camera moved LEFT (target moved left in frame)")
                                print("  -> X axis is INVERTED! Use --invert-x flag")
                        else:
                            print("! Movement too small to determine direction")

                # Return to original
                self.turret.move_to(original_bottom, self.turret.top_pos, force=True)
                time.sleep(1.0)

        # Test vertical movement
        print("\nTesting VERTICAL (top servo)...")
        print("Moving servo +5° - observe if camera moves DOWN")

        frame = self.camera.read_frame()
        if frame is not None:
            detections = self._get_detections(frame)
            target = self.find_target_detection(detections)
            if target:
                _, initial_y, _, _ = target
                print(f"Initial target Y: {initial_y:.1f}")

                # Move +5 degrees
                original_top = self.turret.top_pos
                self.turret.move_to(self.turret.bottom_pos, original_top + 5, force=True)
                time.sleep(1.5)

                # Check new position
                frame = self.camera.read_frame()
                if frame is not None:
                    detections = self._get_detections(frame)
                    target = self.find_target_detection(detections)
                    if target:
                        _, new_y, _, _ = target
                        print(f"New target Y: {new_y:.1f}")
                        delta_y = new_y - initial_y

                        if abs(delta_y) > 20:
                            if delta_y > 0:
                                print("✓ Camera moved DOWN (target moved down in frame)")
                                print("  -> Y axis is CORRECT (no inversion needed)")
                            else:
                                print("✗ Camera moved UP (target moved up in frame)")
                                print("  -> Y axis is INVERTED! Use --invert-y flag")
                        else:
                            print("! Movement too small to determine direction")

                # Return to original
                self.turret.move_to(self.turret.bottom_pos, original_top, force=True)
                time.sleep(1.0)

        print("\nCalibration complete! Press any key to continue tracking...")
        while cv2.waitKey(1) == -1:
            frame = self.camera.read_frame()
            if frame is not None:
                cv2.putText(frame, "Calibration complete - press any key", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("YOLO Gimbal Tracking", frame)

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.running = False

        if self.visualizer:
            print("Stopping 3D visualization...")
            self.visualizer.stop()

        if self.error_plotter:
            print("Stopping error plotting...")
            self.error_plotter.stop()

        if self.turret:
            self.turret.send_command("HOME", read_response=False)
            self.turret.send_command("MOTOR1:0", read_response=False)
            self.turret.send_command("MOTOR2:0", read_response=False)
            self.turret.disconnect()

        if self.rknn_detector:
            self.rknn_detector.release()

        if self.yolo:
            self.yolo.stop()

        if self.camera:
            self.camera.close()

        cv2.destroyAllWindows()
        print("Done")


def list_serial_ports():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
    else:
        print("No serial ports found")




class peripheral_mode:
    def __init__(self,name = "Poorly Configured Turret", **kwargs):
        # overwritable properties
        self.debug_mode = False  # will open numpy and plot
        self.name = name

        for k, v in kwargs.items():  # unpack config into self
            setattr(self, k, v)

        # check for reqired args
        if "port" not in vars(self):
            raise KeyError(f"Port Not specifed for: {name}")
        if "baudrate" not in vars(self):
            raise KeyError(f"baudrate Not specifed for: {name}")
        if 'data_out_label' not in vars(self):
            print(f"data_out_label not setup in config.dill writing as {self.name}")
            self.data_out_label = {self.name}

        # add local properties that cannot be specifed in config file



        #god I hate kyle

        if self.list_ports:
            list_serial_ports()
            return

        # Convert target_class to int if it's a digit
        target_class = self.target_class
        if target_class and target_class.isdigit():
            target_class = int(target_class)

        # Handle distance sensor flag
        enable_distance = self.enable_distance and not self.disable_distance
        try:
            self.gimbal = YOLOGimbal(
                camera_index=self.camera,
                turret_port=self.turret,
                target_class=target_class,
                conf_threshold=self.conf,
                kp=self.kp,
                ki=self.ki,
                kd=self.kd,
                deadzone=self.deadzone,
                deadzone_degrees=self.deadzone_degrees,
                movement_scale=self.movement_scale,
                min_step=self.min_step,
                max_movement=self.max_movement,
                control_rate=self.control_rate,
                camera_fps=self.camera_fps,
                display_fps=self.display_fps,
                disable_display=self.disable_display,
                invert_x=self.invert_x,
                invert_y=self.invert_y,
                swap_servos=self.swap_servos,
                enable_3d_viz=self.enable_3d_viz,
                use_rknn=self.rknn,
                rknn_model=self.rknn_model,
                enable_error_plot=self.enable_error_plot,
                enable_timing=self.enable_timing,
                detection_imgsz=self.detection_imgsz,
                enable_distance=enable_distance,
                fov_horizontal=self.fov_horizontal,
                fov_vertical=self.fov_vertical,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                pid_max_output=self.pid_max_output,
                lost_timeout=self.lost_timeout,
                search_timeout=self.search_timeout,
                sweep_speed=self.sweep_speed,
                sweep_dwell=self.sweep_dwell,
                smooth_window=self.smooth_window,
                smooth_max_jump=self.smooth_max_jump,
                yolo_device=self.yolo_device,
                yolo_half=self.yolo_half
            )
            time.sleep(0.25)
            self.gimbal.initialize()
        except Exception as e: #how fucking stupid is cursor
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def read(self):
        return
    def start(self):
        self.gimbal.run()
        return
    def stop(self):
        return