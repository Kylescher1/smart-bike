#!/usr/bin/env python3
"""
Object Tracking with PI-Controlled Gimbal

Run YOLO inference, click on a detected object to track it.
PI controller with smoothing keeps the selected object centered in frame.

Usage:
    python track_object.py --source 0 --port /dev/ttyUSB0
    
    # Tune for less jitter (slower response):
    python track_object.py --kp 0.08 --ki 0.01 --smoothing 0.15 --deadband 3.0
    
    # Tune for faster tracking (may have some overshoot):
    python track_object.py --kp 0.15 --ki 0.03 --smoothing 0.35 --deadband 1.5
"""

import argparse
import sys
import time
import serial
import threading
from pathlib import Path

import cv2
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

system_packages = '/usr/lib/python3/dist-packages'
if system_packages not in sys.path:
    sys.path.insert(0, system_packages)

from rknnlite.api import RKNNLite
from src.hal.cam.Camera import ThreadedCamera, CAMERA_CONFIG

# Import processing functions from rknn_inference
from rknn_inference import (
    letterbox, process_output, draw_detections, COCO_CLASSES
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolo11n.rknn"


class TurretController:
    """Serial controller for gimbal turret with rate limiting."""
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.lock = threading.Lock()
        self.distance_cm = None
        self.connected = False
        
        # Servo limits
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        self.top_home = 90
        self.bottom_home = 90
        
        # Current positions
        self.top_pos = self.top_home
        self.bottom_pos = self.bottom_home
        self.last_move_time = 0
    
    def connect(self) -> bool:
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)
            self.serial.reset_input_buffer()
            self.connected = True
            print(f"[TURRET] Connected to {self.port}")
            return True
        except Exception as e:
            print(f"[TURRET] Failed to connect: {e}")
            return False
    
    def disconnect(self):
        if self.serial:
            self.serial.close()
            self.connected = False
    
    def send_command(self, cmd: str) -> str:
        if not self.connected:
            return ""
        with self.lock:
            try:
                self.serial.reset_input_buffer()
                self.serial.write(f"{cmd}\n".encode())
                time.sleep(0.03)
                response = ""
                timeout = time.time() + 0.1
                while time.time() < timeout:
                    if self.serial.in_waiting:
                        response += self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                        if '\n' in response:
                            break
                    time.sleep(0.005)
                return response.strip()
            except:
                return ""
    
    def set_position(self, top: int, bottom: int):
        """Set servo positions with rate limiting."""
        top = max(self.top_min, min(self.top_max, top))
        bottom = max(self.bottom_min, min(self.bottom_max, bottom))
        
        top_changed = top != self.top_pos
        bottom_changed = bottom != self.bottom_pos
        
        if top_changed:
            self.send_command(f"TOP:{top}")
            self.top_pos = top
        
        if bottom_changed:
            self.send_command(f"BOTTOM:{bottom}")
            self.bottom_pos = bottom
        
        if top_changed or bottom_changed:
            self.last_move_time = time.time()
    
    def home(self):
        self.send_command("HOME")
        self.top_pos = self.top_home
        self.bottom_pos = self.bottom_home
        self.last_move_time = time.time()
    
    def get_distance(self) -> float:
        if time.time() - self.last_move_time < 0.3:
            return self.distance_cm if self.distance_cm else -1
        
        response = self.send_command("GET_RANGE")
        if "Range:" in response and "in" in response:
            try:
                dist_str = response.split("Range:")[1].split("in")[0].strip()
                self.distance_cm = float(dist_str) * 2.54
                return self.distance_cm
            except:
                pass
        return self.distance_cm if self.distance_cm else -1


class PIController:
    """
    PI controller with smoothing for servo control.
    
    Features:
    - Low-pass filtered output to reduce jitter
    - Accumulated position for sub-degree precision
    - Rate limiting to prevent overshoot
    - Deadband to ignore small errors
    """
    
    def __init__(self, kp: float, ki: float,
                 output_min: float = -30, output_max: float = 30,
                 smoothing: float = 0.3,
                 rate_limit: float = 5.0,
                 deadband: float = 1.5):
        """
        Args:
            kp: Proportional gain
            ki: Integral gain  
            output_min/max: Clamp output range
            smoothing: Low-pass filter factor (0-1, lower = smoother)
            rate_limit: Max degrees per update
            deadband: Ignore errors smaller than this (degrees)
        """
        self.kp = kp
        self.ki = ki
        self.output_min = output_min
        self.output_max = output_max
        self.smoothing = smoothing
        self.rate_limit = rate_limit
        self.deadband = deadband
        
        self.integral = 0.0
        self.prev_time = None
        self.smoothed_output = 0.0
        self.accumulated_pos = 0.0  # Sub-degree accumulator
    
    def reset(self):
        self.integral = 0.0
        self.prev_time = None
        self.smoothed_output = 0.0
        self.accumulated_pos = 0.0
    
    def update(self, error: float) -> tuple:
        """
        Update PI controller.
        
        Returns:
            (int_correction, float_accumulated): Integer correction to apply, 
                                                  and accumulated position
        """
        now = time.time()
        
        if self.prev_time is None:
            dt = 0.033  # Assume 30fps
        else:
            dt = min(now - self.prev_time, 0.1)  # Cap dt to avoid jumps
        
        self.prev_time = now
        
        # Apply deadband - zero out small errors
        if abs(error) < self.deadband:
            # Slowly decay integral when in deadband
            self.integral *= 0.95
            # Apply smoothing toward zero
            self.smoothed_output = self.smoothed_output * (1 - self.smoothing * 0.5)
            return 0, self.accumulated_pos
        
        # Proportional term
        p = self.kp * error
        
        # Integral term with anti-windup
        # Only integrate if not saturated (or error would reduce saturation)
        current_output = p + self.ki * self.integral
        if abs(current_output) < self.output_max or (error * current_output) < 0:
            self.integral += error * dt
            # Clamp integral
            max_integral = self.output_max / max(self.ki, 0.001)
            self.integral = max(-max_integral, min(max_integral, self.integral))
        
        i = self.ki * self.integral
        
        # Raw PI output
        raw_output = p + i
        
        # Rate limiting - prevent sudden large changes
        delta = raw_output - self.smoothed_output
        if abs(delta) > self.rate_limit:
            delta = self.rate_limit if delta > 0 else -self.rate_limit
        
        # Apply smoothing (exponential moving average)
        target = self.smoothed_output + delta
        self.smoothed_output = (self.smoothing * target + 
                                (1 - self.smoothing) * self.smoothed_output)
        
        # Clamp final output
        self.smoothed_output = max(self.output_min, 
                                   min(self.output_max, self.smoothed_output))
        
        # Accumulate sub-degree position
        self.accumulated_pos += self.smoothed_output * dt
        
        # Extract integer part for servo command
        int_correction = int(self.smoothed_output)
        
        return int_correction, self.accumulated_pos


class ObjectTracker:
    """Tracks a selected object across frames."""
    
    def __init__(self):
        self.target_class = None
        self.target_bbox = None  # Last known bbox [x1, y1, x2, y2]
        self.tracking = False
    
    def select_target(self, detection: dict):
        """Select a detection to track."""
        self.target_class = detection['class_id']
        self.target_bbox = detection['bbox']
        self.tracking = True
        print(f"[TRACK] Tracking: {detection['class_name']} (class {self.target_class})")
    
    def clear_target(self):
        """Stop tracking."""
        self.tracking = False
        self.target_bbox = None
        self.target_class = None
        print("[TRACK] Tracking stopped")
    
    def find_target(self, detections: list) -> dict:
        """Find the target object in new detections using IoU matching."""
        if not self.tracking or self.target_bbox is None:
            return None
        
        best_match = None
        best_iou = 0.3  # Minimum IoU threshold
        
        for det in detections:
            # Must be same class
            if det['class_id'] != self.target_class:
                continue
            
            # Calculate IoU
            iou = self._calc_iou(self.target_bbox, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = det
        
        if best_match:
            # Update tracked bbox
            self.target_bbox = best_match['bbox']
        
        return best_match
    
    def _calc_iou(self, box1, box2) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


def find_clicked_detection(x: int, y: int, detections: list) -> dict:
    """Find detection that contains the clicked point."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        if x1 <= x <= x2 and y1 <= y <= y2:
            return det
    return None


def main():
    parser = argparse.ArgumentParser(description="Object Tracking with PID Gimbal")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="RKNN model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    # PI tuning
    parser.add_argument("--kp", type=float, default=0.12, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.02, help="Integral gain")
    parser.add_argument("--smoothing", type=float, default=0.25, help="Output smoothing (0-1, lower=smoother)")
    parser.add_argument("--deadband", type=float, default=2.0, help="Error deadband in degrees")
    args = parser.parse_args()
    
    # Load RKNN model
    print(f"[INFO] Loading model: {args.model}")
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.model)) != 0:
        print("[ERROR] Failed to load model")
        return 1
    if rknn.init_runtime(core_mask=1) != 0:
        print("[ERROR] Failed to init runtime")
        return 1
    
    # Connect to turret
    turret = TurretController(args.port)
    if not turret.connect():
        print("[ERROR] Could not connect to turret")
        rknn.release()
        return 1
    
    turret.home()
    time.sleep(0.5)
    
    # Open camera
    camera_config = CAMERA_CONFIG.copy()
    camera_config.update({"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"})
    
    try:
        camera = ThreadedCamera(index=args.source, config=camera_config)
        camera.open()
        print(f"[INFO] Camera opened ({camera.width}x{camera.height})")
    except Exception as e:
        print(f"[ERROR] Camera failed: {e}")
        turret.disconnect()
        rknn.release()
        return 1
    
    # Initialize components
    tracker = ObjectTracker()
    pi_x = PIController(args.kp, args.ki, smoothing=args.smoothing, 
                        deadband=args.deadband)  # Pan
    pi_y = PIController(args.kp, args.ki, smoothing=args.smoothing,
                        deadband=args.deadband)  # Tilt
    
    # Mouse callback
    click_pos = [None]
    current_detections = []
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            tracker.clear_target()
            pi_x.reset()
            pi_y.reset()
    
    window_name = "Object Tracker - Left click to track, Right click to stop, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    
    # FOV for angle calculation
    fov_h = 50.0
    fov_v = fov_h * (720 / 1280)
    
    # Timing
    last_distance_time = 0
    frame_count = 0
    prev_time = time.time()
    
    print("[INFO] Left-click on an object to track it")
    print("[INFO] Right-click to stop tracking")
    print("[INFO] Press 'h' to home, 'q' to quit")
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_h, frame_w = frame.shape[:2]
            cx, cy = frame_w // 2, frame_h // 2
            
            # Run inference
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(args.imgsz, args.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
            
            outputs = rknn.inference([img_input])
            detections = process_output(outputs, conf_threshold=args.conf, 
                                        img_shape=(args.imgsz, args.imgsz))
            
            # Scale boxes to original frame
            if detections:
                scale = min(args.imgsz / frame_w, args.imgsz / frame_h)
                pad_x = (args.imgsz - frame_w * scale) / 2
                pad_y = (args.imgsz - frame_h * scale) / 2
                
                for det in detections:
                    bbox = np.array(det['bbox'], dtype=np.float32)
                    bbox[[0, 2]] = (bbox[[0, 2]] - pad_x) / scale
                    bbox[[1, 3]] = (bbox[[1, 3]] - pad_y) / scale
                    det['bbox'] = bbox.astype(int).tolist()
            
            current_detections = detections
            
            # Handle click - select target
            if click_pos[0] is not None:
                x, y = click_pos[0]
                clicked_det = find_clicked_detection(x, y, detections)
                if clicked_det:
                    tracker.select_target(clicked_det)
                    pi_x.reset()
                    pi_y.reset()
                click_pos[0] = None
            
            # Track target
            target_det = None
            if tracker.tracking:
                target_det = tracker.find_target(detections)
                
                if target_det:
                    # Calculate error (target center - frame center)
                    tx = (target_det['bbox'][0] + target_det['bbox'][2]) / 2
                    ty = (target_det['bbox'][1] + target_det['bbox'][3]) / 2
                    
                    # Error in pixels
                    error_px_x = tx - cx
                    error_px_y = ty - cy
                    
                    # Convert to degrees
                    error_deg_x = (error_px_x / (frame_w / 2)) * (fov_h / 2)
                    error_deg_y = (error_px_y / (frame_h / 2)) * (fov_v / 2)
                    
                    # PI control with smoothing
                    correction_x, _ = pi_x.update(error_deg_x)
                    correction_y, _ = pi_y.update(error_deg_y)
                    
                    # Apply to servos (PI controller handles deadband internally)
                    if correction_x != 0 or correction_y != 0:
                        new_bottom = turret.bottom_pos + correction_x
                        new_top = turret.top_pos + correction_y  # Inverted
                        turret.set_position(new_top, new_bottom)
                else:
                    # Target lost - slowly decay the PI controllers
                    pi_x.update(0)
                    pi_y.update(0)
            
            # Update distance periodically
            current_time = time.time()
            if current_time - last_distance_time > 0.5:
                turret.get_distance()
                last_distance_time = current_time
            
            # Draw frame
            annotated = frame.copy()
            
            # Draw all detections (dimmed if not target)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                is_target = (target_det and det['bbox'] == target_det['bbox'])
                
                if is_target:
                    color = (0, 255, 0)  # Green for target
                    thickness = 3
                else:
                    color = (128, 128, 128)  # Gray for others
                    thickness = 1
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                label = f"{det['class_name']} {det['score']:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if is_target:
                    # Draw center point and line to frame center
                    tx = int((x1 + x2) / 2)
                    ty = int((y1 + y2) / 2)
                    cv2.circle(annotated, (tx, ty), 8, (0, 255, 0), -1)
                    cv2.line(annotated, (cx, cy), (tx, ty), (0, 255, 255), 2)
            
            # Draw crosshair
            cv2.line(annotated, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 2)
            cv2.line(annotated, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 2)
            
            # Info overlay
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            
            info_lines = [
                f"FPS: {fps:.1f} | Detections: {len(detections)}",
                f"Servos: TOP={turret.top_pos} BOTTOM={turret.bottom_pos}",
            ]
            
            if turret.distance_cm and turret.distance_cm > 0:
                info_lines.append(f"Distance: {turret.distance_cm:.1f} cm")
            
            if tracker.tracking:
                if target_det:
                    error_x = (target_det['bbox'][0] + target_det['bbox'][2]) / 2 - cx
                    error_y = (target_det['bbox'][1] + target_det['bbox'][3]) / 2 - cy
                    info_lines.append(f"TRACKING: {COCO_CLASSES[tracker.target_class]} | Error: ({error_x:.0f}, {error_y:.0f})px")
                else:
                    info_lines.append("TRACKING: Target lost - searching...")
            else:
                info_lines.append("Click on an object to track")
            
            # Draw info background
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 5), (500, 20 + 22 * len(info_lines)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            
            for i, line in enumerate(info_lines):
                cv2.putText(annotated, line, (10, 22 + i * 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            cv2.imshow(window_name, annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                tracker.clear_target()
                turret.home()
                pi_x.reset()
                pi_y.reset()
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        print("[INFO] Cleaning up...")
        camera.close()
        turret.disconnect()
        rknn.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

