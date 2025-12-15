#!/usr/bin/env python3
"""
Fast Center Camera Tracking

Optimized single-camera YOLO tracking - matches rknn_inference.py speed.
Uses threaded frame capture for maximum performance.

Usage:
    python turret_debug/fast_center_track.py
"""

import sys
import time
import threading
from pathlib import Path
from collections import deque

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from src.hal.cam.Camera import Camera, CAMERA_CONFIG
from src.hal.TurretController import TurretController

# Import RKNN post-processing from rknn_inference
from yolo.rknn_inference import (
    letterbox, post_process_yolov8, draw_detections, COCO_CLASSES
)


# ============== CONFIGURATION ==============
CAMERA_INDEX = 5          # Center camera index
ARDUINO_PORT = '/dev/ttyUSB0'
MODEL_PATH = PROJECT_ROOT / "yolo" / "models" / "yolo11n.rknn"
CONF_THRESHOLD = 0.4
TARGET_CLASS = 'person'   # Primary target to track
SHOW_DISPLAY = True       # Set False for max speed (no display overhead)
# ===========================================


class FastTracker:
    """Ultra-fast single camera tracker with threaded capture."""
    
    def __init__(self, camera_index, model_path, conf=0.4):
        self.camera_index = camera_index
        self.model_path = model_path
        self.conf = conf
        self.imgsz = 640
        
        # Camera
        self.camera = None
        
        # Threaded capture
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_running = False
        self.capture_thread = None
        
        # RKNN
        self.rknn = None
        
        # Pre-allocated buffer
        self.img_input_buffer = None
        
        # Timing
        self.inference_time_ms = 0
        
    def start(self):
        """Initialize camera and RKNN."""
        print(f"[FastTracker] Initializing...")
        
        # Open camera with optimized settings
        config = CAMERA_CONFIG.copy()
        config.update({
            "width": 640,    # Match YOLO input size to reduce resize overhead
            "height": 480,
            "fps": 30,
            "fourcc": "MJPG"
        })
        
        self.camera = Camera(index=self.camera_index, config=config)
        self.camera.open()
        print(f"[FastTracker] Camera {self.camera_index} opened ({config['width']}x{config['height']})")
        
        # Start threaded capture
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Wait for first frame
        timeout = time.time() + 2.0
        while self.latest_frame is None and time.time() < timeout:
            time.sleep(0.01)
        
        if self.latest_frame is None:
            raise RuntimeError("Failed to capture initial frame")
        print(f"[FastTracker] Frame capture started")
        
        # Initialize RKNN
        print(f"[FastTracker] Loading RKNN model: {self.model_path}")
        self.rknn = RKNNLite(verbose=False)
        
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        ret = self.rknn.init_runtime(target=None, core_mask=0)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        
        # Pre-allocate input buffer
        self.img_input_buffer = np.zeros((1, self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        print(f"[FastTracker] Ready!")
        
    def _capture_loop(self):
        """Background frame capture - always has fresh frame ready."""
        while self.capture_running:
            try:
                frame = self.camera.read_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.001)
            except:
                time.sleep(0.01)
    
    def detect(self):
        """Run detection on latest frame. Returns (frame, detections, inference_ms)."""
        # Get latest frame (non-blocking)
        with self.frame_lock:
            if self.latest_frame is None:
                return None, [], 0
            frame = self.latest_frame.copy()
        
        # Preprocess
        img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        self.img_input_buffer[0] = img_rgb.astype(np.uint8)
        
        # Inference
        t0 = time.time()
        outputs = self.rknn.inference([self.img_input_buffer])
        self.inference_time_ms = (time.time() - t0) * 1000
        
        if outputs is None:
            return frame, [], self.inference_time_ms
        
        # Post-process
        boxes, classes, scores = post_process_yolov8(
            outputs, self.conf, 0.45, (self.imgsz, self.imgsz)
        )
        
        if len(boxes) == 0:
            return frame, [], self.inference_time_ms
        
        # Scale boxes back to original frame
        h_orig, w_orig = frame.shape[:2]
        scale = min(self.imgsz / w_orig, self.imgsz / h_orig)
        pad_x = (self.imgsz - int(w_orig * scale)) / 2
        pad_y = (self.imgsz - int(h_orig * scale)) / 2
        
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1 = int((x1 - pad_x) / scale)
            y1 = int((y1 - pad_y) / scale)
            x2 = int((x2 - pad_x) / scale)
            y2 = int((y2 - pad_y) / scale)
            
            # Clip to frame
            x1 = max(0, min(w_orig, x1))
            y1 = max(0, min(h_orig, y1))
            x2 = max(0, min(w_orig, x2))
            y2 = max(0, min(h_orig, y2))
            
            class_id = int(classes[i])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f'class_{class_id}'
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(scores[i]),
                'class_id': class_id,
                'class_name': class_name,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2
            })
        
        return frame, detections, self.inference_time_ms
    
    def stop(self):
        """Cleanup."""
        self.capture_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.rknn:
            self.rknn.release()
        if self.camera:
            self.camera.close()


def find_best_target(detections, target_class='person'):
    """Find the best target (largest person, or largest detection if no person)."""
    # Filter for target class
    targets = [d for d in detections if d['class_name'].lower() == target_class.lower()]
    
    if not targets:
        # Fall back to any detection
        targets = detections
    
    if not targets:
        return None
    
    # Return largest by area
    return max(targets, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))


def main():
    print("=" * 60)
    print("FAST CENTER CAMERA TRACKING")
    print("=" * 60)
    print(f"Camera: {CAMERA_INDEX}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Target: {TARGET_CLASS}")
    print()
    
    tracker = FastTracker(CAMERA_INDEX, MODEL_PATH, CONF_THRESHOLD)
    controller = None
    
    try:
        # Initialize tracker
        tracker.start()
        
        # Initialize turret controller
        print(f"[Turret] Connecting to {ARDUINO_PORT}...")
        controller = TurretController(port=ARDUINO_PORT)
        controller.connect()
        controller.home()
        print(f"[Turret] Connected and homed")
        
        print("\n" + "-" * 60)
        print("Tracking started! Press 'q' to quit.")
        print("-" * 60 + "\n")
        
        # Timing stats
        frame_count = 0
        fps_start = time.time()
        total_inference_time = 0
        total_loop_time = 0
        
        # Current turret position
        pan, tilt = 90.0, 90.0
        
        if SHOW_DISPLAY:
            cv2.namedWindow("Fast Track", cv2.WINDOW_NORMAL)
        
        while True:
            loop_start = time.time()
            
            # Run detection
            frame, detections, inference_ms = tracker.detect()
            total_inference_time += inference_ms
            
            if frame is None:
                continue
            
            # Find target
            target = find_best_target(detections, TARGET_CLASS)
            
            if target:
                # Calculate error from center
                h, w = frame.shape[:2]
                center_x, center_y = w / 2, h / 2
                
                error_x = target['center_x'] - center_x
                error_y = target['center_y'] - center_y
                
                # Convert to angle adjustment (proportional control)
                # ~60° FOV horizontal, ~45° vertical
                fov_h, fov_v = 60.0, 45.0
                gain = 0.3  # Smoothing factor
                
                pan_adj = (error_x / w) * fov_h * gain
                tilt_adj = -(error_y / h) * fov_v * gain  # Inverted
                
                # Update target angles
                pan = max(0, min(180, pan + pan_adj))
                tilt = max(60, min(120, tilt + tilt_adj))
                
                # Send to turret
                controller.move_to(pan, tilt)
            
            # Display
            if SHOW_DISPLAY:
                display = frame.copy()
                
                # Draw all detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    color = (0, 255, 0) if det == target else (128, 128, 128)
                    thickness = 3 if det == target else 1
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                    
                    label = f"{det['class_name']} {det['score']:.2f}"
                    cv2.putText(display, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw crosshair
                h, w = display.shape[:2]
                cv2.line(display, (w//2-20, h//2), (w//2+20, h//2), (0, 0, 255), 2)
                cv2.line(display, (w//2, h//2-20), (w//2, h//2+20), (0, 0, 255), 2)
                
                # FPS overlay
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    avg_inference = total_inference_time / frame_count
                    avg_loop = total_loop_time / frame_count if total_loop_time > 0 else 0
                    
                    info = f"FPS: {fps:.1f} | Inf: {avg_inference:.1f}ms | Loop: {avg_loop:.1f}ms"
                    cv2.putText(display, info, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Print stats
                    print(f"FPS: {fps:.1f} | Inference: {avg_inference:.1f}ms | "
                          f"Detections: {len(detections)} | Pan: {pan:.1f}° Tilt: {tilt:.1f}°")
                    
                    # Reset
                    frame_count = 0
                    fps_start = time.time()
                    total_inference_time = 0
                    total_loop_time = 0
                
                cv2.imshow("Fast Track", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # No display - just print stats periodically
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = frame_count / elapsed
                    avg_inference = total_inference_time / frame_count
                    print(f"FPS: {fps:.1f} | Inference: {avg_inference:.1f}ms | Det: {len(detections)}")
                    frame_count = 0
                    fps_start = time.time()
                    total_inference_time = 0
            
            total_loop_time += (time.time() - loop_start) * 1000
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tracker.stop()
        if controller:
            controller.disconnect()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()

