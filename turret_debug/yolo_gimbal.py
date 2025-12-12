#!/usr/bin/env python3
"""
YOLO-Based Automatic Camera Gimbal
Tracks detected objects and keeps them centered using PID servo control

Usage:
    python yolo_gimbal.py --camera 0 --turret COM3 --class person
    python yolo_gimbal.py --camera 1 --turret /dev/ttyUSB0 --class 0
"""

import serial
import sys
import time
import argparse
import threading
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from yolo.live_demo import YOLODetector
from src.hal.cam.Camera import Camera, CAMERA_CONFIG


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
        
        # Clamp outputs to prevent large spikes (FIX #10)
        self.output_x = np.clip(self.output_x, -self.max_output, self.max_output)
        self.output_y = np.clip(self.output_y, -self.max_output, self.max_output)
        
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
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        
        # Rate limiting (FIX #9)
        self.last_command_time = 0.0
        self.command_interval = 0.05  # 20 Hz max command rate
        self.min_angle_change = 0.5  # Don't send if change < 0.5 degrees
        
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
                 kp: float = 0.5, ki: float = 0.01, kd: float = 0.1,
                 deadzone: float = 10.0, deadzone_degrees: float = 1.0,
                 movement_scale: float = 30.0, min_step: float = 1.0,
                 control_rate: float = 30.0,
                 invert_x: bool = False, invert_y: bool = False,
                 swap_servos: bool = False):
        self.camera_index = camera_index
        self.turret_port = turret_port
        self.target_class = target_class
        self.conf_threshold = conf_threshold
        self.deadzone = deadzone  # Pixels - don't move if error is smaller (FIX #12)
        self.deadzone_degrees = deadzone_degrees  # Degrees - minimum movement (FIX #12)
        self.movement_scale = movement_scale  # Scale for normalized error to degrees (FIX #3)
        self.min_step = min_step  # Minimum step to overcome deadband (FIX #2)
        self.control_rate = control_rate  # Control loop rate in Hz (FIX #11)
        self.control_dt = 1.0 / control_rate  # Fixed dt for PID
        self.invert_x = invert_x  # Invert horizontal movement
        self.invert_y = invert_y  # Invert vertical movement
        self.swap_servos = swap_servos  # Swap top and bottom servos
        
        # Initialize components
        self.camera = None
        self.yolo = None
        self.turret = TurretController(turret_port)
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, max_output=10.0)
        
        # State
        self.running = False
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Timing (FIX #4)
        self.last_control_time = 0.0
        self.last_fps_time = 0.0
        self.frame_count = 0
        self.current_fps = 0.0
        
        # Calibration mode (FIX #6)
        self.calibration_mode = False
        
    def initialize(self):
        """Initialize camera, YOLO, and turret"""
        print("Initializing camera...")
        camera_config = CAMERA_CONFIG.copy()
        camera_config.update({
            "width": 640,
            "height": 480,
            "fps": 30,
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
        
        print("Initializing YOLO...")
        yolo_weights = Path(PROJECT_ROOT) / "yolo" / "models" / "yolo11n.pt"
        self.yolo = YOLODetector(
            name="GimbalTracker",
            camera=self.camera,
            weights=str(yolo_weights),
            conf=self.conf_threshold,
            imgsz=640
        )
        self.yolo.start()
        print("YOLO initialized")
        
        print("Connecting to turret...")
        if not self.turret.connect():
            raise RuntimeError(f"Failed to connect to turret on {self.turret_port}")
        print("Turret connected")
        
        # Update status to get current positions and limits
        self.turret.update_status()
        
        # Print current limits
        print(f"\nServo Limits:")
        print(f"  Top: {self.turret.top_min}° - {self.turret.top_max}°")
        print(f"  Bottom: {self.turret.bottom_min}° - {self.turret.bottom_max}°")
        print(f"  Current positions: Top={self.turret.top_pos}°, Bottom={self.turret.bottom_pos}°")
        
        # Check if bottom servo limits are too restrictive
        if self.turret.bottom_max <= 90:
            print(f"\nWARNING: Bottom servo max limit is {self.turret.bottom_max}° (should be 180°)")
            print("  This will prevent tracking to the right side!")
            print("  Fix by running: SET_BOTTOM_MAX:180 on Arduino or reset limits")
        
        # Move to home position
        self.turret.send_command("HOME", read_response=False)
        time.sleep(1)
        self.turret.update_status()
        
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
    
    def calculate_error(self, target_x: float, target_y: float) -> Tuple[float, float, float, float]:
        """Calculate error from center in both pixels and normalized (FIX #3)"""
        # Error: how far target is from center
        # Positive error_x = target is RIGHT of center, need to move turret RIGHT (increase bottom servo)
        # Positive error_y = target is BELOW center, need to move turret DOWN (increase top servo)
        error_x_px = target_x - self.center_x
        error_y_px = target_y - self.center_y
        
        # Normalize error (for PID)
        # Scale by frame dimensions
        error_x_norm = error_x_px / self.frame_width  # -0.5 to 0.5
        error_y_norm = error_y_px / self.frame_height  # -0.5 to 0.5
        
        return error_x_px, error_y_px, error_x_norm, error_y_norm
    
    def run(self):
        """Main tracking loop with all fixes applied"""
        self.running = True
        print("\n=== YOLO Gimbal Tracking Started ===")
        print("Press 'q' to quit, 'r' to reset PID, 'h' to home, 'l' to reset limits")
        print("Press 'c' to enter calibration mode")
        print(f"Tracking: {self.target_class or 'all classes'}")
        print(f"Deadzone: {self.deadzone} pixels, {self.deadzone_degrees} degrees")
        print(f"Control rate: {self.control_rate} Hz")
        print(f"Movement scale: {self.movement_scale}\n")
        
        self.last_control_time = time.time()
        self.last_fps_time = time.time()
        no_target_count = 0
        status_update_counter = 0
        
        try:
            while self.running:
                # Fixed control rate (FIX #11)
                current_time = time.time()
                dt_since_last = current_time - self.last_control_time
                
                # Sleep to maintain control rate
                if dt_since_last < self.control_dt:
                    time.sleep(self.control_dt - dt_since_last)
                    current_time = time.time()
                
                # Read frame (FIX #5 - read fresh frame)
                frame = self.camera.read_frame()
                if frame is None:
                    continue
                
                # Run YOLO detection on THIS frame (FIX #5 - sync detection with frame)
                # Note: For true sync, we'd need to modify YOLODetector to process this frame
                # For now, we ensure we read the most recent result
                result = self.yolo.read()
                if result is None:
                    continue
                
                # Find target
                target = self.find_target_detection(result.detections)
                
                # Periodic status re-sync (FIX #8)
                status_update_counter += 1
                if status_update_counter >= 30:  # Every 30 frames (~1 sec at 30Hz)
                    self.turret.update_status()
                    status_update_counter = 0
                
                # Draw center crosshair
                cv2.line(frame, (self.center_x - 20, self.center_y), 
                        (self.center_x + 20, self.center_y), (0, 255, 0), 2)
                cv2.line(frame, (self.center_x, self.center_y - 20), 
                        (self.center_x, self.center_y + 20), (0, 255, 0), 2)
                
                if target is not None:
                    target_x, target_y, width, height = target
                    no_target_count = 0
                    
                    # Draw target bounding box
                    x1 = int(target_x - width/2)
                    y1 = int(target_y - height/2)
                    x2 = int(target_x + width/2)
                    y2 = int(target_y + height/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (int(target_x), int(target_y)), 5, (0, 0, 255), -1)
                    
                    # Draw line from center to target
                    cv2.line(frame, (self.center_x, self.center_y), 
                            (int(target_x), int(target_y)), (255, 0, 0), 2)
                    
                    # Calculate error (FIX #3 - now returns pixels and normalized)
                    error_x_px, error_y_px, error_x_norm, error_y_norm = self.calculate_error(target_x, target_y)
                    
                    # Check pixel deadzone (FIX #12)
                    if abs(error_x_px) > self.deadzone or abs(error_y_px) > self.deadzone:
                        # Update PID with fixed dt (FIX #11)
                        output_x, output_y = self.pid.update(error_x_norm, error_y_norm, self.control_dt)
                        
                        # Convert PID output to servo movement (FIX #3 - better scaling)
                        move_x = output_x * self.movement_scale
                        move_y = output_y * self.movement_scale
                        
                        # Apply direction inversions
                        if self.invert_x:
                            move_x = -move_x
                        if self.invert_y:
                            move_y = -move_y
                        
                        # Apply minimum step to overcome deadband (FIX #2)
                        if 0 < abs(move_x) < self.min_step:
                            move_x = self.min_step if move_x > 0 else -self.min_step
                        if 0 < abs(move_y) < self.min_step:
                            move_y = self.min_step if move_y > 0 else -self.min_step
                        
                        # Check degree deadzone (FIX #12)
                        if abs(move_x) < self.deadzone_degrees:
                            move_x = 0
                        if abs(move_y) < self.deadzone_degrees:
                            move_y = 0
                        
                        # Calculate target absolute positions (FIX #7)
                        target_bottom = self.turret.bottom_pos + move_x
                        target_top = self.turret.top_pos + move_y
                        
                        # Check limits
                        at_limit_x = (target_bottom <= self.turret.bottom_min or 
                                     target_bottom >= self.turret.bottom_max)
                        at_limit_y = (target_top <= self.turret.top_min or 
                                     target_top >= self.turret.top_max)
                        
                        # Apply servo swap if needed and move (FIX #7 - absolute positioning)
                        if self.swap_servos:
                            self.turret.move_to(target_top, target_bottom)
                        else:
                            self.turret.move_to(target_bottom, target_top)
                        
                        # Better logging (FIX #13)
                        if abs(error_x_px) > 50:  # Significant horizontal error
                            direction = "RIGHT" if error_x_px > 0 else "LEFT"
                            move_dir = "RIGHT" if move_x > 0 else "LEFT"
                            print(f"Target {direction} of center (error={error_x_px:.1f}px), "
                                  f"PID_out={output_x:.3f}, moving turret {move_dir} "
                                  f"(move={move_x:.2f}deg, pos={self.turret.bottom_pos:.1f}°)")
                        
                        # Display info
                        limit_text = ""
                        if at_limit_x:
                            limit_text += " [X LIMIT]"
                        if at_limit_y:
                            limit_text += " [Y LIMIT]"
                        
                        cv2.putText(frame, f"Error: X={error_x_px:.1f}px Y={error_y_px:.1f}px", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"Move: X={move_x:.2f}deg Y={move_y:.2f}deg{limit_text}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"Pos: Bottom={self.turret.bottom_pos:.1f}deg Top={self.turret.top_pos:.1f}deg", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        cv2.putText(frame, "LOCKED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    self.last_control_time = current_time
                else:
                    no_target_count += 1
                    if no_target_count > 30:  # Reset PID after 1 second of no target
                        self.pid.reset()
                    cv2.putText(frame, "NO TARGET", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Calculate FPS properly (FIX #4)
                self.frame_count += 1
                fps_elapsed = current_time - self.last_fps_time
                if fps_elapsed >= 1.0:  # Update FPS every second
                    self.current_fps = self.frame_count / fps_elapsed
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow("YOLO Gimbal Tracking", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
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
                    # Reset limits to full range
                    print("Resetting bottom servo limits to 0-180...")
                    self.turret.send_command("SET_BOTTOM_MIN:0", read_response=False)
                    self.turret.send_command("SET_BOTTOM_MAX:180", read_response=False)
                    time.sleep(0.1)
                    self.turret.update_status()
                    print(f"Limits reset: Bottom now {self.turret.bottom_min}-{self.turret.bottom_max}°")
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
        result = self.yolo.read()
        if result and result.detections:
            target = self.find_target_detection(result.detections)
            if target:
                initial_x, _, _, _ = target
                print(f"Initial target X: {initial_x:.1f}")
                
                # Move +10 degrees
                original_bottom = self.turret.bottom_pos
                self.turret.move_to(original_bottom + 10, self.turret.top_pos, force=True)
                time.sleep(1.5)
                
                # Check new position
                result = self.yolo.read()
                if result and result.detections:
                    target = self.find_target_detection(result.detections)
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
        
        result = self.yolo.read()
        if result and result.detections:
            target = self.find_target_detection(result.detections)
            if target:
                _, initial_y, _, _ = target
                print(f"Initial target Y: {initial_y:.1f}")
                
                # Move +5 degrees
                original_top = self.turret.top_pos
                self.turret.move_to(self.turret.bottom_pos, original_top + 5, force=True)
                time.sleep(1.5)
                
                # Check new position
                result = self.yolo.read()
                if result and result.detections:
                    target = self.find_target_detection(result.detections)
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
        
        if self.turret:
            self.turret.send_command("HOME", read_response=False)
            self.turret.send_command("MOTOR1:0", read_response=False)
            self.turret.send_command("MOTOR2:0", read_response=False)
            self.turret.disconnect()
        
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


def main():
    parser = argparse.ArgumentParser(
        description='YOLO-Based Automatic Camera Gimbal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yolo_gimbal.py --camera 0 --turret COM3 --class person
  python yolo_gimbal.py --camera 1 --turret /dev/ttyUSB0 --class 0
  python yolo_gimbal.py --camera 0 --turret COM3 --class "bottle" --kp 0.8 --ki 0.02
  
  # Fix flipped directions:
  python yolo_gimbal.py --camera 0 --turret COM3 --invert-x  # Flip horizontal
  python yolo_gimbal.py --camera 0 --turret COM3 --invert-y  # Flip vertical
  python yolo_gimbal.py --camera 0 --turret COM3 --swap-servos  # Swap top/bottom
  
Note: To find your camera index, run: python find_camera.py
        """
    )
    parser.add_argument('--camera', '-c', type=int, required=True,
                       help='Camera index (0, 1, 2, etc.)')
    parser.add_argument('--turret', '-t', type=str, required=True,
                       help='Turret serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--class', '-cls', dest='target_class', type=str, default=None,
                       help='Target class name or ID to track (e.g., "person", "0", "bottle")')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--kp', type=float, default=0.5,
                       help='PID proportional gain (default: 0.5)')
    parser.add_argument('--ki', type=float, default=0.01,
                       help='PID integral gain (default: 0.01)')
    parser.add_argument('--kd', type=float, default=0.1,
                       help='PID derivative gain (default: 0.1)')
    parser.add_argument('--deadzone', type=float, default=10.0,
                       help='Deadzone in pixels (default: 10.0)')
    parser.add_argument('--deadzone-degrees', type=float, default=1.0,
                       help='Deadzone in degrees (default: 1.0)')
    parser.add_argument('--movement-scale', type=float, default=30.0,
                       help='Movement scale factor (default: 30.0)')
    parser.add_argument('--min-step', type=float, default=1.0,
                       help='Minimum step size to overcome servo deadband (default: 1.0)')
    parser.add_argument('--control-rate', type=float, default=30.0,
                       help='Control loop rate in Hz (default: 30.0)')
    parser.add_argument('--invert-x', action='store_true',
                       help='Invert horizontal movement direction')
    parser.add_argument('--invert-y', action='store_true',
                       help='Invert vertical movement direction')
    parser.add_argument('--swap-servos', action='store_true',
                       help='Swap top and bottom servos (if they are wired backwards)')
    parser.add_argument('--list-ports', '-l', action='store_true',
                       help='List available serial ports')
    
    args = parser.parse_args()
    
    if args.list_ports:
        list_serial_ports()
        return
    
    # Convert target_class to int if it's a digit
    target_class = args.target_class
    if target_class and target_class.isdigit():
        target_class = int(target_class)
    
    try:
        gimbal = YOLOGimbal(
            camera_index=args.camera,
            turret_port=args.turret,
            target_class=target_class,
            conf_threshold=args.conf,
            kp=args.kp,
            ki=args.ki,
            kd=args.kd,
            deadzone=args.deadzone,
            deadzone_degrees=args.deadzone_degrees,
            movement_scale=args.movement_scale,
            min_step=args.min_step,
            control_rate=args.control_rate,
            invert_x=args.invert_x,
            invert_y=args.invert_y,
            swap_servos=args.swap_servos
        )
        
        gimbal.initialize()
        gimbal.run()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

