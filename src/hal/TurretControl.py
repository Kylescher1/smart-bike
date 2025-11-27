"""
Turret Control Module

Controls a two-servo turret system via Arduino/ESP32 serial communication.
Servo 1: Vertical axis (limited range, typically 15-50 degrees)
Servo 2: Horizontal axis (full range, typically 0-180 degrees)

When run as a script, automatically tracks and follows detected people.
"""

import serial
import time
import threading
import sys
import dill
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2


class TurretControl:
    """
    Controls turret servos via serial communication with Arduino/ESP32.
    
    Communication protocol:
    - Send: "S1:angle,S2:angle\n" (e.g., "S1:35,S2:90\n")
    - Arduino responds with confirmation or current position
    """
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, 
                 servo1_min: int = 15, servo1_max: int = 50, servo1_home: int = 35,
                 servo2_min: int = 0, servo2_max: int = 180, servo2_home: int = 90,
                 deadzone: float = 2.0, kp: float = 0.5, max_speed: float = 5.0,
                 angle_scale_s1: float = 1.0, angle_scale_s2: float = 1.0):
        """
        Initialize turret control.
        
        Args:
            port: Serial port (e.g., "/dev/ttyUSB0" or "COM3")
            baudrate: Serial baudrate (default 115200)
            servo1_min: Minimum angle for servo 1 (vertical, limited range)
            servo1_max: Maximum angle for servo 1
            servo1_home: Home position for servo 1
            servo2_min: Minimum angle for servo 2 (horizontal, full range)
            servo2_max: Maximum angle for servo 2
            servo2_home: Home position for servo 2
            deadzone: Deadzone in degrees (don't move if error < deadzone)
            kp: Proportional gain for control (0.0-1.0, higher = faster response)
            max_speed: Maximum servo movement speed (degrees per update)
        """
        self.port = port
        self.baudrate = baudrate
        
        # Servo limits
        self.servo1_min = servo1_min
        self.servo1_max = servo1_max
        self.servo1_home = servo1_home
        self.servo2_min = servo2_min
        self.servo2_max = servo2_max
        self.servo2_home = servo2_home
        
        # Control parameters
        self.deadzone = deadzone
        self.kp = kp
        self.max_speed = max_speed
        
        # Geometry calibration: scale factors for camera angle to servo angle mapping
        # If camera is offset or servo degrees don't match camera degrees 1:1
        self.angle_scale_s1 = angle_scale_s1  # Vertical axis scaling
        self.angle_scale_s2 = angle_scale_s2  # Horizontal axis scaling
        
        # Smoothing for target angles (minimal smoothing)
        self.smoothed_target_s1 = None
        self.smoothed_target_s2 = None
        self.smoothing_factor = 0.98  # Very high = minimal smoothing, fast response (98% new, 2% old)
        
        # Serial connection
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        
        # Current servo positions
        self.current_s1 = servo1_home
        self.current_s2 = servo2_home
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Rate limiting (prevent sending commands too fast)
        self.last_command_time = 0.0
        self.min_command_interval = 0.02  # 20ms between commands (50 Hz max) - faster updates
    
    def connect(self):
        """Open serial connection to Arduino."""
        if self.connected:
            return
        
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
            time.sleep(2)  # Wait for Arduino to initialize
            self.connected = True
            
            # Move to home position
            self.move_to_absolute(self.servo1_home, self.servo2_home)
            print(f"✅ TurretControl: Connected to {self.port}")
        except Exception as e:
            print(f"❌ TurretControl: Failed to connect: {e}")
            self.connected = False
            self.ser = None
    
    def disconnect(self):
        """Close serial connection."""
        if not self.connected:
            return
        
        self.connected = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        print("TurretControl: Disconnected")
    
    def _send_command(self, s1_angle: Optional[int] = None, s2_angle: Optional[int] = None) -> bool:
        """
        Send servo command to Arduino.
        
        Args:
            s1_angle: Servo 1 angle (None to keep current)
            s2_angle: Servo 2 angle (None to keep current)
        
        Returns:
            True if command sent successfully
        """
        if not self.connected or self.ser is None:
            return False
        
        # Rate limiting (but allow first command)
        current_time = time.time()
        if self.last_command_time > 0 and (current_time - self.last_command_time < self.min_command_interval):
            # Don't block, just skip this command
            if not hasattr(self, '_rate_limit_debug_count'):
                self._rate_limit_debug_count = 0
            self._rate_limit_debug_count += 1
            if self._rate_limit_debug_count % 50 == 0:
                print(f"TurretControl: ⚠️ Rate limited - {current_time - self.last_command_time:.3f}s since last command")
            return False
        
        # Clamp angles to limits
        if s1_angle is not None:
            s1_angle = int(max(self.servo1_min, min(self.servo1_max, s1_angle)))
        if s2_angle is not None:
            s2_angle = int(max(self.servo2_min, min(self.servo2_max, s2_angle)))
        
        # Build command string
        # Note: current_s1/s2 are updated in move_to_absolute, not here
        parts = []
        if s1_angle is not None:
            parts.append(f"S1:{s1_angle}")
        if s2_angle is not None:
            parts.append(f"S2:{s2_angle}")
        
        if not parts:
            return False
        
        command = ",".join(parts) + "\n"
        
        try:
            # Flush input buffer first
            self.ser.reset_input_buffer()
            
            # Send command
            bytes_written = self.ser.write(command.encode('utf-8'))
            self.ser.flush()  # Ensure data is sent immediately
            
            self.last_command_time = current_time
            
            # Debug output (print more frequently for tracking)
            if not hasattr(self, '_cmd_count'):
                self._cmd_count = 0
            self._cmd_count += 1
            if self._cmd_count <= 10 or self._cmd_count % 5 == 0:  # Print first 10, then every 5th
                print(f"TurretControl: 📤 Sent '{command.strip()}' ({bytes_written} bytes)")
            
            # Try to read response (non-blocking) and update positions
            if self.ser.in_waiting > 0:
                try:
                    response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if response:
                        print(f"TurretControl: Arduino response: {response}")
                        # Parse response to update actual servo positions
                        # Format: "S1: 39" or "S1:39,S2:94"
                        import re
                        s1_match = re.search(r'S1:\s*(\d+)', response)
                        s2_match = re.search(r'S2:\s*(\d+)', response)
                        with self.lock:
                            if s1_match:
                                self.current_s1 = float(s1_match.group(1))
                            if s2_match:
                                self.current_s2 = float(s2_match.group(1))
                except Exception as e:
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"TurretControl: Response parse error: {e}")
                    pass
            
            return True
        except Exception as e:
            print(f"TurretControl: Command send error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def move_to_absolute(self, s1_angle: Optional[float] = None, s2_angle: Optional[float] = None):
        """
        Move servos to absolute angles.
        
        Args:
            s1_angle: Servo 1 angle in degrees (None to keep current)
            s2_angle: Servo 2 angle in degrees (None to keep current)
        """
        # Update current positions BEFORE sending command (predictive)
        # This prevents lag in position tracking
        with self.lock:
            if s1_angle is not None:
                self.current_s1 = s1_angle  # Update immediately
            if s2_angle is not None:
                self.current_s2 = s2_angle  # Update immediately
            
            s1_int = int(round(s1_angle)) if s1_angle is not None else None
            s2_int = int(round(s2_angle)) if s2_angle is not None else None
        
        # Send command outside lock
        self._send_command(s1_int, s2_int)
    
    def move_relative(self, s1_delta: float = 0.0, s2_delta: float = 0.0):
        """
        Move servos relative to current position.
        
        Args:
            s1_delta: Servo 1 delta angle in degrees
            s2_delta: Servo 2 delta angle in degrees
        """
        with self.lock:
            new_s1 = self.current_s1 + s1_delta
            new_s2 = self.current_s2 + s2_delta
            self.move_to_absolute(new_s1, new_s2)
    
    def center_object(self, theta_deg: float, alpha_deg: float, 
                     turret_dir_vector: Optional[np.ndarray] = None,
                     object_dir_vector: Optional[np.ndarray] = None):
        """
        Center turret on object using angles from vision system.
        
        Args:
            theta_deg: Horizontal angle in degrees (positive = right)
            alpha_deg: Vertical angle in degrees (positive = up)
            turret_dir_vector: Optional 3D direction vector of current turret aim (for vector-based correction)
            object_dir_vector: Optional 3D direction vector to object (for vector-based correction)
        
        Note:
            - Servo 1 controls vertical (limited range)
            - Servo 2 controls horizontal (full range)
            - Angles are relative to camera center
            - If vectors are provided, uses angle between vectors for more accurate correction
        """
        if not self.connected:
            print(f"TurretControl: ⚠️ Not connected, ignoring command theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°")
            return
        
        # If vectors are provided, calculate correction from angle between vectors
        if turret_dir_vector is not None and object_dir_vector is not None:
            return self._center_object_using_vectors(turret_dir_vector, object_dir_vector)
        
        with self.lock:
            # Simple direct centering: move camera towards frame center
            # Calculate desired servo positions with geometry scaling
            desired_s2 = self.servo2_home + (theta_deg * self.angle_scale_s2)
            desired_s1 = self.servo1_home + (alpha_deg * self.angle_scale_s1)
            
            # Clamp to servo 1's limited range
            desired_s1 = max(self.servo1_min, min(self.servo1_max, desired_s1))
            
            # Apply minimal smoothing to reduce jerkiness
            # With accurate angle calculation, we can use very light smoothing
            if self.smoothed_target_s1 is None:
                self.smoothed_target_s1 = desired_s1
                self.smoothed_target_s2 = desired_s2
            else:
                # Very light smoothing (98% new, 2% old) - allows fast response while reducing noise
                self.smoothed_target_s1 = self.smoothed_target_s1 * (1.0 - self.smoothing_factor) + desired_s1 * self.smoothing_factor
                self.smoothed_target_s2 = self.smoothed_target_s2 * (1.0 - self.smoothing_factor) + desired_s2 * self.smoothing_factor
            
            # Calculate errors from smoothed targets
            error_s1 = self.smoothed_target_s1 - self.current_s1
            error_s2 = self.smoothed_target_s2 - self.current_s2
            
            # Debug output - show detailed centering info (every call when theta is near zero to catch issues)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            should_debug = (self._debug_counter % 5 == 0) or (abs(theta_deg) < 1.0)  # More frequent when near zero
            if should_debug:
                print(f"TurretControl: 📍 Input angles: theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°")
                print(f"TurretControl:    Desired: S1={desired_s1:.2f}° (home={self.servo1_home}° + alpha={alpha_deg:.2f}° * scale={self.angle_scale_s1})")
                print(f"TurretControl:    Desired: S2={desired_s2:.2f}° (home={self.servo2_home}° + theta={theta_deg:.2f}° * scale={self.angle_scale_s2})")
                print(f"TurretControl:    Current: S1={self.current_s1:.2f}°, S2={self.current_s2:.2f}°")
                print(f"TurretControl:    Smoothed targets: S1={self.smoothed_target_s1:.2f}°, S2={self.smoothed_target_s2:.2f}°")
                print(f"TurretControl:    Errors: S1={error_s1:.2f}°, S2={error_s2:.2f}° (deadzone={self.deadzone}°)")
            
            # Simple proportional control with deadzone
            # Move camera towards frame center
            
            # Servo 1 (vertical)
            if abs(error_s1) > self.deadzone:
                move_s1 = error_s1 * self.kp
                move_s1 = max(-self.max_speed, min(self.max_speed, move_s1))
                new_s1 = self.current_s1 + move_s1
            else:
                new_s1 = self.current_s1
            
            # Servo 2 (horizontal)
            if abs(error_s2) > self.deadzone:
                move_s2 = error_s2 * self.kp
                move_s2 = max(-self.max_speed, min(self.max_speed, move_s2))
                new_s2 = self.current_s2 + move_s2
            else:
                # Even if error is small, if it's consistently in one direction, make tiny adjustment
                # This prevents getting stuck just outside deadzone
                if abs(error_s2) > 0.05:  # Very small threshold
                    move_s2 = error_s2 * self.kp * 0.5  # Reduced gain for fine adjustments
                    move_s2 = max(-self.max_speed * 0.3, min(self.max_speed * 0.3, move_s2))
                    new_s2 = self.current_s2 + move_s2
                else:
                    new_s2 = self.current_s2
            
            # Move servos
            should_move = (new_s1 != self.current_s1 or new_s2 != self.current_s2)
            if should_move:
                move_s1_delta = new_s1 - self.current_s1
                move_s2_delta = new_s2 - self.current_s2
                if self._debug_counter % 10 == 0:
                    print(f"TurretControl: 🎯 Moving: S1 {self.current_s1:.1f}° → {new_s1:.1f}° (Δ{move_s1_delta:+.2f}°)")
                    print(f"TurretControl:    Moving: S2 {self.current_s2:.1f}° → {new_s2:.1f}° (Δ{move_s2_delta:+.2f}°)")
                    print(f"TurretControl:    kp={self.kp}, max_speed={self.max_speed}, deadzone={self.deadzone}")
                # Store values we need before releasing lock
                move_s1_val = new_s1
                move_s2_val = new_s2
            else:
                if self._debug_counter % 10 == 0:
                    print(f"TurretControl: ⏸️ No movement - errors too small or filtered")
                move_s1_val = None
                move_s2_val = None
        
        # Call move_to_absolute outside the lock to avoid deadlock
        # (move_to_absolute will acquire its own lock)
        if move_s1_val is not None or move_s2_val is not None:
            self.move_to_absolute(move_s1_val, move_s2_val)
    
    def _center_object_using_vectors(self, turret_dir: np.ndarray, object_dir: np.ndarray):
        """
        Center turret by calculating angle between current aim and object direction vectors.
        This is more robust than relying on potentially incorrect angle calculations.
        
        Args:
            turret_dir: Current turret aim direction vector (normalized)
            object_dir: Direction vector to object (normalized)
        """
        if not self.connected:
            return
        
        # Normalize vectors
        turret_dir = turret_dir / np.linalg.norm(turret_dir)
        object_dir = object_dir / np.linalg.norm(object_dir)
        
        # Calculate angle between vectors
        dot_product = np.clip(np.dot(turret_dir, object_dir), -1.0, 1.0)
        angle_between_rad = np.arccos(dot_product)
        angle_between_deg = np.degrees(angle_between_rad)
        
        # Calculate corrections by projecting onto horizontal and vertical planes
        # Horizontal correction: angle difference in X-Z plane (rotation around Y axis, affects S2)
        # Vertical correction: angle difference in Y-Z plane (rotation around X axis, affects S1)
        
        # Horizontal: calculate yaw angles in X-Z plane
        turret_yaw_rad = np.arctan2(turret_dir[0], turret_dir[2])  # atan2(x, z) for horizontal angle
        object_yaw_rad = np.arctan2(object_dir[0], object_dir[2])
        
        # Calculate horizontal angle difference (how much to rotate around Y axis)
        horizontal_diff_rad = object_yaw_rad - turret_yaw_rad
        
        # Normalize to [-pi, pi] range
        while horizontal_diff_rad > np.pi:
            horizontal_diff_rad -= 2 * np.pi
        while horizontal_diff_rad < -np.pi:
            horizontal_diff_rad += 2 * np.pi
        
        horizontal_correction_deg = np.degrees(horizontal_diff_rad)
        
        # Flip sign for horizontal correction (coordinate system convention)
        # If object is to the right (positive x), we need to rotate right (increase S2)
        # But our calculation gives negative when object is right, so flip it
        horizontal_correction_deg = -horizontal_correction_deg
        
        # Vertical: calculate pitch angles in Y-Z plane
        # Project onto Y-Z plane by normalizing (y, z) components
        turret_yz = np.array([turret_dir[1], turret_dir[2]])
        object_yz = np.array([object_dir[1], object_dir[2]])
        turret_yz_norm = turret_yz / (np.linalg.norm(turret_yz) + 1e-10)
        object_yz_norm = object_yz / (np.linalg.norm(object_yz) + 1e-10)
        
        turret_pitch_rad = np.arctan2(turret_yz_norm[0], turret_yz_norm[1])  # atan2(y, z)
        object_pitch_rad = np.arctan2(object_yz_norm[0], object_yz_norm[1])
        
        # Calculate vertical angle difference
        vertical_diff_rad = object_pitch_rad - turret_pitch_rad
        
        # Normalize to [-pi, pi] range
        while vertical_diff_rad > np.pi:
            vertical_diff_rad -= 2 * np.pi
        while vertical_diff_rad < -np.pi:
            vertical_diff_rad += 2 * np.pi
        
        vertical_correction_deg = np.degrees(vertical_diff_rad)
        
        with self.lock:
            # Apply corrections to current positions
            desired_s2 = self.current_s2 + (horizontal_correction_deg * self.angle_scale_s2)
            desired_s1 = self.current_s1 + (vertical_correction_deg * self.angle_scale_s1)
            
            # Clamp to limits
            desired_s1 = max(self.servo1_min, min(self.servo1_max, desired_s1))
            desired_s2 = max(self.servo2_min, min(self.servo2_max, desired_s2))
            
            # Calculate errors
            error_s1 = desired_s1 - self.current_s1
            error_s2 = desired_s2 - self.current_s2
            
            # Debug output
            if not hasattr(self, '_vector_debug_counter'):
                self._vector_debug_counter = 0
            self._vector_debug_counter += 1
            if self._vector_debug_counter % 5 == 0:
                print(f"TurretControl: 🎯 Vector-based correction")
                print(f"  Turret dir: ({turret_dir[0]:.3f}, {turret_dir[1]:.3f}, {turret_dir[2]:.3f})")
                print(f"  Object dir: ({object_dir[0]:.3f}, {object_dir[1]:.3f}, {object_dir[2]:.3f})")
                print(f"  Turret yaw: {np.degrees(turret_yaw_rad):.2f}°, Object yaw: {np.degrees(object_yaw_rad):.2f}°")
                print(f"  Angle between vectors: {angle_between_deg:.2f}°")
                print(f"  Correction: horizontal={horizontal_correction_deg:.2f}°, vertical={vertical_correction_deg:.2f}°")
                print(f"  Current: S1={self.current_s1:.2f}°, S2={self.current_s2:.2f}°")
                print(f"  Desired: S1={desired_s1:.2f}°, S2={desired_s2:.2f}°")
                print(f"  Errors: S1={error_s1:.2f}°, S2={error_s2:.2f}°")
                print(f"  Servo limits: S1=[{self.servo1_min}-{self.servo1_max}], S2=[{self.servo2_min}-{self.servo2_max}]")
            
            # Apply corrections if above deadzone
            if abs(error_s1) > self.deadzone or abs(error_s2) > self.deadzone:
                move_s1_val = desired_s1
                move_s2_val = desired_s2
            else:
                move_s1_val = None
                move_s2_val = None
        
        # Move servos
        if move_s1_val is not None or move_s2_val is not None:
            self.move_to_absolute(move_s1_val, move_s2_val)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current servo positions."""
        with self.lock:
            return (self.current_s1, self.current_s2)
    
    def query_position(self) -> Tuple[float, float]:
        """
        Query Arduino for actual servo positions and update internal state.
        This is more accurate than predictive tracking.
        """
        if not self.connected or self.ser is None:
            return self.get_position()
        
        try:
            # Send a query command (Arduino might respond with current positions)
            # Try reading any pending responses first
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    import re
                    s1_match = re.search(r'S1:\s*(\d+)', response)
                    s2_match = re.search(r'S2:\s*(\d+)', response)
                    with self.lock:
                        if s1_match:
                            self.current_s1 = float(s1_match.group(1))
                        if s2_match:
                            self.current_s2 = float(s2_match.group(2))
                        return (self.current_s1, self.current_s2)
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"TurretControl: Query position error: {e}")
        
        # Fallback to tracked position
        return self.get_position()
    
    def go_home(self):
        """Move turret to home position."""
        self.move_to_absolute(self.servo1_home, self.servo2_home)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()


class TurretTracker:
    """
    Tracks objects detected by VISION system and centers them using turret servos.
    
    Features:
    - Automatic object selection (largest, highest confidence, or specific class)
    - Smooth tracking with PID control
    - Multiple tracking modes (largest, highest confidence, specific class)
    - Deadzone to prevent jitter
    """
    
    def __init__(self, vision, turret: TurretControl,
                 tracking_mode: str = "largest",
                 target_class: Optional[str] = None,
                 min_confidence: float = 0.3,
                 max_tracking_distance: float = 0.5,
                 camera_config: Optional[Dict] = None):
        """
        Initialize turret tracker.
        
        Args:
            vision: VISION system instance (must be started)
            turret: TurretControl instance (must be connected)
            tracking_mode: "largest", "highest_confidence", or "class"
            target_class: Class name to track (required if mode="class")
            min_confidence: Minimum confidence threshold for tracking
            max_tracking_distance: Maximum angular distance to track (degrees)
        """
        self.vision = vision
        self.turret = turret
        
        # Tracking parameters
        self.tracking_mode = tracking_mode
        self.target_class = target_class
        self.min_confidence = min_confidence
        self.max_tracking_distance = max_tracking_distance
        
        # Load camera intrinsics for accurate angle calculation
        self.camera_config = camera_config
        self.right_K = None  # Camera matrix for right camera (rectified)
        self._load_camera_intrinsics()
        
        # Tracking state
        self.tracking = False
        self.track_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Currently tracked object
        self.tracked_object_id: Optional[int] = None
        self.last_track_time = 0.0
        self.track_timeout = 1.0  # Lose track if no detection for 1 second
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.tracking_stats = {
            'frames_processed': 0,
            'objects_tracked': 0,
            'track_lost_count': 0
        }
    
    def _load_camera_intrinsics(self):
        """Load camera intrinsics from config for accurate angle calculation."""
        if self.camera_config is None:
            # Try to get from vision system's config
            if hasattr(self.vision, 'config'):
                self.camera_config = self.vision.config
            else:
                return
        
        try:
            right_cfg = self.camera_config.get('right', {})
            
            # Try to get rectified camera matrix (newK or from P matrix)
            # Prefer newK (rectified) over original K for better accuracy
            if 'newK' in right_cfg:
                self.right_K = np.asarray(right_cfg['newK'], dtype=np.float64)
            elif 'P' in right_cfg:
                # Extract K from P matrix (first 3 columns)
                P = np.asarray(right_cfg['P'], dtype=np.float64)
                self.right_K = P[:, :3]
            elif 'K' in right_cfg:
                # Use original K if newK/P not available
                self.right_K = np.asarray(right_cfg['K'], dtype=np.float64)
            
            # Also load distortion coefficients if available (for undistortion if needed)
            if 'D' in right_cfg:
                self.right_D = np.asarray(right_cfg['D'], dtype=np.float64)
            else:
                self.right_D = None
            
            if self.right_K is not None:
                fx = self.right_K[0, 0]
                fy = self.right_K[1, 1]
                cx = self.right_K[0, 2]
                cy = self.right_K[1, 2]
                print(f"TurretTracker: Loaded camera intrinsics - fx={fx:.1f}, fy={fy:.1f}, "
                      f"cx={cx:.1f}, cy={cy:.1f}")
                if self.right_D is not None:
                    print(f"TurretTracker: Loaded distortion coefficients D={self.right_D.ravel()}")
        except Exception as e:
            print(f"TurretTracker: Warning - Could not load camera intrinsics: {e}")
            import traceback
            traceback.print_exc()
            self.right_K = None
            self.right_D = None
    
    def start_tracking(self):
        """Start automatic tracking thread."""
        if self.tracking:
            return
        
        if not self.vision.connected:
            raise RuntimeError("VISION system must be started before tracking")
        
        if not self.turret.connected:
            raise RuntimeError("TurretControl must be connected before tracking")
        
        self.tracking = True
        self.stop_event.clear()
        self.track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.track_thread.start()
        print("✅ TurretTracker: Tracking started")
    
    def stop_tracking(self):
        """Stop automatic tracking."""
        if not self.tracking:
            return
        
        self.tracking = False
        self.stop_event.set()
        
        if self.track_thread and self.track_thread.is_alive():
            self.track_thread.join(timeout=2.0)
        
        # Move to home position
        self.turret.go_home()
        print("TurretTracker: Tracking stopped")
    
    def _select_target_object(self, objects: List[Dict]) -> Optional[Dict]:
        """
        Select target object based on tracking mode.
        
        Args:
            objects: List of detected objects from vision.read()
        
        Returns:
            Selected object dict or None
        """
        if not objects:
            return None
        
        # Filter for person class only
        person_objects = [obj for obj in objects if obj.get('type', '').lower() == 'person']
        if not person_objects:
            return None
        
        # Filter by confidence
        valid_objects = [obj for obj in person_objects if obj.get('confidence', 0.0) >= self.min_confidence]
        if not valid_objects:
            return None
        
        # Select highest confidence person
        return max(valid_objects, key=lambda obj: obj.get('confidence', 0.0))
    
    def _bbox_to_angles(self, bbox: List[int]) -> Tuple[float, float]:
        """
        Convert bounding box to turret angles using camera intrinsics.
        
        This method properly handles fisheye cameras by using the camera matrix K
        to convert pixel coordinates to normalized camera coordinates, then calculating
        accurate angles using atan2. This is much more accurate than linear FOV approximation,
        especially for objects far from the image center.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in rectified image coordinates
        
        Returns:
            Tuple of (theta_deg, alpha_deg) - horizontal and vertical angles relative to camera center
        
        Note: Camera center = turret home position. Angles are calculated using camera intrinsics
        for accurate fisheye camera support.
        """
        if len(bbox) != 4:
            return (0.0, 0.0)
        
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Use camera intrinsics if available (more accurate for fisheye)
        if self.right_K is not None:
            # Extract camera parameters from K matrix
            fx = self.right_K[0, 0]  # Focal length x
            fy = self.right_K[1, 1]  # Focal length y
            cx = self.right_K[0, 2]  # Principal point x (for full rectified image)
            cy = self.right_K[1, 2]  # Principal point y (for full rectified image)
            
            # Get actual image dimensions (may be cropped)
            try:
                with self.vision.frame_lock:
                    if self.vision.last_right_frame is not None:
                        h, w = self.vision.last_right_frame.shape[:2]
                    elif self.vision.last_left_frame is not None:
                        h, w = self.vision.last_left_frame.shape[:2]
                    else:
                        h, w = 480, 640
            except:
                h, w = 480, 640  # Default fallback
            
            # Get crop value from config if available
            # If images are cropped, bbox coordinates are in cropped space but principal point is in full rectified space
            crop_value = 0
            if self.camera_config is not None:
                crop_value = self.camera_config.get('crop', 0)
            
            # Adjust bbox coordinates to full rectified image space if cropped
            # The principal point (cx, cy) in K matrix is for the full rectified image
            center_x_full = center_x + crop_value
            center_y_full = center_y + crop_value
            
            # Check if principal point matches expected center (accounting for crop)
            # Full rectified image dimensions would be (w + 2*crop, h + 2*crop)
            expected_cx = (w + 2 * crop_value) / 2.0 if crop_value > 0 else w / 2.0
            expected_cy = (h + 2 * crop_value) / 2.0 if crop_value > 0 else h / 2.0
            
            cx_offset = cx - expected_cx
            cy_offset = cy - expected_cy
            
            # If principal point is way off from expected center, use image center instead
            # The K matrix principal point might be for a different image size/resolution
            if abs(cx_offset) > 100 or abs(cy_offset) > 100:
                # Principal point mismatch - use actual image center
                # This handles cases where K matrix is for different resolution or coordinate system
                cx_adjusted = w / 2.0  # Use actual image center
                cy_adjusted = h / 2.0
                center_x_use = center_x  # Use cropped coordinates (they're relative to cropped image)
                center_y_use = center_y
                
                if not hasattr(self, '_warned_crop_adjustment'):
                    print(f"TurretTracker: ⚠️ Principal point mismatch - using image center")
                    print(f"TurretTracker:    K matrix principal point: ({cx:.1f},{cy:.1f})")
                    print(f"TurretTracker:    Expected for image {w}x{h} (crop={crop_value}): ({expected_cx:.1f},{expected_cy:.1f})")
                    print(f"TurretTracker:    Offset: ({cx_offset:.1f},{cy_offset:.1f})")
                    print(f"TurretTracker:    Using image center: ({cx_adjusted:.1f},{cy_adjusted:.1f})")
                    print(f"TurretTracker:    This suggests K matrix is for different image size - using image center for angle calc")
                    self._warned_crop_adjustment = True
            else:
                # Principal point is reasonable, use it with full rectified coordinates
                cx_adjusted = cx
                cy_adjusted = cy
                center_x_use = center_x_full
                center_y_use = center_y_full
            
            # Convert pixel coordinates to normalized camera coordinates
            # This accounts for the actual camera geometry, not just FOV
            x_norm = (center_x_use - cx_adjusted) / fx  # Normalized x coordinate
            y_norm = (center_y_use - cy_adjusted) / fy  # Normalized y coordinate
            
            # Calculate angles using atan2 for accurate angle calculation
            # atan2 gives us the angle from the camera's optical axis
            theta_rad = np.arctan2(x_norm, 1.0)  # Horizontal angle (yaw)
            alpha_rad = np.arctan2(y_norm, 1.0)  # Vertical angle (pitch)
            
            # Convert to degrees
            theta_deg = np.degrees(theta_rad)
            alpha_deg = np.degrees(alpha_rad)
            
            # Flip signs (object right = turret move left, object up = turret move down)
            theta_deg = -theta_deg
            alpha_deg = -alpha_deg
            
            # Debug output (more frequently to catch issues)
            if not hasattr(self, '_angle_debug_count'):
                self._angle_debug_count = 0
            self._angle_debug_count += 1
            if self._angle_debug_count % 10 == 0:  # Every 10 frames for better visibility
                pixel_error_x = center_x_use - cx_adjusted
                pixel_error_y = center_y_use - cy_adjusted
                print(f"TurretTracker: 🔍 Angle calc (intrinsics)")
                print(f"  Bbox: ({x1},{y1}) to ({x2},{y2}), center (cropped)=({center_x:.1f},{center_y:.1f})")
                print(f"  Image: {w}x{h}, crop={crop_value}, center (full)=({center_x_use:.1f},{center_y_use:.1f})")
                print(f"  Principal point: K=({cx:.1f},{cy:.1f}), adjusted=({cx_adjusted:.1f},{cy_adjusted:.1f})")
                print(f"  Focal: fx={fx:.1f}, fy={fy:.1f}")
                print(f"  Pixel errors: X={pixel_error_x:+.1f}, Y={pixel_error_y:+.1f}")
                print(f"  Normalized: x_norm={x_norm:.4f}, y_norm={y_norm:.4f}")
                print(f"  Angles: θ={theta_deg:.2f}°, α={alpha_deg:.2f}°")
        else:
            # Fallback to FOV-based method if intrinsics not available
            # Get frame dimensions from vision system (using right camera)
            try:
                with self.vision.frame_lock:
                    if self.vision.last_right_frame is not None:
                        h, w = self.vision.last_right_frame.shape[:2]
                    elif self.vision.last_left_frame is not None:
                        h, w = self.vision.last_left_frame.shape[:2]
                    else:
                        h, w = 480, 640
            except:
                h, w = 480, 640  # Default fallback
            
            fov_h = getattr(self.vision, 'fov_horizontal', 126.0)  # degrees
            fov_v = getattr(self.vision, 'fov_vertical', 101.62)  # degrees
            
            # Linear FOV approximation (less accurate, especially for fisheye)
            theta = ((center_x - w / 2.0) / w) * fov_h
            alpha = ((center_y - h / 2.0) / h) * fov_v
            
            theta_deg = -theta
            alpha_deg = -alpha
            
            # Debug output
            if not hasattr(self, '_angle_debug_count'):
                self._angle_debug_count = 0
            self._angle_debug_count += 1
            if self._angle_debug_count % 30 == 0:
                pixel_error_x = center_x - w / 2.0
                pixel_error_y = center_y - h / 2.0
                print(f"TurretTracker: Angle calc (FOV fallback) - bbox center=({center_x:.0f},{center_y:.0f}), "
                      f"frame center=({w/2:.0f},{h/2:.0f}), pixel errors=({pixel_error_x:+.0f},{pixel_error_y:+.0f}), "
                      f"FOV=({fov_h:.1f}°x{fov_v:.1f}°), angles=θ={theta_deg:.2f}° α={alpha_deg:.2f}°")
        
        return (theta_deg, alpha_deg)
    
    def _tracking_loop(self):
        """Main tracking loop running in background thread."""
        print("TurretTracker: Tracking loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get latest vision data
                vision_data = self.vision.read()
                objects = vision_data.get('objects', [])
                
                self.tracking_stats['frames_processed'] += 1
                
                # Debug: Print every frame for now
                print(f"TurretTracker: 🔄 Loop iteration {self.tracking_stats['frames_processed']} - {len(objects)} objects")
                
                # Select target object
                target = self._select_target_object(objects)
                print(f"TurretTracker: 🎯 Selected target: {target.get('id') if target else None}")
                
                current_time = time.time()
                
                if target is None:
                    # No valid target found
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ No valid target selected from {len(objects)} objects")
                    # Check if we should lose track
                    if self.tracked_object_id is not None:
                        if (current_time - self.last_track_time) > self.track_timeout:
                            # Lost track
                            self.tracked_object_id = None
                            self.tracking_stats['track_lost_count'] += 1
                            print(f"TurretTracker: ❌ Lost track (timeout)")
                    time.sleep(0.033)  # ~30 Hz update rate
                    continue
                
                # Check if this is a new object or same as before
                obj_id = target.get('id', None)
                
                # If we have a tracked object, prefer to keep tracking it
                if self.tracked_object_id is not None:
                    # Look for the same object ID
                    same_obj = next((obj for obj in objects if obj.get('id') == self.tracked_object_id), None)
                    if same_obj is not None:
                        target = same_obj
                        obj_id = self.tracked_object_id
                
                # Get bbox directly from cached detections
                # Match by ID first, then fall back to largest detection
                bbox = None
                try:
                    with self.vision.frame_lock:
                        if hasattr(self.vision, 'last_detections_cache') and self.vision.last_detections_cache:
                            # Try to match by track_id or id
                            for det in self.vision.last_detections_cache:
                                det_id = det.get('track_id') or det.get('id')
                                if det_id == obj_id:
                                    bbox = det.get('bbox')
                                    break
                            
                            # If no match by ID, use the largest detection (matches our "largest" selection logic)
                            if bbox is None and self.tracking_mode == "largest":
                                largest_det = None
                                largest_area = 0
                                for det in self.vision.last_detections_cache:
                                    det_bbox = det.get('bbox', [])
                                    if len(det_bbox) == 4:
                                        area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                                        if area > largest_area:
                                            largest_area = area
                                            largest_det = det
                                if largest_det:
                                    bbox = largest_det.get('bbox')
                                    if self.tracking_stats['frames_processed'] % 30 == 0:
                                        print(f"TurretTracker: Using largest detection (area={largest_area})")
                except Exception as e:
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: Bbox lookup error: {e}")
                
                # Calculate angles from bbox
                if bbox is None or len(bbox) != 4:
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ No bbox found for ID={obj_id}, cache has {len(self.vision.last_detections_cache) if hasattr(self.vision, 'last_detections_cache') else 0} detections")
                    time.sleep(0.033)
                    continue
                
                # Calculate angles directly from bbox
                theta_deg, alpha_deg = self._bbox_to_angles(bbox)
                
                # Debug output (print every frame for now)
                print(f"TurretTracker: ✅ Target selected - ID={obj_id}, "
                      f"theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°")
                
                # Check if angles are zero (bbox lookup failed)
                if abs(theta_deg) < 0.01 and abs(alpha_deg) < 0.01:
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ Angles are zero - bbox lookup may have failed for ID={obj_id}")
                    time.sleep(0.033)
                    continue
                
                # Check if object is within tracking distance
                angular_distance = np.sqrt(theta_deg**2 + alpha_deg**2)
                if angular_distance > self.max_tracking_distance:
                    # Object too far, don't track
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ Object too far (distance={angular_distance:.1f}° > max={self.max_tracking_distance}°)")
                    time.sleep(0.033)
                    continue
                
                # Update tracking state
                with self.lock:
                    self.tracked_object_id = obj_id
                    self.last_track_time = current_time
                
                # Debug: Print before sending command
                print(f"TurretTracker: 🎯 Calling center_object(theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°)")
                
                # Calculate direction vectors for vector-based correction
                # This uses the actual geometric relationship, not potentially incorrect angle calculations
                
                # Get current turret position
                s1, s2 = self.turret.get_position()
                
                # Calculate turret direction vector from servo positions
                yaw_deg = s2 - self.turret.servo2_home
                pitch_deg = s1 - self.turret.servo1_home
                yaw_rad = np.deg2rad(yaw_deg)
                pitch_rad = np.deg2rad(pitch_deg)
                turret_dir = np.array([
                    np.sin(yaw_rad) * np.cos(pitch_rad),
                    np.sin(pitch_rad),
                    np.cos(yaw_rad) * np.cos(pitch_rad)
                ])
                turret_dir = turret_dir / np.linalg.norm(turret_dir)
                
                # Calculate object direction vector directly from bbox center
                # This bypasses the potentially incorrect theta/alpha calculation
                # Get image dimensions
                try:
                    with self.vision.frame_lock:
                        if self.vision.last_right_frame is not None:
                            h, w = self.vision.last_right_frame.shape[:2]
                        elif self.vision.last_left_frame is not None:
                            h, w = self.vision.last_left_frame.shape[:2]
                        else:
                            h, w = 480, 640
                except:
                    h, w = 480, 640
                
                # Calculate bbox center
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                
                # Use image center as reference (more reliable than K matrix principal point)
                image_center_x = w / 2.0
                image_center_y = h / 2.0
                
                # Calculate pixel offset from center
                pixel_offset_x = center_x - image_center_x
                pixel_offset_y = center_y - image_center_y
                
                # Convert to normalized coordinates using focal length
                # Use a reasonable focal length estimate (or from K matrix if available)
                if self.right_K is not None:
                    fx = self.right_K[0, 0]
                    fy = self.right_K[1, 1]
                else:
                    # Estimate focal length from image size (common for fisheye)
                    fx = fy = w * 0.8
                
                x_norm = pixel_offset_x / fx
                y_norm = -pixel_offset_y / fy  # Negative because Y points down
                
                # Calculate direction vector from normalized coordinates
                # This gives us the actual direction to the object
                object_dir = np.array([x_norm, y_norm, 1.0])
                object_dir = object_dir / np.linalg.norm(object_dir)
                
                # Debug: show vector-based calculation
                if not hasattr(self, '_vector_calc_count'):
                    self._vector_calc_count = 0
                self._vector_calc_count += 1
                if self._vector_calc_count % 10 == 0:
                    angle_between = np.degrees(np.arccos(np.clip(np.dot(turret_dir, object_dir), -1.0, 1.0)))
                    print(f"TurretTracker: 📐 Vector-based calculation")
                    print(f"  Bbox center: ({center_x:.1f},{center_y:.1f}), image center: ({image_center_x:.1f},{image_center_y:.1f})")
                    print(f"  Pixel offset: ({pixel_offset_x:.1f},{pixel_offset_y:.1f})")
                    print(f"  Normalized: ({x_norm:.4f},{y_norm:.4f})")
                    print(f"  Turret dir: ({turret_dir[0]:.3f},{turret_dir[1]:.3f},{turret_dir[2]:.3f})")
                    print(f"  Object dir: ({object_dir[0]:.3f},{object_dir[1]:.3f},{object_dir[2]:.3f})")
                    print(f"  Angle between: {angle_between:.2f}°")
                
                # Center turret on object using vector-based correction
                try:
                    self.turret.center_object(theta_deg, alpha_deg, 
                                            turret_dir_vector=turret_dir,
                                            object_dir_vector=object_dir)
                    # Don't call get_position() here - it would deadlock since center_object() holds the lock
                    print(f"TurretTracker: ✅ center_object() returned")
                except Exception as e:
                    print(f"TurretTracker: ❌ Error sending command: {e}")
                    import traceback
                    traceback.print_exc()
                
                self.tracking_stats['objects_tracked'] += 1
                
                # Print stats every 60 frames (~2 seconds at 30 Hz)
                if self.tracking_stats['frames_processed'] % 60 == 0:
                    print(f"TurretTracker: Tracking object ID={obj_id}, "
                          f"theta={theta_deg:.1f}°, alpha={alpha_deg:.1f}°, "
                          f"confidence={target.get('confidence', 0.0):.2f}")
                
                time.sleep(0.033)  # ~30 Hz update rate
                
            except Exception as e:
                print(f"TurretTracker: Error in tracking loop: {e}")
                time.sleep(0.1)
        
        print("TurretTracker: Tracking loop stopped")
    
    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        with self.lock:
            return self.tracking_stats.copy()
    
    def set_tracking_mode(self, mode: str, target_class: Optional[str] = None):
        """
        Change tracking mode.
        
        Args:
            mode: "largest", "highest_confidence", or "class"
            target_class: Class name (required if mode="class")
        """
        with self.lock:
            self.tracking_mode = mode
            self.target_class = target_class
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_tracking()


class SimpleTurretTracker:
    """Simple turret tracker that keeps a person centered."""
    
    def __init__(self, vision, turret):
        self.vision = vision
        self.turret = turret
        
        # Get image dimensions (will be set when we get first frame)
        self.image_width = None
        self.image_height = None
        
        # Control parameters
        self.deadzone_pixels = 30  # Don't move if offset is less than this (increased to prevent oscillation)
        self.max_move_speed = 1.0  # Maximum degrees to move per update (reduced for smoother movement)
        self.kp = 0.15  # Proportional gain (reduced to prevent overshoot)
        
        # Damping - reduces movement as we get closer to center
        self.damping_factor = 0.5  # Multiply movement by this when close to center
        self.damping_threshold = 100  # Pixels - start damping when offset < this
        
        # Sweep parameters
        self.last_person_time = time.time()
        self.sweep_interval = 30.0  # Seconds between sweeps when no person detected
        self.sweep_active = False
        self.sweep_start_time = 0
        self.sweep_duration = 5.0  # Seconds for one complete sweep
        self.sweep_range_s2 = 60  # Degrees to sweep horizontally (center ± range)
        
        # Field of view (will use from config if available)
        self.fov_horizontal = 126.0  # degrees
        self.fov_vertical = 101.62  # degrees
    
    def get_image_size(self):
        """Get image dimensions from vision system."""
        debug = self.vision.debug()
        if debug.get('last_left_image') is not None:
            h, w = debug['last_left_image'].shape[:2]
            self.image_width = w
            self.image_height = h
            return True
        return False
    
    def find_person(self, objects):
        """Find the best person to track."""
        people = [obj for obj in objects if obj.get('type', '').lower() == 'person']
        if not people:
            return None
        
        # Return the person with highest confidence
        return max(people, key=lambda obj: obj.get('confidence', 0.0))
    
    def calculate_offset(self, person):
        """
        Calculate pixel offset of person from image center.
        
        Returns:
            (offset_x, offset_y) in pixels, or (None, None) if can't calculate
        """
        if self.image_width is None or self.image_height is None:
            if not self.get_image_size():
                return None, None
        
        # Try to get bbox from detections cache (most accurate)
        bbox = None
        person_id = person.get('id')
        
        try:
            with self.vision.frame_lock:
                if hasattr(self.vision, 'last_detections_cache') and self.vision.last_detections_cache:
                    # Find person by ID
                    for det in self.vision.last_detections_cache:
                        det_id = det.get('track_id') or det.get('id')
                        if det_id == person_id:
                            bbox = det.get('bbox')
                            if bbox and len(bbox) == 4:
                                break
                    
                    # If not found by ID, try to find largest person detection
                    if bbox is None:
                        largest_area = 0
                        for det in self.vision.last_detections_cache:
                            det_bbox = det.get('bbox', [])
                            if len(det_bbox) == 4:
                                area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                                if area > largest_area:
                                    largest_area = area
                                    bbox = det_bbox
        except Exception as e:
            # If bbox access fails, fall back to coordinate method
            pass
        
        if bbox and len(bbox) == 4:
            # We have bbox: [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            
            image_center_x = self.image_width / 2.0
            image_center_y = self.image_height / 2.0
            
            offset_x = center_x - image_center_x
            offset_y = center_y - image_center_y
            
            return offset_x, offset_y
        
        # Fallback: estimate from x, y, z coordinates
        # This uses the unit circle coordinates from vision system
        x = person.get('x', 0)
        y = person.get('y', 0)
        z = person.get('z', 1.0)
        
        # Normalize z to avoid division issues
        if abs(z) < 0.001:
            z = 1.0
        
        # Convert unit circle coordinates to angles
        theta_rad = np.arctan2(x, z)  # Horizontal angle
        alpha_rad = np.arctan2(y, z)  # Vertical angle
        
        # Convert angles to pixel offsets using FOV
        # Formula: pixel_offset = tan(angle) * (image_size / 2) / tan(FOV / 2)
        fov_h_rad = np.radians(self.fov_horizontal / 2.0)
        fov_v_rad = np.radians(self.fov_vertical / 2.0)
        
        offset_x = np.tan(theta_rad) * (self.image_width / 2.0) / np.tan(fov_h_rad)
        offset_y = np.tan(alpha_rad) * (self.image_height / 2.0) / np.tan(fov_v_rad)
        
        return offset_x, offset_y
    
    def pixels_to_servo_angles(self, offset_x, offset_y):
        """
        Convert pixel offset to servo angle changes with damping to prevent oscillation.
        
        Returns:
            (delta_s1, delta_s2) - servo angle changes in degrees
        """
        if self.image_width is None or self.image_height is None:
            return 0, 0
        
        # Calculate distance from center for damping
        distance_from_center = np.sqrt(offset_x**2 + offset_y**2)
        
        # Calculate angles based on FOV
        # Use atan2 to get angle from center, then scale by FOV
        # Formula: angle = atan2(offset, half_image_size) * (FOV / 180)
        
        # Horizontal angle (affects S2 - horizontal servo)
        # Positive offset_x means person is to the right, need to move turret right
        # Note: Sign is flipped in move_x calculation to match servo direction
        angle_horizontal = np.degrees(np.arctan2(offset_x, self.image_width / 2.0)) * (self.fov_horizontal / 180.0)
        
        # Vertical angle (affects S1 - vertical servo)
        # Positive offset_y means person is below center, need to move turret down (decrease S1)
        angle_vertical = np.degrees(np.arctan2(offset_y, self.image_height / 2.0)) * (self.fov_vertical / 180.0)
        
        # Apply proportional control
        move_x = -angle_horizontal * self.kp  # Negative to flip horizontal direction
        move_y = -angle_vertical * self.kp  # Negative because S1 decreases when moving down
        
        # Apply damping - reduce movement when close to center to prevent oscillation
        if distance_from_center < self.damping_threshold:
            # Linear damping: closer to center = less movement
            # Minimum damping is damping_factor, maximum is 1.0 (no damping)
            damping = max(self.damping_factor, (distance_from_center / self.damping_threshold))
            move_x *= damping
            move_y *= damping
        
        # Limit speed
        move_x = max(-self.max_move_speed, min(self.max_move_speed, move_x))
        move_y = max(-self.max_move_speed, min(self.max_move_speed, move_y))
        
        return move_y, move_x  # Note: S1 is vertical, S2 is horizontal
    
    def update(self):
        """Update turret position based on current detections."""
        # Get latest detections
        vision_data = self.vision.read()
        objects = vision_data.get('objects', [])
        
        # Find person
        person = self.find_person(objects)
        
        if person is None:
            # No person detected - check if we should sweep
            current_time = time.time()
            time_since_person = current_time - self.last_person_time
            
            if not self.sweep_active and time_since_person > self.sweep_interval:
                # Start sweep
                self.sweep_active = True
                self.sweep_start_time = current_time
                return None  # Return None to indicate sweep mode
            
            if self.sweep_active:
                # Continue sweep
                return None
            
            return False
        
        # Person detected - stop sweep if active
        if self.sweep_active:
            self.sweep_active = False
        
        self.last_person_time = time.time()
        
        # Calculate offset
        offset_x, offset_y = self.calculate_offset(person)
        
        if offset_x is None or offset_y is None:
            return False
        
        # Check deadzone
        if abs(offset_x) < self.deadzone_pixels and abs(offset_y) < self.deadzone_pixels:
            return True  # Already centered enough
        
        # Convert to servo angles
        delta_s1, delta_s2 = self.pixels_to_servo_angles(offset_x, offset_y)
        
        # Get current position
        current_s1, current_s2 = self.turret.get_position()
        
        # Calculate new position
        new_s1 = current_s1 + delta_s1
        new_s2 = current_s2 + delta_s2
        
        # Move turret
        self.turret.move_to_absolute(new_s1, new_s2)
        
        return True
    
    def update_sweep(self):
        """Perform sweep pattern when no person detected."""
        if not self.sweep_active:
            return False
        
        current_time = time.time()
        elapsed = current_time - self.sweep_start_time
        
        if elapsed > self.sweep_duration:
            # Sweep complete - return to home
            self.sweep_active = False
            self.turret.go_home()
            return False
        
        # Calculate sweep position (sine wave)
        progress = elapsed / self.sweep_duration  # 0 to 1
        sweep_angle = np.sin(progress * 2 * np.pi) * self.sweep_range_s2
        
        # Get home position
        s1_home = self.turret.servo1_home
        s2_home = self.turret.servo2_home
        
        # Calculate new position
        new_s1 = s1_home
        new_s2 = s2_home + sweep_angle
        
        # Clamp to limits
        new_s2 = max(self.turret.servo2_min, min(self.turret.servo2_max, new_s2))
        
        # Move turret
        self.turret.move_to_absolute(new_s1, new_s2)
        
        return True
    
    def show_preview(self):
        """Show live camera preview with YOLO detections."""
        debug = self.vision.debug()
        
        # Try to get YOLO visualization first, fall back to raw camera frames
        display_frame = debug.get('yolo_visualization')
        
        # If no YOLO visualization, use raw camera frame
        if display_frame is None:
            # Prefer right camera (matching turret tracking), fall back to left
            display_frame = debug.get('last_right_image')
            if display_frame is None:
                display_frame = debug.get('last_left_image')
        
        if display_frame is None:
            return False
        
        # Make a copy for drawing
        display_frame = display_frame.copy()
        
        # Get current detections
        vision_data = self.vision.read()
        objects = vision_data.get('objects', [])
        people = [obj for obj in objects if obj.get('type', '').lower() == 'person']
        
        # Draw detections if we're using raw frame (YOLO viz already has them)
        if debug.get('yolo_visualization') is None:
            # Draw bounding boxes and labels for detected objects
            with self.vision.frame_lock:
                if hasattr(self.vision, 'last_detections_cache') and self.vision.last_detections_cache:
                    for det in self.vision.last_detections_cache:
                        bbox = det.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                        
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        
                        # Determine color based on type
                        obj_type = det.get('class_name', '').lower()
                        if obj_type == 'person':
                            color = (0, 255, 0)  # Green for person
                        else:
                            color = (255, 255, 0)  # Cyan for other objects
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        cv2.circle(display_frame, (center_x, center_y), 5, color, -1)
                        
                        # Draw label
                        class_name = det.get('class_name', 'object')
                        score = det.get('score', 0.0)
                        track_id = det.get('track_id') or det.get('id')
                        
                        if track_id is not None:
                            label = f"ID:{track_id} {class_name} {score:.2f}"
                        else:
                            label = f"{class_name} {score:.2f}"
                        
                        # Draw label background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_frame, (x1, y1 - label_h - 5), 
                                    (x1 + label_w, y1), color, -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw center crosshair
        h, w = display_frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        crosshair_color = (255, 0, 255)  # Magenta
        crosshair_size = 15
        cv2.line(display_frame, (center_x - crosshair_size, center_y), 
                (center_x + crosshair_size, center_y), crosshair_color, 2)
        cv2.line(display_frame, (center_x, center_y - crosshair_size), 
                (center_x, center_y + crosshair_size), crosshair_color, 2)
        cv2.circle(display_frame, (center_x, center_y), 3, crosshair_color, -1)
        
        # Add status text overlay
        status_lines = [
            f"FPS: {debug.get('fps', 0):.1f}",
            f"Objects: {len(objects)}",
            f"People: {len(people)}",
        ]
        
        if self.sweep_active:
            status_lines.append("MODE: SWEEP")
        elif people:
            status_lines.append("MODE: TRACKING")
            # Show offset for tracked person
            person = max(people, key=lambda obj: obj.get('confidence', 0.0))
            offset_x, offset_y = self.calculate_offset(person)
            if offset_x is not None:
                status_lines.append(f"Offset: ({offset_x:.0f}, {offset_y:.0f})px")
        else:
            status_lines.append("MODE: SEARCHING")
        
        # Draw status overlay with background
        y_offset = 30
        for i, line in enumerate(status_lines):
            # Draw background rectangle for better readability
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (8, y_offset + i * 25 - text_h - 5), 
                         (12 + text_w, y_offset + i * 25 + 5), (0, 0, 0), -1)
            cv2.putText(display_frame, line, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show preview
        try:
            cv2.imshow('Turret Tracking - Press Q to quit', display_frame)
            return True
        except Exception as e:
            # Silently handle display errors
            return False


def load_config():
    """Load configuration from config.dill."""
    config_path = "config.dill"
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def main():
    print("=" * 60)
    print("Basic Turret Tracking - Simple Person Tracking")
    print("=" * 60)
    
    # Load config
    print("\n📋 Loading configuration...")
    config = load_config()
    if config is None:
        print("❌ Failed to load config.dill")
        return
    
    # Find vision config
    vision_config = None
    for key, value in config.items():
        if isinstance(value, dict) and 'who_to_run' in value:
            if 'VISION' in str(value.get('who_to_run', '')):
                vision_config = value
                break
    
    if vision_config is None:
        print("❌ VISION config not found in config.dill")
        return
    
    # Initialize vision system
    print("\n📹 Initializing VISION system...")
    # Use absolute import when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from hal.VISION.VISION_UPGRADE import VISION
    vision = VISION(name="TurretVision", **vision_config)
    
    try:
        vision.start()
        print("✅ VISION started")
        print("   Waiting for cameras to initialize...")
        time.sleep(2)  # Wait for cameras to initialize
        
        # Verify camera is working
        debug = vision.debug()
        if debug.get('last_left_image') is None:
            print("⚠️  Warning: Camera not providing frames yet")
        else:
            h, w = debug['last_left_image'].shape[:2]
            print(f"   Camera resolution: {w}x{h}")
        
        # Initialize turret
        print("\n🎯 Initializing TurretControl...")
        turret_port = "COM5"  # Windows default
        if sys.platform.startswith('linux'):
            turret_port = "/dev/ttyUSB0"
        
        turret = TurretControl(
            port=turret_port,
            baudrate=115200,
            servo1_min=15, servo1_max=50, servo1_home=35,
            servo2_min=0, servo2_max=180, servo2_home=90,
            deadzone=1.0,  # Deadzone in degrees
            kp=0.5,  # Proportional gain
            max_speed=5.0  # Max degrees per update
        )
        
        turret.connect()
        if not turret.connected:
            print("❌ Failed to connect to turret")
            print(f"   Check if Arduino is connected to {turret_port}")
            return
        
        print("✅ TurretControl connected")
        turret.go_home()
        time.sleep(1)
        
        # Create tracker
        print("\n🎯 Creating SimpleTurretTracker...")
        tracker = SimpleTurretTracker(vision, turret)
        
        # Get FOV from config if available
        if 'fov_horizontal' in vision_config:
            tracker.fov_horizontal = vision_config['fov_horizontal']
        if 'fov_vertical' in vision_config:
            tracker.fov_vertical = vision_config['fov_vertical']
        
        print(f"   FOV: {tracker.fov_horizontal}° x {tracker.fov_vertical}°")
        
        # Main tracking loop
        print("\n🚀 Starting tracking loop...")
        print("   Camera preview window will open")
        print("   Press 'Q' in preview window or Ctrl+C to stop\n")
        
        frame_count = 0
        last_status_time = time.time()
        no_person_count = 0
        
        try:
            while True:
                # Show live preview
                tracker.show_preview()
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                
                # Update tracker or sweep
                if tracker.sweep_active:
                    tracker.update_sweep()
                else:
                    found = tracker.update()
                    
                    if found is False:
                        # No person detected - return to home after a delay
                        no_person_count += 1
                        if no_person_count > 30:  # ~1 second
                            current_s1, current_s2 = turret.get_position()
                            s1_home, s2_home = turret.servo1_home, turret.servo2_home
                            # Only go home if not already there
                            if abs(current_s1 - s1_home) > 2 or abs(current_s2 - s2_home) > 2:
                                turret.go_home()
                                no_person_count = 0
                    else:
                        no_person_count = 0
                
                frame_count += 1
                
                # Print status every 2 seconds
                current_time = time.time()
                if current_time - last_status_time >= 2.0:
                    vision_data = vision.read()
                    objects = vision_data.get('objects', [])
                    people = [obj for obj in objects if obj.get('type', '').lower() == 'person']
                    debug = vision.debug()
                    fps = debug.get('fps', 0)
                    
                    if tracker.sweep_active:
                        elapsed = current_time - tracker.sweep_start_time
                        print(f"📡 Camera: {fps:.1f} FPS | SWEEPING ({elapsed:.1f}s/{tracker.sweep_duration:.1f}s)")
                    elif people:
                        person = max(people, key=lambda obj: obj.get('confidence', 0.0))
                        offset_x, offset_y = tracker.calculate_offset(person)
                        if offset_x is not None:
                            s1, s2 = turret.get_position()
                            print(f"📡 Camera: {fps:.1f} FPS | TRACKING person (conf:{person.get('confidence', 0):.2f}) | "
                                  f"Offset: ({offset_x:.0f}, {offset_y:.0f})px | Turret: S1={s1:.1f}° S2={s2:.1f}°")
                    else:
                        s1, s2 = turret.get_position()
                        time_until_sweep = tracker.sweep_interval - (current_time - tracker.last_person_time)
                        if time_until_sweep > 0:
                            print(f"📡 Camera: {fps:.1f} FPS | SEARCHING | Turret: S1={s1:.1f}° S2={s2:.1f}° | "
                                  f"Sweep in {time_until_sweep:.0f}s")
                        else:
                            print(f"📡 Camera: {fps:.1f} FPS | SEARCHING | Turret: S1={s1:.1f}° S2={s2:.1f}°")
                    
                    last_status_time = current_time
                
                time.sleep(0.033)  # ~30 Hz update rate
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            if 'turret' in locals():
                turret.go_home()
                time.sleep(0.5)
                turret.disconnect()
        except:
            pass
        
        try:
            if 'vision' in locals():
                vision.stop()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("✅ Done")


if __name__ == "__main__":
    main()
