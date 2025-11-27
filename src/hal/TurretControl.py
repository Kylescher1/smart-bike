"""
Turret Control Module

Controls a two-servo turret system via Arduino/ESP32 serial communication.
Servo 1: Vertical axis (limited range, typically 15-50 degrees)
Servo 2: Horizontal axis (full range, typically 0-180 degrees)
"""

import serial
import time
import threading
from typing import Optional, Tuple
import numpy as np


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
                    if self.debug_mode:
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

