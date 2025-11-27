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
                 deadzone: float = 2.0, kp: float = 0.5, max_speed: float = 5.0):
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
        self.min_command_interval = 0.05  # 50ms between commands (20 Hz max)
    
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
            return False
        
        # Clamp angles to limits
        if s1_angle is not None:
            s1_angle = int(max(self.servo1_min, min(self.servo1_max, s1_angle)))
        if s2_angle is not None:
            s2_angle = int(max(self.servo2_min, min(self.servo2_max, s2_angle)))
        
        # Build command string
        parts = []
        if s1_angle is not None:
            parts.append(f"S1:{s1_angle}")
            self.current_s1 = s1_angle
        if s2_angle is not None:
            parts.append(f"S2:{s2_angle}")
            self.current_s2 = s2_angle
        
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
            
            # Try to read response (non-blocking)
            if self.ser.in_waiting > 0:
                try:
                    response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if response:
                        print(f"TurretControl: Arduino response: {response}")
                except:
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
        with self.lock:
            s1_int = int(round(s1_angle)) if s1_angle is not None else None
            s2_int = int(round(s2_angle)) if s2_angle is not None else None
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
    
    def center_object(self, theta_deg: float, alpha_deg: float):
        """
        Center turret on object using angles from vision system.
        
        Args:
            theta_deg: Horizontal angle in degrees (positive = right)
            alpha_deg: Vertical angle in degrees (positive = up)
        
        Note:
            - Servo 1 controls vertical (limited range)
            - Servo 2 controls horizontal (full range)
            - Angles are relative to camera center
        """
        if not self.connected:
            print(f"TurretControl: ⚠️ Not connected, ignoring command theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°")
            return
        
        with self.lock:
            # Calculate desired servo positions
            # Horizontal: theta maps to servo 2 (full range)
            desired_s2 = self.servo2_home + theta_deg
            
            # Vertical: alpha maps to servo 1 (limited range)
            # Clamp to servo 1's limited range
            desired_s1 = self.servo1_home + alpha_deg
            
            # Calculate errors
            error_s1 = desired_s1 - self.current_s1
            error_s2 = desired_s2 - self.current_s2
            
            # Debug output (print every call for now)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            print(f"TurretControl: 📍 center_object called - theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}° -> "
                  f"desired S1={desired_s1:.1f}° S2={desired_s2:.1f}°, "
                  f"current S1={self.current_s1:.1f}° S2={self.current_s2:.1f}°")
            
            # Apply deadzone
            error_s1_before_deadzone = error_s1
            error_s2_before_deadzone = error_s2
            if abs(error_s1) < self.deadzone:
                if self._debug_counter % 10 == 0:
                    print(f"TurretControl: ⏸️ S1 error {abs(error_s1):.2f}° < deadzone {self.deadzone}°, skipping")
                error_s1 = 0.0
            if abs(error_s2) < self.deadzone:
                if self._debug_counter % 10 == 0:
                    print(f"TurretControl: ⏸️ S2 error {abs(error_s2):.2f}° < deadzone {self.deadzone}°, skipping")
                error_s2 = 0.0
            
            # Proportional control with speed limiting
            if error_s1 != 0.0:
                move_s1 = error_s1 * self.kp
                move_s1 = max(-self.max_speed, min(self.max_speed, move_s1))
                new_s1 = self.current_s1 + move_s1
            else:
                new_s1 = self.current_s1
            
            if error_s2 != 0.0:
                move_s2 = error_s2 * self.kp
                move_s2 = max(-self.max_speed, min(self.max_speed, move_s2))
                new_s2 = self.current_s2 + move_s2
            else:
                new_s2 = self.current_s2
            
            # Move servos
            should_move = (new_s1 != self.current_s1 or new_s2 != self.current_s2)
            if should_move:
                print(f"TurretControl: 🎯 Moving to S1={new_s1:.1f}°, S2={new_s2:.1f}° (from S1={self.current_s1:.1f}°, S2={self.current_s2:.1f}°)")
                # Store values we need before releasing lock
                move_s1_val = new_s1
                move_s2_val = new_s2
            else:
                if self._debug_counter % 10 == 0:
                    print(f"TurretControl: ⏸️ No movement needed (S1={self.current_s1:.1f}°, S2={self.current_s2:.1f}°)")
                move_s1_val = None
                move_s2_val = None
        
        # Call move_to_absolute outside the lock to avoid deadlock
        # (move_to_absolute will acquire its own lock)
        if move_s1_val is not None or move_s2_val is not None:
            self.move_to_absolute(move_s1_val, move_s2_val)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current servo positions."""
        with self.lock:
            return (self.current_s1, self.current_s2)
    
    def go_home(self):
        """Move turret to home position."""
        self.move_to_absolute(self.servo1_home, self.servo2_home)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()

