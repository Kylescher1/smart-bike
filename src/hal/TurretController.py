#!/usr/bin/env python3
"""
Turret Controller - Servo and Arduino Communication

Extracted and refactored from yolo_gimbal.py for reusability.
Handles serial communication with Arduino for servo control and sensor reading.
"""

import serial
import time
from typing import Optional


class TurretController:
    """
    Controller for turret servo system via Arduino.
    
    Manages:
    - Serial communication with Arduino
    - Servo position control (pan/tilt)
    - Status reading
    - ToF sensor reading
    """
    
    def __init__(self, port: str, baudrate: int = 115200):
        """
        Initialize turret controller.
        
        Args:
            port: Serial port (e.g., 'COM3', '/dev/ttyUSB0')
            baudrate: Serial baud rate
        """
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        
        # Servo positions (float for precision)
        self.top_pos = 90.0
        self.bottom_pos = 90.0
        
        # Servo limits (will be updated from Arduino)
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        
        # Rate limiting
        self.last_command_time = 0.0
        self.command_interval = 0.05  # 20 Hz max
        self.min_angle_change = 0.5  # Don't send if change < 0.5°
        
        # Last known ToF reading
        self.last_tof_range = None
        self.last_tof_time = 0.0
        
    def connect(self) -> bool:
        """
        Connect to Arduino.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.5,
                write_timeout=0.5
            )
            time.sleep(2)  # Wait for Arduino reset
            self.update_status()
            return True
        except Exception as e:
            print(f"Error connecting to turret: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def send_command(self, command: str, read_response: bool = False) -> Optional[str]:
        """
        Send a command to Arduino.
        
        Args:
            command: Command string
            read_response: If True, wait for and return response
            
        Returns:
            Response string if read_response=True, None otherwise
        """
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
        except Exception as e:
            print(f"Command error: {e}")
            return None
    
    def update_status(self):
        """Update internal state from Arduino STATUS command"""
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
        """
        Move servos to absolute positions.
        
        Args:
            target_bottom: Target angle for bottom servo (pan)
            target_top: Target angle for top servo (tilt)
            force: If True, bypass rate limiting
        """
        # Rate limiting
        current_time = time.time()
        if not force and (current_time - self.last_command_time) < self.command_interval:
            return
        
        # Clamp to limits
        target_bottom = max(self.bottom_min, min(self.bottom_max, target_bottom))
        target_top = max(self.top_min, min(self.top_max, target_top))
        
        # Check if change is significant
        bottom_change = abs(target_bottom - self.bottom_pos)
        top_change = abs(target_top - self.top_pos)
        
        commands_sent = False
        
        if bottom_change >= self.min_angle_change or force:
            bottom_int = round(target_bottom)
            self.send_command(f"BOTTOM:{bottom_int}", read_response=False)
            self.bottom_pos = target_bottom
            commands_sent = True
        
        if top_change >= self.min_angle_change or force:
            top_int = round(target_top)
            self.send_command(f"TOP:{top_int}", read_response=False)
            self.top_pos = target_top
            commands_sent = True
        
        if commands_sent:
            self.last_command_time = current_time
    
    def home(self):
        """Move servos to home position (90°, 90°)"""
        self.send_command("HOME", read_response=False)
        time.sleep(0.5)
        self.update_status()
    
    def get_tof_range(self) -> Optional[float]:
        """
        Get ToF sensor range reading.
        
        Returns:
            Distance in inches, or None if not available
            
        Note: Requires Arduino firmware modification to support GET_RANGE command
        """
        resp = self.send_command("GET_RANGE", read_response=True)
        if resp:
            try:
                # Expected format: "OK: Range: <value> in"
                for line in resp.split('\n'):
                    if 'Range:' in line:
                        value_str = line.split('Range:')[1].strip().split()[0]
                        range_inches = float(value_str)
                        self.last_tof_range = range_inches
                        self.last_tof_time = time.time()
                        return range_inches
            except:
                pass
        
        return None
    
    def get_position(self) -> tuple:
        """
        Get current servo positions.
        
        Returns:
            (pan_angle, tilt_angle): Bottom and top servo angles
        """
        return self.bottom_pos, self.top_pos

