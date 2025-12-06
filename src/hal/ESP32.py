import serial
import time
import threading
from collections import deque
from typing import Optional
import numpy as np


class ESP32:
    """
    ESP32 sensor interface with MPU6050, servos, and vibration motors.
    
    Serial Protocol:
    - READ: Returns accelerometer and gyro data as "ax,ay,az,gx,gy,gz"
    - MOVE,B,ang,T,angle: Moves bottom servo to ang degrees, top servo to angle degrees
    - VIBRATE,L,R: Sets vibration intensity (0-255) for left and right motors
    """
    
    def __init__(self, name="Unnamed ESP32 Sensor", **kwargs):
        """
        Initialize ESP32 sensor instance.
        
        Required kwargs:
        - port: Serial port (e.g., "COM13" or "/dev/ttyUSB0")
        - baudrate: Serial baudrate (default: 115200)
        - BUFFER_SIZE: Buffer size for data storage
        
        Optional kwargs:
        - timeout: Serial timeout in seconds (default: 1.0)
        - position: Position quaternion (required by config system)
        - z_direction: Z-direction quaternion (required by config system)
        """
        self.name = name
        self.debug_mode = True
        
        # Load user-provided configuration
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Validate required fields
        if "port" not in vars(self):
            raise KeyError(f"Port not specified for {self.name}")
        if "baudrate" not in vars(self):
            raise KeyError(f"Baudrate not specified for {self.name}")
        if "BUFFER_SIZE" not in vars(self):
            raise KeyError(f"BUFFER_SIZE not specified for {self.name}")
        
        # Default settings
        self.timeout = getattr(self, "timeout", 1.0)
        self.buffer_size = getattr(self, "buffer_size", self.BUFFER_SIZE)
        
        # Runtime state
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()  # For thread-safe serial access
    
    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    def connect(self):
        """Open serial connection to ESP32."""
        print(f"{self.name}: Connecting to {self.port} at {self.baudrate}...")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Allow ESP32 to initialize
            self.connected = True
            print(f"{self.name}: Connection successful.")
        except Exception as e:
            raise ConnectionError(f"{self.name}: Failed to connect ({e})")
    
    def disconnect(self):
        """Close serial connection."""
        if not self.connected:
            return
        print(f"{self.name}: Disconnecting...")
        self.connected = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        print(f"{self.name}: Disconnected.")
    
    def start(self):
        """Alias for connect()."""
        self.connect()
    
    def stop(self):
        """Alias for disconnect()."""
        self.disconnect()
    
    # -------------------------------------------------------------------------
    # Serial Communication
    # -------------------------------------------------------------------------
    def _send_command(self, command: str, wait_for_response: bool = True) -> Optional[str]:
        """
        Send a command to ESP32 and optionally wait for response.
        
        Args:
            command: Command string to send (without newline)
            wait_for_response: Whether to wait for and return response
            
        Returns:
            Response string if wait_for_response is True, None otherwise
        """
        if not self.connected or not self.ser or not self.ser.is_open:
            raise ConnectionError(f"{self.name}: Not connected")
        
        with self.lock:
            try:
                # Clear input buffer
                self.ser.reset_input_buffer()
                
                # Send command with newline
                self.ser.write((command + "\n").encode('utf-8'))
                self.ser.flush()
                
                if wait_for_response:
                    # Wait for response
                    response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    return response
                return None
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Serial communication error: {e}")
                raise
    
    # -------------------------------------------------------------------------
    # Public Interface Methods
    # -------------------------------------------------------------------------
    def read(self):
        """
        Read accelerometer and gyro data from MPU6050.
        
        Sends READ command, waits for response, parses comma-separated data.
        
        Returns:
            numpy array with [ax, ay, az, gx, gy, gz] or None on error
        """
        try:
            response = self._send_command("READ", wait_for_response=True)
            
            if not response or response == "ERROR":
                if self.debug_mode:
                    print(f"{self.name}: READ command failed or returned ERROR")
                return None
            
            # Parse comma-separated values: ax,ay,az,gx,gy,gz
            try:
                values = [float(x.strip()) for x in response.split(',')]
                if len(values) != 6:
                    if self.debug_mode:
                        print(f"{self.name}: Invalid data format, expected 6 values, got {len(values)}")
                    return None
                
                # Convert to numpy array: [ax, ay, az, gx, gy, gz]
                data = np.array(values)
                
                # Store in buffer
                self.data_buffer.append(data)
                
                return {}
                return data
            except ValueError as e:
                if self.debug_mode:
                    print(f"{self.name}: Failed to parse response '{response}': {e}")
                return None
                
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error in read(): {e}")
            return None
    
    def move(self, B: int, ang: int, T: int, angle: int):
        """
        Move servos to specified angles.
        
        Args:
            B: Bottom servo identifier (unused, kept for API compatibility)
            ang: Bottom servo angle in degrees (15-75)
            T: Top servo identifier (unused, kept for API compatibility)
            angle: Top servo angle in degrees (0-180)
        
        Returns:
            True if command succeeded, False otherwise
        """
        try:
            # Format: MOVE,B,ang,T,angle (B and T are literal strings in command)
            command = f"MOVE,B,{ang},T,{angle}"
            response = self._send_command(command, wait_for_response=True)
            
            if response == "OK":
                return True
            else:
                if self.debug_mode:
                    print(f"{self.name}: MOVE command failed, response: {response}")
                return False
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error in move(): {e}")
            return False
    
    def vibrate(self, L: int, R: int):
        """
        Control vibration motors.
        
        Args:
            L: Left motor intensity (0-255)
            R: Right motor intensity (0-255)
        
        Returns:
            True if command succeeded, False otherwise
        """
        try:
            # Clamp values to valid range
            L = max(0, min(255, int(L)))
            R = max(0, min(255, int(R)))
            
            # Format: VIBRATE,L,R
            command = f"VIBRATE,{L},{R}"
            response = self._send_command(command, wait_for_response=True)
            
            if response == "OK":
                return True
            else:
                if self.debug_mode:
                    print(f"{self.name}: VIBRATE command failed, response: {response}")
                return False
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error in vibrate(): {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def print_status(self):
        """Print current status information."""
        print(f"[{self.name}] Port: {self.port}")
        print(f"[{self.name}] Baudrate: {self.baudrate}")
        print(f"[{self.name}] Connected: {self.connected}")
        print(f"[{self.name}] Buffered Samples: {len(self.data_buffer)}")
    
    def get_buffered_data(self):
        """Return all buffered data."""
        return list(self.data_buffer)
    
    def __repr__(self):
        return f"<ESP32 name={self.name}, port={self.port}, connected={self.connected}>"


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    esp32 = ESP32(name="TestESP32", port="COM7", baudrate=115200, BUFFER_SIZE=200)
    esp32.start()
    try:
        time.sleep(2)
        esp32.print_status()
        
        # Test reading accelerometer data
        data = esp32.read()
        if data is not None:
            print(f"Accel/Gyro data: {data}")
        
        # Test moving servos
        esp32.move(0, 45, 0, 90)  # Bottom to 45°, Top to 90°
        time.sleep(1)
        
        # Test vibration motors
        esp32.vibrate(128, 128)  # 50% intensity on both
        time.sleep(1)
        esp32.vibrate(0, 0)  # Stop
        
    finally:
        esp32.stop()

