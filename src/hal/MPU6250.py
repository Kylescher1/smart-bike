import serial
import time
import threading
import re
import csv
from collections import deque
from datetime import datetime

class MPU6250:
    def __init__(self, name="Unnamed IMU", **kwargs):
        """
        Initialize MPU6250 IMU instance.
        Supports keyword arguments: port, baudrate, timeout, log_file, buffer_size
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

        # Default settings
        self.timeout = getattr(self, "timeout", 1)
        self.log_file = getattr(self, "log_file", f"{self.name}_log.csv")
        self.buffer_size = getattr(self, "buffer_size", 2000)

        # Runtime state
        self.ser = None
        self.connected = False
        self.data_thread = None
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.stop_event = threading.Event()
        self.start_time = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    def connect(self):
        print(f"{self.name}: Connecting to {self.port} at {self.baudrate}...")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            self.connected = True
            print(f"{self.name}: Connection successful.")
            self.data_thread = threading.Thread(target=self._data_collector, daemon=True)
            self.data_thread.start()
        except Exception as e:
            raise ConnectionError(f"{self.name}: Failed to connect ({e})")

    def disconnect(self):
        if not self.connected:
            return
        print(f"{self.name}: Disconnecting...")
        self.connected = False
        self.stop_event.set()
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=2)
        if self.ser and self.ser.is_open:
            self.ser.close()
        print(f"{self.name}: Disconnected.")

    def start(self):
        """Alias for connect()."""
        self.connect()

    def stop(self):
        """Alias for disconnect()."""
        self.disconnect()

    def calibrate(self):
        print(f"Damian needs to make calibration actually do something for {self.name}")
        settings = {"Last_cal":datetime.now()}
        return settings
    def debug(self):
        print(f"Damian needs to make debug actually do something for {self.name}")

    # -------------------------------------------------------------------------
    # Data Collection
    # -------------------------------------------------------------------------
    def _data_collector(self):
        """Background thread to read and buffer data continuously."""
        print(f"{self.name}: Data collector started.")
        pattern = re.compile(
            r"Accel\s*\(g\):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\|\s*Gyro\s*\(°/s\):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)"
        )

        while not self.stop_event.is_set() and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    ax, ay, az, gx, gy, gz = map(float, match.groups())
                    self.data_buffer.append({
                        "timestamp": time.time(),
                        "ax": ax, "ay": ay, "az": az,
                        "gx": gx, "gy": gy, "gz": gz
                    })
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Read error: {e}")
                continue

        print(f"{self.name}: Data collector stopped.")

    def read(self):
        """Return a copy of the most recent buffered IMU data."""
        return list(self.data_buffer)

    # -------------------------------------------------------------------------
    # Data Logging
    # -------------------------------------------------------------------------
    def log_data(self, duration_s=10):
        """Log buffered data for a duration."""
        print(f"{self.name}: Logging data for {duration_s}s...")
        start_time = time.time()
        end_time = start_time + duration_s

        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Ax(g)", "Ay(g)", "Az(g)", "Gx(°/s)", "Gy(°/s)", "Gz(°/s)"])
            while time.time() < end_time:
                if self.data_buffer:
                    d = self.data_buffer[-1]  # latest
                    writer.writerow([
                        f"{d['timestamp'] - start_time:.6f}",
                        d["ax"], d["ay"], d["az"],
                        d["gx"], d["gy"], d["gz"]
                    ])
                time.sleep(0.001)
        print(f"{self.name}: Log saved to {self.log_file}")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def print_status(self):
        print(f"[{self.name}] Port: {self.port}")
        print(f"[{self.name}] Baudrate: {self.baudrate}")
        print(f"[{self.name}] Connected: {self.connected}")
        print(f"[{self.name}] Buffered Samples: {len(self.data_buffer)}")

    def __repr__(self):
        return f"<MPU6250 name={self.name}, port={self.port}, connected={self.connected}>"

# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    imu = MPU6250(name="TestIMU", port="COM7", baudrate=115200)
    imu.start()
    try:
        time.sleep(5)
        imu.print_status()
        print("Sample data:", imu.read()[-3:])
    finally:
        imu.stop()
