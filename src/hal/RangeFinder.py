import serial
import time
import threading
from collections import deque
from typing import Optional
from datetime import datetime

from dill import settings


class RangeFinder:
    """
    TF03 Time-of-Flight Rangefinder interface.
    Designed to be instanced and managed like SpinningLidar.
    """

    FRAME_HEADER = 0x59
    FRAME_LENGTH = 9

    def __init__(self, name="Unnamed RangeFinder", **kwargs):
        """
        Initialize the rangefinder with configurable settings.
        Keyword args can include: port, baudrate, timeout, buffer_size
        """
        self.name = name
        self.debug_mode = True

        # Load configuration from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Required parameters
        if "port" not in vars(self):
            raise KeyError(f"Port not specified for {self.name}")
        if "baudrate" not in vars(self):
            raise KeyError(f"Baudrate not specified for {self.name}")

        # Optional parameters
        self.timeout = getattr(self, "timeout", 0.1)
        self.buffer_size = getattr(self, "buffer_size", 1000)

        # Internal state
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.stop_event = threading.Event()
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.data_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    def connect(self):
        """Open serial connection and start data collection thread."""
        print(f"{self.name}: Connecting to {self.port} at {self.baudrate}...")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1)
            self.connected = True
            self.stop_event.clear()
            self.data_thread = threading.Thread(target=self._data_collector, daemon=True)
            self.data_thread.start()
            print(f"{self.name}: Connection successful.")
        except Exception as e:
            raise ConnectionError(f"{self.name}: Failed to connect ({e})")

    def disconnect(self):
        """Stop data collection and close serial connection."""
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

    # -------------------------------------------------------------------------
    # Data Collection
    # -------------------------------------------------------------------------
    def _data_collector(self):
        """Background thread to continuously read frames from the rangefinder."""
        print(f"{self.name}: Data collector started.")
        while not self.stop_event.is_set():
            frame = self._read_frame()
            if frame:
                frame["timestamp"] = time.time()
                self.data_buffer.append(frame)
            else:
                time.sleep(0.01)  # prevent busy loop
        print(f"{self.name}: Data collector stopped.")

    def _read_frame(self) -> Optional[dict]:
        """Low-level frame read and validation."""
        try:
            if not self.ser or not self.ser.is_open:
                return None

            if self.ser.in_waiting >= self.FRAME_LENGTH:
                data = self.ser.read(self.FRAME_LENGTH)
                self.ser.reset_input_buffer()

                if data[0] == self.FRAME_HEADER and data[1] == self.FRAME_HEADER:
                    checksum = sum(data[0:8]) & 0xFF
                    if checksum != data[8]:
                        return None

                    distance = data[2] + (data[3] << 8)
                    signal_strength = data[4] + (data[5] << 8)

                    return {
                        "distance_cm": distance,
                        "signal_strength": signal_strength,
                    }
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Frame read error - {e}")
        return None

    def read(self):
        """Return a copy of the most recent buffered readings."""
        return list(self.data_buffer)
    def calibrate(self):
        print(f"Damian needs to make calibration actually do something for {self.name}")
        settings = {"Last_cal":datetime.now()}
        return settings
    def debug(self):
        print(f"Damian needs to make debug actually do something for {self.name}")
    # -------------------------------------------------------------------------
    # Debugging & Utilities
    # -------------------------------------------------------------------------
    def print_status(self):
        print(f"[{self.name}] Port: {self.port}")
        print(f"[{self.name}] Baudrate: {self.baudrate}")
        print(f"[{self.name}] Connected: {self.connected}")
        print(f"[{self.name}] Buffered Frames: {len(self.data_buffer)}")

    def debug_print_loop(self, delay=0.1):
        """Print continuous readings for debugging."""
        print(f"{self.name}: Starting debug output (Ctrl+C to stop)")
        try:
            while True:
                if self.data_buffer:
                    frame = self.data_buffer[-1]
                    print(f"Distance: {frame['distance_cm']} cm @ Signal: {frame['signal_strength']}")
                time.sleep(delay)
        except KeyboardInterrupt:
            print(f"\n{self.name}: Debug loop stopped.")
        finally:
            self.disconnect()

    def __repr__(self):
        return f"<RangeFinder name={self.name}, port={self.port}, connected={self.connected}>"

# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    rf = RangeFinder(name="FrontTF03", port="COM5", baudrate=115200)
    rf.start()
    try:
        time.sleep(5)
        rf.print_status()
        print("Sample frames:", rf.read()[-3:])
    finally:
        rf.stop()
