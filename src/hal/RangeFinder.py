# -*- coding: utf-8 -*-
"""
TF03 Time-of-Flight (TOF) Rangefinder Interface

Provides object-oriented access to TF03 sensor over USB/Serial.
"""

import serial
import time
from typing import Optional


class RangeFinder:
    """
    TF03 Time-of-Flight Rangefinder wrapper class.

    Attributes:
        port (str): Serial port (e.g. "/dev/ttyUSB0", "COM5").
        baudrate (int): Serial baudrate (default: 115200).
    """

    FRAME_HEADER = 0x59
    FRAME_LENGTH = 9

    def __init__(self, port: str = "COM5", baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

    def open(self):
        """Open the serial connection."""
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        print(f"[TF03] Opened connection on {self.port} at {self.baudrate} baud")

    def close(self):
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[TF03] Connection closed")

    def read_frame(self) -> Optional[dict[str, float]]:
        """
        Read and parse a full TF03 data frame.

        Returns:
            dict with distance (cm), signal_strength, and temperature (°C),
            or None if frame not valid.
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port not open. Call open() first.")

        if self.ser.in_waiting >= self.FRAME_LENGTH:#Check we have at least the full frame of bytes
            data = self.ser.read(self.FRAME_LENGTH)
            self.ser.reset_input_buffer()#clear buffer for next frame

            # Validate frame header
            if data[0] == self.FRAME_HEADER and data[1] == self.FRAME_HEADER:
                # Compute checksum
                checksum = sum(data[0:8]) & 0xFF #see data sheet for checksum
                if checksum != data[8]:
                    return None  # bad frame

                distance = data[2] + (data[3] << 8)         # cm
                signal_strength = data[4] + (data[5] << 8)

                return {
                    "distance_cm": distance,
                    "signal_strength": signal_strength,
                }
        else: #No Frame collected/building packet
            return None

    def debug_print_loop(self, delay: float = 0.05):
        """
        Continuously print sensor readings for debugging.

        Args:
            delay (float): Delay between reads in seconds.
        """
        print("[TF03] Starting debug output (Ctrl+C to stop)")
        try:
            while True:
                frame = self.read_frame()
                if frame:
                    print(
                        f"Distance: {frame['distance_cm']} cm"
                        f" @ "
                        f"Signal: {frame['signal_strength']} "
                    )
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\n[TF03] Debug loop stopped by user")
        finally:
            self.close()


if __name__ == "__main__":
    sensor = RangeFinder(port="COM5")
    sensor.open()
    sensor.debug_print_loop()
