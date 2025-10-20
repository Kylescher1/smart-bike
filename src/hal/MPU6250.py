import serial
import time
import re

class MPU9250Serial:
    def __init__(self, port="COM7", baudrate=115200, timeout=1):
        """
        Initialize serial connection to Arduino.
        Example port: 'COM3' on Windows or '/dev/ttyUSB0' on Linux
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        """Open the serial port."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Allow Arduino reset
            print(f"Connected to {self.port}")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            self.ser = None

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected.")

    def read_data(self):
        """
        Reads one line of sensor data from Arduino and parses it.
        Returns a dict with accel and gyro values.
        """
        if not self.ser or not self.ser.is_open:
            print("Serial not open. Call connect() first.")
            return None

        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            return None

        # Expected format:
        # Accel (g): ax, ay, az | Gyro (°/s): gx, gy, gz
        match = re.match(
            r"Accel \(g\): ([\-\d\.]+), ([\-\d\.]+), ([\-\d\.]+) \| Gyro \(°/s\): ([\-\d\.]+), ([\-\d\.]+), ([\-\d\.]+)",
            line
        )

        if match:
            ax, ay, az, gx, gy, gz = map(float, match.groups())
            return {
                "ax": ax, "ay": ay, "az": az,
                "gx": gx, "gy": gy, "gz": gz
            }
        else:
            # Print raw line for debugging
            print("Unparsed:", line)
            return None


if __name__ == "__main__":
    # Example usage
    sensor = MPU9250Serial()
    sensor.connect()

    try:
        while True:
            data = sensor.read_data()
            if data:
                print(
                    f"Accel (g): {data['ax']:.3f}, {data['ay']:.3f}, {data['az']:.3f} | "
                    f"Gyro (°/s): {data['gx']:.3f}, {data['gy']:.3f}, {data['gz']:.3f}"
                )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sensor.disconnect()
