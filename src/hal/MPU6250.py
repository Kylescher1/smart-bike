import serial
import time
import re
import csv

class MPU9250Serial:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1, log_file="sensor_log.csv"):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.log_file = log_file

    def connect(self):
        """Open serial port."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
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
        """Read and parse one line of data."""
        if not self.ser or not self.ser.is_open:
            print("Serial not open. Call connect() first.")
            return None

        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            return None

        if not line:
            return None

        match = re.match(
            r"Accel\s*\(g\):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\|\s*Gyro\s*\(°/s\):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)",
            line
        )

        if match:
            ax, ay, az, gx, gy, gz = map(float, match.groups())
            return {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}
        else:
            print("Unparsed:", line)
            return None

    def log_data(self):
        """Log sensor data to CSV, overwriting each run."""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
            print(f"Logging to {self.log_file}")

            try:
                while True:
                    data = self.read_data()
                    if data:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow([
                            timestamp,
                            data["ax"], data["ay"], data["az"],
                            data["gx"], data["gy"], data["gz"]
                        ])
                        f.flush()
                        print(
                            f"{timestamp} | "
                            f"Accel (g): {data['ax']:.3f}, {data['ay']:.3f}, {data['az']:.3f} | "
                            f"Gyro (°/s): {data['gx']:.3f}, {data['gy']:.3f}, {data['gz']:.3f}"
                        )
                    time.sleep(0.25)
            except KeyboardInterrupt:
                print("\nLogging stopped.")


if __name__ == "__main__":
    sensor = MPU9250Serial()
    sensor.connect()
    try:
        sensor.log_data()
    finally:
        sensor.disconnect()
