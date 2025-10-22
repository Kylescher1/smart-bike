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
        self.start_time = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            print(f"Connected to {self.port}")
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            self.ser = None

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected.")

    def read_data(self):
        if not self.ser or not self.ser.is_open:
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
        return None

    def log_data(self, duration_s=10):
        """Logs data at ~1 kHz for the specified duration in seconds."""
        if not self.ser or not self.ser.is_open:
            print("Serial not connected.")
            return

        self.start_time = time.time()
        end_time = self.start_time + duration_s
        sample_interval = 1.0 / 1000.0  # 1 kHz

        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Ax(g)", "Ay(g)", "Az(g)", "Gx(°/s)", "Gy(°/s)", "Gz(°/s)"])

            next_sample_time = self.start_time
            while time.time() < end_time:
                data = self.read_data()
                if data:
                    timestamp = time.time() - self.start_time
                    writer.writerow([
                        f"{timestamp:.6f}",
                        data["ax"], data["ay"], data["az"],
                        data["gx"], data["gy"], data["gz"]
                    ])
                next_sample_time += sample_interval
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(f"Logging complete. Saved to {self.log_file}")


if __name__ == "__main__":
    sensor = MPU9250Serial()
    sensor.connect()
    try:
        sensor.log_data(duration_s=60*10)  # change duration as needed
    finally:
        sensor.disconnect()
