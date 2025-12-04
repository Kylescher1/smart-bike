from time import sleep

import serial
import time
import threading
import re
import csv
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


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
        if 'data_out_label' not in vars(self):
            print(f"data_out_label not setup in config.dill writing as 'IMU'")
            self.data_out_label = 'IMU'

        # Default settings
        self.timeout = getattr(self, "timeout", 1)
        self.log_file = getattr(self, "log_file", f"{self.name}_log.csv")
        self.buffer_size = getattr(self, "buffer_size", self.BUFFER_SIZE)

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

        # self.live_accel_plot()
    def stop(self):
        """Alias for disconnect()."""
        self.disconnect()

    def calibrate(self):
        print(f"HOLD THE BIKE STILL AND UPRIGHT said {self.name}")
        sleep(2)
        print("starting zero read")
        sleep(3)
        data = imu.read()

        settings = {"Last_cal":datetime.now(),
                    "Bias":np.mean(data,axis=0)}

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
                    # self.data_buffer.append({
                    #     "timestamp": time.time(),
                    #     "ax": ax, "ay": ay, "az": az,
                    #     "gx": gx, "gy": gy, "gz": gz
                    # })
                    state_vect = np.array([time.time(),9.81*ax, 9.81*ay, 9.81*az, gx, gy, gz])
                    self.data_buffer.append(state_vect)
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Read error: {e}")
                continue

        print(f"{self.name}: Data collector stopped.")

    def read(self):
        """Return a copy of the most recent buffered IMU data."""
        return {self.data_out_label:list(self.data_buffer)}

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
            writer.writerow(["Time(s)", "Ax(m/s^2)", "Ay(m/s^2)", "Az(m/s^2)", "Gx(°/s)", "Gy(°/s)", "Gz(°/s)"])
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

    def live_accel_plot(imu, interval=50):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Live Acceleration Vector - {imu.name}")

        sleep(3)
        data = imu.read()

        avg_data = np.mean(data,axis=0)
        print(f" BLow me {avg_data}")
        # ax_val_0, ay_val_0, az_val_0 = latest['ax'], latest['ay'], latest['az']
        # ax_val_0, ay_val_0, az_val_0 = ax_val_0, 9.81 * ay_val_0, 9.81 * az_val_0

        # gx_val_0, gy_val_0, gz_val_0 = latest['gx'], latest['gy'], latest['gz']

        quiver = ax.quiver(0, 0, 0, *avg_data[1:4] , color='blue', linewidth=2, arrow_length_ratio=0.2)

        def update(frame):
            data = imu.read()
            if not data:
                return quiver
            latest = np.mean(data[-32:],axis=0)
            latest[1:7] = latest[1:7] - avg_data[1:7]
            print(f"time:{latest[0]}, a: {latest[1:4]}, g {latest[4:7]}")
            # ax_val, ay_val, az_val = latest['ax']-ax_val_0, latest['ay']-ay_val_0, latest['az']-az_val_0
            #
            # gx_val, gy_val, gz_val = latest['gx']-gx_val_0, latest['gy']-gy_val_0, latest['gz']-gz_val_0


            # print(gx_val, gy_val, gz_val)
            gyro_mag = np.sqrt(np.mean(latest[4:7]**2))
            if np.prod(latest[4:7]):  # threshold in °/s (tune as needed)
                # rx,ry,rz = ax_val/gx_val, ay_val/gy_val, az_val/gz_val
                r_vect = latest[1:4]/latest[4:7]
                ax.quiver(0, 0, 0, *r_vect, color='purple', linewidth=3, arrow_length_ratio=0.2)
            # 🔹 Clear entire 3D plot
            ax.clear()

            # 🔹 Reapply labels, limits, title, etc.
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('Ax (g)')
            ax.set_ylabel('Ay (g)')
            ax.set_zlabel('Az (g)')
            ax.set_title(f"Live Acceleration Vector - {imu.name}")


            # draw new vector
            ax.quiver(0, 0, 0, *avg_data[1:4], color='blue', linewidth=2, arrow_length_ratio=0.2)
            ax.quiver(0, 0, 0, *latest[1:4], color='green', linewidth=3, arrow_length_ratio=0.2)


            mag = np.sqrt(np.mean(latest[1:4]**2))
            fig.suptitle(f"Acceleration = {mag:.2f} g", fontsize=12)
            return quiver

        ani = FuncAnimation(fig, update,frames=None, interval=interval,blit=False, cache_frame_data=False)
        plt.show()

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
