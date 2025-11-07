import time
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rplidarc1 import RPLidar
import asyncio
from collections import deque
from datetime import datetime
import matplotlib.cm as cm  # Import colormap module

import serial  # optional, for real lidar connection

class SpinningLidar:
    def __init__(self,name = "Unidentifed Sensor [bozo messed up config file]", **kwargs):
        """
        Initialize the spinning LIDAR sensor.
        """
        #overwritable properties
        self.debug_mode = True #will open numpy and plot
        self.name = name
        for k,v in kwargs.items():#unpack config into self
            setattr(self, k, v)

        # check for reqired args
        if "port" not in vars(self) :
            raise KeyError(f"Port Not specifed for: {name}")
        if "baudrate" not in vars(self):
            raise KeyError(f"baudrate Not specifed for: {name}")

        # add local properties that cannot be specifed in config file
        self.connected = False
        self.last_scan = None
        self.Lidar = None
        self.scan_buffer = deque(maxlen=self.BUFFER_SIZE)

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------
    def connect(self):
        """Attempt to connect to the LIDAR hardware."""
        print(f"{self.name} Connecting to {self.port} at {self.baudrate}...")
        try:
            try:
                self.Lidar = RPLidar(self.port, self.baudrate, timeout=3)
            except Exception as e:
                raise KeyError(f"{self.name} Failed to create rplidar: {e}")
            self.connected = True
            try:
                self.data_thread = threading.Thread(target=self.lidar_data_collector, daemon=True)
                self.data_thread.start()
                print(self.data_thread)
            except Exception as e:
                raise KeyError(f"{self.name} Failed to create thread: {e}")
            print(f"{self.name} Connection successful.")
        except Exception as e:
            raise KeyError(f"{self.name} Failed to connect: {e}")
            self.connected = False

    def disconnect(self):
        if not self.connected:
            return
        print(f"{self.name} Disconnecting...")
        self.connected = False
        if hasattr(self, 'data_thread') and self.data_thread.is_alive():
            self.data_thread.join()  # wait for the collector to finish
        if self.Lidar is not None:
            try:
                self.Lidar.reset()
            except Exception as e:
                print(f"{self.name} Failed to stop Lidar: {e}")
            finally:
                self.Lidar = None
        print(f"{self.name} Disconnected.")

    def start(self):
        self.connect()
    def stop(self):
        self.disconnect()

    def calibrate(self):
        print(f"Damian needs to make calibration actually do something for {self.name}")
        settings = {"Last_cal":datetime.now()}
        return settings
    def debug(self):
        print(f"Damian needs to make debug actually do something for {self.name}")
    # -------------------------------------------------------------------------
    # Data acquisition
    # -------------------------------------------------------------------------
    def lidar_data_collector(self):
        print("Lidar data collector thread started.")

        async def run_the_scan():
            print("Starting Lidar scan...")
            await self.Lidar.simple_scan(make_return_dict=True)

        async def process_the_queue(queue, stop_event):
            while self.connected:
                try:
                    measurement_dict = await asyncio.wait_for(queue.get(), timeout=1.0)
                    self.scan_buffer.append(measurement_dict)
                except asyncio.TimeoutError:
                    continue
            print("Setting stop event for Lidar...")
            stop_event.set()

        async def main_async_loop():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(run_the_scan())
                tg.create_task(process_the_queue(self.Lidar.output_queue, self.Lidar.stop_event))

        try:
            asyncio.run(main_async_loop())
        except ExceptionGroup as eg:
            print(f"LIDAR ERROR: The asyncio TaskGroup failed. Details:")
            for i, error in enumerate(eg.exceptions):
                print(f"  - Sub-exception {i + 1}: {error}")
                import traceback;
                traceback.print_exception(error)
        except Exception as e:
            print(f"Lidar thread encountered a non-TaskGroup error: {e}")
        finally:
            print("Resetting Lidar...")
            self.Lidar.reset()
            print("Lidar thread finished.")

    def read(self):
        """
        Simulate or fetch a single LIDAR scan.
        Returns
        -------
        np.ndarray
            Array of [angle, distance] pairs.
        """
        scan_data_copy = list(self.scan_buffer)
        if scan_data_copy:#new data
            # Extract all angle, distance, and quality values from the list of dictionaries
            angles_deg = [(90 - d['a_deg']) for d in scan_data_copy]
            distances_mm = [d['d_mm'] for d in scan_data_copy]
            quality_values = [d['q'] for d in scan_data_copy]

            angles_rad = np.deg2rad(angles_deg)
            return scan_data_copy

    # -------------------------------------------------------------------------
    # Helpers and simulations
    # -------------------------------------------------------------------------
    def _simulate_scan(self, num_points=360):
        """Simulate a full 360° LIDAR sweep."""
        angles = np.linspace(0, 360, num_points)
        distances = 2 + np.sin(np.radians(angles))  # fake wavy distance data
        return np.column_stack((angles, distances))

    def _parse_raw_data(self, raw_data):
        """Stub for parsing binary data from the real LIDAR."""
        # Implement when you know your device’s data protocol
        return np.zeros((0, 2))

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def print_status(self):
        """Print current configuration and state."""
        print(f"[SpinningLidar] Port: {self.port}")
        print(f"[SpinningLidar] Baudrate: {self.baudrate}")
        print(f"[SpinningLidar] Position: {self.position}")
        print(f"[SpinningLidar] Z Direction: {self.z_direction}")
        print(f"[SpinningLidar] Connected: {self.connected}")

    def __repr__(self):
        return f"<{self.name} port={self.port}, connected={self.connected}>"
