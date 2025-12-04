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
import dill
import quaternion
import numpy as np
import time
import serial  # optional, for real lidar connection
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import importlib
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

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
        if 'data_out_label' not in vars(self):
            print(f"data_out_label not setup in config.dill writing as {self.name}")
            self.data_out_label = {self.name}

        # add local properties that cannot be specifed in config file
        self.connected = False
        self.last_scan = None
        self.Lidar = None
        self.scan_buffer = deque(maxlen=self.BUFFER_SIZE)

        self.fig = None
        self.ax = None
        self.scatter = None
        self.anim = None

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
        self.live_plot()
        # """Show or update a static polar plot for this lidar."""
        # if not hasattr(self, "fig") or self.fig is None or not plt.fignum_exists(self.fig.number):
        #     self._init_debug_plot()
        # self._update_debug_plot()

    def start_debug_plot(self):
        """
        Opens a PyQtGraph window and live-plots LIDAR points from self.debug_buffer.
        Assumes each entry: {"a_deg": float, "d_mm": float, "q": int}
        """

        # ---- Qt App ----
        self._app = QtGui.QApplication.instance() or QtGui.QApplication([])

        # ---- Window ----
        self._win = pg.GraphicsLayoutWidget(title=f"LIDAR Debug – {self.port}")
        self._win.resize(700, 700)
        self._win.show()

        # ---- Plot ----
        self._plot = self._win.addPlot()
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "X (mm)")
        self._plot.setLabel("left", "Y (mm)")
        self._plot.setRange(xRange=[-8000, 8000], yRange=[-8000, 8000])
        self._plot.setBackground("black")

        # ---- Scatter ----
        self._scatter = pg.ScatterPlotItem(size=3)
        self._plot.addItem(self._scatter)

        # ---- Update loop ----
        def update():
            buf = list(self.debug_buffer)
            if not buf:
                return

            angles = np.deg2rad([p["a_deg"] for p in buf])
            dists = np.array([p["d_mm"] for p in buf])
            qual = np.array([p["q"] for p in buf], dtype=float)

            # Cartesian
            xs = dists * np.cos(angles)
            ys = dists * np.sin(angles)

            # Normalize 0–63 -> 0–255
            qual_norm = np.clip((qual / 63) * 255, 0, 255).astype(np.uint8)

            # RGB from quality (blue → red)
            colors = np.zeros((len(xs), 4), dtype=np.uint8)
            colors[:, 0] = qual_norm  # R
            colors[:, 2] = 255 - qual_norm  # B
            colors[:, 1] = 40  # G
            colors[:, 3] = 255  # A

            # Update scatter
            spots = [{"pos": (xs[i], ys[i]), "brush": pg.mkBrush(colors[i])}
                     for i in range(len(xs))]
            self._scatter.setData(spots)

        # ---- Timer ----
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(update)
        self._timer.start(30)  # ~33 FPS

        print("PyQtGraph LIDAR debug plot started.")

    def _init_debug_plot(self):
        """Create a polar plot window unique to this lidar."""
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.fig.canvas.manager.set_window_title(self.name)
        self.scatter = self.ax.scatter([], [], s=15, c=[], cmap=cm.viridis, alpha=0.8)
        self.ax.set_title(f"{self.name} Live LIDAR")
        self.ax.set_rmax(4000)
        self.ax.grid(True)

        # non-blocking show so multiple windows can appear
        plt.show(block=False)
        plt.pause(0.001)

    def _update_debug_plot(self):
        """Refresh the existing plot with the newest scan data."""
        if not self.scan_buffer:
            return

        # copy current data safely
        data = list(self.scan_buffer)

        # filter out invalid measurements (None, NaN, or negative)
        valid_data = [d for d in data if d.get('d_mm') not in (None, 0) and np.isfinite(d.get('d_mm', 0))]
        if not valid_data:
            return  # nothing to plot yet

        angles = np.deg2rad([d['a_deg'] for d in valid_data])
        distances = np.array([d['d_mm'] for d in valid_data])
        quality = np.array([d['q'] for d in valid_data])

        # update scatter data (polar expects angle, radius)
        self.scatter.set_offsets(np.column_stack((angles, distances)))
        self.scatter.set_array(quality)

        # update radial limit safely
        rmax = max(1000, np.nanmax(distances) * 1.1)
        self.ax.set_rmax(rmax)

        # redraw
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # keeps GUI alive

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
        np.ndarray 4xn
            Array of [x,y,z,q]x# of samples .
        """
        if not self.scan_buffer:
            return None

        xs = []
        ys = []
        zs = []
        qs = []

        for pkt in self.scan_buffer:
            dist_mm = pkt.get("d_mm", None)
            if dist_mm is None:
                continue

            angle_deg = pkt.get("a_deg", 0.0)
            q_val = pkt.get("q", 0)

            angle = np.deg2rad(angle_deg)
            dist_m = dist_mm / 1000.0

            # Polar → Cartesian
            x = dist_m * np.cos(angle)
            y = dist_m * np.sin(angle)
            z = 0  # definition of 2d lidar

            xs.append(x)
            ys.append(y)
            zs.append(z)
            qs.append(q_val)

        if len(xs) == 0:
            return {self.data_out_label:None} #No new data for this type

        return {self.data_out_label:np.vstack([xs, ys, zs, qs])}

    def get_latest_frame(self):
        """
        Returns a 4×N numpy array:
            frame[0] = x  (meters)
            frame[1] = y  (meters)
            frame[2] = z  (always 0)
            frame[3] = q  (quality)

        Skips samples where distance is None.
        """

        if not self.scan_buffer:
            return np.zeros((4, 0))

        xs = []
        ys = []
        zs = []
        qs = []

        for pkt in self.scan_buffer:
            dist_mm = pkt.get("d_mm", None)
            if dist_mm is None:
                continue

            angle_deg = pkt.get("a_deg", 0.0)
            q_val = pkt.get("q", 0)

            angle = np.deg2rad(angle_deg)
            dist_m = dist_mm / 1000.0

            # Polar → Cartesian
            x = dist_m * np.cos(angle)
            y = dist_m * np.sin(angle)
            z = 0 #definition of 2d lidar

            xs.append(x)
            ys.append(y)
            zs.append(z)
            qs.append(q_val)

        if len(xs) == 0:
            return np.zeros((4, 0))

        return np.vstack([xs, ys, zs, qs])

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

    def live_plot(self, interval_ms=100):
        """
        Launch a live polar plot window for THIS LIDAR instance.
        Matches the behavior of your working standalone script.
        """

        # Create figure + axis
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # ---- Aesthetics (BLACK THEME) ----
        ax.set_facecolor("black")
        fig.set_facecolor("black")
        ax.title.set_color("white")
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['polar'].set_edgecolor('white')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        ax.set_title(f"{self.name} — LIDAR Scan (Quality Map)", pad=20)

        # ---- Set default radius ----
        ax.set_rlim(0, 8000)

        # ---- Colormap ----
        color_map = cm.get_cmap("jet")

        # ---- Initial empty scatter plot ----
        scatter_artist = ax.scatter(
            [], [], c=[], cmap=color_map,
            vmin=0, vmax=63, s=5
        )

        # ---- Colorbar ----
        cbar = fig.colorbar(scatter_artist, ax=ax, orientation="vertical", pad=0.1)
        cbar.set_label("Measurement Quality (0 = low, 63 = high)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # ---- Update function EXACTLY like your working script ----
        def update_plot(frame):
            scan_data = list(self.scan_buffer)
            if not scan_data:
                return scatter_artist,

            angles = [d["a_deg"] for d in scan_data]
            distances = [d["d_mm"] for d in scan_data]
            quality = [d["q"] for d in scan_data]

            # Convert to radians
            angles_rad = np.deg2rad(angles)

            scatter_artist.set_offsets(np.c_[angles_rad, distances])
            scatter_artist.set_array(quality)

            return scatter_artist,

        # ---- Animation ----
        ani = animation.FuncAnimation(
            fig,
            update_plot,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False
        )

        plt.show(block=False)

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


def load_class_from_path(path: str):
    """Load a class given its full import path"""
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def collect_all_sensors(sensors):
    """
    Collects the latest frames from all sensors.
    sensors: list of sensor objects with .get_latest_frame() -> 4xN array
    Returns: list of frames, one per sensor
    """
    frames = [s.get_latest_frame() for s in sensors]
    return frames

def plot_sensor_frames(frames, ax, scatters, process_funcs=None):
    """
    Updates the scatter plots for each sensor.

    frames: list of 4xN arrays [x, y, z, q]
    ax: matplotlib Axes
    scatters: list of scatter artists (one per sensor)
    process_funcs: optional list of functions, one per sensor, to transform frames
    """
    for i, (scatter, frame) in enumerate(zip(scatters, frames)):
        if frame.size == 0:
            continue

        # Apply processing function if provided
        if process_funcs and process_funcs[i] is not None:
            frame = process_funcs[i](frame)

        x = frame[0]
        y = frame[1]

        scatter.set_offsets(np.column_stack([x, y]))

    plt.pause(0.05)  # adjust delay as needed


def multi_sensor_live_plot(sensors, process_funcs=None, update_delay=0.05):
    """
    Main live plot loop for multiple sensors.

    sensors: list of sensor objects with .get_latest_frame() -> 4xN array
    process_funcs: optional list of callables per sensor: func(frame) -> frame
    update_delay: time between updates in seconds
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_title("Live Multi-Sensor Viewer", color="white")

    max_r = 1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)

    # Polar-style concentric circles
    for r in np.linspace(2, max_r, 5):
        ax.add_artist(plt.Circle((0, 0), r, color="gray", fill=False, lw=0.6, alpha=0.5))
    # Radial lines every 30°
    for deg in range(0, 360, 30):
        rad = np.deg2rad(deg)
        ax.plot([0, max_r * np.cos(rad)], [0, max_r * np.sin(rad)], color="gray", lw=0.4, alpha=0.5)

    # scatters = [ax.scatter([], [], s=10, color='cyan') for _ in sensors]
    scatters = [ax.scatter([], [], s=10) for _ in sensors]

    while True:
        frames = collect_all_sensors(sensors)
        plot_sensor_frames(frames, ax, scatters, process_funcs)
        plt.pause(update_delay)



if __name__ == "__main__":
    kwargs = {"port": "COM6",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            # "orientation": np.quaternion(0.7071, 0, 0, -0.7071),#w,x,y,z
            "orientation": np.quaternion(np.cos(-np.pi/2), 0, 0, np.sin(-np.pi/2)),#w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "data_out_label":"point_cloud",
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",}
    Lidar = SpinningLidar(name= "horizontal_lidar",**kwargs)
    kwargs2 = {"port": "COM13",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            "orientation": np.quaternion(np.cos(np.pi/2), 0, 0, np.sin(np.pi/2))*np.quaternion(np.cos(np.pi/2),  np.sin(np.pi/2), 0, 0),#w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "data_out_label":"ground_edge_detect",
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",}
    Lidar2 = SpinningLidar(name="ground_lidar", **kwargs2)
    Lidar.start()
    Lidar2.start()
    try:
        sensors = [Lidar, Lidar2]


        # Optional: define a processing function per sensor
        def filter_quality(frame):
            # remove points with quality < 10
            mask = frame[3] >= 10
            return frame[:, mask]
        def pass_go(frame):
            print(frame)
            return frame

        process_funcs = [filter_quality, pass_go]  # first sensor filtered, second untouched

        # multi_sensor_live_plot(sensors, process_funcs=process_funcs, update_delay=0.05)
        Sonar = load_class_from_path("src.Debug_Tools.PlotTools.Sonar")()

        x = 0
        while True:
            if x == 0:
                print("stat")
            x = 1
            # print("looping")
            for perhp in sensors:
                Sonar.update_plot(perhp.read())
                # print()
        #
        # Lidar.live_plot()
        # Lidar2.live_plot()
        # print("Sample data:", Lidar.read())
        # print("Sample data2:", Lidar2.read())
    finally:
        Lidar.stop()
        Lidar2.stop()