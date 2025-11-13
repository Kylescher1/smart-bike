import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.cm as cm

class SensorDashboard:
    def __init__(self, sensors):
        """
        sensors : list of SpinningLidar (or similar) objects
        Automatically creates a subplot per LIDAR.
        """
        self.sensors = sensors
        self.num_sensors = len(sensors)
        self.fig = plt.figure(figsize=(8, 4 * self.num_sensors))
        self.gs = gridspec.GridSpec(self.num_sensors, 1, figure=self.fig)
        self.axes = []
        self.scatters = []

        for i, sensor in enumerate(sensors):
            ax = self.fig.add_subplot(self.gs[i, 0], projection='polar')
            ax.set_title(sensor.name)
            ax.set_rmax(4000)
            ax.grid(True)
            sc = ax.scatter([], [], s=8, c=[], cmap=cm.viridis, alpha=0.8)
            self.axes.append(ax)
            self.scatters.append(sc)

        self.fig.suptitle("Multi-LIDAR Dashboard", fontsize=16)
        plt.tight_layout()
        plt.show(block=False)

    def update(self):
        """Fetch data from each sensor and update its subplot."""
        for i, sensor in enumerate(self.sensors):
            data = sensor.read()
            if not data:
                continue

            # filter invalid data
            valid = [d for d in data if d.get('d_mm') not in (None, 0) and np.isfinite(d.get('d_mm', 0))]
            if not valid:
                continue

            angles = np.deg2rad([d['a_deg'] for d in valid])
            distances = np.array([d['d_mm'] for d in valid])
            quality = np.array([d['q'] for d in valid])

            sc = self.scatters[i]
            ax = self.axes[i]
            sc.set_offsets(np.column_stack((angles, distances)))
            sc.set_array(quality)

            # auto-scale radius
            if len(distances) > 0:
                rmax = max(1000, np.nanmax(distances) * 1.1)
                ax.set_rmax(rmax)

        # redraw once per frame for all sensors
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
