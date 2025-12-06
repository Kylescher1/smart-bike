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


class Sonar:
    def __init__(self, **kwargs):
        #make obj
        self.scatter = None
        self.anim = None
        self.update_delay = 0.05

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7), facecolor="black")
        self.ax.set_facecolor("black")
        self.ax.set_aspect("equal")
        self.ax.set_title("Live Multi-Sensor Viewer", color="white")

        self.max_r = 2
        self.ax.set_xlim(-self.max_r, self.max_r)
        self.ax.set_ylim(-self.max_r, self.max_r)

        # Polar-style concentric circles
        for r in np.linspace(2, self.max_r, 15):
            self.ax.add_artist(plt.Circle((0, 0), r, color="gray", fill=False, lw=0.6, alpha=0.5))
        self.ax.add_artist(plt.Circle((0, 0), 0.25, color="red", fill=False, lw=0.6, alpha=0.5))
        # Radial lines every 30°
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            self.ax.plot([0, self.max_r * np.cos(rad)], [0,self. max_r * np.sin(rad)], color="gray", lw=0.4, alpha=0.5)

        ground_data = {
            "data_out_label": "ground_edge_detect",
            "color":"grey"
        }
        pointcloud_data = {
            "data_out_label": "point_cloud",
            "color": "cyan"
        }

        self.plot_types = [ground_data,pointcloud_data]
        self.scatters = [self.ax.scatter([], [], s=10) for _ in self.plot_types]

    def update_plot(self,data):#call each loop
        """
            Updates the scatter plots for each sensor.

            frames: list of 4xN arrays [x, y, z, q]
            data:dict with information of frames stacked based on type
            ax: matplotlib Axes
            scatters: list of scatter artists (one per sensor)
            process_funcs: optional list of functions, one per sensor, to transform frames
            """
        #make frames from dict

        for i,plot_ty in enumerate(self.plot_types):
            key = plot_ty["data_out_label"]
            scatter = self.scatters[i]

            try:
                frame = data[key]
            except:
                continue
            x = frame[0]
            y = frame[1]

            scatter.set_offsets(np.column_stack([x, y]))

        plt.pause(0.05)
        plt.pause(self.update_delay)
