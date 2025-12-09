import time
import numpy as np
import threading
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend for better interactive performance
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rplidarc1 import RPLidar
import asyncio
from collections import deque
from datetime import datetime
import matplotlib.cm as cm  # Import colormap module
import dill
import quaternion
import serial  # optional, for real lidar connection
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtCore
    PYQTPGRAPH_AVAILABLE = True
except ImportError:
    PYQTPGRAPH_AVAILABLE = False
    pg = None
    QtGui = None
    QtCore = None


class Sonar:
    def __init__(self, **kwargs):
        #make obj
        self.scatter = None
        self.anim = None
        self.update_delay = 0.05

        # Enable interactive mode for non-blocking plotting
        print(f"[SONAR INIT] Using matplotlib backend: {matplotlib.get_backend()}")
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 9), facecolor="black")
        self.ax.set_facecolor("black")
        self.ax.set_aspect("equal")
        self.ax.set_title("Live Multi-Sensor Viewer (LIDAR)", color="white", fontsize=14)
        self.ax.set_xlabel("X (meters)", color="white")
        self.ax.set_ylabel("Y (meters)", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, color="gray", linestyle="--", linewidth=0.3, alpha=0.3)
        
        # Show the figure immediately
        self.fig.show()
        # Keep window responsive
        plt.pause(0.001)
        print("[SONAR INIT] Plot window created in interactive mode")

        # Start with a larger view that will auto-adjust
        self.max_r = 10  # Increased from 2 to 10 meters
        self.ax.set_xlim(-self.max_r, self.max_r)
        self.ax.set_ylim(-self.max_r, self.max_r)
        print(f"[SONAR INIT] Initial plot limits: [{-self.max_r}, {self.max_r}]")

        # Polar-style concentric circles - draw more circles at different radii
        for r in [1, 2, 3, 5, 7, 10]:
            self.ax.add_artist(plt.Circle((0, 0), r, color="gray", fill=False, lw=0.6, alpha=0.5))
            # Add radius labels
            self.ax.text(0, r, f'{r}m', color='gray', fontsize=8, ha='center', va='bottom', alpha=0.7)
        self.ax.add_artist(plt.Circle((0, 0), 0.5, color="red", fill=False, lw=1.0, alpha=0.8))
        
        # Radial lines every 30°
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            self.ax.plot([0, self.max_r * np.cos(rad)], [0, self.max_r * np.sin(rad)], color="gray", lw=0.4, alpha=0.5)
        
        # Enable auto-scaling
        self.auto_scale = True
        self.update_count = 0  # Track number of updates

        ground_data = {
            "data_out_label": "ground_edge_detect",
            "color": "grey"
        }
        pointcloud_data = {
            "data_out_label": "point_cloud",
            "color": "cyan"
        }

        self.plot_types = [ground_data, pointcloud_data]
        # Create scatter plots with proper colors and larger size for visibility
        self.scatters = [
            self.ax.scatter([], [], s=30, c=plot_type["color"], alpha=0.8, edgecolors='white', linewidths=0.5) 
            for plot_type in self.plot_types
        ]
        print(f"[SONAR INIT] Created {len(self.scatters)} scatter plots for: {[pt['data_out_label'] for pt in self.plot_types]}")

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
        self.update_count += 1
        print(f"\n[SONAR UPDATE_PLOT #{self.update_count}] Received data keys: {data.keys()}")

        for i,plot_ty in enumerate(self.plot_types):
            key = plot_ty["data_out_label"]
            scatter = self.scatters[i]
            
            print(f"[SONAR UPDATE_PLOT] Processing plot type {i}: '{key}'")

            try:
                frame = data[key]
                print(f"[SONAR UPDATE_PLOT] Got frame for '{key}', id: {id(frame)}, shape: {np.shape(frame)}")
            except Exception as e:
                print(f"[SONAR UPDATE_PLOT] No data for '{key}': {e}")
                continue
            
            if frame is None:
                print(f"[SONAR UPDATE_PLOT] Frame is None for '{key}', skipping")
                continue
                
            print(f"[SONAR UPDATE_PLOT] Frame for '{key}' - shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"[SONAR UPDATE_PLOT] Frame id: {id(frame)}, frame.base: {frame.base}")
            
            x = frame[0]
            y = frame[1]
            
            print(f"[SONAR UPDATE_PLOT] Extracted x (id={id(x)}), y (id={id(y)})")
            print(f"[SONAR UPDATE_PLOT] x shape: {x.shape}, x is view: {x.base is not None}")
            print(f"[SONAR UPDATE_PLOT] y shape: {y.shape}, y is view: {y.base is not None}")
            print(f"[SONAR UPDATE_PLOT] x stats: len={len(x)}, min={np.min(x) if len(x) > 0 else 'N/A'}, max={np.max(x) if len(x) > 0 else 'N/A'}")
            print(f"[SONAR UPDATE_PLOT] y stats: len={len(y)}, min={np.min(y) if len(y) > 0 else 'N/A'}, max={np.max(y) if len(y) > 0 else 'N/A'}")
            
            offsets = np.column_stack([x, y])
            print(f"[SONAR UPDATE_PLOT] Created offsets, shape: {offsets.shape}, id: {id(offsets)}")
            print(f"[SONAR UPDATE_PLOT] Offsets stats: min={np.min(offsets)}, max={np.max(offsets)}")
            
            scatter.set_offsets(offsets)
            # Force scatter to be visible
            scatter.set_visible(True)
            print(f"[SONAR UPDATE_PLOT] Updated scatter plot for '{key}' with {len(offsets)} points")
            
            # Update title with frame count to verify updates are happening
            self.ax.set_title(f"Live Multi-Sensor Viewer (LIDAR) - Frame {self.update_count} - {len(offsets)} points", 
                            color="white", fontsize=14)
            
            # Auto-adjust plot limits to fit all data with some margin
            if self.auto_scale and len(offsets) > 0:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                
                # Add 20% margin
                margin = 0.2
                x_range = max(abs(x_max - x_min), 1.0)  # at least 1m
                y_range = max(abs(y_max - y_min), 1.0)
                
                x_center = (x_max + x_min) / 2
                y_center = (y_max + y_min) / 2
                
                half_x = x_range * (1 + margin) / 2
                half_y = y_range * (1 + margin) / 2
                
                # Make it symmetric around origin for radar-like view
                max_half = max(half_x, half_y, 2.0)  # at least 2m radius
                
                new_xlim = (-max_half, max_half)
                new_ylim = (-max_half, max_half)
                
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
                
                # Only update if significantly different to avoid jitter
                if abs(current_xlim[0] - new_xlim[0]) > 0.5 or abs(current_xlim[1] - new_xlim[1]) > 0.5:
                    self.ax.set_xlim(new_xlim)
                    self.ax.set_ylim(new_ylim)
                    print(f"[SONAR UPDATE_PLOT] Adjusted limits to: x={new_xlim}, y={new_ylim}")

        print(f"[SONAR UPDATE_PLOT] Forcing display redraw...")
        # Force immediate redraw - draw_idle() wasn't working
        try:
            self.fig.canvas.draw()  # Force immediate draw (not idle)
            self.fig.canvas.flush_events()  # Process any pending GUI events
            plt.pause(0.001)  # Small pause to let GUI process the update
            print(f"[SONAR UPDATE_PLOT] Redraw successful")
        except Exception as e:
            print(f"[SONAR UPDATE_PLOT] WARNING: Draw failed: {e}")
        print(f"[SONAR UPDATE_PLOT] Update complete")
