#!/usr/bin/env python3
"""
ESP32 IMU Live Plotter

Real-time plotting of accelerometer and gyroscope data from ESP32.
Shows 6 subplots: 3 for accelerometer (ax, ay, az) and 3 for gyroscope (gx, gy, gz).

Usage:
    python plot_esp32_imu.py [port]
    
    If port is not specified, will prompt for it or use default "COM5"
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Try to import quaternion library for np.quaternion support
try:
    import quaternion
    # After importing quaternion, np.quaternion becomes available
    has_quaternion = True
except ImportError:
    has_quaternion = False
    print("Warning: quaternion module not found. Using None for quaternion parameters.")
    print("  Install with: pip install numpy-quaternion")

from hal.ESP32 import ESP32


class IMUPlotter:
    """Real-time IMU data plotter."""
    
    def __init__(self, esp32, max_points=500):
        """
        Initialize plotter.
        
        Args:
            esp32: ESP32 instance
            max_points: Maximum number of data points to display
        """
        self.esp32 = esp32
        self.max_points = max_points
        
        # Data storage
        self.times = []
        self.accel_data = {'x': [], 'y': [], 'z': []}
        self.gyro_data = {'x': [], 'y': [], 'z': []}
        
        # Setup figure and subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('ESP32 IMU Data - Live Plot', fontsize=16)
        
        # Accelerometer plots
        self.accel_lines = {}
        self.accel_lines['x'] = self.axes[0, 0].plot([], [], 'r-', label='ax', linewidth=1.5)[0]
        self.accel_lines['y'] = self.axes[0, 1].plot([], [], 'g-', label='ay', linewidth=1.5)[0]
        self.accel_lines['z'] = self.axes[0, 2].plot([], [], 'b-', label='az', linewidth=1.5)[0]
        
        # Gyroscope plots
        self.gyro_lines = {}
        self.gyro_lines['x'] = self.axes[1, 0].plot([], [], 'r-', label='gx', linewidth=1.5)[0]
        self.gyro_lines['y'] = self.axes[1, 1].plot([], [], 'g-', label='gy', linewidth=1.5)[0]
        self.gyro_lines['z'] = self.axes[1, 2].plot([], [], 'b-', label='gz', linewidth=1.5)[0]
        
        # Configure axes
        self.setup_axes()
        
        # Start time
        self.start_time = time.time()
        
    def setup_axes(self):
        """Configure plot axes labels and limits."""
        # Accelerometer axes
        for i, axis in enumerate(['x', 'y', 'z']):
            self.axes[0, i].set_title(f'Accelerometer {axis.upper()} (g)')
            self.axes[0, i].set_xlabel('Time (s)')
            self.axes[0, i].set_ylabel('Acceleration (g)')
            self.axes[0, i].grid(True, alpha=0.3)
            self.axes[0, i].legend()
            self.axes[0, i].set_ylim(-2, 2)  # Typical range for ±2g
        
        # Gyroscope axes
        for i, axis in enumerate(['x', 'y', 'z']):
            self.axes[1, i].set_title(f'Gyroscope {axis.upper()} (deg/s)')
            self.axes[1, i].set_xlabel('Time (s)')
            self.axes[1, i].set_ylabel('Angular Velocity (deg/s)')
            self.axes[1, i].grid(True, alpha=0.3)
            self.axes[1, i].legend()
            self.axes[1, i].set_ylim(-250, 250)  # Typical range for ±250 deg/s
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Update plot with new data."""
        try:
            # Read data from ESP32
            data = self.esp32.read()
            
            if data is not None and len(data) == 6:
                # Extract values
                ax, ay, az, gx, gy, gz = data
                
                # Calculate time since start
                current_time = time.time() - self.start_time
                
                # Store data
                self.times.append(current_time)
                self.accel_data['x'].append(ax)
                self.accel_data['y'].append(ay)
                self.accel_data['z'].append(az)
                self.gyro_data['x'].append(gx)
                self.gyro_data['y'].append(gy)
                self.gyro_data['z'].append(gz)
                
                # Limit data points
                if len(self.times) > self.max_points:
                    self.times.pop(0)
                    for key in ['x', 'y', 'z']:
                        self.accel_data[key].pop(0)
                        self.gyro_data[key].pop(0)
                
                # Update plots
                if len(self.times) > 0:
                    # Accelerometer
                    self.accel_lines['x'].set_data(self.times, self.accel_data['x'])
                    self.accel_lines['y'].set_data(self.times, self.accel_data['y'])
                    self.accel_lines['z'].set_data(self.times, self.accel_data['z'])
                    
                    # Gyroscope
                    self.gyro_lines['x'].set_data(self.times, self.gyro_data['x'])
                    self.gyro_lines['y'].set_data(self.times, self.gyro_data['y'])
                    self.gyro_lines['z'].set_data(self.times, self.gyro_data['z'])
                    
                    # Update axis limits
                    if len(self.times) > 1:
                        time_range = [max(0, self.times[-1] - 10), self.times[-1] + 1]
                        
                        for i, axis in enumerate(['x', 'y', 'z']):
                            # Accelerometer
                            accel_range = [
                                min(self.accel_data[axis]) - 0.5,
                                max(self.accel_data[axis]) + 0.5
                            ]
                            self.axes[0, i].set_xlim(time_range)
                            self.axes[0, i].set_ylim(accel_range)
                            
                            # Gyroscope
                            gyro_range = [
                                min(self.gyro_data[axis]) - 10,
                                max(self.gyro_data[axis]) + 10
                            ]
                            self.axes[1, i].set_xlim(time_range)
                            self.axes[1, i].set_ylim(gyro_range)
                
                # Print current values
                print(f"\rTime: {current_time:.2f}s | "
                      f"Accel: ax={ax:7.3f}, ay={ay:7.3f}, az={az:7.3f} | "
                      f"Gyro: gx={gx:7.2f}, gy={gy:7.2f}, gz={gz:7.2f}", end='', flush=True)
            else:
                print("\rWaiting for data...", end='', flush=True)
                
        except Exception as e:
            print(f"\nError reading data: {e}")
        
        return list(self.accel_lines.values()) + list(self.gyro_lines.values())


def main():
    """Main function."""
    print("ESP32 IMU Live Plotter")
    print("=" * 60)
    
    # Get port from command line or prompt
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = input("Enter ESP32 serial port (e.g., COM7 or /dev/ttyUSB0) [default: COM5]: ").strip()
        if not port:
            port = "COM5"
    
    print(f"\nUsing port: {port}")
    print("Make sure the ESP32 is connected and the firmware is uploaded!")
    input("Press Enter to continue...")
    
    # Create ESP32 instance
    try:
        # Create quaternion objects if available, otherwise use None
        # Note: ESP32 class may accept None for these parameters
        if has_quaternion:
            try:
                position = np.quaternion(1, 0, 0, 0)
                z_direction = np.quaternion(0, 0, 0, 1)
            except AttributeError:
                # Fallback if np.quaternion not available even after import
                position = None
                z_direction = None
        else:
            position = None
            z_direction = None
        
        esp32 = ESP32(
            name="IMUPlotter",
            port=port,
            baudrate=115200,
            BUFFER_SIZE=200,
            position=position,
            z_direction=z_direction
        )
    except Exception as e:
        print(f"✗ Failed to create ESP32 instance: {e}")
        return
    
    # Connect
    try:
        print("\nConnecting to ESP32...")
        esp32.start()
        time.sleep(2)  # Give ESP32 time to initialize
        
        # Test connection
        print("Testing connection...")
        test_data = esp32.read()
        if test_data is None:
            print("⚠ Warning: Could not read initial data. Plotting will continue anyway.")
        else:
            print("✓ Connection successful!")
        
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that the ESP32 is connected")
        print("  2. Verify the port name is correct")
        print("  3. Make sure no other program is using the serial port")
        print("  4. Check that the firmware is uploaded to the ESP32")
        return
    
    # Create plotter
    plotter = IMUPlotter(esp32, max_points=500)
    
    # Start animation
    print("\n" + "=" * 60)
    print("Starting live plot...")
    print("Close the plot window to stop.")
    print("=" * 60 + "\n")
    
    try:
        # Animate at ~20 Hz (50ms interval)
        ani = animation.FuncAnimation(
            plotter.fig,
            plotter.update_plot,
            interval=50,  # 50ms = 20 Hz update rate
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n\nStopping plotter...")
    except Exception as e:
        print(f"\n✗ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nDisconnecting...")
        try:
            esp32.vibrate(0, 0)  # Stop vibration motors
            time.sleep(0.5)
            esp32.stop()
            print("✓ Disconnected successfully")
        except:
            pass


if __name__ == "__main__":
    main()

