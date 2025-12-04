import dill
import importlib
import sys
from pathlib import Path
import dashboard
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import quaternion
# from termcolor import colored, cprint #earn this buddy

# make sure src folder is on sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def load_class_from_path(path: str):
    """Load a class given its full import path"""
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def instantiate_sensors(config):
    sensors = {}
    for name, params in config.items():
        print(f"Loading: {name}")
        cls = load_class_from_path(params['who_to_run'])

        try:#load class
            sensor = cls(name=name,**params)
        except Exception as e:
            raise KeyError(f"{name} threw an error: {e}")
        print(f"Loaded: {name}")

        try:#startup
            sensor.start()
            print(f"Started: {name}")
        except Exception as e:
            raise KeyError(f"{name}.start() threw an error: {e}")

        sensors[name] = sensor
    return sensors

def main():
    #Load condfig
    print("Loading Config...")
    try:
        with open("config.dill", "rb") as f:
            config = dill.load(f)
        print("Loaded Config")
        for k, v in config.items():
            print(f"Device: {k} | Properties: {v}")
    except Exception as e:
        raise KeyError(f"An unexpected error occurred Loading config.dill: {e}")

    print("==="*20)
    print("Enabling Sensors...")
    try:
        sensors = instantiate_sensors(config)
        print("Sensors Enabled")
    except Exception as e:
        raise KeyError(f"An unexpected error occurred with instantiate_sensors(), blame Damian: {e}")

    print("===" * 20)
    print("Sensor Check Would go here")
    print("===" * 20)
    # dash = dashboard.SensorDashboard(list(sensors.values()))

    try:
        # Optional: define a processing function per sensor
        def filter_quality(frame):
            # remove points with quality < 10
            mask = frame[3] >= 10
            return frame[:, mask]

        def shift_x_by_sensor_position(frame, sensor):
            """
            Shift X by sensor's x-position.
            """
            if frame.size == 0:
                return frame
            new_frame = frame.copy()
            # new_frame[0] += sensor.position[0]  # assuming sensor.position = np.array([x,y,z])
            new_frame[0] += 3  # assuming sensor.position = np.array([x,y,z])
            return new_frame

        def shift_sensor_data(frame, sensor):
            """
            Shift by sensor's x,y,z-position.
            """
            if frame.size == 0:
                return frame
            new_frame = frame.copy()

            rot = sensor.orientation  # quaternion

            # Rotate each column vector (x,y,z) by the quaternion
            points = new_frame[:3, :].T  # shape (N,3)
            rotated_points = quaternion.as_rotation_matrix(rot) @ points.T  # shape (3, N)
            new_frame[:3, :] = rotated_points

            #translate
            new_frame[:3, :] -= sensor.sensor_location[:, np.newaxis]# sensor.position = np.array([x,y,z])
            return new_frame

        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_title("Live Multi-Sensor Viewer", color="white")

        max_r = 7
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_aspect('equal')
        ax.set_xlabel("X (m)", color='white')
        ax.set_ylabel("Y (m)", color='white')

        # Polar-style concentric circles
        for r in np.linspace(2, max_r, 5):
            ax.add_artist(plt.Circle((0, 0), r, color="gray", fill=False, lw=0.6, alpha=0.5))
        # Radial lines every 30°
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            ax.plot([0, max_r * np.cos(rad)], [0, max_r * np.sin(rad)], color="gray", lw=0.4, alpha=0.5)
        ax.plot([0, 0], [0, max_r], color="white", lw=0.4, alpha=0.9)

        # scatters = [ax.scatter([], [], s=10, color='cyan') for _ in sensors]
        # Assign a unique color to each sensor
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
        print(colors)
        scatters = []

        for color, (name, sensor) in zip(colors, sensors.items()):
            print(color)
            scat = ax.scatter([], [], c=[], s=10, cmap='turbo', label=name)
            scatters.append(scat)

        # Add legend to show which color corresponds to which sensor
        ax.legend(loc='upper right', facecolor='black', labelcolor='white')


        def plot_sensor_frames(frames, ax, scatters):
            """
            Updates the scatter plots for each sensor.

            frames: list of 4xN arrays [x, y, z, q]
            ax: matplotlib Axes
            scatters: list of scatter artists (one per sensor)
            """

            def update_scatter(scat, x,y, color_func):
                """
                scat       = a matplotlib PathCollection (scatter plot)
                points     = Nx2 array of x,y coordinates
                color_func = function(x, y) -> array of color values
                """
                # Update positions
                scat.set_offsets(np.column_stack((x, y)))


                cvalues = color_func(x, y)

                # Apply to scatter
                scat.set_array(cvalues)

            def color_fn(x, y):
                # Example: color by quadrant

                # return (x > 0).astype(int) + 2 * (y > 0).astype(int)
                return np.sqrt((x**2)+(y+2)**2)*(2.71828182846)**(-1*((0.6*x)**2+(0.5*y)**2))


            for scat, (name, sensor) in zip(scatters, sensors.items()):
                update_scatter(scat, frame[0],frame[1], color_fn)

            plt.pause(0.05)  # adjust delay as needed

        while True:
            while True:
                update_delay = 0.05  # 50 ms
                frames = []

                for name, sensor in sensors.items():
                    frame = sensor.get_latest_frame()
                    frame = shift_sensor_data(frame, sensor)
                    frames.append(frame)

                # update all plots at once
                plot_sensor_frames(frames, ax, scatters)

                plt.pause(update_delay)

    except KeyboardInterrupt: #Closed file
        print("\nStopping sensors...")
        for sensor in sensors.values():
            sensor.stop()

if __name__ == "__main__":
    main()
