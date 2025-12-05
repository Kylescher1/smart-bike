import dill
import importlib
import sys
from pathlib import Path
import time
import quaternion
import numpy as np



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
def simple_point2obsticle(data):
    obsticle_arr = None
    #use magnitude of distance to origin

    for key,key_data in data.items():
        if key in ['point_cloud']:
            """
            RETURN THE CLOSEST POINT
            """
            coords = key_data[:3, :]  # shape (3, N)
            norms = np.linalg.norm(coords, axis=0)  # Euclidean norm per column
            idx = np.argmin(norms)  # index of column with smallest norm

            obsticle_arr = key_data[:, idx]  # the whole m×1 column
    return obsticle_arr

def simple_obsticle_response(obsticle_arr):
    if obsticle_arr is None:
        return #escapes function if there is no obsticles

    coords = obsticle_arr[:3, :]  # shape (3, N)
    norms = np.linalg.norm(coords, axis=0)  # Euclidean norm per column

    idx = np.argmin(norms)  # index of column with smallest norm

    closeest = obsticle_arr[:, idx]  # the whole m×1 column

    # B is shape (5, N)
    x = closeest[0, :]
    y = closeest[1, :]

    # Distance in XY plane
    dist = np.sqrt(x ** 2 + y ** 2)
    angle_deg = np.degrees(np.arctan2(x, y))

    L,R = calculate_haptics(dist,angle_deg)

    #do we vibe?

    #do we play sound?

    #do we activate breaks?

def Plot_Obsticles(obsticle_arr):
    return

def transform_to_cordnate(arr,Q=np.quaternion(1,0,0,0),Z=np.array([0,0,0])):
    """
    arr: list of MxN arrays [x, y, z, ...]
    Q: matplotlib quats of new WCS
    Z: offset vector in new wcs of old origin
    """
    V = arr[:3].T # shape (N,3)

    # Convert vectors to quaternion array (0, x, y, z) efficiently
    Vq = quaternion.from_vector_part(V)  # shape (N,) dtype=quaternion

    # Rotate: v' = q * v * q.conjugate()
    Vq_rot = Q * Vq * Q.conjugate()

    # Convert back to Nx3 array
    V_rot = quaternion.as_vector_part(Vq_rot)  # drop scalar part

    out = arr.copy()
    out[:3] = (V_rot + Z).T
    return out

def calculate_haptics(r, theta):
    """
    Calculates Left and Right PWM (0-255) based on Front Cone Priority logic.
    Args:
        r (float): Distance in meters.
        theta (float): Angle in degrees (Negative=Left, Positive=Right).
    Returns:
        tuple: (left_pwm, right_pwm) as integers.
    """
    # --- CONFIGURATION ---
    MIN_DIST_M = 2.0
    MAX_DIST_M = 5.0
    PAN_ANGLE = 25.0
    # 1. Filter: Out of range
    if r > MAX_DIST_M or r <= 0:
        return 0, 0
    # 2. Step A: Normalize Distance (0.0 = Far, 1.0 = Close)
    norm_dist = (MAX_DIST_M - r) / (MAX_DIST_M - MIN_DIST_M)
    norm_dist = max(0.0, min(norm_dist, 1.0)) # Clamp
    # 3. Step B: Determine Exponent (k)
    abs_theta = abs(theta)
    exponent = 1.0
    if abs_theta <= PAN_ANGLE:
        # Front Cone: Strict Linear Response
        exponent = 1.0
    else:
        # Side Zone: Ramp from 1.0 to 8.0
        # We map the remaining angle (25 to 90) to the range (0.0 to 1.0)
        side_ratio = (abs_theta - PAN_ANGLE) / (90.0 - PAN_ANGLE)
        exponent = 1.0 + (side_ratio * 7.0)
    # 4. Step C: Calculate Base Intensity
    base_intensity = (norm_dist ** exponent) * 255.0
    # 5. Step D: Stereo Panning
    if theta < 0:
        # Object is Left
        left_mix = 1.0
        # Fade right motor out as we approach 25 degrees left
        right_mix = 1.0 - (abs_theta / PAN_ANGLE)
    else:
        # Object is Right
        right_mix = 1.0
        # Fade left motor out as we approach 25 degrees right
        left_mix = 1.0 - (abs_theta / PAN_ANGLE)
    # Clamp mixes to 0.0 - 1.0
    left_mix = max(0.0, min(left_mix, 1.0))
    right_mix = max(0.0, min(right_mix, 1.0))
    # Calculate final integer PWM
    left_pwm = int(base_intensity * left_mix)
    right_pwm = int(base_intensity * right_mix)
    return left_pwm, right_pwm

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
    try:
        #Runs once before main loop
        display = True
        if display:
            Sonar = load_class_from_path("src.Debug_Tools.PlotTools.Sonar")()
        while True:
            #MAIN LOOP

            #SENSOR DATA COLLECTION
            new_data = {}
            for name, sensor in sensors.items(): #make a data dict that aggreegates data by type/use
                start = time.time()
                this_sensor_data = sensor.read() #{data_goal:data,...}
                if this_sensor_data is None:continue #only data thats valid gets passed

                for key,key_data in this_sensor_data.items():
                    if key not in new_data:
                        if key in ['point_cloud','ground_edge_detect']:#3d points
                            #data moves from sensor cords to bike cords (config dill specified)
                            new_data[key] = transform_to_cordnate(key_data,Q=sensor.orientation,Z=sensor.sensor_location)
                        else:
                            new_data[key] = key_data
                    else:
                        new_data[key].append(key_data)
                    # print(f"shape of data {np.shape(key_data)}")
                # print(f"{name} took {time.time()-start} s")

            #STATE EST
            """
            here we would put the gyro/accel intergration step to find the transfrom from last measurement to now in x,y,z
            and Q
            """

            #Lower quality of all data by factor
            #CULL DATA HER BY QUALITY

            """
            Here we would apply measured Q,Z (from accel) to historgrapgical data to make a unfied time evolution
            """



            #process raw sensor data into labeled groups
            obsticle_arr = simple_point2obsticle(new_data)

            #decide
            simple_obsticle_response(obsticle_arr)

            if display:
                Sonar.update_plot(new_data)
                # plot_sonar("data")
            # print(f"{data.keys()} → {data}")
    except KeyboardInterrupt: #Closed file
        print("\nStopping sensors...")
        for sensor in sensors.values():
            sensor.stop()

if __name__ == "__main__":
    main()
