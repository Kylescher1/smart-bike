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

def instantiate_Peripherals(config):
    Peripherals = {}
    for name, params in config.items():
        print(f"Loading: {name}")
        cls = load_class_from_path(params['who_to_run'])

        try:#load class
            Peripheral = cls(name=name,**params)
        except Exception as e:
            raise KeyError(f"{name} threw an error: {e}")
        print(f"Loaded: {name}")

        try:#startup
            Peripheral.start()
            print(f"Started: {name}")
        except Exception as e:
            raise KeyError(f"{name}.start() threw an error: {e}")

        Peripherals[name] = Peripheral
    return Peripherals
def simple_point2obsticle(data,k=10):
    obsticle_arr = None

    for key, key_data in data.items():
        if key == 'point_cloud':
            # point_cloud is M×N
            coords = key_data[:3, :]  # (3, N)
            norms = np.linalg.norm(coords, axis=0)  # (N,)

            # indices of the closest K points
            k = min(k, norms.size)  # avoid overflow if <10 points
            idxs = np.argpartition(norms, k)[:k]  # unsorted K closest

            # If you want them sorted from nearest → farthest:
            idxs = idxs[np.argsort(norms[idxs])]

            # Select the columns (M × K)
            obsticle_arr = key_data[:, idxs]  # (M, K)

            return obsticle_arr  # stop after processing point_cloud

    return None  # if no point_cloud found

def simple_obsticle_response(obsticle_arr,Peripherals):
    if obsticle_arr is None:
        return #escapes function if there is no obsticles

    #more than 1
    if obsticle_arr.shape[0] > 1:
        obsticle_arr = np.atleast_2d(obsticle_arr)
        coords = obsticle_arr[:3, :]  # shape (3, N)
        norms = np.linalg.norm(coords, axis=0)  # Euclidean norm per column

        idx = np.argmin(norms)  # index of column with smallest norm

        closeest = obsticle_arr[:,idx]  # the whole m×1 column
    else:
        closeest = obsticle_arr

    x = closeest[0]
    y = closeest[1]

    # Distance in XY plane
    dist = np.sqrt(x ** 2 + y ** 2)
    angle_deg = np.degrees(np.arctan2(x, y))

    L,R = calculate_haptics(dist,angle_deg)


    print(f"Left:{L},Right:{R},dist:{dist},angle:{angle_deg}")


    #do we vibe?
    if 'esp32' in Peripherals:
        Peripherals['esp32'].vibrate(L, R)
    else:
        print("No esp/haptics device detected!")
    # do we play sound?

    #do we activate breaks?
    brake_mindist = 1# (m)
    if dist < brake_mindist:
        if 'Brakes' in Peripherals:
            Peripherals['Brakes'].dontdie()
        else:
            print("No esp/brake device detected!")


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
    MAX_DIST_M = 10.0
    # 1. Filter: Out of range
    if r > MAX_DIST_M or r <= 0:
        return 0, 0
    if theta <=-80 or theta >=80:
        return 0,0

    if r <= MIN_DIST_M: #R too close
        I = 1
    else: #In btw use linear aprox
        I = 1 + (MIN_DIST_M-r)/(MAX_DIST_M-MIN_DIST_M)

    M = theta/90

    L = I*(1-M)/2
    R = I*(1+M)/2
    return int(255*R),int(255*L)#we wired the motors backwords



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
    print("Enabling Peripherals...")
    try:
        Peripherals = instantiate_Peripherals(config)
        print("Peripherals Enabled")
    except Exception as e:
        raise KeyError(f"An unexpected error occurred with instantiate_sensors(), blame Damian: {e}")

    print("===" * 20)
    print("Peripheral Check Would go here")
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
            for name, Peripheral in Peripherals.items(): #make a data dict that aggreegates data by type/use
                start = time.time()
                if hasattr(Peripheral, 'read') and callable(getattr(Peripheral, 'read')):
                    this_Peripheral_data = Peripheral.read() #{data_goal:data,...}

                    if this_Peripheral_data is None:continue #only data thats valid gets passed

                    for key,key_data in this_Peripheral_data.items():
                        if key not in new_data:
                            if key in ['point_cloud','ground_edge_detect']:#3d points
                                #data moves from sensor cords to bike cords (config dill specified)
                                new_data[key] = transform_to_cordnate(key_data,Q=Peripheral.orientation,Z=Peripheral.sensor_location)
                            else:
                                new_data[key] = key_data
                        else:
                            new_data[key].append(key_data)
                        # print(f"shape of data {np.shape(key_data)}")
                    # print(f"{name} took {time.time()-start} s")
                else:#No data to produce IE output only
                    continue
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
            simple_obsticle_response(obsticle_arr,Peripherals)

            if display:
                Sonar.update_plot(new_data)
                # plot_sonar("data")
            # print(f"{data.keys()} → {data}")
    except KeyboardInterrupt: #Closed file
        print("\nStopping Peripherals...")
        for sensor in Peripherals.values():
            sensor.stop()

if __name__ == "__main__":
    main()
