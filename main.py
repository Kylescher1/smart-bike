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
    
    print(f"[POINT2OBSTICLE] Processing data with keys: {data.keys()}")

    for key, key_data in data.items():
        if key == 'point_cloud':
            print(f"[POINT2OBSTICLE] Found point_cloud, id: {id(key_data)}, shape: {key_data.shape}")
            print(f"[POINT2OBSTICLE] point_cloud is view: {key_data.base is not None}")
            
            # point_cloud is M×N
            coords = key_data[:3, :]  # (3, N)
            print(f"[POINT2OBSTICLE] Extracted coords (view), id: {id(coords)}, shape: {coords.shape}")
            
            norms = np.linalg.norm(coords, axis=0)  # (N,)
            print(f"[POINT2OBSTICLE] Computed norms, shape: {norms.shape}, min: {np.min(norms)}, max: {np.max(norms)}")

            # indices of the closest K points
            k = min(k, norms.size)  # avoid overflow if <10 points
            idxs = np.argpartition(norms, k)[:k]  # unsorted K closest
            print(f"[POINT2OBSTICLE] Selected top {k} closest points")

            # If you want them sorted from nearest → farthest:
            idxs = idxs[np.argsort(norms[idxs])]

            # Select the columns (M × K) - THIS IS A VIEW, so make a COPY!
            obsticle_arr = key_data[:, idxs].copy()  # (M, K) - COPY to prevent mutation
            print(f"[POINT2OBSTICLE] Created obsticle_arr (COPY), id: {id(obsticle_arr)}, shape: {obsticle_arr.shape}")
            print(f"[POINT2OBSTICLE] obsticle_arr.base is None (independent): {obsticle_arr.base is None}")

            return obsticle_arr  # stop after processing point_cloud

    return None  # if no point_cloud found

def simple_obsticle_response(obsticle_arr,Peripherals):
    if obsticle_arr is None:
        print(f"[OBSTICLE RESPONSE] No obstacles detected")
        return #escapes function if there is no obsticles

    print(f"[OBSTICLE RESPONSE] Processing obstacles, shape: {obsticle_arr.shape}, id: {id(obsticle_arr)}")

    #more than 1
    if obsticle_arr.shape[0] > 1:
        obsticle_arr = np.atleast_2d(obsticle_arr)
        coords = obsticle_arr[:3, :]  # shape (3, N)
        norms = np.linalg.norm(coords, axis=0)  # Euclidean norm per column

        idx = np.argmin(norms)  # index of column with smallest norm
        print(f"[OBSTICLE RESPONSE] Closest obstacle at index {idx}, distance: {norms[idx]:.3f}m")

        closeest = obsticle_arr[:,idx]  # the whole m×1 column
    else:
        closeest = obsticle_arr

    x = closeest[0]
    y = closeest[1]

    # Distance in XY plane
    dist = np.sqrt(x ** 2 + y ** 2)
    angle_deg = np.degrees(np.arctan2(x, y))

    L,R = calculate_haptics(dist,angle_deg)


    print(f"[OBSTICLE RESPONSE] Left:{L},Right:{R},dist:{dist:.3f}m,angle:{angle_deg:.1f}°")


    #do we vibe?
    if 'esp32' in Peripherals:
        Peripherals['esp32'].vibrate(L, R)
    else:
        print("No esp/haptics device detected!")
    # do we play sound?

    #do we activate breaks?
    brake_mindist = 1# (m)
    if dist < brake_mindist:
        print(f" I am seeing {dist} meters away, activating brakes")
        if 'Brakes' in Peripherals:
            #Peripherals['Brakes'].dontdie()
            print("Brakes are not enabled ")
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
    print(f"[TRANSFORM] Input arr id: {id(arr)}, shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"[TRANSFORM] Input arr is C-contiguous: {arr.flags['C_CONTIGUOUS']}, arr.base is: {arr.base}")
    
    V = arr[:3].T # shape (N,3)
    print(f"[TRANSFORM] V shape: {V.shape}, V is view: {V.base is not None}")

    # Convert vectors to quaternion array (0, x, y, z) efficiently
    Vq = quaternion.from_vector_part(V)  # shape (N,) dtype=quaternion

    # Rotate: v' = q * v * q.conjugate()
    Vq_rot = Q * Vq * Q.conjugate()

    # Convert back to Nx3 array
    V_rot = quaternion.as_vector_part(Vq_rot)  # drop scalar part

    # Make a DEEP copy to ensure no mutation
    out = arr.copy()
    print(f"[TRANSFORM] Output arr id: {id(out)}, is same as input: {id(out) == id(arr)}")
    out[:3] = (V_rot + Z).T
    print(f"[TRANSFORM] After modification, output arr stats: min={np.min(out[:2])}, max={np.max(out[:2])}")
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
        
        loop_count = 0
        while True:
            #MAIN LOOP
            loop_count += 1
            print(f"\n{'='*60}")
            print(f"[MAIN LOOP {loop_count}] Starting new iteration")
            print(f"{'='*60}")

            #SENSOR DATA COLLECTION
            new_data = {}
            for name, Peripheral in Peripherals.items(): #make a data dict that aggreegates data by type/use
                start = time.time()
                if hasattr(Peripheral, 'read') and callable(getattr(Peripheral, 'read')):
                    this_Peripheral_data = Peripheral.read() #{data_goal:data,...}
                    
                    print(f"[{name}] Read data: {this_Peripheral_data.keys() if this_Peripheral_data else 'None'}")

                    if this_Peripheral_data is None:continue #only data thats valid gets passed

                    for key,key_data in this_Peripheral_data.items():
                        if key_data is None:
                            print(f"[{name}] Key '{key}' has None value, skipping")
                            continue
                        
                        print(f"[{name}] Processing key '{key}', data shape: {np.shape(key_data)}, data id: {id(key_data)}")
                        
                        if key not in new_data:
                            if key in ['point_cloud','ground_edge_detect']:#3d points
                                print(f"[{name}] Transforming '{key}' data")
                                print(f"[{name}] BEFORE transform - data stats: min={np.min(key_data[:2]) if key_data.size > 0 else 'N/A'}, max={np.max(key_data[:2]) if key_data.size > 0 else 'N/A'}, shape={key_data.shape}")
                                #data moves from sensor cords to bike cords (config dill specified)
                                transformed_data = transform_to_cordnate(key_data,Q=Peripheral.orientation,Z=Peripheral.sensor_location)
                                print(f"[{name}] AFTER transform - data stats: min={np.min(transformed_data[:2]) if transformed_data.size > 0 else 'N/A'}, max={np.max(transformed_data[:2]) if transformed_data.size > 0 else 'N/A'}, shape={transformed_data.shape}, id={id(transformed_data)}")
                                new_data[key] = transformed_data
                            else:
                                new_data[key] = key_data
                                print(f"[{name}] Added '{key}' directly (not point cloud)")
                        else:
                            print(f"[{name}] WARNING: Key '{key}' already exists in new_data! Attempting to append...")
                            print(f"[{name}] Current new_data['{key}'] type: {type(new_data[key])}, shape: {np.shape(new_data[key])}")
                            # This is problematic - can't append to numpy array like this!
                            # new_data[key].append(key_data)
                            print(f"[{name}] ERROR: Cannot append to numpy array! Skipping this data.")
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
            print(f"\n[OBSTICLE DETECTION] Processing new_data keys: {new_data.keys()}")
            for key, val in new_data.items():
                if val is not None:
                    print(f"[OBSTICLE DETECTION] '{key}' shape: {np.shape(val)}, id: {id(val)}")
                    if isinstance(val, np.ndarray) and val.size > 0:
                        print(f"[OBSTICLE DETECTION] '{key}' stats: min={np.min(val[:2] if len(val) >= 2 else val)}, max={np.max(val[:2] if len(val) >= 2 else val)}")
            
            obsticle_arr = simple_point2obsticle(new_data)
            
            if obsticle_arr is not None:
                print(f"[OBSTICLE DETECTION] Found obstacles, shape: {obsticle_arr.shape}")

            #decide
            simple_obsticle_response(obsticle_arr,Peripherals)

            if display:
                print(f"\n[PLOTTING] Calling Sonar.update_plot with data keys: {new_data.keys()}")
                for key, val in new_data.items():
                    if val is not None and isinstance(val, np.ndarray):
                        print(f"[PLOTTING] Before plot - '{key}' id: {id(val)}, shape: {val.shape}")
                        if val.size > 0:
                            print(f"[PLOTTING] Before plot - '{key}' stats: min={np.min(val[:2])}, max={np.max(val[:2])}")
                
                Sonar.update_plot(new_data)
                
                print(f"[PLOTTING] After plot - checking if data was mutated:")
                for key, val in new_data.items():
                    if val is not None and isinstance(val, np.ndarray) and val.size > 0:
                        print(f"[PLOTTING] After plot - '{key}' id: {id(val)}, stats: min={np.min(val[:2])}, max={np.max(val[:2])}")
                
                # Small sleep to control loop rate and prevent overwhelming the system
                time.sleep(0.01)
            else:
                time.sleep(0.05)
                # plot_sonar("data")
            # print(f"{data.keys()} → {data}")
    except KeyboardInterrupt: #Closed file
        print("\nStopping Peripherals...")
        for sensor in Peripherals.values():
            sensor.stop()

if __name__ == "__main__":
    main()
