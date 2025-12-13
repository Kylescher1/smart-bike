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
    timings = {}
    
    for key, key_data in data.items():
        if key == 'point_cloud':
            # Check if point cloud is empty after filtering
            if key_data is None or key_data.size == 0 or key_data.shape[1] == 0:
                return None  # No points to process
            
            t0 = time.time()
            # point_cloud is M×N
            coords = key_data[:3, :]  # (3, N)
            timings['slice'] = (time.time() - t0) * 1000
            
            t0 = time.time()
            norms = np.linalg.norm(coords, axis=0)  # (N,)
            timings['norm'] = (time.time() - t0) * 1000

            t0 = time.time()
            # indices of the closest K points
            k = min(k, norms.size)  # avoid overflow if <10 points
            if k == 0:
                return None  # No points to process
            if k < norms.size:
                idxs = np.argpartition(norms, k)[:k]  # unsorted K closest
            else:
                idxs = np.arange(norms.size)  # take all points when k >= size
            timings['argpartition'] = (time.time() - t0) * 1000

            t0 = time.time()
            # If you want them sorted from nearest → farthest:
            idxs = idxs[np.argsort(norms[idxs])]
            timings['argsort'] = (time.time() - t0) * 1000

            t0 = time.time()
            # Select the columns (M × K) - make a COPY to prevent mutation
            obsticle_arr = key_data[:, idxs].copy()  # (M, K)
            timings['copy'] = (time.time() - t0) * 1000
            
            # Store timings for external access
            simple_point2obsticle.last_timings = timings

            return obsticle_arr  # stop after processing point_cloud

    return None  # if no point_cloud found

def simple_obsticle_response(obsticle_arr,Peripherals):
    if obsticle_arr is None:
        return #escapes function if there is no obsticles
    
    # Check if array is empty or has no columns
    if obsticle_arr.size == 0 or obsticle_arr.shape[1] == 0:
        return #escapes function if there is no obsticles after filtering

    timings = {}
    t0 = time.time()
    
    #more than 1
    if obsticle_arr.shape[0] > 1:
        obsticle_arr = np.atleast_2d(obsticle_arr)
        coords = obsticle_arr[:3, :]  # shape (3, N)
        norms = np.linalg.norm(coords, axis=0)  # Euclidean norm per column

        idx = np.argmin(norms)  # index of column with smallest norm

        closeest = obsticle_arr[:,idx]  # the whole m×1 column
    else:
        closeest = obsticle_arr
    
    timings['find_closest'] = (time.time() - t0) * 1000

    t0 = time.time()
    x = closeest[0]
    y = closeest[1]

    # Distance in XY plane
    dist = np.sqrt(x ** 2 + y ** 2)
    angle_deg = np.degrees(np.arctan2(x, y))
    timings['calc_dist_angle'] = (time.time() - t0) * 1000

    t0 = time.time()
    L,R = calculate_haptics(dist,angle_deg)
    timings['haptics'] = (time.time() - t0) * 1000
    
    # Store timings for external access
    simple_obsticle_response.last_timings = timings

    print(f"Obstacle: {dist:.2f}m @ {angle_deg:.0f}° | Haptics L:{L} R:{R}")


    #do we vibe?
    if 'esp32' in Peripherals:
        Peripherals['esp32'].vibrate(L, R)
    else:
        print("No esp/haptics device detected!")
    # do we play sound?

    #do we activate breaks?
    brake_mindist = 5.0# (m)
    if dist < brake_mindist:
        print(f" I am seeing {dist} meters away, activating brakes")
        if 'Brakes' in Peripherals:
            Peripherals['Brakes'].dont_die()
            # print("Brakes are not enabled ")
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
    # Store timing breakdown if needed for debugging
    transform_timings = {}
    
    t0 = time.time()
    V = arr[:3].T # shape (N,3)
    transform_timings['slice'] = (time.time() - t0) * 1000

    # Convert vectors to quaternion array (0, x, y, z) efficiently
    t0 = time.time()
    Vq = quaternion.from_vector_part(V)  # shape (N,) dtype=quaternion
    transform_timings['to_quat'] = (time.time() - t0) * 1000

    # Rotate: v' = q * v * q.conjugate()
    t0 = time.time()
    Vq_rot = Q * Vq * Q.conjugate()
    transform_timings['rotate'] = (time.time() - t0) * 1000

    # Convert back to Nx3 array
    t0 = time.time()
    V_rot = quaternion.as_vector_part(Vq_rot)  # drop scalar part
    transform_timings['from_quat'] = (time.time() - t0) * 1000

    # Make a DEEP copy to ensure no mutation
    t0 = time.time()
    out = arr.copy()
    out[:3] = (V_rot + Z).T
    transform_timings['copy_assign'] = (time.time() - t0) * 1000
    
    # Store timings for external access
    transform_to_cordnate.last_timings = transform_timings
    
    return out

def filter_forward_cone(arr):
    """
    Filters to only keep data points within a 90-degree cone (±45°) from the forward direction.
    This keeps only relevant data in front of the bike for obstacle detection.
    
    Args:
        arr: MxN array where first 3 rows are [x, y, z, ...]
    
    Returns:
        MxN array with filtered columns (points within the forward cone)
    """
    if arr is None or arr.size == 0:
        return arr
    
    # Extract x, y coordinates
    x = arr[0, :]  # shape (N,)
    y = arr[1, :]  # shape (N,)
    
    # Calculate angle from positive x-axis in degrees
    # arctan2(y, x) gives angle where:
    #   0° = positive x (right)
    #   90° = positive y (forward)
    #   -90° or 270° = negative y (backward)
    angles = np.degrees(np.arctan2(y, x))
    
    # Define the forward cone: ±45° from positive y direction (90°)
    # This means angles between 45° and 135°
    # Keep points WITHIN this range
    mask = (angles >= 85) & (angles <= 95)
    
    # Filter the array to keep only points within the forward cone
    filtered_arr = arr[:, mask]
    
    return filtered_arr

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
        display = False
        last_plot_data = None
        plot_skip_counter = 0
        
        if display:
            Sonar = load_class_from_path("src.Debug_Tools.PlotTools.Sonar")()
            print("[PLOT] Sonar display initialized")
        
        loop_count = 0
        timing_stats = {
            'read_sensors': [],
            'transform': [],
            'obstacle_detection': [],
            'response': [],
            'plot': [],
            'sleep': [],
            'total': []
        }
        
        while True:
            #MAIN LOOP
            loop_start = time.time()
            loop_count += 1

            #SENSOR DATA COLLECTION
            read_start = time.time()
            new_data = {}
            sensor_read_times = {}
            transform_times = {}
            
            for name, Peripheral in Peripherals.items(): #make a data dict that aggreegates data by type/use
                sensor_start = time.time()
                if hasattr(Peripheral, 'read') and callable(getattr(Peripheral, 'read')):
                    read_call_start = time.time()
                    this_Peripheral_data = Peripheral.read() #{data_goal:data,...}
                    read_call_time = (time.time() - read_call_start) * 1000
                    sensor_read_times[name] = read_call_time

                    if this_Peripheral_data is None:
                        continue #only data thats valid gets passed

                    for key,key_data in this_Peripheral_data.items():
                        if key_data is None:
                            continue
                        
                        if key not in new_data:
                            if key in ['point_cloud','ground_edge_detect']:#3d points
                                transform_start = time.time()
                                #data moves from sensor cords to bike cords (config dill specified)
                                transformed_data = transform_to_cordnate(key_data,Q=Peripheral.orientation,Z=Peripheral.sensor_location)
                                # Filter to keep only forward cone data (±45° from positive y direction)
                                filtered_data = filter_forward_cone(transformed_data)
                                transform_time = (time.time() - transform_start) * 1000
                                transform_times[f"{name}.{key}"] = transform_time
                                new_data[key] = filtered_data
                            else:
                                new_data[key] = key_data
                        else:
                            print(f"[{name}] WARNING: Key '{key}' already exists in new_data! Skipping duplicate.")
                
                sensor_time = (time.time() - sensor_start) * 1000
                if name in sensor_read_times:
                    sensor_read_times[name] = sensor_time
                else:
                    sensor_read_times[name] = sensor_time
                    
            read_time = (time.time() - read_start) * 1000
            timing_stats['read_sensors'].append(read_time)
            
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
            obsticle_start = time.time()
            obsticle_arr = simple_point2obsticle(new_data)
            # print(f"Obsticle array: {obsticle_arr}")
            obsticle_time = (time.time() - obsticle_start) * 1000
            timing_stats['obstacle_detection'].append(obsticle_time)

            #decide
            response_start = time.time()
            simple_obsticle_response(obsticle_arr,Peripherals)
            response_time = (time.time() - response_start) * 1000
            timing_stats['response'].append(response_time)

            if display:
                plot_start = time.time()
                # Update plot every 3rd frame to reduce overhead
                plot_skip_counter += 1
                if plot_skip_counter >= 3:
                    plot_skip_counter = 0
                    Sonar.update_plot(new_data)
                    plot_time = (time.time() - plot_start) * 1000
                else:
                    plot_time = 0  # Skipped this frame
                timing_stats['plot'].append(plot_time)
                
                sleep_start = time.time()
                # Small sleep to control loop rate and prevent overwhelming the system
                time.sleep(0.01)
                sleep_time = (time.time() - sleep_start) * 1000
                timing_stats['sleep'].append(sleep_time)
            else:
                sleep_start = time.time()
                time.sleep(0.05)
                sleep_time = (time.time() - sleep_start) * 1000
                timing_stats['sleep'].append(sleep_time)
                timing_stats['plot'].append(0)
            
            # Calculate total loop time
            loop_time = (time.time() - loop_start) * 1000
            timing_stats['total'].append(loop_time)
            
            # Print detailed timing every 10 loops
            if loop_count % 10 == 0:
                print(f"\n{'='*70}")
                print(f"[TIMING ANALYSIS] Loop {loop_count} - Last 10 loops average:")
                print(f"{'='*70}")
                
                # Calculate averages
                avg_read = np.mean(timing_stats['read_sensors'][-10:])
                avg_transform = np.mean([sum(transform_times.values())] if transform_times else [0])
                avg_obstacle = np.mean(timing_stats['obstacle_detection'][-10:])
                avg_response = np.mean(timing_stats['response'][-10:])
                avg_plot = np.mean(timing_stats['plot'][-10:])
                avg_sleep = np.mean(timing_stats['sleep'][-10:])
                avg_total = np.mean(timing_stats['total'][-10:])
                
                # Calculate percentages
                pct_read = (avg_read / avg_total * 100) if avg_total > 0 else 0
                pct_transform = (avg_transform / avg_total * 100) if avg_total > 0 else 0
                pct_obstacle = (avg_obstacle / avg_total * 100) if avg_total > 0 else 0
                pct_response = (avg_response / avg_total * 100) if avg_total > 0 else 0
                pct_plot = (avg_plot / avg_total * 100) if avg_total > 0 else 0
                pct_sleep = (avg_sleep / avg_total * 100) if avg_total > 0 else 0
                
                print(f"Total Loop Time: {avg_total:.2f}ms ({1000/avg_total:.1f} FPS)")
                print(f"\nBreakdown:")
                print(f"  ├─ Read Sensors:     {avg_read:7.2f}ms ({pct_read:5.1f}%)", end="")
                if sensor_read_times:
                    print(" [", end="")
                    sensor_details = []
                    for name, t in sensor_read_times.items():
                        sensor_details.append(f"{name}: {t:.1f}ms")
                    print(" | ".join(sensor_details), end="")
                    print("]")
                else:
                    print()
                
                if transform_times:
                    total_transform = sum(transform_times.values())
                    print(f"  ├─ Transform:        {total_transform:7.2f}ms ({pct_transform:5.1f}%)", end="")
                    print(" [", end="")
                    transform_details = []
                    for key, t in transform_times.items():
                        transform_details.append(f"{key}: {t:.1f}ms")
                    print(" | ".join(transform_details), end="")
                    print("]")
                    
                    # Show transform breakdown if available
                    if hasattr(transform_to_cordnate, 'last_timings'):
                        tf_times = transform_to_cordnate.last_timings
                        print(f"      └─ Breakdown: slice={tf_times.get('slice', 0):.2f}ms, "
                              f"to_quat={tf_times.get('to_quat', 0):.2f}ms, "
                              f"rotate={tf_times.get('rotate', 0):.2f}ms, "
                              f"from_quat={tf_times.get('from_quat', 0):.2f}ms, "
                              f"copy={tf_times.get('copy_assign', 0):.2f}ms")
                
                print(f"  ├─ Obstacle Detect:  {avg_obstacle:7.2f}ms ({pct_obstacle:5.1f}%)", end="")
                if hasattr(simple_point2obsticle, 'last_timings'):
                    obs_times = simple_point2obsticle.last_timings
                    print(f" [slice={obs_times.get('slice', 0):.2f}ms, "
                          f"norm={obs_times.get('norm', 0):.2f}ms, "
                          f"argpartition={obs_times.get('argpartition', 0):.2f}ms, "
                          f"argsort={obs_times.get('argsort', 0):.2f}ms, "
                          f"copy={obs_times.get('copy', 0):.2f}ms]")
                else:
                    print()
                print(f"  ├─ Response Logic:   {avg_response:7.2f}ms ({pct_response:5.1f}%)", end="")
                if hasattr(simple_obsticle_response, 'last_timings'):
                    resp_times = simple_obsticle_response.last_timings
                    print(f" [find_closest={resp_times.get('find_closest', 0):.2f}ms, "
                          f"calc={resp_times.get('calc_dist_angle', 0):.2f}ms, "
                          f"haptics={resp_times.get('haptics', 0):.2f}ms]")
                else:
                    print()
                print(f"  ├─ Plot Update:       {avg_plot:7.2f}ms ({pct_plot:5.1f}%)", end="")
                if display and hasattr(Sonar, 'last_update_times'):
                    plot_times = Sonar.last_update_times
                    print(f" [process: {plot_times['process']:.1f}ms, draw: {plot_times['draw']:.1f}ms]")
                else:
                    print()
                print(f"  └─ Sleep:             {avg_sleep:7.2f}ms ({pct_sleep:5.1f}%)")
                
                # Show bottleneck
                times_dict = {
                    'Read Sensors': avg_read,
                    'Transform': avg_transform if transform_times else 0,
                    'Obstacle Detection': avg_obstacle,
                    'Response Logic': avg_response,
                    'Plot Update': avg_plot,
                    'Sleep': avg_sleep
                }
                bottleneck = max(times_dict.items(), key=lambda x: x[1])
                print(f"\n[BOTTLENECK] {bottleneck[0]}: {bottleneck[1]:.2f}ms ({bottleneck[1]/avg_total*100:.1f}% of total)")
                print(f"{'='*70}\n")
                
                # Keep only last 100 entries to prevent memory growth
                for key in timing_stats:
                    if len(timing_stats[key]) > 100:
                        timing_stats[key] = timing_stats[key][-100:]
    except KeyboardInterrupt: #Closed file
        print("\nStopping Peripherals...")
        for sensor in Peripherals.values():
            sensor.stop()

if __name__ == "__main__":
    main()
