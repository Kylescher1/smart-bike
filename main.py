import dill
import importlib
import sys
from pathlib import Path
import time

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
    
    return obsticle_arr

def simple_obsticle_response(obsticle_arr):
    if obsticle_arr is None:
        return #escapes function if there is no obsticles

    #do we vibe?

    #do we play sound?

    #do we activate breaks?

def Plot_Obsticles(obsticle_arr):
    return

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
            sonar = "init"
        while True:
            #MAIN LOOP

            #SENSOR DATA COLLECTION
            data = {}
            for name, sensor in sensors.items(): #make a data dict that aggreegates data by type/use
                start = time.time()
                this_sensor_data = sensor.read() #{data_goal:data,...}
                if this_sensor_data is None:continue #only data thats valid gets passed

                for key,key_data in this_sensor_data.items():
                    if key not in data:
                        data[key] = key_data
                    else:
                        data[key].append(key_data)
                    # print(f"shape of data {np.shape(key_data)}")
                # print(f"{name} took {time.time()-start} s")

            #process raw sensor data into labeled groups
            obsticle_arr = simple_point2obsticle(data)

            #decide
            simple_obsticle_response(obsticle_arr)

            if display:
                Plot_Obsticles(obsticle_arr)
                # plot_sonar("data")
            # print(f"{data.keys()} → {data}")
            time.sleep(1)
    except KeyboardInterrupt: #Closed file
        print("\nStopping sensors...")
        for sensor in sensors.values():
            sensor.stop()

if __name__ == "__main__":
    main()
