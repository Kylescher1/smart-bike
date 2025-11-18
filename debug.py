import dill
import importlib
import sys
from pathlib import Path
import dashboard
import time
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
        # Check if camera sensor has debug_visual method and call it
        camera_sensor = sensors.get('camera')
        if camera_sensor and hasattr(camera_sensor, 'debug_visual'):
            camera_sensor.debug_visual()
        else:
            # Fallback to regular debug loop
            while True:
                sensor_data = {}
                for name, sensor in sensors.items():
                    sensor.debug()
                    sensor_data.update({name: sensor.read()})
                # time.sleep(1)
                # dash.update()
    except KeyboardInterrupt: #Closed file
        print("\nStopping sensors...")
        for sensor in sensors.values():
            sensor.stop()

if __name__ == "__main__":
    main()
