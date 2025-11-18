"""
Safe debug mode - runs with minimal features to isolate segfault sources.
Disables depth computation and other risky operations.
"""
import dill
import importlib
import sys
from pathlib import Path
import time
import faulthandler
import traceback
import signal

# Enable faulthandler to get stack traces on segfaults
faulthandler.enable()
faulthandler.enable(file=open('crash_log.txt', 'w'), all_threads=True)

# make sure src folder is on sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def signal_handler(sig, frame):
    """Handle signals gracefully"""
    print("\n⚠️ Received signal, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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

        try:
            # Enable safe mode for camera/VISION sensors
            if 'VISION' in params.get('who_to_run', '') or 'camera' in name.lower():
                params['safe_mode'] = True
                print(f"  ⚠️ Safe mode enabled for {name}")
            
            sensor = cls(name=name, **params)
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            traceback.print_exc()
            raise KeyError(f"{name} threw an error: {e}")
        print(f"Loaded: {name}")

        try:
            sensor.start()
            print(f"Started: {name}")
        except Exception as e:
            print(f"❌ Error starting {name}: {e}")
            traceback.print_exc()
            raise KeyError(f"{name}.start() threw an error: {e}")

        sensors[name] = sensor
    return sensors

def main():
    print("=" * 60)
    print("SAFE DEBUG MODE - Minimal features to isolate segfaults")
    print("=" * 60)
    
    #Load config
    print("Loading Config...")
    try:
        with open("config.dill", "rb") as f:
            config = dill.load(f)
        print("Loaded Config")
        for k, v in config.items():
            print(f"Device: {k} | Properties: {v}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        traceback.print_exc()
        raise KeyError(f"An unexpected error occurred Loading config.dill: {e}")

    print("===" * 20)
    print("Enabling Sensors (SAFE MODE)...")
    try:
        sensors = instantiate_sensors(config)
        print("Sensors Enabled")
    except Exception as e:
        print(f"❌ Error enabling sensors: {e}")
        traceback.print_exc()
        raise KeyError(f"An unexpected error occurred with instantiate_sensors(): {e}")

    print("===" * 20)
    print("Starting debug visualization...")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        # Check if camera sensor has debug_visual method and call it
        camera_sensor = sensors.get('camera')
        if camera_sensor and hasattr(camera_sensor, 'debug_visual'):
            print("⚠️ Running in SAFE MODE - depth computation disabled")
            camera_sensor.debug_visual()
        else:
            print("⚠️ No debug_visual method found, running basic loop")
            # Fallback to regular debug loop
            while True:
                sensor_data = {}
                for name, sensor in sensors.items():
                    try:
                        sensor.debug()
                        sensor_data.update({name: sensor.read()})
                    except Exception as e:
                        print(f"❌ Error in {name}.debug(): {e}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping sensors...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        for sensor in sensors.values():
            try:
                sensor.stop()
            except Exception as e:
                print(f"⚠️ Error stopping sensor: {e}")
        print("Done.")

if __name__ == "__main__":
    main()

