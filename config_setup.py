import dill
import quaternion
import numpy as np
import time

#EXAMPLE  of FEATURES
examp = {
    "name": "Damian",
    "callback": lambda x: x**2,  # function reference
    "config": {"a": 1, "b": 2}
}
"""
Example sensor
"senosor_name":
        {
            "port":"COM13",#or /dev/whatever if in debian
            "baudrate" : 460800, 
            "BUFFER_SIZE" : 600,
            "position": np.quaternion(1, 0, 0, 0),#w,x,y,z <- REQUIRED
            "z_direction":np.quaternion(0, 0, 0, 1),#w,x,y,z <- REQUIRED
            "class": "src.hal.SpinningLidar.SpinningLidar",#  <- REQUIRED
         },
"""
print("Writing config file as")

config = {
    "horizontal_lidar":
        {
            "port": "COM13",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            "position": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "z_direction":np.quaternion(0, 0, 0, 1),#w,x,y,z
            "class": "src.hal.SpinningLidar.SpinningLidar",
         },
    "ground_lidar":
        {
            "port": "COM6",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            "position": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "z_direction":np.quaternion(0, 0, 0, 1),#w,x,y,z
            "class": "src.hal.SpinningLidar.SpinningLidar",
         },
    "arduino_breakout":
        {
            "port": "COM7",
            "baudrate": 115200,
            "BUFFER_SIZE": 200,
            "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
            "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
            "class": "src.hal.MPU6250.MPU6250",
        },
    "RangeFinder":
        {
            "port": "COM5",
            "baudrate" : 115200,
            "BUFFER_SIZE" : 200,
            "position": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "z_direction":np.quaternion(0, 0, 0, 1),#w,x,y,z
            "class": "src.hal.RangeFinder.RangeFinder",
         },
}

config = {
    "arduino_breakout":
        {
            "port": "/dev/ttyUSB0",
            "baudrate": 115200,
            "BUFFER_SIZE": 200,
            "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
            "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
            "class": "src.hal.MPU6250.MPU6250",
        },
    "camera":
        {
            "left":
                {
                    "port": 1,
                    "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
                    "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
                },
            "right":
                {
                    "port": 3,
                    "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
                    "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
                },
            "class": "src.hal.VISION.VISION",
            
        },
}

#Check you have all reqired fields
required_keys = {"class","port", "position", "z_direction"}

for name,cfg in config.items():
    # Handle nested structure for camera
    if name == "camera" and isinstance(cfg, dict) and "left" in cfg and "right" in cfg:
        # Check camera.left
        left_cfg = cfg["left"]
        left_missing = {"port", "position", "z_direction"} - ({"port", "position", "z_direction"} & left_cfg.keys())
        if left_missing:
            raise KeyError(f"{name}.left is missing required config items: {left_missing} ")
        # Check camera.right
        right_cfg = cfg["right"]
        right_missing = {"port", "position", "z_direction"} - ({"port", "position", "z_direction"} & right_cfg.keys())
        if right_missing:
            raise KeyError(f"{name}.right is missing required config items: {right_missing} ")
        # Check class field at camera level
        if "class" not in cfg:
            raise KeyError(f"{name} is missing required config item: class")
    else:
        # Standard flat structure
        missing = required_keys - (required_keys & cfg.keys())
        if missing:
            raise KeyError(f"{name} is missing required config items: {missing} ")

with open("config.dill", "wb") as f: #place data in
    dill.dump(config, f)

time.sleep(1)

with open("config.dill", "rb") as f:
    config_loaded = dill.load(f)


#Show us
for k,v in config_loaded.items():
    print(f"Device: {k} | Properties: {v}")
