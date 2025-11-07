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
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",#  <- REQUIRED
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
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",
         },
    "ground_lidar":
        {
            "port": "COM6",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            "position": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "z_direction":np.quaternion(0, 0, 0, 1),#w,x,y,z
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",
         },
    "arduino_breakout":
        {
            "port": "COM7",
            "baudrate": 115200,
            "BUFFER_SIZE": 200,
            "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
            "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
            "who_to_run": "src.hal.MPU6250.MPU6250",
        },
    "RangeFinder":
        {
            "port": "COM5",
            "baudrate" : 115200,
            "BUFFER_SIZE" : 200,
            "position": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "z_direction":np.quaternion(0, 0, 0, 1),#w,x,y,z
            "who_to_run": "src.hal.RangeFinder.RangeFinder",
         },
}

config = {
    "camera":
        {
            "left":
                {
                    "port": 1,
                    "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
                    "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
                    "map_x": None,  # Placeholder - will be set by calibration (leftMapX)
                    "map_y": None,  # Placeholder - will be set by calibration (leftMapY)
                },
            "right":
                {
                    "port": 2,
                    "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
                    "z_direction": np.quaternion(0, 0, 0, 1),  # w,x,y,z
                    "map_x": None,  # Placeholder - will be set by calibration (rightMapX)
                    "map_y": None,  # Placeholder - will be set by calibration (rightMapY)
                },
            "who_to_run": "src.hal.VISION.VISION.VISION",

            # Stereo block matcher core parameters
            "minDisparity": 0,
            "numDisparitiesK": 2,
            "numDisparities": 4,
            "blockSize": 11,
            "preFilterCap": 43,
            "uniquenessRatio": 1,
            "speckleWindowSize": 196,
            "speckleRange": 34,
            "disp12MaxDiff": 18,

            # Pre-processing & scaling
            "medianBlurK": 0,
            "downSample": 57,
            "crop": 128,
            "farEnhance": 27,
            "nearCutoff": 72,
            "farCutoff": 5,

            # Morphological filtering
            "useMorph": True,
            "morphIter": 5,

            # Bilateral smoothing
            "useBilateral": False,
            "bilateralStrength": 20,

            # Weighted least squares refinement
            "useWLS": True,
            "wlsLambda": 2389,
            "wlsSigma": 2.1,

            # Object detection thresholds
            "objectThresholdMM": 1095,
            "wsSigma": 2,
            "wsMinArea": 800,

            # Edge enhancement
            "edgeEqualize": True,
            "edgeBilateralD": 13,
            "edgeBilateralSigma": 200,
            "edgeCannyKLow": 3.0,
            "edgeCannyKHigh": 4.0,
            "edgeUseScharr": True,

            # Color segmentation
            "colorFocusMM": 10612,
            "colorSpanMM": 17800,
            "segMode": 0,
            "kmK": 4,
            "kmSpatialX100": 50,
            "rgTau": 50,
            "rgSeedStep": 16,

            # Calibration placeholders (shared between cameras)
            "imageSize": None,  # Set by calibration
            "Q": None,  # Set by calibration
        },
}

#Check you have all reqired fields
required_keys = {"who_to_run","port", "position", "z_direction"}

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
        # Check who_to_run field at camera level
        if "who_to_run" not in cfg:
            raise KeyError(f"{name} is missing required config item: who_to_run")
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
