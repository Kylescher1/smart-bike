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
            "orientation": np.quaternion(1, 0, 0, 0),#w,x,y,z <- REQUIRED
            "sensor_location":np.array([0, 0, 0]),#x,y,z <- REQUIRED
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",#  <- REQUIRED
         },
"""
print("Writing config file as")

config = {
    "horizontal_lidar":
        {
            "port": "/dev/ttyUSB0",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            # "orientation": np.quaternion(0.7071, 0, 0, -0.7071),#w,x,y,z
            "orientation": np.quaternion(np.cos(-np.pi/2), 0, 0, np.sin(-np.pi/2)),#w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "data_out_label":"point_cloud",
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",
         },
    "ground_lidar":
        {
            "port": "/dev/ttyUSB0",
            "baudrate" : 460800,
            "BUFFER_SIZE" : 600,
            "orientation": np.quaternion(np.cos(np.pi/2), 0, 0, np.sin(np.pi/2))*np.quaternion(np.cos(np.pi/2),  np.sin(np.pi/2), 0, 0),#w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "data_out_label":"ground_edge_detect",
            "who_to_run": "src.hal.SpinningLidar.SpinningLidar",
         },
    "arduino_breakout":
        {
            "port": "COM13",
            "baudrate": 115200,
            "BUFFER_SIZE": 200,
            "Bias":np.array([0,0,0,0,0,0,0]), # time, ax,ay,az, gx,gy,gz
            "orientation": np.quaternion(1, 0, 0, 0),  # w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "data_out_label":"IMU",
            "who_to_run": "src.hal.MPU6250.MPU6250",
        },
    "RangeFinder":
        {
            "port": "COM5",
            "baudrate" : 115200,
            "BUFFER_SIZE" : 200,
            "orientation": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "data_out_label":"point_cloud",
            "who_to_run": "src.hal.RangeFinder.RangeFinder",
         },
    "esp32":
        {
            "port": "/dev/ttyUSB0",
            "baudrate": 115200,
            "BUFFER_SIZE": 200,
            "orientation": np.quaternion(1, 0, 0, 0),#w,x,y,z
            "sensor_location":np.array([0, 0, 0]),#x,y,z
            "who_to_run": "src.hal.ESP32.ESP32",
            "debug_mode":False,
        },
    "StatusLED":
        {
            "port": None,
            "orientation": None,
            "sensor_location":None,
            "who_to_run": "src.hal.LED.GPIO_LED",
            "debug_mode":False,
        },
    "Brakes":
        {
            "port": None,
            "orientation": None,
            "sensor_location":None,
            "who_to_run": "src.hal.BrakeRoutines.BrakeRoutines",
            "debug_mode":False,
            "abs_enabled":True,
            "chip_num": 4,
            "line_num": 11,
        },
    "camera":
        {
            "left":
                {
                    "port": 2,
                    "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
                    "orientation":np.array([0, 0, 0]),#x,y,z
                    "map_x": None,  # Placeholder - will be set by calibration (leftMapX)
                    "map_y": None,  # Placeholder - will be set by calibration (leftMapY)
                },
            "right":
                {
                    "port": 1,
                    "position": np.quaternion(1, 0, 0, 0),  # w,x,y,z
                    "orientation":np.array([0, 0, 0]),#x,y,z
                    "map_x": None,  # Placeholder - will be set by calibration (rightMapX)
                    "map_y": None,  # Placeholder - will be set by calibration (rightMapY)
                },
            "who_to_run": "src.hal.VISION.VISION_UPGRADE.VISION",

            # YOLO Object Detection Configuration
            "yolo":
                {
                    "model_path": "yolo/models/yolo11n.rknn",  # Path to RKNN model file
                    "conf_threshold": 0.25,  # Confidence threshold for detections
                    "imgsz": 640,  # Input image size for YOLO
                    "track_enabled": True,  # Enable object tracking
                    "track_thresh": 0.5,  # Tracking confidence threshold
                    "track_high_thresh": 0.6,  # High confidence threshold for tracking
                    "track_match_thresh": 0.8,  # IoU threshold for track matching
                    "frame_rate": 30,  # Frame rate for tracking
                    "track_buffer": 30,  # Number of frames to keep lost tracks
                },

            # Depth Estimation Parameters
            "baseline": 0.23,  # Stereo baseline in meters
            "focal_length_px": 489.14,  # Focal length in pixels
            "ema_alpha": 0.3,  # EMA smoothing factor (0-1, higher = less smoothing)
            "roi_expansion": 10,  # Pixels to expand ROI around bounding box

            # Camera Field of View (for angle calculation)
            "fov_horizontal": 126.0,  # Horizontal FOV in degrees
            "fov_vertical": 101.62,  # Vertical FOV in degrees

            # Buffer Configuration
            "buffer_size": 10,  # Circular buffer size for object data

            # Stereo block matcher core parameters
            "minDisparity": 0,
            "numDisparitiesK": 2,
            "numDisparities": 4,
            "blockSize": 11,
            "P1": 968,  # Penalty for disparity change by 1 (typically 8 * channels * blockSize^2)
            "P2": 3872,  # Penalty for disparity change by more than 1 (typically 32 * channels * blockSize^2)
            "preFilterCap": 43,
            "uniquenessRatio": 1,
            "speckleWindowSize": 196,
            "speckleRange": 34,
            "disp12MaxDiff": 18,
            "sgbmMode": 2,  # SGBM mode: 0=SGBM, 1=HH, 2=SGBM_3WAY (default)

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

            # Post-processing filters
            "smoothingKernel": 0,
            "confidenceWindow": 5,
            "confidenceThreshold": 0.0,

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
            "segMode": 0, #1-3
            "kmK": 4,
            "kmSpatialX100": 50,
            "rgTau": 50,
            "rgSeedStep": 16,

            # Calibration placeholders (shared between cameras)
            "imageSize": None,  # Set by calibration
            "Q": None,  # Set by calibration
            "Yolo":{
                "seting1"
            }
        },
    "Turret":
        {
            "port": None,
            "orientation": None,
            "sensor_location":None,
            "who_to_run": "src.hal.Turret.peripheral_mode",
            "debug_mode":False,
            "camera": 1,
            "turret":"/dev/ttyUSB0",
            "invert-y":True,
            "timing":True,
            "rknn-model": "/home/radxa/smart-bike/yolo/models/yolo11n.rknn",
            "deadzone": 50,
            "pid-max-output": 3,
            "rknn":True,
            "kp": 0.1,
            "ki": 0.1,
            "kd": 0.15,
            "max-movement": 5,
            "control-rate": 60,
            "detection-imgsz": 320,
            "yolo-half":True
        }
}
config = {
    'horizontal_lidar':config['horizontal_lidar'],
    # 'esp32':config['esp32'],
    'Brakes':config['Brakes'],
          }

#Check you have all reqired fields
required_keys = {"who_to_run","port", "sensor_location", "orientation"}

for name,cfg in config.items():
    # Handle nested structure for camera
    if name == "camera" and isinstance(cfg, dict) and "left" in cfg and "right" in cfg:
        # Check camera.left
        left_cfg = cfg["left"]
        left_missing = {"port", "sensor_location", "orientation"} - ({"port", "sensor_location", "orientation"} & left_cfg.keys())
        if left_missing:
            raise KeyError(f"{name}.left is missing required config items: {left_missing} ")
        # Check camera.right
        right_cfg = cfg["right"]
        right_missing = {"port", "sensor_location", "orientation"} - ({"port", "sensor_location", "orientation"} & right_cfg.keys())
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
