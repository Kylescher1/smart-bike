import cv2
import numpy as np
import os

LEFT_IMAGE_PATH = os.path.join("src", "hal", "cam", "calibrate", "stereo_pairs", "left_000.png")
RIGHT_IMAGE_PATH = os.path.join("src", "hal", "cam", "calibrate", "stereo_pairs", "right_000.png")

left_frame = cv2.imread(LEFT_IMAGE_PATH)
right_frame = cv2.imread(RIGHT_IMAGE_PATH)

if left_frame is None:
    print(f"Failed to load {LEFT_IMAGE_PATH}")
    exit()

if right_frame is None:
    print(f"Failed to load {RIGHT_IMAGE_PATH}")
    exit()

import dill

# Load rectification maps from config.dill
config_path = "config.dill"
with open(config_path, "rb") as f:
    config = dill.load(f)

camera_cfg = config.get("camera", {})
left_cfg = camera_cfg.get("left", {})
right_cfg = camera_cfg.get("right", {})

left_map_x = left_cfg.get("map_x")
left_map_y = left_cfg.get("map_y")
right_map_x = right_cfg.get("map_x")
right_map_y = right_cfg.get("map_y")

if left_map_x is None or left_map_y is None or right_map_x is None or right_map_y is None:
    print("One or more rectification maps are missing in config.dill under camera.left/right map_x/map_y.")
    exit()

left_map_x = np.asarray(left_map_x, dtype=np.float32)
left_map_y = np.asarray(left_map_y, dtype=np.float32)
right_map_x = np.asarray(right_map_x, dtype=np.float32)
right_map_y = np.asarray(right_map_y, dtype=np.float32)

# Rectify the images and convert to grayscale (StereoBM requires CV_8UC1)
left_rectified = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)

left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# Create and compute disparity map
stereo_bm = cv2.StereoBM_create(numDisparities=16 * 6, blockSize=15)
disparity = stereo_bm.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Normalize for display
disparity_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_display = np.uint8(disparity_display)

cv2.imshow("Disparity Map", disparity_display)

print("Press any key to close")
cv2.waitKey(0)
cv2.destroyAllWindows()
