imports

setup

from src.hal.cam.Camera import Camera

left, right = open_stereo_pair()

while True:
    left_frame = left.read_frame()
    right_frame = right.read_frame()

    calib = load_calibration()
    settings = load_settings()
    depth = compute_depth(left_frame, right_frame,calib,settings)

    if preview == True:
        depth.preview.show()
        depth.preview.size = (1280, 720)
