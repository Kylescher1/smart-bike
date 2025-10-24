imports

setup

from src.hal.cam.Camera import Camera

left, right = open_stereo_pair()

while True:

    read frames
    send frames to Depth class, with settings , calibration, options
    Depth class returns depth map

    if preview I can show depth map
    if save I can save depth map npz file to folder 


    
