# src/hal/cam/depth.py
import cv2
import numpy as np
from .Camera import Camera

def compute_depth_map(cam: Camera):
    frame = cam.read_frame()
    if frame is None:
        return None
    # Placeholder depth map = grayscale
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
