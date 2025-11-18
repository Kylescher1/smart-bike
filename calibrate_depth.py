#!/usr/bin/env python3
"""
Depth Map Calibration Tool

Interactive depth map calibration similar to web_stream.py but runs locally.
Allows tuning parameters and saving to config.dill.

Usage:
    python calibrate_depth.py
"""

import sys
import os
from pathlib import Path

# Fix Qt/Wayland issues BEFORE importing cv2
if 'WAYLAND_DISPLAY' in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    print("Detected Wayland, forcing X11 backend")

# Make sure src folder is on sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.hal.VISION.VISION import DepthCalibrator

def main():
    """Run depth calibration tool."""
    calibrator = DepthCalibrator(config_path="config.dill", camera_name="camera")
    calibrator.run()

if __name__ == "__main__":
    main()

