# src/hal/cam/tools/rectify_overlay.py
import cv2
import numpy as np
import os
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Camera import open_stereo_pair
from src.hal.config import LEFT_INDEX, RIGHT_INDEX, SWAP_LR

def draw_epilines(img, step=40):
    """Draw horizontal green lines every `step` pixels."""
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    return img

def tint_red(img):
    """Return a red-tinted version of grayscale or color image."""
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(img)
    red[:, :, 2] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return red

def rectify_pair(left_frame, right_frame, calib):
    """Rectify a stereo pair using calibration maps."""
    left_map_x, left_map_y, right_map_x, right_map_y, _image_size, _Q = calib
    rectL = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)
    return rectL, rectR

def main():
    calib = load_calibration()

    # open stereo cameras
    left_cam, right_cam = open_stereo_pair(LEFT_INDEX, RIGHT_INDEX)

    try:
        while True:
            left_frame = left_cam.get_frame()
            right_frame = right_cam.get_frame()
            if left_frame is None or right_frame is None:
                continue

            if SWAP_LR:
                left_frame, right_frame = right_frame, left_frame

            rectL, rectR = rectify_pair(left_frame, right_frame, calib)

            # tint right frame red
            rectR_red = tint_red(rectR)

            # overlay left and red-tinted right
            overlay = cv2.addWeighted(rectL, 0.5, rectR_red, 0.5, 0)

            # add epipolar lines
            overlay = draw_epilines(overlay, step=40)

            overlay = cv2.resize(overlay, (800, 600))
            cv2.imshow("Rectification Overlay", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
