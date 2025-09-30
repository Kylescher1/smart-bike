# src/hal/cam/rectify_overlay.py
import cv2
import numpy as np
import os
from src.hal.cam.depth import load_calibration, rectify_pair
from src.hal.cam.Camera import open_stereo_pair

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

def main():
    calib = load_calibration()

    # open stereo cameras
    left_cam, right_cam = open_stereo_pair()

    try:
        while True:
            left_frame = left_cam.read_frame()
            right_frame = right_cam.read_frame()
            if left_frame is None or right_frame is None:
                continue

            rectL, rectR = rectify_pair(left_frame, right_frame, calib)

            # tint right frame red
            rectR_red = tint_red(rectR)

            # overlay left and red-tinted right
            overlay = cv2.addWeighted(rectL, 0.5, rectR_red, 0.5, 0)

            # add epipolar lines
            overlay = draw_epilines(overlay, step=40)

            cv2.imshow("Rectification Overlay", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
