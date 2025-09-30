import cv2
import numpy as np
import sys
import os

from Camera import Camera



class StereoDepth:
    def __init__(self, cam_left_index=1, cam_right_index=3):
        self.cam_left = Camera(cam_left_index)
        self.cam_right = Camera(cam_right_index)

        # Open cameras
        self.cam_left.open()
        self.cam_right.open()

        # Stereo matcher (basic block matching)
        self.stereo = cv2.StereoBM_create(
            numDisparities=16*6,  # must be multiple of 16
            blockSize=15          # odd number between 5..255
        )

    def get_depth_map(self):
        # Capture frames
        retL, frameL = self.cam_left.cap.read()
        retR, frameR = self.cam_right.cap.read()
        if not retL or not retR:
            return None

        # Convert to grayscale
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(grayL, grayR)

        # Normalize for visualization (0–255)
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)

        return disp_norm

    def release(self):
        self.cam_left.close()
        self.cam_right.close()


if __name__ == "__main__":
    stereo = StereoDepth()

    while True:
        depth_map = stereo.get_depth_map()
        if depth_map is None:
            break

        cv2.imshow("Depth Map (uncalibrated)", depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stereo.release()
    cv2.destroyAllWindows()
