# -*- coding: utf-8 -*-
"""
Camera Interface Wrapper

Provides object-oriented access to USB cameras using OpenCV.
"""

import cv2
from typing import Optional


class Camera:
    """
    Camera wrapper class for OpenCV VideoCapture.

    Attributes:
        index (int): Camera index for OpenCV (e.g. 0, 1, 2...).
        cap (cv2.VideoCapture): OpenCV capture object.
    """

    def __init__(self, index: int = 0, backend: int = cv2.CAP_V4L2):
        self.index = index
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera stream."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index, self.backend)
        if self.cap.isOpened():
            print(f"[Camera {self.index}] Opened successfully.")
            return True
        else:
            print(f"[Camera {self.index}] Failed to open.")
            return False

    def close(self):
        """Release the camera resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print(f"[Camera {self.index}] Released.")

    def read(self):
        """
        Capture a single frame from the camera.

        Returns:
            (ret, frame): ret is True if successful, frame is the image array.
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.index}] Not open. Call open() first.")
        return self.cap.read()

    def debug_show_loop(self, window_name: str = "Camera", exit_key: str = "q"):
        """
        Continuously show camera feed for debugging.

        Args:
            window_name (str): Name of the OpenCV window.
            exit_key (str): Key to press to exit loop.
        """
        if not self.open():
            return
        print(f"[Camera {self.index}] Starting debug feed (press '{exit_key}' to quit).")
        try:
            while True:
                ret, frame = self.read()
                if ret:
                    cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord(exit_key):
                    print(f"[Camera {self.index}] Exiting debug loop.")
                    break
        finally:
            self.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: test two cameras
    cam1 = Camera(index=1)
    cam2 = Camera(index=3)

    if cam1.open() and cam2.open():
        while True:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            if ret1:
                cv2.imshow("Camera 1", frame1)
            if ret2:
                cv2.imshow("Camera 2", frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    cam1.close()
    cam2.close()
    cv2.destroyAllWindows()
