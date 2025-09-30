# -*- coding: utf-8 -*-
"""
Camera Interface Wrapper

Provides object-oriented access to USB cameras using OpenCV.
"""

import cv2
import time
from typing import Optional


class Camera:
    """
    Camera wrapper class for OpenCV VideoCapture.

    Attributes:
        index (int): Camera index for OpenCV (e.g. 0, 1, 2...).
        cap (cv2.VideoCapture): OpenCV capture object.
    """

    def __init__(self, index: int = 0, backend: int = cv2.CAP_V4L2,
                 width: int = 980, height: int = 720):
        self.index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the camera stream."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index, self.backend)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Camera {self.index}] Opened at {w}x{h}")
            else:
                raise RuntimeError(f"[Camera {self.index}] Failed to open.")

    def close(self) -> None:
        """Release the camera resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print(f"[Camera {self.index}] Released.")

    def read_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            frame: image array or None if failed
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.index}] Not open. Call open() first.")
        ret, frame = self.cap.read()
        return frame if ret else None

    def debug_print_loop(self, delay: float = 0.05):
        """
        Continuously display frames for debugging.

        Args:
            delay (float): Delay between reads in seconds.
        """
        print(f"[Camera {self.index}] Starting debug output (Ctrl+C to stop)")
        try:
            while True:
                frame = self.read_frame()
                if frame is not None:
                    cv2.imshow(f"Camera {self.index}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[Camera] Debug loop stopped by user")
                    break
                time.sleep(delay)
        finally:
            self.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = Camera(index=0)
    cam.open()
    cam.debug_print_loop()
