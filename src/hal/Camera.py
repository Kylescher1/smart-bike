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

    def __init__(self, index: int = 0, backend: int = cv2.CAP_V4L2,
                 width: int = 1280, height: int = 720):
        self.index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera stream."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index, self.backend)

            # Request resolution
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                # Confirm actual resolution
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Camera {self.index}] Opened at {w}x{h}")
        if self.cap.isOpened():
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

if __name__ == "__main__":
    # Try some likely indexes (you can adjust if needed)
    camera_indexes = [0, 1, 2, 3]
    cameras = []

    for idx in camera_indexes:
        cam = Camera(index=idx)
        if cam.open():
            cameras.append(cam)

    if len(cameras) == 0:
        print("❌ No cameras could be opened. Exiting.")
    elif len(cameras) == 1:
        print("✅ Using single camera mode.")
        cam = cameras[0]
        try:
            while True:
                ret, frame = cam.read()
                if ret:
                    cv2.imshow(f"Camera {cam.index}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
        finally:
            cam.close()
            cv2.destroyAllWindows()
    else:
        print(f"✅ Using multi-camera mode with {len(cameras)} cameras.")
        try:
            while True:
                for cam in cameras:
                    ret, frame = cam.read()
                    if ret:
                        cv2.imshow(f"Camera {cam.index}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
        finally:
            for cam in cameras:
                cam.close()
            cv2.destroyAllWindows()
