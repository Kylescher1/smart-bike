# src/hal/cam/Camera.py
import cv2
from typing import Optional

class Camera:
    def __init__(self, index: int, backend: int = cv2.CAP_V4L2,
                 width: int = 800, height: int = 600, fps: int = 90):
        self.index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)  # force V4L2
        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.index}] Failed to open.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.index}] Not open. Call open() first.")
        ret, frame = self.cap.read()
        return frame if ret else None


def open_stereo_pair(left_idx=3, right_idx=1):
    """
    Convenience: open two cameras as a stereo pair.
    Uses defaults defined in Camera.__init__.
    """
    left = Camera(index=left_idx)
    right = Camera(index=right_idx)
    left.open()
    right.open()
    return left, right


if __name__ == "__main__":
    # Demo: try to open two cameras and show their streams
    cam_indices = [3, 1]  # adjust if your devices use different indices
    cameras = []

    try:
        for idx in cam_indices:
            try:
                cam = Camera(index=idx, backend=cv2.CAP_ANY, width=800, height=600, fps=90)
                cam.open()
                cameras.append(cam)
                print(f"✅ Opened camera {idx}")
            except RuntimeError as e:
                print(e)

        if not cameras:
            print("❌ No cameras opened.")
        else:
            while True:
                for cam in cameras:
                    frame = cam.read_frame()
                    if frame is not None:
                        cv2.imshow(f"Camera {cam.index}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for cam in cameras:
            cam.close()
        cv2.destroyAllWindows()
