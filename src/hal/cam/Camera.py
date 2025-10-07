# src/hal/cam/Camera.py
import cv2
from typing import Optional, Dict


# Centralized configuration
CAMERA_CONFIG: Dict[str, int | str] = {
    "backend": cv2.CAP_V4L2,
    "width": 1600,
    "height": 1200,
    "fps": 60,
    "fourcc": "MJPG",  # string form for clarity
}


class Camera:
    def __init__(self, index: int, config: Dict[str, int | str] = CAMERA_CONFIG):
        self.index = index
        self.backend = config["backend"]
        self.width = config["width"]
        self.height = config["height"]
        self.fps = config["fps"]
        self.fourcc = config["fourcc"]
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, self.backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.index}] Failed to open.")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
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


def open_stereo_pair(max_index: int = 10, config: Dict[str, int | str] = CAMERA_CONFIG):
    """
    Try all /dev/video indices up to `max_index` and open the first two that work.
    Returns (left, right) Camera objects.
    """
    opened = []
    for idx in range(max_index):
        try:
            cam = Camera(index=idx, config=config)
            cam.open()
            print(f"✅ Opened camera {idx}")
            opened.append(cam)
            if len(opened) == 2:
                break
        except RuntimeError:
            pass

    if len(opened) < 2:
        for cam in opened:
            cam.close()
        raise RuntimeError("❌ Could not find two working cameras.")

    return opened[0], opened[1]


if __name__ == "__main__":
    try:
        # Override config here if needed
        stereo_config = CAMERA_CONFIG.copy()
        stereo_config.update({"width": 800, "height": 600, "fps": 90})

        left, right = open_stereo_pair(config=stereo_config)
        while True:
            frameL = left.read_frame()
            frameR = right.read_frame()
            if frameL is not None:
                cv2.imshow(f"Camera {left.index}", frameL)
            if frameR is not None:
                cv2.imshow(f"Camera {right.index}", frameR)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        left.close()
        right.close()
        cv2.destroyAllWindows()
