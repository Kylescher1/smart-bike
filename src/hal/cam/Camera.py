# src/hal/cam/Camera.py
import cv2
import platform
from typing import Optional, Dict

def get_default_backend():
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW  # or cv2.CAP_MSMF
    else:  # Linux
        return cv2.CAP_V4L2

# Centralized configuration
CAMERA_CONFIG: Dict[str, int | str] = {
    #"backend": cv2.CAP_V4L2,
    "backend": get_default_backend(),
    "width": 1920,
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
        
        # CRITICAL: Set buffer size to 1 to always get latest frame (reduces lag significantly)
        # This prevents frame buffering which causes delays
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.index}] Not open. Call open() first.")
        # Optimize capture: grab frames until we get the latest one
        # This prevents using stale buffered frames
        self.cap.grab()  # Discard any old frame
        ret, frame = self.cap.retrieve()
        if not ret:
            # Fallback to regular read
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
        stereo_config.update({"width": 1024, "height": 768, "fps": 90})

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
