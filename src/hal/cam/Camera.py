# src/hal/cam/Camera.py
import cv2
import platform
import threading
import time
from typing import Optional, Dict

def get_default_backend():
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW  # or cv2.CAP_MSMF
    else:  # Linux
        return cv2.CAP_V4L2

# Centralized configuration
# Using 640x480@30fps for USB 2.0 compatibility (~63 Mbps for 2 cameras)
# Can increase to 60fps if using USB 3.0 ports (~126 Mbps for 2 cameras)
# YOLO resizes to 640x640 anyway, so resolution is fine
CAMERA_CONFIG: Dict[str, int | str] = {
    #"backend": cv2.CAP_V4L2,
    "backend": get_default_backend(),
    "width": 640,
    "height": 480,
    "fps": 30,  # Reduced from 60 to 30 for USB 2.0 compatibility
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
        # With buffer size = 1, single read should get latest frame
        ret, frame = self.cap.read()
        return frame if ret else None


class ThreadedCamera:
    """
    Threaded camera capture for low-latency frame acquisition.
    Captures frames in a background thread so read_frame() returns immediately
    with the latest available frame, eliminating capture wait time.
    """
    def __init__(self, index: int, config: Dict[str, int | str] = CAMERA_CONFIG):
        self.index = index
        self.backend = config["backend"]
        self.width = config["width"]
        self.height = config["height"]
        self.fps = config["fps"]
        self.fourcc = config["fourcc"]
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Thread synchronization
        self._frame = None
        self._frame_ready = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def open(self):
        self.cap = cv2.VideoCapture(self.index, self.backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"[ThreadedCamera {self.index}] Failed to open.")
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # Wait for first frame
        timeout = 2.0
        start = time.time()
        while not self._frame_ready and (time.time() - start) < timeout:
            time.sleep(0.01)
        
        if not self._frame_ready:
            self.close()
            raise RuntimeError(f"[ThreadedCamera {self.index}] Timeout waiting for first frame.")
    
    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        while self._running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._frame_ready = True
            else:
                time.sleep(0.001)
    
    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    def read_frame(self):
        """Returns the latest captured frame immediately (non-blocking)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


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
