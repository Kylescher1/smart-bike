import cv2
import os
import time
import logging
from typing import List, Optional, Tuple
import numpy as np

class CameraStreamer:
    """
    Utility class for managing USB camera streams (ELP AR0234) 
    with support for single and stereo camera configurations.
    """
    
    def __init__(self, 
                 camera_indices: List[int] = [0, 1], 
                 resolution: Tuple[int, int] = (1280, 720), 
                 fps: int = 30):
        """
        Initialize camera stream with configurable parameters.
        
        :param camera_indices: List of camera device indices to use
        :param resolution: Desired camera resolution (width, height)
        :param fps: Frames per second
        """
        self.camera_indices = camera_indices
        self.resolution = resolution
        self.fps = fps
        self.cameras = []
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def init_cameras(self) -> bool:
        """
        Initialize cameras and verify connectivity.
        
        :return: True if all cameras are successfully initialized, False otherwise
        """
        self.cameras.clear()
        
        for index in self.camera_indices:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    self.logger.error(f"Cannot open camera {index}")
                    return False
                
                # Configure camera settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                self.cameras.append(cap)
                self.logger.info(f"Camera {index} initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing camera {index}: {e}")
                return False
        
        return True
    
    def capture_frame(self, camera_index: int = 0) -> Optional[np.ndarray]:
        """
        Capture a single frame from a specified camera.
        
        :param camera_index: Index of camera to capture from
        :return: Captured frame or None if capture fails
        """
        if camera_index >= len(self.cameras):
            self.logger.error(f"Invalid camera index: {camera_index}")
            return None
        
        camera = self.cameras[camera_index]
        ret, frame = camera.read()
        
        if not ret:
            self.logger.error(f"Failed to capture frame from camera {camera_index}")
            return None
        
        return frame
    
    def capture_stereo_frames(self) -> Optional[List[np.ndarray]]:
        """
        Capture synchronized frames from multiple cameras.
        
        :return: List of captured frames or None if capture fails
        """
        if len(self.cameras) < 2:
            self.logger.error("Not enough cameras initialized for stereo capture")
            return None
        
        frames = []
        for camera in self.cameras:
            ret, frame = camera.read()
            if not ret:
                self.logger.error("Failed to capture stereo frames")
                return None
            frames.append(frame)
        
        return frames
    
    def show_frame(self, frame: np.ndarray, window_name: str = "Camera Stream"):
        """
        Display a frame in a window.
        
        :param frame: Frame to display
        :param window_name: Name of the display window
        """
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    
    def save_frame(self, 
                   frame: np.ndarray, 
                   output_dir: str = "captured_frames", 
                   prefix: str = "frame"):
        """
        Save a frame to disk with timestamp.
        
        :param frame: Frame to save
        :param output_dir: Directory to save frames
        :param prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{prefix}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self.logger.info(f"Frame saved: {filename}")
    
    def release_cameras(self):
        """
        Release all camera resources.
        """
        for camera in self.cameras:
            camera.release()
        cv2.destroyAllWindows()

def main():
    """
    Example usage of CameraStreamer
    """
    streamer = CameraStreamer()
    
    if not streamer.init_cameras():
        print("Failed to initialize cameras")
        return
    
    try:
        while True:
            # Capture and display frames from first camera
            frame = streamer.capture_frame(0)
            if frame is not None:
                streamer.show_frame(frame)
                streamer.save_frame(frame)
            
            # Optional: Capture stereo frames
            stereo_frames = streamer.capture_stereo_frames()
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        streamer.release_cameras()

if __name__ == "__main__":
    main()
