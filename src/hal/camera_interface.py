"""
Camera Interface Module for Smart Bike Project

Handles USB3 camera (AR0234) video capture and basic sensor configuration.
Responsible for frame grabbing, exposure control, and timestamp management.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

class CameraInterface:
    """
    Manages USB3 stereo camera pair (Left and Right)
    
    Attributes:
        - Handles frame capture at 30 Hz
        - Locks exposure, gain, white balance
        - Provides timestamped frames
    """
    
    def __init__(self, left_camera_index: int = 0, right_camera_index: int = 1):
        """
        Initialize stereo camera capture
        
        Args:
            left_camera_index: USB index for left camera
            right_camera_index: USB index for right camera
        """
        self.left_capture = cv2.VideoCapture(left_camera_index)
        self.right_capture = cv2.VideoCapture(right_camera_index)
        
        # Configure camera parameters
        self._configure_cameras()
    
    def _configure_cameras(self):
        """
        Set fixed camera parameters for consistent capture
        
        - Lock exposure
        - Set fixed gain
        - Set white balance
        """
        # TODO: Implement specific camera configuration for AR0234
        pass
    
    def grab_stereo_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Capture synchronized stereo frames
        
        Returns:
            Tuple of (left_frame, right_frame) or None if capture fails
        """
        ret_left = self.left_capture.grab()
        ret_right = self.right_capture.grab()
        
        if not (ret_left and ret_right):
            return None
        
        _, left_frame = self.left_capture.retrieve()
        _, right_frame = self.right_capture.retrieve()
        
        return left_frame, right_frame
    
    def __del__(self):
        """
        Release camera resources on object destruction
        """
        self.left_capture.release()
        self.right_capture.release()
