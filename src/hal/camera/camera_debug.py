import cv2
import time
import logging
from camera_interface import CameraStreamer  # Updated import

class CameraDebugger:
    """
    Advanced camera debugging utility for Smart Bike project
    """
    
    def __init__(self, camera_indices=None):
        """
        Initialize camera debugger
        
        :param camera_indices: List of camera indices to debug (default: [0])
        """
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Use provided indices or default to [0]
        self.camera_indices = camera_indices or [0]
        
        # Create CameraStreamer instance
        self.streamer = CameraStreamer(camera_indices=self.camera_indices)
    
    def detect_cameras(self):
        """
        Detect and list available cameras
        """
        self.logger.info("Scanning for available cameras...")
        available_cameras = []
        
        for index in range(10):  # Check first 10 possible camera indices
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                self.logger.info(f"Camera {index} detected:")
                self.logger.info(f"  Resolution: {width}x{height}")
                self.logger.info(f"  FPS: {fps}")
                
                available_cameras.append(index)
                cap.release()
        
        return available_cameras
    
    def camera_stress_test(self, duration=60, save_frames=True):
        """
        Perform a stress test on cameras using CameraStreamer
        
        :param duration: Test duration in seconds
        :param save_frames: Whether to save frames during test
        """
        self.logger.info(f"Starting {duration}-second camera stress test...")
        
        # Initialize cameras
        if not self.streamer.init_cameras():
            self.logger.error("Failed to initialize cameras")
            return
        
        try:
            start_time = time.time()
            frame_counts = [0] * len(self.streamer.cameras)
            
            while time.time() - start_time < duration:
                for i in range(len(self.streamer.cameras)):
                    frame = self.streamer.capture_frame(i)
                    
                    if frame is not None:
                        frame_counts[i] += 1
                        
                        # Optional: save frames
                        if save_frames:
                            self.streamer.save_frame(frame, prefix=f'camera_{i}_frame')
                        
                        # Display frames
                        self.streamer.show_frame(frame, f'Camera {i} Stream')
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Log results
            for i, count in enumerate(frame_counts):
                self.logger.info(f"Camera {self.camera_indices[i]}: {count} frames captured")
        
        except Exception as e:
            self.logger.error(f"Stress test error: {e}")
        
        finally:
            # Clean up
            self.streamer.release_cameras()
    
    def camera_property_explorer(self):
        """
        Explore and log detailed camera properties
        """
        self.logger.info("Exploring camera properties...")
        
        # Initialize cameras to ensure we can access them
        if not self.streamer.init_cameras():
            self.logger.error("Failed to initialize cameras")
            return
        
        try:
            for i, camera in enumerate(self.streamer.cameras):
                self.logger.info(f"\n--- Camera {self.camera_indices[i]} Properties ---")
                
                # Capture a frame to ensure camera is working
                frame = self.streamer.capture_frame(i)
                if frame is not None:
                    self.logger.info(f"Frame shape: {frame.shape}")
                    self.logger.info(f"Frame dtype: {frame.dtype}")
        
        finally:
            # Clean up
            self.streamer.release_cameras()

def main():
    # Create debugger
    debugger = CameraDebugger()
    
    # Detect available cameras
    available_cameras = debugger.detect_cameras()
    
    # Explore camera properties
    debugger.camera_property_explorer()
    
    # Optional: Run stress test on detected cameras
    if available_cameras:
        debugger.camera_stress_test(duration=10, save_frames=True)

if __name__ == "__main__":
    main()
