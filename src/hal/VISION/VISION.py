import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import os
import glob
import dill
import copy

from ..cam.Camera import Camera, CAMERA_CONFIG
from .depth_processing import DepthProcessor
# from .calibration import run_calibration


class VISION:
    BOOLEAN_FIELDS = {
        "useMorph",
        "useBilateral",
        "useWLS",
        "edgeEqualize",
        "edgeUseScharr",
    }
    """
    Vision system for stereo camera depth processing.
    Follows sensor pattern with start(), stop(), and read() methods.
    """
    
    def __init__(self, name="Unnamed VISION", **kwargs):
        """
        Initialize VISION instance.
        Supports nested config structure: camera.left and camera.right
        Also supports flat structure where left/right are at top level
        """
        self.name = name
        self.debug_mode = True
        
        # Load user-provided configuration (following MPU6250 pattern)
        for k, v in kwargs.items():
            setattr(self, k, v)
        #     print(f"self.{k} returns:{v}")
        # print(f"kyle is evil and am {vars(self)}")

        # Normalize boolean toggles (config may use 0/1 or True/False)
        for field in self.BOOLEAN_FIELDS:
            if hasattr(self, field):
                setattr(self, field, bool(getattr(self, field)))

        if not hasattr(self, 'smoothingKernel'):
            self.smoothingKernel = 0
        if not hasattr(self, 'confidenceWindow'):
            self.confidenceWindow = 5
        if not hasattr(self, 'confidenceThreshold'):
            self.confidenceThreshold = 0.0

        # Extract calibration maps from camera.left and camera.right
        # Maps are stored as map_x and map_y under each camera

        # Extract shared calibration data (imageSize and Q) from camera level
        # These are set via kwargs unpacking, but we can also get them from camera config
        if not hasattr(self, 'imageSize'):
            self.imageSize = kwargs.get('imageSize', None)
        if not hasattr(self, 'Q'):
            self.Q = kwargs.get('Q', None)
        
        # Initialize camera objects
        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        
        # Depth processing components (initialized in start())
        # Calibration maps are already set via kwargs (leftMapX, leftMapY, rightMapX, rightMapY, imageSize, Q)
        self.stereo = None
        self.depth_processor: Optional[DepthProcessor] = None
        
        # Runtime state
        self.connected = False
    
    def start(self):
        """Open cameras and initialize depth processor."""
        if self.connected:
            return
        
        print(f"{self.name}: Starting vision system...")
        
        # Initialize cameras
        self.left_camera = Camera(self.left['port'], CAMERA_CONFIG)
        self.right_camera = Camera(self.right['port'], CAMERA_CONFIG)
        
        # Open cameras
        try:
            self.left_camera.open()
            print(f"{self.name}: Left camera opened on port {self.left['port']}")
        except Exception as e:
            raise RuntimeError(f"{self.name}: Failed to open left camera: {e}")
        
        try:
            self.right_camera.open()
            print(f"{self.name}: Right camera opened on port {self.right['port']}")
        except Exception as e:
            self.left_camera.close()
            raise RuntimeError(f"{self.name}: Failed to open right camera: {e}")
        

        # Calibration maps are already set via kwargs unpacking
        # Use them directly (they're stored as leftMapX, leftMapY, etc.)
        print(f"{self.name}: Calibration data loaded from config")
        
        # Initialize stereo matcher using config settings
        block_size = self.blockSize
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        num_disparities = max(16, 16 * self.numDisparitiesK)
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.minDisparity,
            numDisparities=num_disparities,
            blockSize=max(3, block_size),
            preFilterCap=self.preFilterCap,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            disp12MaxDiff=self.disp12MaxDiff,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        
        self._refresh_depth_processor()
        self.connected = True
        print(f"{self.name}: Vision system started successfully")
    
    def stop(self):
        """Close cameras and cleanup resources."""
        if not self.connected:
            return
        
        print(f"{self.name}: Stopping vision system...")
        
        if self.left_camera:
            self.left_camera.close()
            self.left_camera = None
        
        if self.right_camera:
            self.right_camera.close()
            self.right_camera = None
        
        # Clear stereo matcher (calibration maps remain as they're from config)
        self.stereo = None
        self.depth_processor = None
        
        self.connected = False
        print(f"{self.name}: Vision system stopped")
    
    def read(self):
        """
        Capture frames, process depth, and return dictionary with depth_map and metadata.
        
        Returns:
            dict: {
                'depth_map': np.ndarray,
                'disparity_map': np.ndarray,
                'metadata': {
                    'timestamp': str,
                    'num_disparities': int
                }
            }
        """
        if not self.connected:
            raise RuntimeError(f"{self.name}: Not connected. Call start() first.")
        
        # Capture frames
        left_frame = self.left_camera.read_frame()
        right_frame = self.right_camera.read_frame()
        
        if left_frame is None or right_frame is None:
            empty = np.array([])
            return {
                'depth_map': empty,
                'disparity_map': empty,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_disparities': 0,
                    'error': 'Failed to capture frames'
                }
            }
        
        # Process depth pipeline
        try:
            if self.depth_processor is None:
                raise RuntimeError("Depth processor not initialized. Call start() first.")

            depth_map, disparity_map, metadata = self.depth_processor.process_frames(left_frame, right_frame)
            
            return {
                'depth_map': depth_map,
                'disparity_map': disparity_map,
                'metadata': metadata
            }
        except Exception as e:
            if self.debug_mode:
                print(f"{self.name}: Error processing depth: {e}")
            empty = np.array([])
            return {
                'depth_map': empty,
                'disparity_map': empty,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_disparities': 0,
                    'error': str(e)
                }
            }
    
    def calibrate(self, checkerboard=(7, 10), square_size=20.0, min_pairs=10):
        print("VISION/calibrate.py has been deprecated I hate kyle")
        return {}
        # """
        # Perform stereo calibration by capturing images from cameras and return updated config dictionary.
        #
        # Args:
        #     checkerboard: Tuple of (cols, rows) for checkerboard pattern
        #     square_size: Size of checkerboard squares in mm
        #     min_pairs: Minimum number of valid stereo pairs required for calibration
        #
        # Returns:
        #     dict: Dictionary with updated calibration data in the format expected by Calibrate.py.
        #           Structure: {self.name: {"left": {...}, "right": {...}, "imageSize": ..., "Q": ...}}
        # """
        # result = run_calibration(self, checkerboard=checkerboard, square_size=square_size, min_pairs=min_pairs)
        # self._refresh_depth_processor()
        # return result
    
    def debug(self):
        """
        Debug mode that continuously displays the depth map.
        Press 'o' to start recording, 'p' to stop recording, 'q' to quit.
        """
        if not self.connected:
            print(f"{self.name}: Cannot start debug mode. Vision system not connected. Call start() first.")
            return
        
        print(f"{self.name}: Starting debug mode - displaying depth map...")
        print(f"{self.name}: Press 'o' to start recording | 'p' to stop | 'q' to exit")
        
        window_name = f"{self.name} - Depth Map Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Video recording state
        video_writer = None
        recording = False
        video_counter = 0
        frame_size = None
        
        try:
            while True:
                # Get depth data
                result = self.read()
                depth_map = result.get('depth_map')
                metadata = result.get('metadata', {})
                
                # Check for errors
                if 'error' in metadata:
                    print(f"{self.name}: Error in debug mode: {metadata['error']}")
                    continue
                
                if depth_map is None or depth_map.size == 0:
                    print(f"{self.name}: No depth data available")
                    continue
                
                # Normalize depth map for visualization (0-255 range)
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_normalized.astype(np.uint8)
                
                # Apply colormap for better visualization
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                # Store frame size for video writer initialization
                if frame_size is None:
                    frame_size = (depth_colored.shape[1], depth_colored.shape[0])
                
                # Add metadata text overlay
                timestamp = metadata.get('timestamp', 'N/A')
                num_disp = metadata.get('num_disparities', 'N/A')
                cv2.putText(depth_colored, f"Timestamp: {timestamp}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(depth_colored, f"Num Disparities: {num_disp}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add recording indicator
                if recording:
                    cv2.circle(depth_colored, (depth_colored.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(depth_colored, "REC", (depth_colored.shape[1] - 70, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Write frame to video if recording
                if recording and video_writer is not None:
                    video_writer.write(depth_colored)
                
                # Display the depth map
                cv2.imshow(window_name, depth_colored)
                
                # Check for keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    if not recording:
                        # Start recording
                        video_counter += 1
                        filename = f"disparity_recording_{video_counter}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
                        recording = True
                        print(f"🔴 Recording started: {filename}")
                    else:
                        print("⚠️ Already recording. Press 'p' to stop first.")
                elif key == ord('p'):
                    if recording:
                        # Stop recording
                        recording = False
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
                        print(f"⏹️ Recording stopped: disparity_recording_{video_counter}.avi")
                    else:
                        print("⚠️ Not currently recording.")
                    
        except KeyboardInterrupt:
            print(f"\n{self.name}: Debug mode interrupted by user")
        except Exception as e:
            print(f"{self.name}: Error in debug mode: {e}")
        finally:
            # Clean up video writer if still recording
            if video_writer is not None:
                video_writer.release()
                print(f"⏹️ Recording stopped (cleanup): disparity_recording_{video_counter}.avi")
            
            cv2.destroyWindow(window_name)
            print(f"{self.name}: Debug mode ended")
    
    def __repr__(self):
        return f"<VISION name={self.name}, left_port={self.left['port']}, right_port={self.right['port']}, connected={self.connected}>"

    def _refresh_depth_processor(self):

        #check if left and right are in the config
        if not hasattr(self, 'left') or not hasattr(self, 'right'):
            self.depth_processor = None
            return
        #check if stereo is in the config
        if self.stereo is None:
            self.depth_processor = None
            return
        
        # Recreate the depth processor so it always uses the latest stereo matcher and calibration maps.
        self.depth_processor = DepthProcessor(
            self.left,
            self.right,
            self.stereo,
            getattr(self, 'Q', None),
            debug=self.debug_mode,
            downSample=getattr(self, 'downSample', 0),
            crop=getattr(self, 'crop', 0),
            nearCutoff=getattr(self, 'nearCutoff', 0),
            farCutoff=getattr(self, 'farCutoff', 0),
            useMorph=getattr(self, 'useMorph', False),
            morphIter=getattr(self, 'morphIter', 5),
            useWLS=getattr(self, 'useWLS', False),
            wlsLambda=getattr(self, 'wlsLambda', 8000.0),
            wlsSigma=getattr(self, 'wlsSigma', 1.5),
            smoothingKernel=getattr(self, 'smoothingKernel', 0),
            confidenceWindow=getattr(self, 'confidenceWindow', 5),
            confidenceThreshold=getattr(self, 'confidenceThreshold', 0.0),
        )

