#!/usr/bin/env python3
"""
Multi-Camera YOLO Detection Manager

Manages 3 cameras running YOLO detection in parallel:
- Left fisheye (peripheral detection)
- Right fisheye (peripheral detection)
- Center camera (active tracking)
"""

import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from yolo.live_demo import YOLODetector
from src.hal.cam.Camera import Camera, CAMERA_CONFIG

# Try to import RKNN detector
try:
    from yolo.rknn_detector import RKNNYOLODetector
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    RKNNYOLODetector = None


@dataclass
class Detection:
    """Single object detection"""
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    center_x: float
    center_y: float
    width: float
    height: float
    camera_id: str  # 'left', 'right', 'center'
    timestamp: float


@dataclass
class CameraDetections:
    """Detections from one camera"""
    camera_id: str
    detections: List[Detection]
    timestamp: float
    frame_width: int
    frame_height: int


class MultiCameraYOLO:
    """
    Manages 3 YOLO detectors running on different cameras.
    
    Fisheye cameras (left/right) scout for targets.
    Center camera actively tracks selected target.
    """
    
    def __init__(self, camera_indices: Dict[str, int], 
                 yolo_model: str = "yolo11n.pt",
                 conf_threshold: float = 0.5,
                 target_classes: Optional[List[str]] = None):
        """
        Initialize multi-camera YOLO system.
        
        Args:
            camera_indices: Dict mapping 'left', 'right', 'center' to camera indices
            yolo_model: Path to YOLO weights
            conf_threshold: Detection confidence threshold
            target_classes: List of class names to detect (None = all classes)
        """
        self.camera_indices = camera_indices
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        
        # Camera and detector instances
        self.cameras: Dict[str, Camera] = {}
        self.detectors: Dict[str, YOLODetector] = {}
        
        # Latest detections from each camera
        self.latest_detections: Dict[str, CameraDetections] = {}
        self.detection_locks: Dict[str, threading.Lock] = {
            'left': threading.Lock(),
            'right': threading.Lock(),
            'center': threading.Lock()
        }
        
        # Control flags
        self.running = False
        self.threads: Dict[str, threading.Thread] = {}
        
    def start(self):
        """Initialize cameras and start detection threads"""
        print("Starting multi-camera YOLO system...")
        
        # Find model path
        model_path = Path(PROJECT_ROOT) / "yolo" / "models" / self.yolo_model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize each camera
        for cam_id in ['left', 'right', 'center']:
            if cam_id not in self.camera_indices:
                print(f"Warning: {cam_id} camera index not provided, skipping")
                continue
            
            camera_index = self.camera_indices[cam_id]
            print(f"Initializing {cam_id} camera (index {camera_index})...")
            
            # Configure camera (optimized for speed)
            config = CAMERA_CONFIG.copy()
            config.update({
                "width": 640,   # Can reduce to 320 for even faster processing
                "height": 480,  # Can reduce to 240 for even faster processing
                "fps": 30,
                "fourcc": "MJPG"
            })
            
            # Create camera
            camera = Camera(index=camera_index, config=config)
            camera.open()
            self.cameras[cam_id] = camera
            
            # Choose detector based on model file extension
            use_rknn = str(model_path).endswith('.rknn') and RKNN_AVAILABLE
            
            if use_rknn:
                print(f"  Using RKNN detector for {cam_id} camera")
                # RKNN models are compiled for specific input sizes
                # yolov8n.rknn and yolo11n.rknn are compiled for 640x640
                # Must match the model's compiled size exactly
                inference_size = 640  # Model was compiled for this size
                detector = RKNNYOLODetector(
                    name=f"{cam_id.title()}Detector",
                    camera=camera,
                    weights=str(model_path),
                    conf=self.conf_threshold,
                    imgsz=inference_size
                )
            else:
                if str(model_path).endswith('.rknn'):
                    print(f"  Warning: RKNN model specified but RKNN not available. Falling back to Ultralytics.")
                detector = YOLODetector(
                    name=f"{cam_id.title()}Detector",
                    camera=camera,
                    weights=str(model_path),
                    conf=self.conf_threshold,
                    imgsz=640
                )
            
            detector.start()
            self.detectors[cam_id] = detector
            
            print(f"  {cam_id} camera ready")
        
        # Start detection reading threads
        self.running = True
        for cam_id in self.detectors.keys():
            thread = threading.Thread(
                target=self._detection_loop,
                args=(cam_id,),
                daemon=True
            )
            thread.start()
            self.threads[cam_id] = thread
        
        print("Multi-camera YOLO system started")
    
    def _detection_loop(self, camera_id: str):
        """Background thread that reads detections from one camera"""
        detector = self.detectors[camera_id]
        
        while self.running:
            try:
                # Read detection result
                result = detector.read()
                if result is None:
                    time.sleep(0.01)
                    continue
                
                # Process detections
                detections = []
                if result.detections:
                    for det in result.detections:
                        # Filter by target classes if specified
                        if self.target_classes is not None:
                            class_name = getattr(det, 'label', getattr(det, 'class_name', '')).lower()
                            if not any(tc.lower() in class_name for tc in self.target_classes):
                                continue
                        
                        # Extract detection info
                        x1, y1, x2, y2 = det.bbox
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Handle both YOLODetector (label) and RKNNYOLODetector (label) formats
                        class_name = getattr(det, 'label', getattr(det, 'class_name', 'unknown'))
                        
                        detections.append(Detection(
                            class_name=class_name,
                            class_id=getattr(det, 'class_id', -1),
                            confidence=getattr(det, 'confidence', 0.0),
                            bbox=(x1, y1, x2, y2),
                            center_x=center_x,
                            center_y=center_y,
                            width=width,
                            height=height,
                            camera_id=camera_id,
                            timestamp=time.time()
                        ))
                
                # Get frame dimensions
                frame = result.frame
                frame_height, frame_width = frame.shape[:2] if frame is not None else (480, 640)
                
                # Store latest detections
                camera_detections = CameraDetections(
                    camera_id=camera_id,
                    detections=detections,
                    timestamp=time.time(),
                    frame_width=frame_width,
                    frame_height=frame_height
                )
                
                with self.detection_locks[camera_id]:
                    self.latest_detections[camera_id] = camera_detections
                
            except Exception as e:
                print(f"Error in {camera_id} detection loop: {e}")
                time.sleep(0.1)
    
    def read_detections(self, camera_id: str) -> Optional[CameraDetections]:
        """
        Read latest detections from a specific camera.
        
        Args:
            camera_id: 'left', 'right', or 'center'
            
        Returns:
            CameraDetections object or None if no detections yet
        """
        with self.detection_locks[camera_id]:
            return self.latest_detections.get(camera_id)
    
    def read_all_detections(self) -> Dict[str, CameraDetections]:
        """
        Read latest detections from all cameras.
        
        Returns:
            Dict mapping camera_id to CameraDetections
        """
        result = {}
        for cam_id in self.detectors.keys():
            det = self.read_detections(cam_id)
            if det is not None:
                result[cam_id] = det
        return result
    
    def get_all_detections_list(self) -> List[Detection]:
        """
        Get a flat list of all current detections from all cameras.
        
        Returns:
            List of Detection objects
        """
        all_dets = []
        for cam_dets in self.read_all_detections().values():
            all_dets.extend(cam_dets.detections)
        return all_dets
    
    def stop(self):
        """Stop all cameras and detectors"""
        print("Stopping multi-camera YOLO system...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads.values():
            thread.join(timeout=1.0)
        
        # Stop detectors
        for detector in self.detectors.values():
            detector.stop()
        
        # Close cameras
        for camera in self.cameras.values():
            camera.close()
        
        print("Multi-camera YOLO system stopped")

