#!/usr/bin/env python3
"""
Turret System - 3D Object Tracking

This script controls the turret system. Its goal is to use 3 cameras, 
2 servos and a Time of Flight sensor to detect objects, track them,
get depth and return 3D positions.

Structure:
- 2 static fisheye cameras (left/right) at z=0, pointing 30° outward
- 1 center camera on turret platform (active tracking)
- 1 ToF sensor for depth measurement
- Arduino controls servos via serial

The fisheye cameras scout for targets and tell the turret where to point.
The center camera actively tracks the selected target.
When locked on target, ToF sensor measures depth.
3D positions are computed and mapped to spherical coordinates (unit sphere).

Usage:
    from src.hal.Turret import Turret
    
    turret = Turret(
        port='COM3',
        cameras={'left': 0, 'right': 1, 'center': 2}
    )
    turret.start()
    
    # Read detections with 3D positions
    output = turret.read()
    for det3d in output.detections_3d:
        az, el, dist = det3d.position_spherical
        print(f"{det3d.detection.class_name} at azimuth={az:.1f}°, "
              f"elevation={el:.1f}°, distance={dist:.2f}in")
    
    turret.stop()

See turret_demo.py for complete example.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import supporting classes
from .TurretGeometry import TurretGeometry
from .TurretController import TurretController
from .PositionCalculator import Position3D, PositionCalculator
from .MultiCameraYOLO import MultiCameraYOLO, Detection
from .TargetSelector import TargetSelector, Target


@dataclass
class TurretPose:
    """Current turret pose (pan and tilt angles)"""
    pan_angle: float  # Bottom servo angle (0-180°)
    tilt_angle: float  # Top servo angle (60-120°)


@dataclass
class TurretOutput:
    """Output from turret system containing detections and state"""
    detections_3d: List[Position3D]  # 3D positions with depth
    all_detections: List[Detection]  # All 2D detections from all cameras
    current_target: Optional[Target]  # Active target being tracked
    turret_pose: TurretPose  # Current pan/tilt angles
    is_locked: bool  # Locked on target?
    timestamp: float


class Turret:
    """
    Main turret system class that integrates all subsystems.
    
    Coordinates:
    - MultiCameraYOLO for object detection
    - TargetSelector for target prioritization and tracking
    - TurretController for servo control
    - PositionCalculator for 3D position computation
    - TurretGeometry for coordinate transformations
    """
    
    def __init__(self, 
                 port: str,
                 cameras: Dict[str, int],
                 yolo_model: str = 'yolo11n.pt',
                 conf_threshold: float = 0.5,
                 target_classes: Optional[List[str]] = None,
                 tracking_enabled: bool = True,
                 lock_threshold: float = 30.0):
        """
        Initialize turret system.
        
        Args:
            port: Serial port for Arduino (e.g., 'COM3', '/dev/ttyUSB0')
            cameras: Dict mapping 'left', 'right', 'center' to camera indices
            yolo_model: Path to YOLO model file (relative to yolo/models/)
            conf_threshold: Detection confidence threshold (0.0-1.0)
            target_classes: List of class names to track (None = all classes)
            tracking_enabled: Whether to actively track targets
            lock_threshold: Pixel error threshold for "locked" state
        """
        self.port = port
        self.cameras = cameras
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.tracking_enabled = tracking_enabled
        self.lock_threshold = lock_threshold
        
        # Subsystem instances (initialized in start())
        self.geometry: Optional[TurretGeometry] = None
        self.controller: Optional[TurretController] = None
        self.yolo: Optional[MultiCameraYOLO] = None
        self.target_selector: Optional[TargetSelector] = None
        self.position_calculator: Optional[PositionCalculator] = None
        
        # State
        self.running = False
        self.position_buffer: List[Position3D] = []  # Rolling buffer for 3D positions
        self.buffer_max_age = 2.0  # Keep positions for 2 seconds
        
    def start(self):
        """Initialize and start all subsystems"""
        print("Starting turret system...")
        
        # Initialize geometry
        self.geometry = TurretGeometry()
        
        # Initialize controller
        self.controller = TurretController(port=self.port)
        if not self.controller.connect():
            raise RuntimeError(f"Failed to connect to Arduino on {self.port}")
        print("  ✓ Connected to Arduino")
        
        # Initialize YOLO detection system
        self.yolo = MultiCameraYOLO(
            camera_indices=self.cameras,
            yolo_model=self.yolo_model,
            conf_threshold=self.conf_threshold,
            target_classes=self.target_classes
        )
        self.yolo.start()
        print("  ✓ Multi-camera YOLO system started")
        
        # Initialize target selector with person as primary class
        self.target_selector = TargetSelector(
            geometry=self.geometry,
            controller=self.controller,
            lock_threshold=self.lock_threshold,
            min_confidence=self.conf_threshold * 0.8,  # Slightly lower for selection
            priority_classes=self.target_classes,
            primary_class='person'  # Prioritize person class above all others
        )
        print("  ✓ Target selector initialized")
        
        # Initialize position calculator
        self.position_calculator = PositionCalculator(
            geometry=self.geometry,
            controller=self.controller
        )
        print("  ✓ Position calculator initialized")
        
        # Home the turret
        self.controller.home()
        print("  ✓ Turret homed")
        
        self.running = True
        print("Turret system started successfully")
    
    def read(self) -> TurretOutput:
        """
        Read current turret state and detections.
        
        Returns:
            TurretOutput containing detections, 3D positions, and state
        """
        if not self.running:
            raise RuntimeError("Turret system not started. Call start() first.")
        
        current_time = time.time()
        
        # Read all detections from cameras
        all_detections_dict = self.yolo.read_all_detections()
        all_detections_list = self.yolo.get_all_detections_list()
        
        # Select target
        selected_target = None
        if self.tracking_enabled:
            selected_target = self.target_selector.select_target(all_detections_dict)
            
            # Move turret to target if we have one
            if selected_target:
                self.controller.move_to(
                    selected_target.estimated_pan,
                    selected_target.estimated_tilt
                )
        
        # Compute 3D positions for center camera detections when locked
        detections_3d = []
        is_locked = self.target_selector.is_locked()
        
        if is_locked and selected_target:
            # Get center camera detections
            center_dets = all_detections_dict.get('center')
            if center_dets and center_dets.detections:
                # Compute 3D position for the locked target
                locked_detection = selected_target.detection
                if locked_detection.camera_id == 'center':
                    pos3d = self.position_calculator.compute_3d_position(
                        locked_detection,
                        center_dets.frame_width,
                        center_dets.frame_height,
                        force_depth_read=False
                    )
                    if pos3d:
                        detections_3d.append(pos3d)
                        # Add to rolling buffer
                        self.position_buffer.append(pos3d)
        
        # Clean up old positions from buffer
        self.position_buffer = [
            p for p in self.position_buffer
            if (current_time - p.timestamp) < self.buffer_max_age
        ]
        
        # Get current turret pose
        pan, tilt = self.controller.get_position()
        turret_pose = TurretPose(pan_angle=pan, tilt_angle=tilt)
        
        return TurretOutput(
            detections_3d=detections_3d,
            all_detections=all_detections_list,
            current_target=selected_target,
            turret_pose=turret_pose,
            is_locked=is_locked,
            timestamp=current_time
        )
    
    def read_buffer(self, max_age: float = 1.0) -> List[Position3D]:
        """
        Read 3D positions from rolling buffer within time window.
        
        Args:
            max_age: Maximum age of positions to include (seconds)
            
        Returns:
            List of Position3D objects within time window
        """
        current_time = time.time()
        return [
            p for p in self.position_buffer
            if (current_time - p.timestamp) <= max_age
        ]
    
    def stop(self):
        """Stop and cleanup all subsystems"""
        print("Stopping turret system...")
        self.running = False
        
        if self.yolo:
            self.yolo.stop()
        
        if self.controller:
            self.controller.disconnect()
        
        self.position_buffer.clear()
        print("Turret system stopped")


__all__ = [
    'Turret',
    'TurretOutput',
    'TurretPose',
    'Position3D',
    'TurretGeometry',
    'TurretController'
]
