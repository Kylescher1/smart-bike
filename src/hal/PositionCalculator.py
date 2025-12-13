#!/usr/bin/env python3
"""
3D Position Calculator

Computes 3D world positions from center camera detections + ToF depth.
Maps positions to unit sphere (spherical coordinates).
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

from src.hal.TurretGeometry import TurretGeometry
from src.hal.TurretController import TurretController
from src.hal.MultiCameraYOLO import Detection


@dataclass
class Position3D:
    """3D position of a detected object"""
    detection: Detection
    position_xyz: np.ndarray  # Cartesian coordinates (x, y, z) in inches
    position_spherical: Tuple[float, float, float]  # (azimuth, elevation, distance)
    pan_angle: float  # Servo angle when detected
    tilt_angle: float  # Servo angle when detected
    depth: float  # ToF depth reading in inches
    timestamp: float
    has_valid_depth: bool
    
    def __repr__(self):
        az, el, dist = self.position_spherical
        return (f"Position3D({self.detection.class_name}, "
                f"xyz={self.position_xyz}, "
                f"spherical=(az={az:.1f}°, el={el:.1f}°, r={dist:.2f}in))")


class PositionCalculator:
    """
    Calculates 3D positions from center camera detections + ToF depth.
    
    Only works for center camera (where depth sensor is aimed).
    Fisheye cameras are for scouting only, no 3D position calculation.
    """
    
    def __init__(self, geometry: TurretGeometry, controller: TurretController,
                 depth_smoothing: int = 5, max_depth: float = 200.0):
        """
        Initialize position calculator.
        
        Args:
            geometry: Turret geometry instance
            controller: Turret controller for depth reading
            depth_smoothing: Number of depth samples to average
            max_depth: Maximum valid depth in inches
        """
        self.geometry = geometry
        self.controller = controller
        self.max_depth = max_depth
        
        # Depth smoothing buffer
        self.depth_buffer = deque(maxlen=depth_smoothing)
        self.last_depth_read = 0.0
        self.depth_read_interval = 0.1  # Read depth every 100ms
        
    def compute_3d_position(self, detection: Detection, 
                           frame_width: int, frame_height: int,
                           force_depth_read: bool = False) -> Optional[Position3D]:
        """
        Compute 3D position for a detection from center camera.
        
        Args:
            detection: Detection from center camera
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            force_depth_read: Force a new ToF reading (otherwise uses cached)
            
        Returns:
            Position3D object or None if depth unavailable
        """
        # Only works for center camera
        if detection.camera_id != 'center':
            return None
        
        # Get current servo angles
        pan_angle, tilt_angle = self.controller.get_position()
        
        # Read ToF depth
        current_time = time.time()
        depth = None
        
        if force_depth_read or (current_time - self.last_depth_read) > self.depth_read_interval:
            raw_depth = self.controller.get_tof_range()
            if raw_depth is not None and 0 < raw_depth < self.max_depth:
                self.depth_buffer.append(raw_depth)
                self.last_depth_read = current_time
        
        # Use smoothed depth if available
        if len(self.depth_buffer) > 0:
            depth = np.mean(self.depth_buffer)
        else:
            # No depth available
            return None
        
        # Compute 3D position
        try:
            xyz = self.geometry.compute_3d_position(
                detection.center_x,
                detection.center_y,
                frame_width,
                frame_height,
                pan_angle,
                tilt_angle,
                depth
            )
            
            # Convert to spherical coordinates
            azimuth, elevation, distance = self.geometry.cartesian_to_spherical(xyz)
            
            return Position3D(
                detection=detection,
                position_xyz=xyz,
                position_spherical=(azimuth, elevation, distance),
                pan_angle=pan_angle,
                tilt_angle=tilt_angle,
                depth=depth,
                timestamp=time.time(),
                has_valid_depth=True
            )
            
        except Exception as e:
            print(f"Error computing 3D position: {e}")
            return None
    
    def estimate_3d_position_no_depth(self, detection: Detection,
                                     frame_width: int, frame_height: int,
                                     assumed_depth: float = 24.0) -> Optional[Position3D]:
        """
        Estimate 3D position with assumed depth (for detections without ToF reading).
        
        Args:
            detection: Detection object
            frame_width, frame_height: Frame dimensions
            assumed_depth: Assumed distance in inches (default 24" = 2 feet)
            
        Returns:
            Position3D with estimated position (has_valid_depth=False)
        """
        if detection.camera_id != 'center':
            return None
        
        pan_angle, tilt_angle = self.controller.get_position()
        
        try:
            xyz = self.geometry.compute_3d_position(
                detection.center_x,
                detection.center_y,
                frame_width,
                frame_height,
                pan_angle,
                tilt_angle,
                assumed_depth
            )
            
            azimuth, elevation, distance = self.geometry.cartesian_to_spherical(xyz)
            
            return Position3D(
                detection=detection,
                position_xyz=xyz,
                position_spherical=(azimuth, elevation, distance),
                pan_angle=pan_angle,
                tilt_angle=tilt_angle,
                depth=assumed_depth,
                timestamp=time.time(),
                has_valid_depth=False
            )
            
        except Exception as e:
            print(f"Error estimating 3D position: {e}")
            return None
    
    def clear_depth_buffer(self):
        """Clear depth smoothing buffer (call when target changes)"""
        self.depth_buffer.clear()

