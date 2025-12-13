#!/usr/bin/env python3
"""
Turret Geometry and Coordinate Transformations

Defines the physical geometry of the turret system and provides utilities
for coordinate transformations between camera frames and world space.

Coordinate System:
  - Origin: Turret rotation center (xy = 0, 0)
  - +X: Forward
  - +Y: Right  
  - +Z: Up
  - Units: inches
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class CameraInfo:
    """Information about a camera's position and orientation"""
    name: str
    position: np.ndarray  # (x, y, z) in inches
    rotation: Rotation    # Rotation from world frame to camera frame
    fov_horizontal: float  # Field of view in degrees
    fov_vertical: float
    is_fisheye: bool = False


class TurretGeometry:
    """
    Manages turret geometry and coordinate transformations.
    
    Camera Layout:
    - 2 fisheye cameras (left/right) at z=0, pointing outward 30°
    - 1 center camera on turret platform at z=5.1415"
    - 1 ToF/LIDAR sensor at z=3.8017"
    - Turret rotates about Z-axis (pan) and Y-axis (tilt)
    """
    
    # Geometry specifications (from user measurements)
    FISHEYE_RADIAL_DIST = 1.9929  # inches from origin
    FISHEYE_ANGLE = 110.0  # degrees from first line
    FISHEYE_OFFSET_DIST = 2.1044  # inches from angle point to lens center
    FISHEYE_Z = 0.0  # height of fisheye cameras
    
    TURRET_PLATFORM_Z = 2.7918  # inches
    LIDAR_Z_OFFSET = 1.0099  # inches from platform
    CAMERA_Z_OFFSET = 2.3497  # inches from platform
    
    FISHEYE_FOV_OUTWARD = 30.0  # degrees - fisheyes point 30° outward
    
    def __init__(self):
        """Initialize turret geometry with computed camera positions"""
        # Calculate fisheye camera positions
        # From origin, go 1.9929" at some angle, then 110° turn, then 2.1044" to lens
        # Assuming symmetric left/right configuration
        
        # Left fisheye (negative Y side)
        # First segment: assume it goes at 180° - offset angle
        base_angle_left = np.radians(180.0 - 30.0)  # Pointing left-back
        p1_left = np.array([
            np.cos(base_angle_left) * self.FISHEYE_RADIAL_DIST,
            np.sin(base_angle_left) * self.FISHEYE_RADIAL_DIST,
            self.FISHEYE_Z
        ])
        
        # Second segment: 110° turn, then 2.1044" to lens
        lens_angle_left = base_angle_left + np.radians(self.FISHEYE_ANGLE)
        left_pos = p1_left + np.array([
            np.cos(lens_angle_left) * self.FISHEYE_OFFSET_DIST,
            np.sin(lens_angle_left) * self.FISHEYE_OFFSET_DIST,
            0.0
        ])
        
        # Right fisheye (positive Y side) - mirror of left
        base_angle_right = np.radians(30.0)  # Pointing right-back
        p1_right = np.array([
            np.cos(base_angle_right) * self.FISHEYE_RADIAL_DIST,
            np.sin(base_angle_right) * self.FISHEYE_RADIAL_DIST,
            self.FISHEYE_Z
        ])
        
        lens_angle_right = base_angle_right - np.radians(self.FISHEYE_ANGLE)
        right_pos = p1_right + np.array([
            np.cos(lens_angle_right) * self.FISHEYE_OFFSET_DIST,
            np.sin(lens_angle_right) * self.FISHEYE_OFFSET_DIST,
            0.0
        ])
        
        # Camera rotations (pointing direction)
        # Left camera points 30° left of forward
        left_rotation = Rotation.from_euler('z', -self.FISHEYE_FOV_OUTWARD, degrees=True)
        
        # Right camera points 30° right of forward  
        right_rotation = Rotation.from_euler('z', self.FISHEYE_FOV_OUTWARD, degrees=True)
        
        # Center camera is on turret platform (position varies with pan/tilt)
        # At home position (pan=90°, tilt=90°), camera points forward
        center_base_pos = np.array([0.0, 0.0, self.TURRET_PLATFORM_Z + self.CAMERA_Z_OFFSET])
        
        # LIDAR position
        lidar_base_pos = np.array([0.0, 0.0, self.TURRET_PLATFORM_Z + self.LIDAR_Z_OFFSET])
        
        # Store camera info
        self.cameras = {
            'left': CameraInfo(
                name='left',
                position=left_pos,
                rotation=left_rotation,
                fov_horizontal=170.0,  # Typical fisheye FOV
                fov_vertical=170.0,
                is_fisheye=True
            ),
            'right': CameraInfo(
                name='right',
                position=right_pos,
                rotation=right_rotation,
                fov_horizontal=170.0,
                fov_vertical=170.0,
                is_fisheye=True
            ),
            'center': CameraInfo(
                name='center',
                position=center_base_pos,  # Base position, varies with pan/tilt
                rotation=Rotation.identity(),  # Updated based on servo angles
                fov_horizontal=60.0,  # Typical camera FOV
                fov_vertical=45.0,
                is_fisheye=False
            )
        }
        
        self.lidar_position = lidar_base_pos
        
    def get_camera_info(self, camera_id: str) -> CameraInfo:
        """Get information about a specific camera"""
        return self.cameras[camera_id]
    
    def estimate_target_angles(self, camera_id: str, pixel_x: float, pixel_y: float,
                              frame_width: int, frame_height: int) -> Tuple[float, float]:
        """
        Estimate pan/tilt angles to point turret at a detection from fisheye camera.
        
        Args:
            camera_id: 'left' or 'right'
            pixel_x, pixel_y: Detection center in pixels
            frame_width, frame_height: Frame dimensions
            
        Returns:
            (pan_angle, tilt_angle): Estimated servo angles in degrees
        """
        if camera_id not in ['left', 'right']:
            raise ValueError("Only fisheye cameras (left/right) need angle estimation")
        
        cam = self.cameras[camera_id]
        
        # Normalize pixel coordinates to -0.5 to +0.5
        norm_x = (pixel_x / frame_width) - 0.5
        norm_y = (pixel_y / frame_height) - 0.5
        
        # Estimate angles based on FOV (simple linear approximation for fisheye)
        # This is approximate - good enough to point turret in right direction
        horizontal_angle = norm_x * (cam.fov_horizontal / 2.0)
        vertical_angle = norm_y * (cam.fov_vertical / 2.0)
        
        # Add camera's base pointing direction
        if camera_id == 'left':
            base_pan = 90.0 - self.FISHEYE_FOV_OUTWARD  # 60°
        else:  # right
            base_pan = 90.0 + self.FISHEYE_FOV_OUTWARD  # 120°
        
        target_pan = base_pan + horizontal_angle
        target_tilt = 90.0 + vertical_angle  # 90° is horizontal
        
        return target_pan, target_tilt
    
    def get_center_camera_position(self, pan_angle: float, tilt_angle: float) -> Tuple[np.ndarray, Rotation]:
        """
        Get the current position and orientation of the center camera.
        
        Args:
            pan_angle: Bottom servo angle (0-180°, 90° is forward)
            tilt_angle: Top servo angle (60-120°, 90° is horizontal)
            
        Returns:
            (position, rotation): Camera position in world frame and rotation
        """
        # Convert servo angles to rotation
        # Pan: rotation around Z-axis (90° = forward = 0° rotation)
        pan_rot = Rotation.from_euler('z', -(pan_angle - 90.0), degrees=True)
        
        # Tilt: rotation around Y-axis (90° = horizontal = 0° rotation)
        tilt_rot = Rotation.from_euler('y', -(tilt_angle - 90.0), degrees=True)
        
        # Combined rotation: first tilt, then pan
        combined_rotation = pan_rot * tilt_rot
        
        # Position is at the base plus any offset from tilt
        # For simplicity, assume camera stays at fixed height (small tilt range)
        position = self.cameras['center'].position.copy()
        
        return position, combined_rotation
    
    def pixel_to_world_ray(self, pixel_x: float, pixel_y: float,
                          frame_width: int, frame_height: int,
                          pan_angle: float, tilt_angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a pixel in the center camera to a ray in world coordinates.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            frame_width, frame_height: Frame dimensions
            pan_angle, tilt_angle: Current servo angles
            
        Returns:
            (origin, direction): Ray origin and unit direction vector in world frame
        """
        # Get camera position and rotation
        cam_pos, cam_rot = self.get_center_camera_position(pan_angle, tilt_angle)
        
        # Normalize pixel to -0.5 to +0.5
        norm_x = (pixel_x / frame_width) - 0.5
        norm_y = (pixel_y / frame_height) - 0.5
        
        # Convert to camera angles (simple pinhole model)
        cam = self.cameras['center']
        h_angle = norm_x * (cam.fov_horizontal / 2.0)
        v_angle = norm_y * (cam.fov_vertical / 2.0)
        
        # Create ray in camera frame (camera looks along +X axis)
        # Ray direction based on angles
        ray_cam = np.array([
            1.0,
            np.tan(np.radians(h_angle)),
            np.tan(np.radians(v_angle))
        ])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)  # Normalize
        
        # Transform ray to world frame
        ray_world = cam_rot.apply(ray_cam)
        
        return cam_pos, ray_world
    
    def compute_3d_position(self, pixel_x: float, pixel_y: float,
                           frame_width: int, frame_height: int,
                           pan_angle: float, tilt_angle: float,
                           depth: float) -> np.ndarray:
        """
        Compute 3D world position from pixel coordinates and depth.
        
        Args:
            pixel_x, pixel_y: Detection center in pixels
            frame_width, frame_height: Frame dimensions
            pan_angle, tilt_angle: Current servo angles in degrees
            depth: Distance to object in inches (from ToF sensor)
            
        Returns:
            xyz: 3D position in world frame (inches)
        """
        origin, direction = self.pixel_to_world_ray(
            pixel_x, pixel_y, frame_width, frame_height, pan_angle, tilt_angle
        )
        
        # Position = origin + depth * direction
        position_3d = origin + depth * direction
        
        return position_3d
    
    @staticmethod
    def cartesian_to_spherical(xyz: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert cartesian coordinates to spherical (azimuth, elevation, distance).
        
        Args:
            xyz: 3D position (x, y, z)
            
        Returns:
            (azimuth, elevation, distance): 
                - azimuth: angle in XY plane from +X axis (degrees, 0-360)
                - elevation: angle from XY plane (degrees, -90 to +90)
                - distance: radial distance (same units as xyz)
        """
        x, y, z = xyz
        
        distance = np.linalg.norm(xyz)
        
        if distance < 1e-6:
            return 0.0, 0.0, 0.0
        
        # Azimuth: angle in XY plane from +X axis
        azimuth = np.degrees(np.arctan2(y, x))
        if azimuth < 0:
            azimuth += 360.0
        
        # Elevation: angle from XY plane
        elevation = np.degrees(np.arcsin(z / distance))
        
        return azimuth, elevation, distance
    
    @staticmethod
    def spherical_to_cartesian(azimuth: float, elevation: float, distance: float) -> np.ndarray:
        """
        Convert spherical coordinates to cartesian.
        
        Args:
            azimuth: angle in XY plane from +X axis (degrees)
            elevation: angle from XY plane (degrees)
            distance: radial distance
            
        Returns:
            xyz: 3D position
        """
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)
        
        return np.array([x, y, z])

