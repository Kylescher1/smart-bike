#!/usr/bin/env python3
"""
Target Selection and Tracking Logic

Prioritizes and selects targets from multiple camera streams,
then commands turret to point at selected target.
"""

import time
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from src.hal.MultiCameraYOLO import Detection, CameraDetections
from src.hal.TurretGeometry import TurretGeometry
from src.hal.TurretController import TurretController


@dataclass
class Target:
    """Selected tracking target"""
    detection: Detection
    estimated_pan: float  # Estimated pan angle to point at target
    estimated_tilt: float  # Estimated tilt angle to point at target
    priority_score: float  # Higher = more important
    source: str  # 'fisheye_scout' or 'center_locked'


class TargetSelector:
    """
    Selects best target from multi-camera detections and maintains tracking.
    
    Strategy:
    1. If currently tracking a target (locked) → maintain lock and actively center it
    2. If target lost but visible in center camera → reacquire and center
    3. If no center target → scout from fisheye cameras (left/right)
    4. Point turret at selected target
    
    How Fisheye Cameras Help:
    - Left/Right fisheye cameras have wide FOV (~120° each) pointing outward
    - They detect targets outside the center camera's narrow FOV
    - When a target is detected in fisheye, the system estimates pan/tilt angles
    - Turret moves to point center camera at estimated target location
    - Once center camera sees the target, it takes over with precise tracking
    - This allows the system to find and track targets across a ~240° field of view
    """
    
    def __init__(self, geometry: TurretGeometry, controller: TurretController,
                 lock_threshold: float = 30.0, 
                 min_confidence: float = 0.4,
                 priority_classes: Optional[List[str]] = None,
                 primary_class: Optional[str] = 'person'):
        """
        Initialize target selector.
        
        Args:
            geometry: Turret geometry instance
            controller: Turret controller instance
            lock_threshold: Pixel error threshold for "locked" state
            min_confidence: Minimum detection confidence to consider
            priority_classes: List of class names to prioritize (e.g., ['person', 'car'])
            primary_class: Primary class to track (e.g., 'person') - gets highest priority
        """
        self.geometry = geometry
        self.controller = controller
        self.lock_threshold = lock_threshold
        self.min_confidence = min_confidence
        self.priority_classes = priority_classes or []
        self.primary_class = primary_class.lower() if primary_class else None
        
        # Active tracking parameters
        self.centering_gain = 0.3  # Proportional gain for centering (0.0-1.0)
        self.max_centering_speed = 5.0  # Maximum degrees per adjustment
        
        # Tracking state
        self.current_target: Optional[Target] = None
        self.locked = False
        self.lock_time = 0.0
        self.lost_time = 0.0
        self.lock_hold_duration = 0.5  # Hold lock for 0.5s after loss
        
    def select_target(self, all_detections: dict) -> Optional[Target]:
        """
        Select best target from all camera detections.
        
        Args:
            all_detections: Dict[camera_id, CameraDetections] from all cameras
            
        Returns:
            Target object or None if no suitable target
        """
        current_time = time.time()
        
        # Get detections from each camera
        center_dets = all_detections.get('center')
        left_dets = all_detections.get('left')
        right_dets = all_detections.get('right')
        
        # Strategy 1: Maintain lock on current target if visible in center camera
        if self.current_target and center_dets and center_dets.detections:
            center_target = self._find_best_center_detection(center_dets)
            if center_target:
                # Check if locked (error within threshold)
                error = self._compute_center_error(center_target, center_dets)
                if error < self.lock_threshold:
                    self.locked = True
                    self.lock_time = current_time
                    self.current_target = center_target
                    return center_target
                else:
                    # Not locked but visible, continue tracking
                    self.locked = False
                    self.current_target = center_target
                    return center_target
        
        # Strategy 2: Check if we recently lost lock (hold for a bit)
        if self.locked and (current_time - self.lock_time) < self.lock_hold_duration:
            # Give it a moment to reacquire
            if center_dets and center_dets.detections:
                center_target = self._find_best_center_detection(center_dets)
                if center_target:
                    self.current_target = center_target
                    return center_target
            # Still within hold period, return current target position estimate
            return self.current_target
        
        # Strategy 3: Lost lock, look for new target in center camera
        if center_dets and center_dets.detections:
            center_target = self._find_best_center_detection(center_dets)
            if center_target:
                self.locked = False
                self.current_target = center_target
                return center_target
        
        # Strategy 4: No center detections, scout from fisheye cameras
        fisheye_targets = []
        
        if left_dets and left_dets.detections:
            for det in left_dets.detections:
                if det.confidence < self.min_confidence:
                    continue
                target = self._create_fisheye_target(det, left_dets)
                if target:
                    fisheye_targets.append(target)
        
        if right_dets and right_dets.detections:
            for det in right_dets.detections:
                if det.confidence < self.min_confidence:
                    continue
                target = self._create_fisheye_target(det, right_dets)
                if target:
                    fisheye_targets.append(target)
        
        if fisheye_targets:
            # Select highest priority fisheye target
            best_fisheye = max(fisheye_targets, key=lambda t: t.priority_score)
            self.locked = False
            self.current_target = best_fisheye
            return best_fisheye
        
        # No targets found anywhere
        self.locked = False
        if not self.current_target or (current_time - self.lock_time) > 2.0:
            self.current_target = None
        return self.current_target
    
    def _find_best_center_detection(self, center_dets: CameraDetections) -> Optional[Target]:
        """Find best detection in center camera (largest, highest confidence)"""
        valid_dets = [d for d in center_dets.detections if d.confidence >= self.min_confidence]
        if not valid_dets:
            return None
        
        # Score based on size, confidence, and class priority
        def score_center_det(det: Detection) -> float:
            area = det.width * det.height
            conf = det.confidence
            class_name_lower = det.class_name.lower()
            
            # Primary class gets highest priority (10x bonus)
            if self.primary_class and class_name_lower == self.primary_class:
                priority_bonus = 10.0
            # Other priority classes get 2x bonus
            elif class_name_lower in [c.lower() for c in self.priority_classes]:
                priority_bonus = 2.0
            else:
                priority_bonus = 1.0
            
            return area * conf * priority_bonus
        
        best_det = max(valid_dets, key=score_center_det)
        
        # Calculate target angles to center the detection
        pan, tilt = self._calculate_centering_angles(best_det, center_dets)
        
        return Target(
            detection=best_det,
            estimated_pan=pan,
            estimated_tilt=tilt,
            priority_score=score_center_det(best_det),
            source='center_locked'
        )
    
    def _create_fisheye_target(self, det: Detection, cam_dets: CameraDetections) -> Optional[Target]:
        """Create target from fisheye detection with estimated angles"""
        try:
            # Estimate pan/tilt angles to point at this detection
            pan, tilt = self.geometry.estimate_target_angles(
                det.camera_id,
                det.center_x,
                det.center_y,
                cam_dets.frame_width,
                cam_dets.frame_height
            )
            
            # Score based on size, confidence, and class priority
            area = det.width * det.height
            class_name_lower = det.class_name.lower()
            
            # Primary class gets highest priority (10x bonus)
            if self.primary_class and class_name_lower == self.primary_class:
                priority_bonus = 10.0
            # Other priority classes get 2x bonus
            elif class_name_lower in [c.lower() for c in self.priority_classes]:
                priority_bonus = 2.0
            else:
                priority_bonus = 1.0
            
            score = area * det.confidence * priority_bonus
            
            return Target(
                detection=det,
                estimated_pan=pan,
                estimated_tilt=tilt,
                priority_score=score,
                source='fisheye_scout'
            )
        except Exception as e:
            print(f"Error creating fisheye target: {e}")
            return None
    
    def _compute_center_error(self, target: Target, center_dets: CameraDetections) -> float:
        """Compute pixel error from center of frame"""
        det = target.detection
        center_x = center_dets.frame_width / 2.0
        center_y = center_dets.frame_height / 2.0
        
        error_x = det.center_x - center_x
        error_y = det.center_y - center_y
        
        return np.sqrt(error_x**2 + error_y**2)
    
    def _calculate_centering_angles(self, detection: Detection, center_dets: CameraDetections) -> Tuple[float, float]:
        """
        Calculate pan/tilt angles to center a detection in the center camera frame.
        
        Uses proportional control to adjust turret position based on pixel error.
        """
        # Get current turret position
        current_pan, current_tilt = self.controller.get_position()
        
        # Calculate pixel error from center
        center_x = center_dets.frame_width / 2.0
        center_y = center_dets.frame_height / 2.0
        
        error_x = detection.center_x - center_x  # Positive = target is right of center
        error_y = detection.center_y - center_y  # Positive = target is below center
        
        # Convert pixel error to angle adjustment
        # Approximate: 1 pixel ≈ 0.1 degrees at 640x480 (adjust based on FOV)
        # Assuming ~60° horizontal FOV: 640 pixels = 60°, so 1 pixel ≈ 0.094°
        fov_h = 60.0  # Approximate horizontal FOV
        fov_v = 45.0  # Approximate vertical FOV
        
        angle_per_pixel_h = fov_h / center_dets.frame_width
        angle_per_pixel_v = fov_v / center_dets.frame_height
        
        # Calculate desired angle adjustment (proportional control)
        pan_adjustment = error_x * angle_per_pixel_h * self.centering_gain
        tilt_adjustment = -error_y * angle_per_pixel_v * self.centering_gain  # Negative because tilt is inverted
        
        # Limit maximum adjustment speed
        pan_adjustment = np.clip(pan_adjustment, -self.max_centering_speed, self.max_centering_speed)
        tilt_adjustment = np.clip(tilt_adjustment, -self.max_centering_speed, self.max_centering_speed)
        
        # Calculate new target angles
        target_pan = current_pan + pan_adjustment
        target_tilt = current_tilt + tilt_adjustment
        
        return target_pan, target_tilt
    
    def is_locked(self) -> bool:
        """Check if currently locked on target"""
        return self.locked
    
    def get_current_target(self) -> Optional[Target]:
        """Get current target being tracked"""
        return self.current_target

