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
    1. If currently tracking a target (locked) → maintain lock
    2. If target lost but visible in center camera → reacquire
    3. If no center target → scout from fisheye cameras (largest/closest)
    4. Point turret at selected target
    """
    
    def __init__(self, geometry: TurretGeometry, controller: TurretController,
                 lock_threshold: float = 30.0, 
                 min_confidence: float = 0.4,
                 priority_classes: Optional[List[str]] = None):
        """
        Initialize target selector.
        
        Args:
            geometry: Turret geometry instance
            controller: Turret controller instance
            lock_threshold: Pixel error threshold for "locked" state
            min_confidence: Minimum detection confidence to consider
            priority_classes: List of class names to prioritize (e.g., ['person', 'car'])
        """
        self.geometry = geometry
        self.controller = controller
        self.lock_threshold = lock_threshold
        self.min_confidence = min_confidence
        self.priority_classes = priority_classes or []
        
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
        
        # Score based on size and confidence
        def score_center_det(det: Detection) -> float:
            area = det.width * det.height
            conf = det.confidence
            priority_bonus = 2.0 if det.class_name.lower() in self.priority_classes else 1.0
            return area * conf * priority_bonus
        
        best_det = max(valid_dets, key=score_center_det)
        
        # For center camera, target angles are current servo positions
        pan, tilt = self.controller.get_position()
        
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
            
            # Score based on size and confidence
            area = det.width * det.height
            priority_bonus = 2.0 if det.class_name.lower() in self.priority_classes else 1.0
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
    
    def is_locked(self) -> bool:
        """Check if currently locked on target"""
        return self.locked
    
    def get_current_target(self) -> Optional[Target]:
        """Get current target being tracked"""
        return self.current_target

