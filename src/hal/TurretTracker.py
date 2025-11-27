"""
Turret Tracker Module

Integrates VISION system with TurretControl to automatically track and center
detected objects using servo control.
"""

import time
import threading
from typing import Optional, Dict, List, Tuple
import numpy as np

from .TurretControl import TurretControl
from .VISION.VISION_UPGRADE import VISION


class TurretTracker:
    """
    Tracks objects detected by VISION system and centers them using turret servos.
    
    Features:
    - Automatic object selection (largest, highest confidence, or specific class)
    - Smooth tracking with PID control
    - Multiple tracking modes (largest, highest confidence, specific class)
    - Deadzone to prevent jitter
    """
    
    def __init__(self, vision: VISION, turret: TurretControl,
                 tracking_mode: str = "largest",
                 target_class: Optional[str] = None,
                 min_confidence: float = 0.3,
                 max_tracking_distance: float = 0.5,
                 camera_config: Optional[Dict] = None):
        """
        Initialize turret tracker.
        
        Args:
            vision: VISION system instance (must be started)
            turret: TurretControl instance (must be connected)
            tracking_mode: "largest", "highest_confidence", or "class"
            target_class: Class name to track (required if mode="class")
            min_confidence: Minimum confidence threshold for tracking
            max_tracking_distance: Maximum angular distance to track (degrees)
        """
        self.vision = vision
        self.turret = turret
        
        # Tracking parameters
        self.tracking_mode = tracking_mode
        self.target_class = target_class
        self.min_confidence = min_confidence
        self.max_tracking_distance = max_tracking_distance
        
        # Load camera intrinsics for accurate angle calculation
        self.camera_config = camera_config
        self.right_K = None  # Camera matrix for right camera (rectified)
        self._load_camera_intrinsics()
        
        # Tracking state
        self.tracking = False
        self.track_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Currently tracked object
        self.tracked_object_id: Optional[int] = None
        self.last_track_time = 0.0
        self.track_timeout = 1.0  # Lose track if no detection for 1 second
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.tracking_stats = {
            'frames_processed': 0,
            'objects_tracked': 0,
            'track_lost_count': 0
        }
    
    def _load_camera_intrinsics(self):
        """Load camera intrinsics from config for accurate angle calculation."""
        if self.camera_config is None:
            # Try to get from vision system's config
            if hasattr(self.vision, 'config'):
                self.camera_config = self.vision.config
            else:
                return
        
        try:
            right_cfg = self.camera_config.get('right', {})
            
            # Try to get rectified camera matrix (newK or from P matrix)
            if 'newK' in right_cfg:
                self.right_K = np.asarray(right_cfg['newK'], dtype=np.float64)
            elif 'P' in right_cfg:
                # Extract K from P matrix (first 3 columns)
                P = np.asarray(right_cfg['P'], dtype=np.float64)
                self.right_K = P[:, :3]
            elif 'K' in right_cfg:
                # Use original K if newK/P not available
                self.right_K = np.asarray(right_cfg['K'], dtype=np.float64)
            
            if self.right_K is not None:
                print(f"TurretTracker: Loaded camera intrinsics - fx={self.right_K[0,0]:.1f}, fy={self.right_K[1,1]:.1f}, "
                      f"cx={self.right_K[0,2]:.1f}, cy={self.right_K[1,2]:.1f}")
        except Exception as e:
            print(f"TurretTracker: Warning - Could not load camera intrinsics: {e}")
            self.right_K = None
    
    def start_tracking(self):
        """Start automatic tracking thread."""
        if self.tracking:
            return
        
        if not self.vision.connected:
            raise RuntimeError("VISION system must be started before tracking")
        
        if not self.turret.connected:
            raise RuntimeError("TurretControl must be connected before tracking")
        
        self.tracking = True
        self.stop_event.clear()
        self.track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.track_thread.start()
        print("✅ TurretTracker: Tracking started")
    
    def stop_tracking(self):
        """Stop automatic tracking."""
        if not self.tracking:
            return
        
        self.tracking = False
        self.stop_event.set()
        
        if self.track_thread and self.track_thread.is_alive():
            self.track_thread.join(timeout=2.0)
        
        # Move to home position
        self.turret.go_home()
        print("TurretTracker: Tracking stopped")
    
    def _select_target_object(self, objects: List[Dict]) -> Optional[Dict]:
        """
        Select target object based on tracking mode.
        
        Args:
            objects: List of detected objects from vision.read()
        
        Returns:
            Selected object dict or None
        """
        if not objects:
            return None
        
        # Filter for person class only
        person_objects = [obj for obj in objects if obj.get('type', '').lower() == 'person']
        if not person_objects:
            return None
        
        # Filter by confidence
        valid_objects = [obj for obj in person_objects if obj.get('confidence', 0.0) >= self.min_confidence]
        if not valid_objects:
            return None
        
        # Select highest confidence person
        return max(valid_objects, key=lambda obj: obj.get('confidence', 0.0))
    
    def _bbox_to_angles(self, bbox: List[int]) -> Tuple[float, float]:
        """
        Convert bounding box to turret angles using pixel coordinates.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Tuple of (theta_deg, alpha_deg) - horizontal and vertical angles relative to camera center
        
        Note: Camera center = turret home position. Angles are calculated directly from
        pixel coordinates using the same formula as VISION system:
        - theta = ((center_x - w/2) / w) * fov_h  (horizontal)
        - alpha = ((center_y - h/2) / h) * fov_v  (vertical)
        """
        if len(bbox) != 4:
            return (0.0, 0.0)
        
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Get frame dimensions from vision system (using right camera)
        try:
            with self.vision.frame_lock:
                if self.vision.last_right_frame is not None:
                    h, w = self.vision.last_right_frame.shape[:2]
                elif self.vision.last_left_frame is not None:
                    # Fallback to left if right not available
                    h, w = self.vision.last_left_frame.shape[:2]
                else:
                    # Default dimensions if frame not available
                    h, w = 480, 640
        except:
            h, w = 480, 640  # Default fallback
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Convert pixel coordinates to angles
        # For rectified fisheye images, use camera intrinsics if available
        # Otherwise use FOV-based linear approximation
        if self.right_K is not None:
            # Use camera intrinsics for accurate angle calculation
            fx = self.right_K[0, 0]  # Focal length x
            fy = self.right_K[1, 1]  # Focal length y
            # For rectified images, principal point is typically at image center
            cx = w / 2.0  # Use image center, not K matrix cx (rectified images are centered)
            cy = h / 2.0  # Use image center, not K matrix cy
            
            # Convert to normalized camera coordinates (relative to principal point)
            x_norm = (center_x - cx) / fx
            y_norm = (center_y - cy) / fy
            
            # Convert to angles using atan (for pinhole/rectified model)
            # atan gives accurate angles for rectified fisheye images
            theta_rad = np.arctan(x_norm)  # Horizontal angle
            alpha_rad = np.arctan(y_norm)  # Vertical angle
            
            # Convert to degrees and flip (object right = turret move left)
            theta_deg = -np.degrees(theta_rad)
            alpha_deg = -np.degrees(alpha_rad)
        else:
            # Fallback to linear FOV model if intrinsics not available
            fov_h = getattr(self.vision, 'fov_horizontal', 126.0)  # degrees
            fov_v = getattr(self.vision, 'fov_vertical', 101.62)  # degrees
            
            # Linear approximation (less accurate at edges)
            theta_deg = -((center_x - w / 2.0) / w) * fov_h
            alpha_deg = -((center_y - h / 2.0) / h) * fov_v
        
        return (theta_deg, alpha_deg)
    
    def _tracking_loop(self):
        """Main tracking loop running in background thread."""
        print("TurretTracker: Tracking loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get latest vision data
                vision_data = self.vision.read()
                objects = vision_data.get('objects', [])
                
                self.tracking_stats['frames_processed'] += 1
                
                # Debug: Print every frame for now
                print(f"TurretTracker: 🔄 Loop iteration {self.tracking_stats['frames_processed']} - {len(objects)} objects")
                
                # Select target object
                target = self._select_target_object(objects)
                print(f"TurretTracker: 🎯 Selected target: {target.get('id') if target else None}")
                
                current_time = time.time()
                
                if target is None:
                    # No valid target found
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ No valid target selected from {len(objects)} objects")
                    # Check if we should lose track
                    if self.tracked_object_id is not None:
                        if (current_time - self.last_track_time) > self.track_timeout:
                            # Lost track
                            self.tracked_object_id = None
                            self.tracking_stats['track_lost_count'] += 1
                            print(f"TurretTracker: ❌ Lost track (timeout)")
                    time.sleep(0.033)  # ~30 Hz update rate
                    continue
                
                # Check if this is a new object or same as before
                obj_id = target.get('id', None)
                
                # If we have a tracked object, prefer to keep tracking it
                if self.tracked_object_id is not None:
                    # Look for the same object ID
                    same_obj = next((obj for obj in objects if obj.get('id') == self.tracked_object_id), None)
                    if same_obj is not None:
                        target = same_obj
                        obj_id = self.tracked_object_id
                
                # Get bbox directly from cached detections
                # Match by ID first, then fall back to largest detection
                bbox = None
                try:
                    with self.vision.frame_lock:
                        if hasattr(self.vision, 'last_detections_cache') and self.vision.last_detections_cache:
                            # Try to match by track_id or id
                            for det in self.vision.last_detections_cache:
                                det_id = det.get('track_id') or det.get('id')
                                if det_id == obj_id:
                                    bbox = det.get('bbox')
                                    break
                            
                            # If no match by ID, use the largest detection (matches our "largest" selection logic)
                            if bbox is None and self.tracking_mode == "largest":
                                largest_det = None
                                largest_area = 0
                                for det in self.vision.last_detections_cache:
                                    det_bbox = det.get('bbox', [])
                                    if len(det_bbox) == 4:
                                        area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                                        if area > largest_area:
                                            largest_area = area
                                            largest_det = det
                                if largest_det:
                                    bbox = largest_det.get('bbox')
                                    if self.tracking_stats['frames_processed'] % 30 == 0:
                                        print(f"TurretTracker: Using largest detection (area={largest_area})")
                except Exception as e:
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: Bbox lookup error: {e}")
                
                # Calculate angles from bbox
                if bbox is None or len(bbox) != 4:
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ No bbox found for ID={obj_id}, cache has {len(self.vision.last_detections_cache) if hasattr(self.vision, 'last_detections_cache') else 0} detections")
                    time.sleep(0.033)
                    continue
                
                # Calculate angles directly from bbox
                theta_deg, alpha_deg = self._bbox_to_angles(bbox)
                
                # Debug output (print every frame for now)
                print(f"TurretTracker: ✅ Target selected - ID={obj_id}, "
                      f"theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°")
                
                # Check if angles are zero (bbox lookup failed)
                if abs(theta_deg) < 0.01 and abs(alpha_deg) < 0.01:
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ Angles are zero - bbox lookup may have failed for ID={obj_id}")
                    time.sleep(0.033)
                    continue
                
                # Check if object is within tracking distance
                angular_distance = np.sqrt(theta_deg**2 + alpha_deg**2)
                if angular_distance > self.max_tracking_distance:
                    # Object too far, don't track
                    if self.tracking_stats['frames_processed'] % 30 == 0:
                        print(f"TurretTracker: ⚠️ Object too far (distance={angular_distance:.1f}° > max={self.max_tracking_distance}°)")
                    time.sleep(0.033)
                    continue
                
                # Update tracking state
                with self.lock:
                    self.tracked_object_id = obj_id
                    self.last_track_time = current_time
                
                # Debug: Print before sending command
                print(f"TurretTracker: 🎯 Calling center_object(theta={theta_deg:.2f}°, alpha={alpha_deg:.2f}°)")
                
                # Center turret on object
                try:
                    self.turret.center_object(theta_deg, alpha_deg)
                    # Don't call get_position() here - it would deadlock since center_object() holds the lock
                    print(f"TurretTracker: ✅ center_object() returned")
                except Exception as e:
                    print(f"TurretTracker: ❌ Error sending command: {e}")
                    import traceback
                    traceback.print_exc()
                
                self.tracking_stats['objects_tracked'] += 1
                
                # Print stats every 60 frames (~2 seconds at 30 Hz)
                if self.tracking_stats['frames_processed'] % 60 == 0:
                    print(f"TurretTracker: Tracking object ID={obj_id}, "
                          f"theta={theta_deg:.1f}°, alpha={alpha_deg:.1f}°, "
                          f"confidence={target.get('confidence', 0.0):.2f}")
                
                time.sleep(0.033)  # ~30 Hz update rate
                
            except Exception as e:
                print(f"TurretTracker: Error in tracking loop: {e}")
                time.sleep(0.1)
        
        print("TurretTracker: Tracking loop stopped")
    
    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        with self.lock:
            return self.tracking_stats.copy()
    
    def set_tracking_mode(self, mode: str, target_class: Optional[str] = None):
        """
        Change tracking mode.
        
        Args:
            mode: "largest", "highest_confidence", or "class"
            target_class: Class name (required if mode="class")
        """
        with self.lock:
            self.tracking_mode = mode
            self.target_class = target_class
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_tracking()

