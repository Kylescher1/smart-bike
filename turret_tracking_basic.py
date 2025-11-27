"""
Basic Turret Tracking - Simple and Reliable
Keeps a person centered in frame using a 2-axis turret with camera mounted on top.

This is a simplified version that focuses on:
1. Detecting people in the camera frame
2. Calculating offset from center
3. Moving turret to center the person

Usage:
    python turret_tracking_basic.py

Requirements:
    - config.dill file with VISION configuration
    - Arduino/ESP32 connected to COM port (COM5 on Windows, /dev/ttyUSB0 on Linux)
    - Camera system initialized and working

The script will:
    - Connect to vision system and turret
    - Find the highest confidence person detection
    - Calculate pixel offset from image center
    - Move turret servos to center the person
    - Update at ~30 Hz

Press Ctrl+C to stop.
"""

import time
import sys
import dill
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.VISION.VISION_UPGRADE import VISION
from hal.TurretControl import TurretControl


class SimpleTurretTracker:
    """Simple turret tracker that keeps a person centered."""
    
    def __init__(self, vision, turret):
        self.vision = vision
        self.turret = turret
        
        # Get image dimensions (will be set when we get first frame)
        self.image_width = None
        self.image_height = None
        
        # Control parameters
        self.deadzone_pixels = 30  # Don't move if offset is less than this (increased to prevent oscillation)
        self.max_move_speed = 1.0  # Maximum degrees to move per update (reduced for smoother movement)
        self.kp = 0.15  # Proportional gain (reduced to prevent overshoot)
        
        # Damping - reduces movement as we get closer to center
        self.damping_factor = 0.5  # Multiply movement by this when close to center
        self.damping_threshold = 100  # Pixels - start damping when offset < this
        
        # Field of view (will use from config if available)
        self.fov_horizontal = 126.0  # degrees
        self.fov_vertical = 101.62  # degrees
    
    def get_image_size(self):
        """Get image dimensions from vision system."""
        debug = self.vision.debug()
        if debug.get('last_left_image') is not None:
            h, w = debug['last_left_image'].shape[:2]
            self.image_width = w
            self.image_height = h
            return True
        return False
    
    def find_person(self, objects):
        """Find the best person to track."""
        people = [obj for obj in objects if obj.get('type', '').lower() == 'person']
        if not people:
            return None
        
        # Return the person with highest confidence
        return max(people, key=lambda obj: obj.get('confidence', 0.0))
    
    def calculate_offset(self, person):
        """
        Calculate pixel offset of person from image center.
        
        Returns:
            (offset_x, offset_y) in pixels, or (None, None) if can't calculate
        """
        if self.image_width is None or self.image_height is None:
            if not self.get_image_size():
                return None, None
        
        # Try to get bbox from detections cache (most accurate)
        bbox = None
        person_id = person.get('id')
        
        try:
            with self.vision.frame_lock:
                if hasattr(self.vision, 'last_detections_cache') and self.vision.last_detections_cache:
                    # Find person by ID
                    for det in self.vision.last_detections_cache:
                        det_id = det.get('track_id') or det.get('id')
                        if det_id == person_id:
                            bbox = det.get('bbox')
                            if bbox and len(bbox) == 4:
                                break
                    
                    # If not found by ID, try to find largest person detection
                    if bbox is None:
                        largest_area = 0
                        for det in self.vision.last_detections_cache:
                            det_bbox = det.get('bbox', [])
                            if len(det_bbox) == 4:
                                area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                                if area > largest_area:
                                    largest_area = area
                                    bbox = det_bbox
        except Exception as e:
            # If bbox access fails, fall back to coordinate method
            pass
        
        if bbox and len(bbox) == 4:
            # We have bbox: [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            
            image_center_x = self.image_width / 2.0
            image_center_y = self.image_height / 2.0
            
            offset_x = center_x - image_center_x
            offset_y = center_y - image_center_y
            
            return offset_x, offset_y
        
        # Fallback: estimate from x, y, z coordinates
        # This uses the unit circle coordinates from vision system
        x = person.get('x', 0)
        y = person.get('y', 0)
        z = person.get('z', 1.0)
        
        # Normalize z to avoid division issues
        if abs(z) < 0.001:
            z = 1.0
        
        # Convert unit circle coordinates to angles
        theta_rad = np.arctan2(x, z)  # Horizontal angle
        alpha_rad = np.arctan2(y, z)  # Vertical angle
        
        # Convert angles to pixel offsets using FOV
        # Formula: pixel_offset = tan(angle) * (image_size / 2) / tan(FOV / 2)
        fov_h_rad = np.radians(self.fov_horizontal / 2.0)
        fov_v_rad = np.radians(self.fov_vertical / 2.0)
        
        offset_x = np.tan(theta_rad) * (self.image_width / 2.0) / np.tan(fov_h_rad)
        offset_y = np.tan(alpha_rad) * (self.image_height / 2.0) / np.tan(fov_v_rad)
        
        return offset_x, offset_y
    
    def pixels_to_servo_angles(self, offset_x, offset_y):
        """
        Convert pixel offset to servo angle changes with damping to prevent oscillation.
        
        Returns:
            (delta_s1, delta_s2) - servo angle changes in degrees
        """
        if self.image_width is None or self.image_height is None:
            return 0, 0
        
        # Calculate distance from center for damping
        distance_from_center = np.sqrt(offset_x**2 + offset_y**2)
        
        # Calculate angles based on FOV
        # Use atan2 to get angle from center, then scale by FOV
        # Formula: angle = atan2(offset, half_image_size) * (FOV / 180)
        
        # Horizontal angle (affects S2 - horizontal servo)
        # Positive offset_x means person is to the right, need to move turret right
        # Note: Sign is flipped in move_x calculation to match servo direction
        angle_horizontal = np.degrees(np.arctan2(offset_x, self.image_width / 2.0)) * (self.fov_horizontal / 180.0)
        
        # Vertical angle (affects S1 - vertical servo)
        # Positive offset_y means person is below center, need to move turret down (decrease S1)
        angle_vertical = np.degrees(np.arctan2(offset_y, self.image_height / 2.0)) * (self.fov_vertical / 180.0)
        
        # Apply proportional control
        move_x = -angle_horizontal * self.kp  # Negative to flip horizontal direction
        move_y = -angle_vertical * self.kp  # Negative because S1 decreases when moving down
        
        # Apply damping - reduce movement when close to center to prevent oscillation
        if distance_from_center < self.damping_threshold:
            # Linear damping: closer to center = less movement
            # Minimum damping is damping_factor, maximum is 1.0 (no damping)
            damping = max(self.damping_factor, (distance_from_center / self.damping_threshold))
            move_x *= damping
            move_y *= damping
        
        # Limit speed
        move_x = max(-self.max_move_speed, min(self.max_move_speed, move_x))
        move_y = max(-self.max_move_speed, min(self.max_move_speed, move_y))
        
        return move_y, move_x  # Note: S1 is vertical, S2 is horizontal
    
    def update(self):
        """Update turret position based on current detections."""
        # Get latest detections
        vision_data = self.vision.read()
        objects = vision_data.get('objects', [])
        
        # Find person
        person = self.find_person(objects)
        
        if person is None:
            return False
        
        # Calculate offset
        offset_x, offset_y = self.calculate_offset(person)
        
        if offset_x is None or offset_y is None:
            return False
        
        # Check deadzone
        if abs(offset_x) < self.deadzone_pixels and abs(offset_y) < self.deadzone_pixels:
            return True  # Already centered enough
        
        # Convert to servo angles
        delta_s1, delta_s2 = self.pixels_to_servo_angles(offset_x, offset_y)
        
        # Get current position
        current_s1, current_s2 = self.turret.get_position()
        
        # Calculate new position
        new_s1 = current_s1 + delta_s1
        new_s2 = current_s2 + delta_s2
        
        # Move turret
        self.turret.move_to_absolute(new_s1, new_s2)
        
        return True


def load_config():
    """Load configuration from config.dill."""
    config_path = "config.dill"
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def main():
    print("=" * 60)
    print("Basic Turret Tracking - Simple Person Tracking")
    print("=" * 60)
    
    # Load config
    print("\n📋 Loading configuration...")
    config = load_config()
    if config is None:
        print("❌ Failed to load config.dill")
        return
    
    # Find vision config
    vision_config = None
    for key, value in config.items():
        if isinstance(value, dict) and 'who_to_run' in value:
            if 'VISION' in str(value.get('who_to_run', '')):
                vision_config = value
                break
    
    if vision_config is None:
        print("❌ VISION config not found in config.dill")
        return
    
    # Initialize vision system
    print("\n📹 Initializing VISION system...")
    vision = VISION(name="TurretVision", **vision_config)
    
    try:
        vision.start()
        print("✅ VISION started")
        time.sleep(2)  # Wait for cameras to initialize
        
        # Initialize turret
        print("\n🎯 Initializing TurretControl...")
        turret_port = "COM5"  # Windows default
        if sys.platform.startswith('linux'):
            turret_port = "/dev/ttyUSB0"
        
        turret = TurretControl(
            port=turret_port,
            baudrate=115200,
            servo1_min=15, servo1_max=50, servo1_home=35,
            servo2_min=0, servo2_max=180, servo2_home=90,
            deadzone=1.0,  # Deadzone in degrees
            kp=0.5,  # Proportional gain
            max_speed=5.0  # Max degrees per update
        )
        
        turret.connect()
        if not turret.connected:
            print("❌ Failed to connect to turret")
            print(f"   Check if Arduino is connected to {turret_port}")
            return
        
        print("✅ TurretControl connected")
        turret.go_home()
        time.sleep(1)
        
        # Create tracker
        print("\n🎯 Creating SimpleTurretTracker...")
        tracker = SimpleTurretTracker(vision, turret)
        
        # Get FOV from config if available
        if 'fov_horizontal' in vision_config:
            tracker.fov_horizontal = vision_config['fov_horizontal']
        if 'fov_vertical' in vision_config:
            tracker.fov_vertical = vision_config['fov_vertical']
        
        print(f"   FOV: {tracker.fov_horizontal}° x {tracker.fov_vertical}°")
        
        # Main tracking loop
        print("\n🚀 Starting tracking loop...")
        print("   Press Ctrl+C to stop\n")
        
        frame_count = 0
        
        try:
            while True:
                # Update tracker
                found = tracker.update()
                
                frame_count += 1
                
                # Print status every 30 frames (~1 second at 30fps)
                if frame_count % 30 == 0:
                    vision_data = vision.read()
                    objects = vision_data.get('objects', [])
                    people = [obj for obj in objects if obj.get('type', '').lower() == 'person']
                    
                    if people:
                        person = max(people, key=lambda obj: obj.get('confidence', 0.0))
                        offset_x, offset_y = tracker.calculate_offset(person)
                        if offset_x is not None:
                            print(f"📊 Frame {frame_count}: Found {len(people)} person(s), "
                                  f"offset: ({offset_x:.1f}, {offset_y:.1f}) px, "
                                  f"conf: {person.get('confidence', 0):.2f}")
                    else:
                        print(f"📊 Frame {frame_count}: No person detected")
                
                time.sleep(0.033)  # ~30 Hz update rate
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            if 'turret' in locals():
                turret.go_home()
                time.sleep(0.5)
                turret.disconnect()
        except:
            pass
        
        try:
            if 'vision' in locals():
                vision.stop()
        except:
            pass
        
        print("✅ Done")


if __name__ == "__main__":
    main()

