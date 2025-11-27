"""
Test script to debug turret centering
Shows bbox center, frame center, pixel errors, calculated angles, and servo movements
"""

import dill
import time
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.VISION.VISION_UPGRADE import VISION
from hal.TurretControl import TurretControl


def load_config(config_path: str = "config.dill"):
    """Load configuration from config.dill file."""
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def main():
    print("=" * 60)
    print("Turret Centering Test")
    print("=" * 60)
    
    # Load config
    print("\n📋 Loading configuration...")
    config = load_config()
    if config is None:
        return
    
    # Find vision config
    vision_config = None
    for key, value in config.items():
        if isinstance(value, dict) and 'who_to_run' in value:
            if 'VISION' in str(value.get('who_to_run', '')):
                vision_config = value
                break
    
    if vision_config is None:
        print("❌ VISION config not found")
        return
    
    # Initialize VISION
    print("\n📹 Initializing VISION system...")
    vision = VISION(name="TestVision", **vision_config)
    
    try:
        vision.start()
        print("✅ VISION started")
        time.sleep(2)
        
        # Test parameters (adjust these)
        angle_scale_s2 = 1.5  # Horizontal scaling - increase if not moving enough
        angle_scale_s1 = 1.0  # Vertical scaling
        kp = 1.5
        deadzone = 0.5
        
        print(f"\n⚙️ Test parameters:")
        print(f"   angle_scale_s2 (horizontal) = {angle_scale_s2}")
        print(f"   angle_scale_s1 (vertical) = {angle_scale_s1}")
        print(f"   kp = {kp}")
        print(f"   deadzone = {deadzone}°")
        print(f"\nPress 'q' to quit, 's' to increase scale, 'd' to decrease scale")
        print("Creating preview window...")
        
        # Create window
        window_name = "Centering Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.moveWindow(window_name, 100, 100)
        
        frame_count = 0
        no_frame_count = 0
        print("Window created. Waiting for frames...")
        
        while True:
            # Get frame and detections first (they're synchronized)
            frame = None
            detections = []
            
            try:
                with vision.frame_lock:
                    if vision.last_right_frame is not None:
                        frame = vision.last_right_frame.copy()
                    elif vision.last_left_frame is not None:
                        frame = vision.last_left_frame.copy()
                    
                    # Get cached detections (these have the bbox!)
                    if hasattr(vision, 'last_detections_cache') and vision.last_detections_cache:
                        detections = vision.last_detections_cache.copy()
            except Exception as e:
                print(f"Error getting frame: {e}")
                time.sleep(0.1)
                continue
            
            # Get vision data for object info (but bbox comes from detections)
            vision_data = vision.read()
            objects = vision_data.get('objects', [])
            
            if frame is None:
                no_frame_count += 1
                if no_frame_count % 30 == 0:
                    print(f"Waiting for frame... (count: {no_frame_count})")
                time.sleep(0.1)
                continue
            
            no_frame_count = 0  # Reset counter when we get a frame
            
            if frame.size == 0:
                print("Empty frame received")
                time.sleep(0.1)
                continue
            
            # Ensure frame is valid
            if len(frame.shape) < 2:
                print(f"Invalid frame shape: {frame.shape}")
                time.sleep(0.1)
                continue
            
            h, w = frame.shape[:2]
            frame_center_x = w / 2.0
            frame_center_y = h / 2.0
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw frame center (green crosshair) - always visible
            cv2.line(display_frame, (int(frame_center_x - 20), int(frame_center_y)), 
                    (int(frame_center_x + 20), int(frame_center_y)), (0, 255, 0), 2)
            cv2.line(display_frame, (int(frame_center_x), int(frame_center_y - 20)), 
                    (int(frame_center_x), int(frame_center_y + 20)), (0, 255, 0), 2)
            cv2.circle(display_frame, (int(frame_center_x), int(frame_center_y)), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, "Frame Center", (int(frame_center_x + 10), int(frame_center_y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Find person detections directly (they have bbox!)
            person_detections = [det for det in detections if det.get('type', '').lower() == 'person']
            
            # Also get person objects from vision.read() for confidence comparison
            person_objects = [obj for obj in objects if obj.get('type', '').lower() == 'person']
            
            # Debug: print frame info occasionally
            if frame_count % 60 == 0:
                print(f"Frame {frame_count}: shape={display_frame.shape}, dtype={display_frame.dtype}")
                print(f"  Person detections: {len(person_detections)}, Person objects: {len(person_objects)}")
                if person_detections:
                    print(f"  First person detection: {person_detections[0]}")
            
            # Use detections directly - they have bbox!
            if not person_detections:
                cv2.putText(display_frame, "No person detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Ensure we show the frame with drawings
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                frame_count += 1
                continue
            
            # Select highest confidence person detection (has bbox!)
            target_detection = max(person_detections, key=lambda det: det.get('score', det.get('confidence', 0.0)))
            bbox = target_detection.get('bbox')
            
            if bbox is None or len(bbox) != 4:
                if frame_count % 60 == 0:
                    print(f"ERROR: No bbox in detection: {target_detection}")
                cv2.putText(display_frame, "No bbox in detection", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                frame_count += 1
                continue
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            bbox_center_x = (x1 + x2) / 2.0
            bbox_center_y = (y1 + y2) / 2.0
            
            # Calculate pixel errors
            pixel_error_x = bbox_center_x - frame_center_x
            pixel_error_y = bbox_center_y - frame_center_y
            
            # Calculate angles
            fov_h = getattr(vision, 'fov_horizontal', 126.0)
            fov_v = getattr(vision, 'fov_vertical', 101.62)
            
            theta = ((bbox_center_x - frame_center_x) / w) * fov_h
            alpha = ((bbox_center_y - frame_center_y) / h) * fov_v
            
            # Flip signs
            theta_deg = -theta
            alpha_deg = -alpha
            
            # Calculate what servo positions would be
            servo2_home = 90
            servo1_home = 35
            desired_s2 = servo2_home + (theta_deg * angle_scale_s2)
            desired_s1 = servo1_home + (alpha_deg * angle_scale_s1)
            
            # Draw bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Draw bbox center (purple crosshair - same as YOLO preview)
            cv2.circle(display_frame, (int(bbox_center_x), int(bbox_center_y)), 8, (255, 0, 255), -1)
            cv2.circle(display_frame, (int(bbox_center_x), int(bbox_center_y)), 10, (255, 0, 255), 1)
            cv2.line(display_frame, (int(bbox_center_x - 15), int(bbox_center_y)), 
                    (int(bbox_center_x + 15), int(bbox_center_y)), (255, 0, 255), 2)
            cv2.line(display_frame, (int(bbox_center_x), int(bbox_center_y - 15)), 
                    (int(bbox_center_x), int(bbox_center_y + 15)), (255, 0, 255), 2)
            
            # Draw line from frame center to bbox center
            cv2.line(display_frame, (int(frame_center_x), int(frame_center_y)), 
                    (int(bbox_center_x), int(bbox_center_y)), (0, 255, 255), 2)
            
            # Add text info
            info_y = 20
            cv2.putText(display_frame, f"Frame center: ({int(frame_center_x)},{int(frame_center_y)})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            info_y += 20
            cv2.putText(display_frame, f"Bbox center: ({int(bbox_center_x)},{int(bbox_center_y)})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            info_y += 20
            cv2.putText(display_frame, f"Pixel error: X={pixel_error_x:+.1f}, Y={pixel_error_y:+.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(display_frame, f"Angles: theta={theta_deg:.2f}deg, alpha={alpha_deg:.2f}deg", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(display_frame, f"Desired servo: S1={desired_s1:.1f}deg, S2={desired_s2:.1f}deg", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(display_frame, f"Scale factors: S1={angle_scale_s1:.2f}, S2={angle_scale_s2:.2f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            info_y += 20
            cv2.putText(display_frame, f"Press 's'=increase scale, 'd'=decrease scale, 'q'=quit", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show frame - ensure it's valid
            try:
                if display_frame is None or display_frame.size == 0:
                    print("Invalid display_frame")
                    time.sleep(0.1)
                    continue
                
                # Ensure frame is BGR format
                if len(display_frame.shape) == 2:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                elif len(display_frame.shape) == 3 and display_frame.shape[2] == 4:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGRA2BGR)
                
                # Display frame
                cv2.imshow(window_name, display_frame)
                
                # Handle keys - waitKey must be called for window to update
                # Use longer delay to ensure window updates
                key = cv2.waitKey(30) & 0xFF  # 30ms delay for ~30 FPS
                if key != 255:  # 255 means no key pressed
                    print(f"Key pressed: {chr(key) if key < 128 else 'non-ASCII'}")
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    angle_scale_s2 += 0.1
                    print(f"Scale S2 increased to {angle_scale_s2:.2f}")
                elif key == ord('d'):
                    angle_scale_s2 = max(0.1, angle_scale_s2 - 0.1)
                    print(f"Scale S2 decreased to {angle_scale_s2:.2f}")
                
                frame_count += 1
            except Exception as e:
                print(f"Error showing frame: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"\nFrame {frame_count}:")
                print(f"  Bbox center: ({bbox_center_x:.1f}, {bbox_center_y:.1f})")
                print(f"  Frame center: ({frame_center_x:.1f}, {frame_center_y:.1f})")
                print(f"  Pixel error: X={pixel_error_x:+.1f}, Y={pixel_error_y:+.1f}")
                print(f"  Angles: theta={theta_deg:.2f}deg, alpha={alpha_deg:.2f}deg")
                print(f"  Desired servo: S1={desired_s1:.1f}deg, S2={desired_s2:.1f}deg")
                print(f"  Scale: S2={angle_scale_s2:.2f}")
        
        cv2.destroyAllWindows()
        vision.stop()
        print("\n✅ Test complete")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        cv2.destroyAllWindows()
        vision.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
            vision.stop()
        except:
            pass


if __name__ == "__main__":
    main()

