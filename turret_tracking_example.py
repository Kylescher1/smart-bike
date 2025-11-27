"""
Turret Tracking Example

Demonstrates how to use the VISION system with TurretControl to automatically
track and center detected objects.

Usage:
    python turret_tracking_example.py

Requirements:
    - Arduino/ESP32 running turret_control.ino
    - Camera connected and configured
    - YOLO model available
"""

import dill
import time
import sys
import cv2
import numpy as np
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.VISION.VISION_UPGRADE import VISION
from hal.TurretControl import TurretControl
from hal.TurretTracker import TurretTracker


def load_config(config_path: str = "config.dill"):
    """Load configuration from config.dill file."""
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("Please run config_setup.py first to create config.dill")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def main():
    """Main function to run turret tracking."""
    print("=" * 60)
    print("Turret Tracking System")
    print("=" * 60)
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_config()
    if config is None:
        return
    
    # Find camera/VISION config
    vision_config = None
    for key, value in config.items():
        if isinstance(value, dict) and 'who_to_run' in value:
            if 'VISION' in str(value.get('who_to_run', '')):
                vision_config = value
                break
    
    if vision_config is None:
        print("❌ VISION config not found in config.dill")
        return
    
    # Initialize VISION system
    print("\n📹 Initializing VISION system...")
    vision = VISION(name="TurretVision", **vision_config)
    
    try:
        # Start VISION system
        vision.start()
        print("✅ VISION system started")
        
        # Wait a moment for cameras to initialize
        time.sleep(2)
        
        # Initialize TurretControl
        print("\n🎯 Initializing TurretControl...")
        # Update port based on your system (Windows: "COM3", Linux: "/dev/ttyUSB0")
        turret_port = "COM5"  # Change this to your Arduino port
        if sys.platform.startswith('linux'):
            turret_port = "/dev/ttyUSB0"
        elif sys.platform.startswith('darwin'):
            turret_port = "/dev/cu.usbserial-*"  # macOS
        
        turret = TurretControl(
            port=turret_port,
            baudrate=115200,
            servo1_min=15,
            servo1_max=50,
            servo1_home=35,
            servo2_min=0,
            servo2_max=180,
            servo2_home=90,
            deadzone=2.0,  # Don't move if error < 2 degrees
            kp=0.5,  # Proportional gain (0.0-1.0)
            max_speed=5.0  # Max degrees per update
        )
        
        # Connect to turret
        turret.connect()
        if not turret.connected:
            print("❌ Failed to connect to turret")
            print("   Make sure Arduino is connected and running turret_control.ino")
            return
        
        print("✅ TurretControl connected")
        
        # Quick turret test
        print("\n🧪 Testing turret movement...")
        turret.go_home()
        time.sleep(0.3)
        print("✅ Turret test complete")
        
        # Initialize TurretTracker
        print("\n🎯 Initializing TurretTracker...")
        tracker = TurretTracker(
            vision=vision,
            turret=turret,
            tracking_mode="largest",  # Options: "largest", "highest_confidence", "class"
            target_class=None,  # Set to "person", "car", etc. if mode="class"
            min_confidence=0.3,  # Minimum confidence to track
            max_tracking_distance=60.0  # Max angular distance in degrees (increased for testing)
        )
        
        # Start tracking FIRST (before visualization)
        print("\n🚀 Starting automatic tracking...")
        print("   Tracking mode: largest object")
        tracker.start_tracking()
        print("✅ Tracking started")
        
        # Wait a moment for tracking to initialize
        time.sleep(1)
        
        # Start visualization window
        print("\n📺 Starting visualization window...")
        print("   Press 'q' in window or Ctrl+C to stop")
        visualization_running = [True]  # Use list to make it mutable across threads
        visualization_started = [False]
        
        def visualization_loop():
            """Simplified visualization loop running in separate thread."""
            window_name = "Turret Tracking - Camera View"
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
                print("✅ Visualization window created")
                visualization_started[0] = True
            except Exception as e:
                print(f"❌ Warning: Could not create window: {e}")
                print("   Visualization disabled - tracking will continue")
                return
            
            frame_count = 0
            last_error_time = 0
            
            while visualization_running[0]:
                try:
                    # Get frame directly from vision system (thread-safe, quick access)
                    frame = None
                    try:
                        with vision.frame_lock:
                            if vision.last_left_frame is not None:
                                frame = vision.last_left_frame.copy()
                    except Exception as e:
                        if time.time() - last_error_time > 5:  # Print error max once per 5 seconds
                            print(f"Frame access error: {e}")
                            last_error_time = time.time()
                        time.sleep(0.1)
                        continue
                    
                    if frame is None:
                        # Show "Waiting for frame..." message
                        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank_frame, "Waiting for camera frame...", (50, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        try:
                            cv2.imshow(window_name, blank_frame)
                            cv2.waitKey(1)
                        except:
                            pass
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    
                    # Create display frame
                    try:
                        display_frame = frame.copy()
                        h, w = display_frame.shape[:2]
                    except Exception as e:
                        print(f"Frame copy error: {e}")
                        time.sleep(0.1)
                        continue
                    
                    # Get current vision data (non-blocking)
                    try:
                        vision_data = vision.read()
                        objects = vision_data.get('objects', [])
                    except Exception as e:
                        objects = []
                        if frame_count % 60 == 0:
                            print(f"Vision read error: {e}")
                    
                    # Get tracked object ID (quick, non-blocking)
                    tracked_id = None
                    try:
                        # Try to get tracked_id without blocking
                        if tracker.lock.acquire(blocking=False):
                            try:
                                tracked_id = tracker.tracked_object_id
                            finally:
                                tracker.lock.release()
                    except:
                        pass
                    
                    # Get cached detections for bounding boxes (quick access)
                    detections = []
                    try:
                        with vision.frame_lock:
                            if hasattr(vision, 'last_detections_cache') and vision.last_detections_cache:
                                detections = vision.last_detections_cache[:]  # Quick copy
                    except:
                        pass
                    
                    # Draw detections (simplified)
                    try:
                        for det in detections[:10]:  # Limit to 10 detections
                            bbox = det.get('bbox', [])
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = [int(c) for c in bbox]
                                det_id = det.get('track_id') or det.get('id')
                                is_tracked = (det_id == tracked_id)
                                color = (0, 255, 0) if is_tracked else (255, 0, 0)
                                thickness = 3 if is_tracked else 2
                                
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Center point
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                cv2.circle(display_frame, (cx, cy), 5, color, -1)
                                
                                # Label
                                label = f"{det.get('class_name', 'obj')} {det.get('score', 0):.2f}"
                                cv2.putText(display_frame, label, (x1, y1 - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        if frame_count % 60 == 0:
                            print(f"Drawing error: {e}")
                    
                    # Draw center crosshair
                    try:
                        cx_img, cy_img = w // 2, h // 2
                        cv2.line(display_frame, (cx_img - 20, cy_img), (cx_img + 20, cy_img), (255, 255, 255), 1)
                        cv2.line(display_frame, (cx_img, cy_img - 20), (cx_img, cy_img + 20), (255, 255, 255), 1)
                    except:
                        pass
                    
                    # Draw simple info overlay
                    try:
                        turret_pos = turret.get_position()
                        info_text = [
                            f"FPS: {vision.current_fps:.1f}",
                            f"Objects: {len(objects)}",
                            f"Tracked: {tracked_id if tracked_id else 'None'}",
                            f"S1: {turret_pos[0]:.1f} S2: {turret_pos[1]:.1f}",
                        ]
                        y = 25
                        for text in info_text:
                            cv2.putText(display_frame, text, (10, y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            y += 25
                    except:
                        pass
                    
                    # Show frame
                    try:
                        cv2.imshow(window_name, display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            visualization_running[0] = False
                            break
                    except Exception as e:
                        if frame_count % 60 == 0:
                            print(f"imshow error: {e}")
                    
                    # Debug output every 60 frames
                    if frame_count % 60 == 0:
                        print(f"Viz: Frame {frame_count}, Objs: {len(objects)}, Tracked: {tracked_id}, "
                              f"Servos: S1={turret.get_position()[0]:.1f} S2={turret.get_position()[1]:.1f}")
                    
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    print(f"Visualization loop error: {e}")
                    import traceback
                    if frame_count % 60 == 0:  # Only print traceback occasionally
                        traceback.print_exc()
                    time.sleep(0.1)
            
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
            print("Visualization thread ended")
        
        # Start visualization in separate thread (non-blocking)
        viz_thread = threading.Thread(target=visualization_loop, daemon=True)
        viz_thread.start()
        time.sleep(0.5)  # Brief wait for window to open
        
        print("✅ System ready!")
        print("   - Camera preview should be visible")
        print("   - Objects will be tracked automatically")
        print("   - Press 'q' in preview window or Ctrl+C to stop\n")
        
        # Main monitoring loop
        try:
            last_stats_print = 0
            while visualization_running[0]:
                current_time = time.time()
                
                # Print stats every 2 seconds
                if current_time - last_stats_print >= 2.0:
                    stats = tracker.get_stats()
                    vision_data = vision.read()
                    objects = vision_data.get('objects', [])
                    
                    with tracker.lock:
                        tracked_id = tracker.tracked_object_id
                    
                    print(f"📊 Status: Frames={stats['frames_processed']}, "
                          f"Objects={len(objects)}, "
                          f"Tracked={stats['objects_tracked']}, "
                          f"Current ID={tracked_id if tracked_id else 'None'}")
                    
                    if objects:
                        for obj in objects[:2]:
                            print(f"   - {obj.get('type', 'unknown')} "
                                  f"(ID:{obj.get('id', '?')}, "
                                  f"conf:{obj.get('confidence', 0.0):.2f})")
                    
                    last_stats_print = current_time
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping tracking...")
            visualization_running[0] = False
            tracker.stop_tracking()
            time.sleep(0.5)  # Give visualization thread time to cleanup
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        turret.disconnect()
        vision.stop()
        print("✅ Done")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            if 'tracker' in locals():
                tracker.stop_tracking()
            if 'turret' in locals():
                turret.disconnect()
            if 'vision' in locals():
                vision.stop()
        except:
            pass


if __name__ == "__main__":
    main()

