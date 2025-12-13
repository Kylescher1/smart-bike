#!/usr/bin/env python3
"""
Turret System Demo

Example usage of the turret tracking system.
Shows how to read detections and 3D positions.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.Turret import Turret


def main():
    print("=" * 60)
    print("TURRET 3D TRACKING SYSTEM DEMO")
    print("=" * 60)
    
    # ============================================
    # CONFIGURE YOUR TURRET HERE:
    # ============================================
    
    # STEP 1: Set your Arduino port
    # Windows: 'COM3', 'COM4', etc.
    # Linux: '/dev/ttyUSB0', '/dev/ttyACM0', etc.
    PORT = 'COM3'  # <<< CHANGE THIS
    
    # STEP 2: Set your camera indices (from camera finder tool)
    # Run: python src/Debug_Tools/fuck_you_camerafinder.py
    # Then plug in the indices here:
    CAMERAS = {
        'left': 0,    # <<< Left fisheye camera index
        'right': 1,   # <<< Right fisheye camera index
        'center': 2   # <<< Center tracking camera index
    }
    
    # STEP 3: (Optional) Only track specific objects
    # Leave as None to track everything, or set list of classes:
    TARGET_CLASSES = ['person', 'bottle', 'cup']  # Or None for all
    # TARGET_CLASSES = None  # Track everything
    
    # ============================================
    
    print(f"\nConfiguration:")
    print(f"  Port: {PORT}")
    print(f"  Cameras: Left={CAMERAS['left']}, Right={CAMERAS['right']}, Center={CAMERAS['center']}")
    print(f"  Target Classes: {TARGET_CLASSES or 'All'}")
    print()
    
    # Create and start turret
    try:
        turret = Turret(
            port=PORT,
            cameras=CAMERAS,
            yolo_model='yolo11n.pt',
            conf_threshold=0.5,
            target_classes=TARGET_CLASSES,
            tracking_enabled=True
        )
        
        turret.start()
        
        print("\nTurret is now tracking. Press Ctrl+C to stop.\n")
        print("-" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Read turret state
            output = turret.read()
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Print status every 30 frames (~1 second at 30fps)
            if frame_count % 30 == 0:
                print(f"\n[{time.strftime('%H:%M:%S')}] Status Update:")
                print(f"  FPS: {fps:.1f}")
                print(f"  Turret: Pan={output.turret_pose.pan_angle:.1f}°, "
                      f"Tilt={output.turret_pose.tilt_angle:.1f}°")
                print(f"  Locked: {'YES' if output.is_locked else 'NO'}")
                
                # Show all detections
                if output.all_detections:
                    print(f"  Detections ({len(output.all_detections)}):")
                    for det in output.all_detections[:5]:  # Show first 5
                        print(f"    - {det.class_name} ({det.camera_id} cam, "
                              f"conf={det.confidence:.2f})")
                else:
                    print("  Detections: None")
                
                # Show 3D positions
                if output.detections_3d:
                    print(f"  3D Positions ({len(output.detections_3d)}):")
                    for pos3d in output.detections_3d[-3:]:  # Show last 3
                        az, el, dist = pos3d.position_spherical
                        print(f"    - {pos3d.detection.class_name}: "
                              f"azimuth={az:.1f}°, elevation={el:.1f}°, "
                              f"distance={dist:.2f}in")
                else:
                    print("  3D Positions: None (waiting for lock + depth)")
                
                # Show current target
                if output.current_target:
                    tgt = output.current_target
                    print(f"  Target: {tgt.detection.class_name} "
                          f"({tgt.source}, priority={tgt.priority_score:.1f})")
                
                print("-" * 60)
            
            time.sleep(0.033)  # ~30 Hz
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        turret.stop()
        print("Done!")


if __name__ == '__main__':
    main()
