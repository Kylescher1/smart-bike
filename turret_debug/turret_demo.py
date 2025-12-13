#!/usr/bin/env python3
"""
Turret System Demo

Example usage of the turret tracking system.
Shows how to read detections and 3D positions.
"""

import sys
import time
import cv2
import numpy as np
from collections import deque
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    # To find your port on Linux, run: ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
    PORT = '/dev/ttyUSB0'  # <<< CHANGE THIS (or /dev/ttyACM0, etc.)
    
    # STEP 2: Set your camera indices (from camera finder tool)
    # Run: python src/Debug_Tools/fuck_you_camerafinder.py
    # Then plug in the indices here:
    CAMERAS = {
        'left': 1,    # <<< Left fisheye camera index
        'right': 3,   # <<< Right fisheye camera index
        'center': 5,   # <<< Center tracking camera index
    }
    
    # STEP 3: (Optional) Only track specific objects
    # Leave as None to track everything, or set list of classes:
    TARGET_CLASSES = ['person', 'bottle', 'cup']  # Or None for all
    # TARGET_CLASSES = None  # Track everything
    
    # STEP 4: Enable camera view debugging
    SHOW_CAMERA_VIEWS = True  # Set to False to disable camera windows
    
    # STEP 5: Enable performance profiling
    ENABLE_PROFILING = True  # Set to False to disable timing measurements
    
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
        if SHOW_CAMERA_VIEWS:
            print("Camera views are displayed. Press 'q' in any window to quit.\n")
        if ENABLE_PROFILING:
            print("Performance profiling enabled. Timing stats will be shown periodically.\n")
        print("-" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        # Performance profiling data
        timing_stats = {
            'turret_read': deque(maxlen=60),
            'frame_capture': deque(maxlen=60),
            'yolo_inference': deque(maxlen=60),
            'target_selection': deque(maxlen=60),
            'turret_movement': deque(maxlen=60),
            'position_3d': deque(maxlen=60),
            'display': deque(maxlen=60),
            'total_frame': deque(maxlen=60)
        }
        
        while True:
            frame_start = time.time()
            
            # Read turret state
            turret_read_start = time.time()
            output = turret.read()
            turret_read_time = (time.time() - turret_read_start) * 1000  # ms
            
            # Measure frame capture and YOLO inference times
            frame_capture_time = 0
            yolo_inference_time = 0
            
            # Display camera views with detections
            display_start = time.time()
            if SHOW_CAMERA_VIEWS:
                for cam_id in ['left', 'right', 'center']:
                    if cam_id in turret.yolo.detectors:
                        detector = turret.yolo.detectors[cam_id]
                        
                        # Measure frame capture time
                        frame_cap_start = time.time()
                        result = detector.read()
                        frame_capture_time += (time.time() - frame_cap_start) * 1000  # ms
                        
                        if result is not None and result.annotated_frame is not None:
                            # Accumulate YOLO inference time (from result)
                            if result.inference_time_ms > 0:
                                yolo_inference_time += result.inference_time_ms
                            # Get annotated frame with detections
                            display_frame = result.annotated_frame.copy()
                            
                            # Add camera label
                            cv2.putText(display_frame, f"{cam_id.upper()} Camera", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Highlight current target if this is the center camera
                            if cam_id == 'center' and output.current_target:
                                tgt = output.current_target
                                if tgt.detection.camera_id == 'center':
                                    det = tgt.detection
                                    x1, y1, x2, y2 = map(int, det.bbox)
                                    # Draw thicker border for current target
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                                    # Add lock indicator
                                    lock_text = "LOCKED" if output.is_locked else "TRACKING"
                                    cv2.putText(display_frame, lock_text, (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # Add detection count
                            cam_dets = turret.yolo.read_detections(cam_id)
                            det_count = len(cam_dets.detections) if cam_dets else 0
                            cv2.putText(display_frame, f"Detections: {det_count}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Add FPS
                            fps_text = f"FPS: {result.fps:.1f}" if result.fps > 0 else "FPS: --"
                            cv2.putText(display_frame, fps_text, 
                                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Add YOLO inference time if profiling enabled
                            if ENABLE_PROFILING and result.inference_time_ms > 0:
                                yolo_time_text = f"YOLO: {result.inference_time_ms:.1f}ms"
                                cv2.putText(display_frame, yolo_time_text, 
                                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Show frame
                            window_name = f"Turret - {cam_id.title()} Camera"
                            cv2.imshow(window_name, display_frame)
                
                # Check for 'q' key to quit (once per frame, not per camera)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
            
            display_time = (time.time() - display_start) * 1000  # ms
            
            # Measure individual components (we'll need to instrument Turret.read() for this)
            # For now, we'll estimate based on available data
            total_frame_time = (time.time() - frame_start) * 1000  # ms
            
            # Store timing stats
            if ENABLE_PROFILING:
                timing_stats['turret_read'].append(turret_read_time)
                timing_stats['frame_capture'].append(frame_capture_time)
                timing_stats['yolo_inference'].append(yolo_inference_time)
                timing_stats['display'].append(display_time)
                timing_stats['total_frame'].append(total_frame_time)
            
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
                
                # Performance profiling output
                if ENABLE_PROFILING and len(timing_stats['total_frame']) > 0:
                    print("\n  Performance Profile (last 30 frames):")
                    
                    # Calculate averages
                    avg_turret_read = np.mean(list(timing_stats['turret_read'])[-30:]) if timing_stats['turret_read'] else 0
                    avg_frame_capture = np.mean(list(timing_stats['frame_capture'])[-30:]) if timing_stats['frame_capture'] else 0
                    avg_yolo = np.mean(list(timing_stats['yolo_inference'])[-30:]) if timing_stats['yolo_inference'] else 0
                    avg_display = np.mean(list(timing_stats['display'])[-30:]) if timing_stats['display'] else 0
                    avg_total = np.mean(list(timing_stats['total_frame'])[-30:]) if timing_stats['total_frame'] else 0
                    
                    if avg_total > 0:
                        pct_turret = (avg_turret_read / avg_total) * 100
                        pct_capture = (avg_frame_capture / avg_total) * 100
                        pct_yolo = (avg_yolo / avg_total) * 100
                        pct_display = (avg_display / avg_total) * 100
                        
                        print(f"  ├─ Turret.read():     {avg_turret_read:7.2f}ms ({pct_turret:5.1f}%)")
                        print(f"  │  └─ Includes: target selection, turret movement, 3D calc")
                        print(f"  ├─ YOLO Inference:    {avg_yolo:7.2f}ms ({pct_yolo:5.1f}%)")
                        print(f"  │  └─ Total for all 3 cameras (runs in parallel)")
                        print(f"  ├─ Frame Buffer Read:  {avg_frame_capture:7.2f}ms ({pct_capture:5.1f}%)")
                        print(f"  │  └─ Time to read buffered frames (should be <1ms)")
                        print(f"  ├─ Display/Render:    {avg_display:7.2f}ms ({pct_display:5.1f}%)")
                        print(f"  │  └─ OpenCV drawing and imshow")
                        print(f"  └─ Total Frame Time:  {avg_total:7.2f}ms ({1000/avg_total:.1f} FPS)")
                        
                        # Show bottleneck
                        times_dict = {
                            'Turret.read() (target select + movement)': avg_turret_read,
                            'YOLO Inference (3 cameras)': avg_yolo,
                            'Frame Buffer Read': avg_frame_capture,
                            'Display/Render': avg_display
                        }
                        bottleneck = max(times_dict.items(), key=lambda x: x[1])
                        print(f"\n  ⚠ BOTTLENECK: {bottleneck[0]}")
                        print(f"     {bottleneck[1]:.2f}ms ({bottleneck[1]/avg_total*100:.1f}% of total time)")
                        
                        # Additional insights
                        if avg_yolo > avg_turret_read * 0.5:
                            print(f"\n  💡 YOLO inference is likely the main bottleneck")
                            print(f"     Consider: reducing image size, using smaller model, or NPU acceleration")
                        elif avg_turret_read > avg_yolo * 2:
                            print(f"\n  💡 Turret processing (target selection/movement) is slower than YOLO")
                            print(f"     Consider: optimizing target selection logic or reducing movement frequency")
                
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
        if SHOW_CAMERA_VIEWS:
            cv2.destroyAllWindows()
        turret.stop()
        print("Done!")


if __name__ == '__main__':
    main()
