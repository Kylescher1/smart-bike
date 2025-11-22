"""
YOLO Tracking Parameter Calibration Tool

Interactive tool to tune YOLO tracking parameters with sliders.
Press 'S' to save parameters to config.dill
Press 'Q' to quit
"""

import cv2
import numpy as np
import dill
import sys
import os
import time

config_path = "config.dill"

def load_config():
    """Load config.dill and return camera config."""
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        
        # Find camera config
        camera_config = None
        for key, value in config.items():
            if isinstance(value, dict) and 'who_to_run' in value:
                camera_config = value
                break
        
        if camera_config is None:
            raise ValueError("Camera config not found in config.dill")
        
        return config, camera_config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def save_config(config, yolo_params):
    """Save updated YOLO parameters to config.dill."""
    try:
        # Find camera config and update YOLO parameters
        camera_config = None
        for key, value in config.items():
            if isinstance(value, dict) and 'who_to_run' in value:
                camera_config = value
                break
        
        if camera_config is None:
            print("❌ Camera config not found")
            return False
        
        # Update YOLO config
        if 'yolo' not in camera_config:
            camera_config['yolo'] = {}
        
        camera_config['yolo'].update(yolo_params)
        
        # Save to file
        with open(config_path, "wb") as f:
            dill.dump(config, f)
        
        print(f"✅ YOLO parameters saved to {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_trackbars(window_name, yolo_params):
    """Create OpenCV trackbars for YOLO tracking parameters."""
    
    # Confidence threshold (0-100, representing 0.0-1.0)
    conf_val = int(yolo_params.get('conf_threshold', 0.25) * 100)
    cv2.createTrackbar("Conf Threshold", window_name, conf_val, 100, lambda x: None)
    
    # Track threshold (0-100, representing 0.0-1.0)
    track_thresh_val = int(yolo_params.get('track_thresh', 0.5) * 100)
    cv2.createTrackbar("Track Thresh", window_name, track_thresh_val, 100, lambda x: None)
    
    # High threshold (0-100, representing 0.0-1.0)
    high_thresh_val = int(yolo_params.get('track_high_thresh', 0.6) * 100)
    cv2.createTrackbar("High Thresh", window_name, high_thresh_val, 100, lambda x: None)
    
    # Match threshold (0-100, representing 0.0-1.0)
    match_thresh_val = int(yolo_params.get('track_match_thresh', 0.8) * 100)
    cv2.createTrackbar("Match Thresh", window_name, match_thresh_val, 100, lambda x: None)
    
    # Track buffer (frames to keep lost tracks)
    track_buffer_val = yolo_params.get('track_buffer', 30)
    cv2.createTrackbar("Track Buffer", window_name, track_buffer_val, 150, lambda x: None)
    
    # Frame rate (for tracking)
    frame_rate_val = yolo_params.get('frame_rate', 30)
    cv2.createTrackbar("Frame Rate", window_name, frame_rate_val, 60, lambda x: None)

def get_trackbar_values(window_name):
    """Read all trackbar values and return updated YOLO parameters."""
    params = {}
    
    # Read trackbar values (convert back from 0-100 to 0.0-1.0 for thresholds)
    params['conf_threshold'] = cv2.getTrackbarPos("Conf Threshold", window_name) / 100.0
    params['track_thresh'] = cv2.getTrackbarPos("Track Thresh", window_name) / 100.0
    params['track_high_thresh'] = cv2.getTrackbarPos("High Thresh", window_name) / 100.0
    params['track_match_thresh'] = cv2.getTrackbarPos("Match Thresh", window_name) / 100.0
    params['track_buffer'] = cv2.getTrackbarPos("Track Buffer", window_name)
    params['frame_rate'] = cv2.getTrackbarPos("Frame Rate", window_name)
    
    return params

def run_yolo_calibration():
    """Run interactive YOLO parameter calibration."""
    print("=" * 60)
    print("YOLO Tracking Parameter Calibration Tool")
    print("=" * 60)
    print("\nControls:")
    print("  - Drag sliders to adjust parameters")
    print("  - Parameters update in real-time")
    print("  - Press 'S' to save parameters to config.dill")
    print("  - Press 'Q' to quit")
    print("=" * 60 + "\n")
    
    # Load config
    config, camera_config = load_config()
    yolo_params = camera_config.get('yolo', {})
    
    if not yolo_params:
        print("❌ YOLO config not found in config.dill")
        sys.exit(1)
    
    # Import vision system
    try:
        from src.hal.VISION.VISION_UPGRADE import VISION
    except ImportError as e:
        print(f"❌ Error importing VISION: {e}")
        sys.exit(1)
    
    # Initialize vision system
    print("📦 Initializing vision system...")
    vision = VISION(name="YOLO Calibration", **camera_config)
    
    try:
        vision.start()
        print("✅ Vision system started")
    except Exception as e:
        print(f"❌ Error starting vision system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create windows
    trackbar_window = "YOLO Parameters"
    cv2.namedWindow(trackbar_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(trackbar_window, 400, 400)
    
    # Create trackbars
    create_trackbars(trackbar_window, yolo_params)
    
    # Create info window
    info_img = np.zeros((200, 400, 3), dtype=np.uint8)
    
    print("\n💡 TIP: Adjust parameters and watch the visualization update in real-time!")
    print("   When satisfied, press 'S' to save.\n")
    
    last_update_time = time.time()
    update_interval = 0.5  # Update tracker every 0.5 seconds
    
    try:
        while True:
            # Read trackbar values
            new_params = get_trackbar_values(trackbar_window)
            
            # Check if parameters changed
            params_changed = False
            for key, value in new_params.items():
                if abs(value - yolo_params.get(key, 0)) > 0.01:
                    params_changed = True
                    break
            
            # Update tracker if parameters changed and enough time has passed
            current_time = time.time()
            if params_changed and (current_time - last_update_time) >= update_interval:
                # Update YOLO config
                yolo_params.update(new_params)
                
                # Update YOLO config in vision system
                vision.yolo_config.update(new_params)
                
                # Reinitialize tracker with new parameters
                if vision.yolo and vision.yolo.track_enabled:
                    try:
                        # Try to import ByteTrackerWrapper (try both paths)
                        ByteTracker = None
                        try:
                            from yolo.yolo import ByteTrackerWrapper
                            ByteTracker = ByteTrackerWrapper
                        except ImportError:
                            try:
                                from yolo.rknn_inference import ByteTrackerWrapper
                                ByteTracker = ByteTrackerWrapper
                            except ImportError:
                                pass
                        
                        if ByteTracker:
                            vision.yolo.tracker = ByteTracker(
                                track_thresh=yolo_params['track_thresh'],
                                high_thresh=yolo_params['track_high_thresh'],
                                match_thresh=yolo_params['track_match_thresh'],
                                frame_rate=yolo_params['frame_rate'],
                                track_buffer=yolo_params['track_buffer']
                            )
                            vision.yolo.conf_threshold = yolo_params['conf_threshold']
                            print(f"🔄 Updated: conf={yolo_params['conf_threshold']:.2f}, "
                                  f"track_thresh={yolo_params['track_thresh']:.2f}, "
                                  f"buffer={yolo_params['track_buffer']}")
                    except Exception as e:
                        print(f"⚠️ Error updating tracker: {e}")
                
                last_update_time = current_time
            
            # Update info window
            info_img.fill(0)
            cv2.putText(info_img, "YOLO Parameter Tuning", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_img, f"Conf: {yolo_params['conf_threshold']:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_img, f"Track Thresh: {yolo_params['track_thresh']:.2f}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_img, f"High Thresh: {yolo_params['track_high_thresh']:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_img, f"Match Thresh: {yolo_params['track_match_thresh']:.2f}", (10, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_img, f"Buffer: {yolo_params['track_buffer']} frames", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_img, "Press 'S' to save, 'Q' to quit", (10, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            cv2.imshow(trackbar_window, info_img)
            
            # Get latest frame and detections for visualization
            latest = vision.read()
            if latest:
                objects = latest.get('objects', [])
                with vision.frame_lock:
                    if vision.last_left_frame is not None:
                        frame = vision.last_left_frame.copy()
                    else:
                        frame = None
                    
                    # Get cached detections
                    current_time = time.time()
                    if hasattr(vision, 'last_detections_cache') and \
                       (current_time - vision.last_detections_time) < 1.0:
                        detections = vision.last_detections_cache.copy() if vision.last_detections_cache else []
                    else:
                        detections = []
                
                if frame is not None:
                    # Draw detections
                    display_frame = frame.copy()
                    for det in detections:
                        bbox = det.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                        
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        color = (0, 255, 0)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        class_name = det.get('class_name', 'object')
                        score = det.get('score', 0.0)
                        track_id = det.get('track_id')
                        
                        label = f"{class_name} {score:.2f}"
                        if track_id is not None:
                            label += f" ID:{track_id}"
                        
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_frame, (x1, y1 - label_h - 5),
                                    (x1 + label_w, y1), color, -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Add stats
                    cv2.putText(display_frame, f"FPS: {vision.current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Detections: {len(detections)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Objects: {len(objects)}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("YOLO Visualization", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                # Save current parameters
                final_params = get_trackbar_values(trackbar_window)
                if save_config(config, final_params):
                    # Visual feedback
                    info_img.fill(0)
                    cv2.putText(info_img, "SAVED!", (150, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow(trackbar_window, info_img)
                    cv2.waitKey(500)  # Show feedback for 0.5 seconds
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        vision.stop()
        cv2.destroyAllWindows()
        print("\n✅ Calibration tool closed")

if __name__ == "__main__":
    run_yolo_calibration()

