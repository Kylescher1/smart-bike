"""
Simple camera test - verify camera feed works
"""

import sys
import time
import cv2
import dill
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.VISION.VISION_UPGRADE import VISION

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
    print("Simple Camera Test")
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
    
    # Initialize vision
    print("\n📹 Initializing VISION system...")
    vision = VISION(name="CameraTest", **vision_config)
    
    try:
        vision.start()
        print("✅ VISION started")
        time.sleep(2)  # Wait for cameras
        
        print("\n📺 Opening camera preview window...")
        print("   Press 'q' to quit")
        
        window_name = "Camera Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while True:
            try:
                # Get frame
                frame = None
                with vision.frame_lock:
                    if vision.last_left_frame is not None:
                        frame = vision.last_left_frame.copy()
                
                if frame is None:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for frame...", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(window_name, blank)
                else:
                    frame_count += 1
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {vision.current_fps:.1f}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.033)
                
            except Exception as e:
                print(f"Error in loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print(f"\n✅ Test complete! Processed {frame_count} frames")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vision.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np
    main()

