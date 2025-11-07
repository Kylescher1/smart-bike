import cv2
import platform


def get_camera_info(index):
    """Get device name and backend info for a camera."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    
    info = {
        "index": index,
        "backend": cap.getBackendName(),
    }
    
    # Try to get device path/name (platform specific)
    if platform.system() == "Windows":
        # Get CAP_PROP_SETTINGS to trigger device info
        try:
            # Some properties that might give us info
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except:
            pass
    
    cap.release()
    return info


def find_and_preview_cameras():
    """Find all cameras and show preview windows."""
    available_cameras = []
    
    # Check indices 0-2 as requested
    for index in range(3):
        print(f"\n--- Checking Camera {index} ---")
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print(f"❌ Camera {index}: Cannot be opened")
            continue
            
        # Get camera info
        info = get_camera_info(index)
        if info:
            print(f"✅ Camera {index}:")
            print(f"   Backend: {info['backend']}")
            if 'width' in info and 'height' in info:
                print(f"   Resolution: {info['width']}x{info['height']}")
            if 'fps' in info:
                print(f"   FPS: {info['fps']}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            available_cameras.append((index, cap, frame))
            print(f"   📹 Frame captured successfully")
        else:
            print(f"   ⚠️  Could not read frame")
            cap.release()
    
    if not available_cameras:
        print("\n❌ No cameras found that can capture frames.")
        return
    
    print(f"\n✅ Found {len(available_cameras)} working camera(s)")
    print("\n" + "="*50)
    print("PREVIEW MODE - Press keys to navigate:")
    print("  - Number keys (0-2): View specific camera")
    print("  - 'a': View ALL cameras at once")
    print("  - 'q' or ESC: Quit")
    print("="*50 + "\n")
    
    current_mode = 'all'  # Start with all cameras view
    
    while True:
        # Capture fresh frames
        frames = []
        for idx, cap, _ in available_cameras:
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))
        
        if current_mode == 'all':
            # Show all cameras
            for idx, frame in frames:
                # Add label to frame
                labeled_frame = frame.copy()
                label = f"Camera {idx}"
                cv2.putText(labeled_frame, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera {idx}", labeled_frame)
        else:
            # Show only selected camera
            for idx, frame in frames:
                if idx == current_mode:
                    labeled_frame = frame.copy()
                    label = f"Camera {idx} (SELECTED)"
                    cv2.putText(labeled_frame, label, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f"Camera {idx}", labeled_frame)
                else:
                    cv2.destroyWindow(f"Camera {idx}")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n👋 Closing preview...")
            break
        elif key == ord('a'):
            current_mode = 'all'
            print("Showing all cameras")
        elif key == ord('0'):
            current_mode = 0
            print("Showing camera 0")
        elif key == ord('1'):
            current_mode = 1
            print("Showing camera 1")
        elif key == ord('2'):
            current_mode = 2
            print("Showing camera 2")
    
    # Cleanup
    for idx, cap, _ in available_cameras:
        cap.release()
    cv2.destroyAllWindows()
    
    print("✅ All cameras released")


if __name__ == "__main__":
    find_and_preview_cameras()