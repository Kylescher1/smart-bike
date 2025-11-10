import cv2
import dill
import numpy as np
import sys
import platform

config_path = r"config.dill"

def get_default_backend():
    """Get the default camera backend for the current platform."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW  # or cv2.CAP_MSMF
    else:  # Linux
        return cv2.CAP_V4L2

def open_camera(port, backend):
    """Open a camera and configure it."""
    cap = cv2.VideoCapture(port, backend)
    if not cap.isOpened():
        return None
    
    # Try to set some common properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    return cap

def assign_cameras():
    """
    Open both cameras and let the user decide which should be left (1) and which should be right (2).
    Updates the config.dill file with the user's choice.
    """
    
    # Load config
    print("Loading Config...")
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        print("✅ Loaded config.dill")
        camera_config = config['camera']
        print("✅ Loaded Camera Config")
    except FileNotFoundError:
        print(f"❌ Error: config.dill not found at {config_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: 'camera' key not found in config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred loading config.dill: {e}")
        sys.exit(1)

    # Get current camera ports
    left_port = camera_config['left']['port']
    right_port = camera_config['right']['port']
    
    print("\n" + "="*60)
    print("CURRENT CAMERA CONFIGURATION")
    print("="*60)
    print(f"Left Camera (1):  Port {left_port}")
    print(f"Right Camera (2): Port {right_port}")
    print("="*60 + "\n")
    
    # Get backend
    backend = get_default_backend()
    
    # Open both cameras
    print(f"Opening cameras on ports {left_port} and {right_port}...")
    
    cap_left = open_camera(left_port, backend)
    if cap_left is None:
        print(f"❌ Error: Could not open camera on port {left_port}")
        sys.exit(1)
    print(f"✅ Opened camera on port {left_port}")
    
    cap_right = open_camera(right_port, backend)
    if cap_right is None:
        print(f"❌ Error: Could not open camera on port {right_port}")
        cap_left.release()
        sys.exit(1)
    print(f"✅ Opened camera on port {right_port}")
    
    print("\n" + "="*60)
    print("CAMERA PREVIEW")
    print("="*60)
    print("Look at both camera feeds to determine which is which.")
    print("")
    print("Instructions:")
    print("  - Left window shows camera currently assigned as LEFT (1)")
    print("  - Right window shows camera currently assigned as RIGHT (2)")
    print("")
    print("Press:")
    print("  - 'k' or 'y': KEEP current assignment (left stays left, right stays right)")
    print("  - 's' or 'n': SWAP cameras (left becomes right, right becomes left)")
    print("  - 'q': Quit without making changes")
    print("="*60 + "\n")
    
    # Create windows
    cv2.namedWindow("Current LEFT (1) Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Current RIGHT (2) Camera", cv2.WINDOW_NORMAL)
    
    # Position windows side by side
    cv2.moveWindow("Current LEFT (1) Camera", 50, 50)
    cv2.moveWindow("Current RIGHT (2) Camera", 900, 50)
    
    user_choice = None
    
    try:
        while True:
            # Read frames
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                print("⚠️  Failed to read from one or both cameras")
                continue
            
            # Create display frames with labels
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            # Add labels
            cv2.putText(display_left, f"Current LEFT (Port {left_port})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_left, "This is CAMERA 1 (left)", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.putText(display_right, f"Current RIGHT (Port {right_port})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_right, "This is CAMERA 2 (right)", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add instructions at bottom
            cv2.putText(display_left, "Press: k/y = KEEP | s/n = SWAP | q = QUIT", 
                       (10, display_left.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_right, "Press: k/y = KEEP | s/n = SWAP | q = QUIT", 
                       (10, display_right.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frames
            cv2.imshow("Current LEFT (1) Camera", display_left)
            cv2.imshow("Current RIGHT (2) Camera", display_right)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('k') or key == ord('y'):
                user_choice = 'keep'
                print("\n✅ You chose to KEEP the current camera assignment")
                break
            elif key == ord('s') or key == ord('n'):
                user_choice = 'swap'
                print("\n🔄 You chose to SWAP the camera assignment")
                break
            elif key == ord('q'):
                user_choice = 'quit'
                print("\n❌ Quitting without making changes")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        user_choice = 'quit'
    
    finally:
        # Cleanup
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
    
    # Exit if user quit
    if user_choice == 'quit' or user_choice is None:
        print("No changes made to configuration.")
        sys.exit(0)
    
    # If keeping, no changes needed
    if user_choice == 'keep':
        print("\n" + "="*60)
        print("KEEPING CURRENT CONFIGURATION")
        print("="*60)
        print(f"Left Camera (1):  Port {left_port} ✓")
        print(f"Right Camera (2): Port {right_port} ✓")
        print("="*60 + "\n")
        print("✅ No changes made to config.dill")
        return
    
    # If swapping, swap the ports and calibration maps
    if user_choice == 'swap':
        print("\n" + "="*60)
        print("SWAPPING CAMERA CONFIGURATION")
        print("="*60)
        
        # Swap camera ports
        camera_config['left']['port'], camera_config['right']['port'] = right_port, left_port
        print(f"✅ Swapped ports: {left_port} ↔ {right_port}")
        
        # Swap calibration maps
        left_map_x = camera_config['left']['map_x']
        left_map_y = camera_config['left']['map_y']
        right_map_x = camera_config['right']['map_x']
        right_map_y = camera_config['right']['map_y']
        
        camera_config['left']['map_x'] = right_map_x
        camera_config['left']['map_y'] = right_map_y
        camera_config['right']['map_x'] = left_map_x
        camera_config['right']['map_y'] = left_map_y
        
        print(f"✅ Swapped calibration maps")
        print(f"   Left map shape:  {camera_config['left']['map_x'].shape}")
        print(f"   Right map shape: {camera_config['right']['map_x'].shape}")
        
        # Update the config with swapped camera settings
        config['camera'] = camera_config
        
        # Save updated config back to dill file
        print("\n" + "="*60)
        print("SAVING UPDATED CONFIGURATION")
        print("="*60)
        
        try:
            with open(config_path, "wb") as f:
                dill.dump(config, f)
            print("✅ Configuration saved successfully to config.dill!")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            sys.exit(1)
        
        # Display new configuration
        print("\n" + "="*60)
        print("NEW CAMERA CONFIGURATION")
        print("="*60)
        print(f"Left Camera (1):  Port {camera_config['left']['port']}")
        print(f"Right Camera (2): Port {camera_config['right']['port']}")
        print("="*60 + "\n")
        
        print("✅ Camera swap complete!")

if __name__ == "__main__":
    try:
        assign_cameras()
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during camera assignment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

