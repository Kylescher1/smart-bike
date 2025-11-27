import cv2
import dill
import numpy as np
import sys
import platform
from collections import OrderedDict
from typing import Optional, Sequence, Tuple

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
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(* "MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_FPS, 60)

    return cap


def probe_ports(
    backend: int,
    candidate_ports: Optional[Sequence[int]] = None,
    max_ports: int = 8,
) -> list[int]:
    """Probe camera ports and return those that appear to provide frames."""
    if candidate_ports is None:
        candidate_ports = list(range(max_ports))

    available_ports: list[int] = []
    for port in OrderedDict.fromkeys(candidate_ports):
        cap = open_camera(port, backend)
        if cap is None:
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            available_ports.append(port)
        cap.release()
    return available_ports


def interactive_port_selection(backend: int, port_options: Sequence[int]) -> Tuple[int, int]:
    """Allow the user to preview ports and assign them to LEFT and RIGHT roles."""
    if not port_options:
        raise RuntimeError("No camera ports were detected. Please connect cameras and try again.")

    left_port: Optional[int] = None
    right_port: Optional[int] = None

    print("\n" + "=" * 60)
    print("CAMERA PORT SELECTION")
    print("=" * 60)
    print("Detected ports:", ", ".join(map(str, port_options)))
    print("Instructions:")
    print("  • Preview each port feed and press:")
    print("      - 'l' to assign the current port as LEFT camera")
    print("      - 'r' to assign the current port as RIGHT camera")
    print("      - 'u' to clear the current LEFT/RIGHT selection")
    print("      - space or 'n' to move to the next detected port")
    print("      - 'c' (or Enter) to confirm once both LEFT and RIGHT are assigned")
    print("      - 'q' to cancel without making changes")
    print("=" * 60 + "\n")

    window_name = "Port Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 200, 120)

    try:
        index = 0
        while True:
            if not port_options:
                break
            port = port_options[index % len(port_options)]
            cap = open_camera(port, backend)
            if cap is None:
                print(f"⚠️  Skipping port {port}: unable to open.")
                index += 1
                continue

            print(f"🔍 Previewing port {port}.")

            stay_on_port = True
            while stay_on_port:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"⚠️  Failed to read frame from port {port}. Moving on.")
                    break

                display = frame.copy()
                height, width = display.shape[:2]

                status_text = [
                    f"Previewing port {port}",
                    f"Assigned LEFT:  {left_port if left_port is not None else 'None'}",
                    f"Assigned RIGHT: {right_port if right_port is not None else 'None'}",
                    "Keys: [L]=assign left  [R]=assign right  [U]=clear",
                    "      [N/Space]=next port  [C]=confirm  [Q]=quit",
                ]

                y = 30
                for line in status_text:
                    cv2.putText(
                        display,
                        line,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    y += 32

                if width > 0:
                    cv2.resizeWindow(window_name, min(width, 960), min(height, 720))
                cv2.imshow(window_name, display)

                key = cv2.waitKey(30) & 0xFF
                if key in (ord("l"), ord("L")):
                    if right_port == port:
                        print("⚠️  This port is already assigned as RIGHT. Reassign RIGHT first.")
                    else:
                        left_port = port
                        print(f"✅ Assigned port {port} as LEFT camera.")
                elif key in (ord("r"), ord("R")):
                    if left_port == port:
                        print("⚠️  This port is already assigned as LEFT. Reassign LEFT first.")
                    else:
                        right_port = port
                        print(f"✅ Assigned port {port} as RIGHT camera.")
                elif key in (ord("u"), ord("U")):
                    cleared = []
                    if left_port == port:
                        left_port = None
                        cleared.append("LEFT")
                    if right_port == port:
                        right_port = None
                        cleared.append("RIGHT")
                    if cleared:
                        print(f"🔁 Cleared assignment for port {port}: {', '.join(cleared)}.")
                    else:
                        print("ℹ️  No assignments cleared.")
                elif key in (ord("n"), ord("N"), ord(" "), 9):
                    stay_on_port = False
                elif key in (ord("c"), ord("C"), 13):
                    if left_port is not None and right_port is not None and left_port != right_port:
                        print("✅ Confirmed selections.")
                        return left_port, right_port
                    print("⚠️  Assign both LEFT and RIGHT (must be different ports) before confirming.")
                elif key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt("User cancelled selection.")

            cap.release()
            index += 1

    finally:
        cv2.destroyWindow(window_name)

    if left_port is None or right_port is None:
        raise RuntimeError("Unable to complete selection of LEFT and RIGHT camera ports.")

    return left_port, right_port

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
    current_left_port = camera_config['left']['port']
    current_right_port = camera_config['right']['port']

    print("\n" + "="*60)
    print("CURRENT CAMERA CONFIGURATION")
    print("="*60)
    print(f"Left Camera (1):  Port {current_left_port}")
    print(f"Right Camera (2): Port {current_right_port}")
    print("="*60 + "\n")

    backend = get_default_backend()

    candidate_ports = list(range(10))
    candidate_ports.extend([current_left_port, current_right_port])
    available_ports = probe_ports(backend, candidate_ports=candidate_ports)
    if not available_ports:
        print("❌ Error: Unable to detect any active camera ports.")
        sys.exit(1)

    try:
        left_port, right_port = interactive_port_selection(backend, available_ports)
    except KeyboardInterrupt:
        print("\n❌ Selection cancelled by user. No changes made.")
        sys.exit(0)
    except RuntimeError as err:
        print(f"\n❌ {err}")
        sys.exit(1)

    print("\n" + "="*60)
    print("SELECTED CAMERA PORTS")
    print("="*60)
    print(f"Left Camera (1):  Port {left_port}")
    print(f"Right Camera (2): Port {right_port}")
    print("="*60 + "\n")

    print(f"Opening cameras on selected ports {left_port} and {right_port} for confirmation...")

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
    print("Look at both camera feeds to verify the LEFT/RIGHT assignments.")
    print("")
    print("Instructions:")
    print("  - Left window shows camera selected as LEFT (1)")
    print("  - Right window shows camera selected as RIGHT (2)")
    print("")
    print("Press:")
    print("  - 'k' or 'y': KEEP this assignment")
    print("  - 's' or 'n': SWAP the left/right roles")
    print("  - 'q': Quit without saving changes")
    print("="*60 + "\n")

    cv2.namedWindow("Selected LEFT (1) Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Selected RIGHT (2) Camera", cv2.WINDOW_NORMAL)

    cv2.moveWindow("Selected LEFT (1) Camera", 50, 50)
    cv2.moveWindow("Selected RIGHT (2) Camera", 900, 50)

    user_choice = None

    try:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                print("⚠️  Failed to read from one or both cameras")
                continue

            display_left = frame_left.copy()
            display_right = frame_right.copy()

            cv2.putText(display_left, f"Selected LEFT (Port {left_port})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_left, "Camera 1 (left)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(display_right, f"Selected RIGHT (Port {right_port})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_right, "Camera 2 (right)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(display_left, "Press: k/y = KEEP | s/n = SWAP | q = QUIT",
                        (10, display_left.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_right, "Press: k/y = KEEP | s/n = SWAP | q = QUIT",
                        (10, display_right.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Selected LEFT (1) Camera", display_left)
            cv2.imshow("Selected RIGHT (2) Camera", display_right)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('k'), ord('y')):
                user_choice = 'keep'
                print("\n✅ You chose to KEEP the selected camera assignment")
                break
            elif key in (ord('s'), ord('n')):
                user_choice = 'swap'
                print("\n🔄 You chose to SWAP the camera assignment")
                break
            elif key == ord('q'):
                user_choice = 'quit'
                print("\n❌ Quitting without saving changes")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        user_choice = 'quit'

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    if user_choice == 'quit' or user_choice is None:
        print("No changes made to configuration.")
        sys.exit(0)

    if user_choice == 'swap':
        print("\n" + "="*60)
        print("SWAPPING CAMERA CONFIGURATION")
        print("="*60)
        left_port, right_port = right_port, left_port
        print(f"✅ Swapped ports: left→{left_port}, right→{right_port}")

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

    if user_choice == 'keep':
        print("\n" + "="*60)
        print("KEEPING SELECTED CONFIGURATION")
        print("="*60)
        print(f"Left Camera (1):  Port {left_port} ✓")
        print(f"Right Camera (2): Port {right_port} ✓")
        print("="*60 + "\n")

    camera_config['left']['port'] = left_port
    camera_config['right']['port'] = right_port
    config['camera'] = camera_config

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

    print("\n" + "="*60)
    print("FINAL CAMERA CONFIGURATION")
    print("="*60)
    print(f"Left Camera (1):  Port {camera_config['left']['port']}")
    print(f"Right Camera (2): Port {camera_config['right']['port']}")
    print("="*60 + "\n")

    print("✅ Camera assignment complete!")

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

