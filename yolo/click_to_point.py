#!/usr/bin/env python3
"""
Click-to-Point Turret Control

Click anywhere on the camera feed to point the gimbal at that location.
Displays TF03 LiDAR distance reading on screen.

Usage:
    python click_to_point.py --source 0 --port /dev/ttyUSB0
"""

import argparse
import sys
import time
import serial
import threading
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hal.cam.Camera import ThreadedCamera, CAMERA_CONFIG


class TurretController:
    """Simple serial controller for the gimbal turret."""
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.lock = threading.Lock()
        self.distance_cm = None
        self.connected = False
        
        # Servo limits (from turret_debug.ino)
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        self.top_home = 90
        self.bottom_home = 90
        
        # Current positions
        self.top_pos = self.top_home
        self.bottom_pos = self.bottom_home
    
    def connect(self) -> bool:
        """Connect to the turret controller."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)  # Wait for Arduino reset
            self.serial.reset_input_buffer()
            self.connected = True
            print(f"[TURRET] Connected to {self.port}")
            return True
        except Exception as e:
            print(f"[TURRET] Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the turret controller."""
        if self.serial:
            self.serial.close()
            self.connected = False
    
    def send_command(self, cmd: str) -> str:
        """Send a command and return response."""
        if not self.connected:
            return ""
        
        with self.lock:
            try:
                self.serial.reset_input_buffer()
                self.serial.write(f"{cmd}\n".encode())
                time.sleep(0.05)
                
                response = ""
                timeout = time.time() + 0.2
                while time.time() < timeout:
                    if self.serial.in_waiting:
                        response += self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                        if '\n' in response:
                            break
                    time.sleep(0.01)
                return response.strip()
            except Exception as e:
                print(f"[TURRET] Command error: {e}")
                return ""
    
    def set_position(self, top: int, bottom: int):
        """Set both servo positions."""
        top = max(self.top_min, min(self.top_max, top))
        bottom = max(self.bottom_min, min(self.bottom_max, bottom))
        
        self.send_command(f"TOP:{top}")
        self.send_command(f"BOTTOM:{bottom}")
        self.top_pos = top
        self.bottom_pos = bottom
    
    def home(self):
        """Move to home position."""
        self.send_command("HOME")
        self.top_pos = self.top_home
        self.bottom_pos = self.bottom_home
    
    def get_distance(self) -> float:
        """Get distance from TF03 LiDAR in cm."""
        response = self.send_command("GET_RANGE")
        
        if "Range:" in response and "in" in response:
            try:
                dist_str = response.split("Range:")[1].split("in")[0].strip()
                distance_inches = float(dist_str)
                self.distance_cm = distance_inches * 2.54
                return self.distance_cm
            except:
                pass
        return self.distance_cm if self.distance_cm else -1


def pixel_to_servo(x: int, y: int, frame_w: int, frame_h: int, 
                   turret: TurretController,
                   fov_h: float = 50.0) -> tuple:
    """
    Convert pixel coordinates to servo angles based on camera FOV.
    
    Click offset from frame center = angular offset to move from CURRENT position.
    This makes clicking center do nothing, clicking off-center moves to center on that point.
    
    Args:
        x, y: Click position in pixels
        frame_w, frame_h: Frame dimensions (e.g., 1280x720)
        turret: TurretController instance
        fov_h: Horizontal field of view in degrees (default 50°)
    """
    # Calculate vertical FOV from aspect ratio
    aspect_ratio = frame_h / frame_w
    fov_v = fov_h * aspect_ratio  # ~28° for 1280x720 with 50° horizontal
    
    # Pixel offset from center of frame
    dx = x - frame_w / 2  # Positive = right of center
    dy = y - frame_h / 2  # Positive = below center
    
    # Convert pixel offset to angular offset (degrees)
    # At frame edge: dx = frame_w/2 corresponds to fov_h/2 degrees
    angle_x = (dx / (frame_w / 2)) * (fov_h / 2)  # Horizontal angle offset
    angle_y = (dy / (frame_h / 2)) * (fov_v / 2)  # Vertical angle offset
    
    # Add angular offset to CURRENT position (not home)
    # Bottom servo (pan): click right of center → increase servo angle
    bottom_angle = turret.bottom_pos + int(angle_x)
    
    # Top servo (tilt): click below center → decrease servo angle (inverted)
    top_angle = turret.top_pos - int(angle_y)
    
    # Clamp to limits
    bottom_angle = max(turret.bottom_min, min(turret.bottom_max, bottom_angle))
    top_angle = max(turret.top_min, min(turret.top_max, top_angle))
    
    return top_angle, bottom_angle


def main():
    parser = argparse.ArgumentParser(description="Click-to-Point Turret Control")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port for turret")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    args = parser.parse_args()
    
    # Connect to turret
    turret = TurretController(args.port, args.baudrate)
    if not turret.connect():
        print("[ERROR] Could not connect to turret. Check port and connection.")
        return 1
    
    # Home the turret
    print("[INFO] Homing turret...")
    turret.home()
    time.sleep(1)
    
    # Open camera (1280x720 to match 50° FOV calibration)
    camera_config = CAMERA_CONFIG.copy()
    camera_config.update({"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"})
    
    try:
        camera = ThreadedCamera(index=args.source, config=camera_config)
        camera.open()
        print(f"[INFO] Camera {args.source} opened ({camera.width}x{camera.height})")
    except Exception as e:
        print(f"[ERROR] Failed to open camera: {e}")
        turret.disconnect()
        return 1
    
    # Mouse callback
    click_pos = [None]
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
    
    window_name = "Click to Point - Press 'q' to quit, 'h' to home"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    
    # Distance update timing
    last_distance_time = 0
    distance_interval = 0.5  # Update distance every 500ms
    
    print("[INFO] Click on the video to point the camera there.")
    print("[INFO] Press 'h' to home, 'q' to quit.")
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_h, frame_w = frame.shape[:2]
            
            # Handle click
            if click_pos[0] is not None:
                x, y = click_pos[0]
                top_angle, bottom_angle = pixel_to_servo(x, y, frame_w, frame_h, turret)
                
                print(f"[CLICK] ({x}, {y}) -> TOP:{top_angle}, BOTTOM:{bottom_angle}")
                turret.set_position(top_angle, bottom_angle)
                
                # Draw click marker
                cv2.circle(frame, (x, y), 20, (0, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                
                click_pos[0] = None
            
            # Update distance periodically
            current_time = time.time()
            if current_time - last_distance_time > distance_interval:
                turret.get_distance()
                last_distance_time = current_time
            
            # Draw crosshair at center
            cx, cy = frame_w // 2, frame_h // 2
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
            
            # Draw info overlay
            info_lines = [
                f"TOP: {turret.top_pos}  BOTTOM: {turret.bottom_pos}",
            ]
            
            if turret.distance_cm and turret.distance_cm > 0:
                info_lines.append(f"Distance: {turret.distance_cm:.1f} cm ({turret.distance_cm/100:.2f} m)")
            else:
                info_lines.append("Distance: --")
            
            # Draw semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 25 + 25 * len(info_lines)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("[INFO] Homing...")
                turret.home()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        print("[INFO] Cleaning up...")
        camera.close()
        turret.disconnect()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

