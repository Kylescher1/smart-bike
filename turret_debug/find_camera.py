#!/usr/bin/env python3
"""
Find Camera Index by Name
Helps identify which camera index corresponds to a device name

Usage:
    python find_camera.py
    python find_camera.py --name "HD Camera"
"""

import cv2
import sys
import platform
import subprocess
from typing import Optional, Dict, List


def get_windows_camera_names() -> Dict[int, str]:
    """Get camera names on Windows using PowerShell"""
    camera_names = {}
    
    if platform.system() != "Windows":
        return camera_names
    
    try:
        # Use PowerShell to get camera device names
        ps_command = """
        Get-PnpDevice | Where-Object {
            $_.Class -eq 'Camera' -or $_.FriendlyName -like '*Camera*'
        } | Select-Object -Property FriendlyName, InstanceId | Format-Table -AutoSize
        """
        
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Camera' in line or 'camera' in line.lower():
                    # Try to extract name
                    parts = line.strip().split()
                    if parts:
                        name = ' '.join(parts)
                        camera_names[len(camera_names)] = name
    except Exception as e:
        print(f"Warning: Could not get camera names via PowerShell: {e}")
    
    return camera_names


def probe_cameras(max_index: int = 10) -> List[Dict]:
    """Probe all camera indices and return info"""
    cameras = []
    
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            continue
        
        # Try to get camera info
        info = {
            "index": index,
            "backend": cap.getBackendName(),
            "name": None,
            "width": None,
            "height": None,
            "fps": None,
            "works": False
        }
        
        # Try to read a frame to verify it works
        ret, frame = cap.read()
        if ret:
            info["works"] = True
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to get device name (Windows DirectShow)
            if platform.system() == "Windows":
                try:
                    # On Windows, we can try to get the device name
                    # This is a bit hacky but works sometimes
                    backend_name = cap.getBackendName()
                    if 'DSHOW' in backend_name or 'MSMF' in backend_name:
                        # Try to get device name via CAP_PROP_SETTINGS
                        # This sometimes triggers device selection dialog which shows name
                        pass
                except:
                    pass
        
        cap.release()
        cameras.append(info)
    
    return cameras


def find_camera_by_name(search_name: str, max_index: int = 10) -> Optional[int]:
    """Find camera index by searching for name in device list"""
    search_name_lower = search_name.lower()
    
    # On Windows, try to get camera names
    if platform.system() == "Windows":
        # Try using DirectShow to enumerate devices
        try:
            import pywintypes
            import win32com.client
            
            # Use Windows COM to access DirectShow
            system_devices = win32com.client.Dispatch("System.Devices.DeviceInformation")
            # This is complex, so we'll use a simpler approach
        except ImportError:
            pass
    
    # Fallback: probe all cameras and let user identify
    cameras = probe_cameras(max_index)
    
    print(f"\nSearching for camera matching '{search_name}'...")
    print("=" * 60)
    
    matching_indices = []
    for cam in cameras:
        if cam["works"]:
            # Check if name matches (if we have it)
            if cam["name"] and search_name_lower in cam["name"].lower():
                matching_indices.append(cam["index"])
            print(f"\nMATCH FOUND!")
            print(f"   Index: {cam['index']}")
            print(f"   Name: {cam['name']}")
            print(f"   Resolution: {cam['width']}x{cam['height']}")
            print(f"   Backend: {cam['backend']}")
    
    if matching_indices:
        return matching_indices[0]
    
    # If no match found, show all available cameras
    print(f"\nNo exact match found. Available cameras:")
    print("=" * 60)
    
    for cam in cameras:
        if cam["works"]:
            print(f"\nCamera Index {cam['index']}:")
            print(f"   Backend: {cam['backend']}")
            print(f"   Resolution: {cam['width']}x{cam['height']}")
            if cam['fps']:
                print(f"   FPS: {cam['fps']:.1f}")
            print(f"   Status: Working")
    
    print(f"\nTip: Try each index (0, 1, 2, etc.) with --camera flag")
    print(f"   Example: python yolo_gimbal.py --camera 0 --turret COM3")
    
    return None


def list_all_cameras(max_index: int = 10):
    """List all available cameras"""
    cameras = probe_cameras(max_index)
    
    print("\n" + "=" * 60)
    print("AVAILABLE CAMERAS")
    print("=" * 60)
    
    found_any = False
    for cam in cameras:
        if cam["works"]:
            found_any = True
            print(f"\nCamera Index {cam['index']}:")
            print(f"   Backend: {cam['backend']}")
            print(f"   Resolution: {cam['width']}x{cam['height']}")
            if cam['fps']:
                print(f"   FPS: {cam['fps']:.1f}")
            print(f"   Status: Working")
    
    if not found_any:
        print("\nNo working cameras found!")
        print("   Make sure your camera is connected and not in use by another application.")
    else:
        print(f"\nUse --camera <index> with your scripts")
        print(f"   Example: python yolo_gimbal.py --camera 0 --turret COM3")


def preview_camera(index: int):
    """Preview a specific camera"""
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        return
    
    print(f"\nPreviewing Camera {index}")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Add label
        cv2.putText(frame, f"Camera {index}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Camera {index} Preview", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find Camera Index by Name',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_camera.py
  python find_camera.py --name "HD Camera"
  python find_camera.py --preview 0
        """
    )
    parser.add_argument('--name', '-n', type=str, default=None,
                       help='Search for camera by name (e.g., "HD Camera")')
    parser.add_argument('--preview', '-p', type=int, default=None,
                       help='Preview camera at specified index')
    parser.add_argument('--max-index', type=int, default=10,
                       help='Maximum camera index to check (default: 10)')
    
    args = parser.parse_args()
    
    if args.preview is not None:
        preview_camera(args.preview)
        return
    
    if args.name:
        result = find_camera_by_name(args.name, args.max_index)
        if result is not None:
            print(f"\n✅ Use --camera {result} in your scripts")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        list_all_cameras(args.max_index)


if __name__ == '__main__':
    main()

