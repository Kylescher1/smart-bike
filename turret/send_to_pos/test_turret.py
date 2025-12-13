#!/usr/bin/env python3
"""
Turret System Test

Minimal test to verify turret system is working.
Tests each component individually before full integration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from src.hal.TurretGeometry import TurretGeometry
        from src.hal.TurretController import TurretController
        from src.hal.MultiCameraYOLO import MultiCameraYOLO
        from src.hal.TargetSelector import TargetSelector
        from src.hal.PositionCalculator import PositionCalculator
        from src.hal.Turret import Turret, TurretOutput, Position3D
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_geometry():
    """Test geometry calculations"""
    print("\nTesting geometry...")
    try:
        from src.hal.TurretGeometry import TurretGeometry
        import numpy as np
        
        geo = TurretGeometry()
        
        # Test camera info
        left_cam = geo.get_camera_info('left')
        print(f"  ✓ Left camera at {left_cam.position}")
        
        right_cam = geo.get_camera_info('right')
        print(f"  ✓ Right camera at {right_cam.position}")
        
        center_cam = geo.get_camera_info('center')
        print(f"  ✓ Center camera at {center_cam.position}")
        
        # Test angle estimation
        pan, tilt = geo.estimate_target_angles('left', 320, 240, 640, 480)
        print(f"  ✓ Estimated angles: pan={pan:.1f}°, tilt={tilt:.1f}°")
        
        # Test 3D position calculation
        xyz = geo.compute_3d_position(320, 240, 640, 480, 90.0, 90.0, 24.0)
        print(f"  ✓ 3D position: {xyz}")
        
        # Test spherical conversion
        az, el, dist = geo.cartesian_to_spherical(xyz)
        print(f"  ✓ Spherical: az={az:.1f}°, el={el:.1f}°, dist={dist:.2f}in")
        
        return True
    except Exception as e:
        print(f"  ✗ Geometry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controller(port):
    """Test Arduino connection"""
    print(f"\nTesting Arduino connection on {port}...")
    try:
        from src.hal.TurretController import TurretController
        import time
        
        controller = TurretController(port)
        
        if not controller.connect():
            print("  ✗ Failed to connect")
            return False
        
        print("  ✓ Connected to Arduino")
        
        # Get status
        controller.update_status()
        print(f"  ✓ Current position: pan={controller.bottom_pos}°, tilt={controller.top_pos}°")
        print(f"  ✓ Limits: pan {controller.bottom_min}-{controller.bottom_max}°, "
              f"tilt {controller.top_min}-{controller.top_max}°")
        
        # Test movement (small)
        print("  Testing small movement...")
        original_pan = controller.bottom_pos
        controller.move_to(original_pan + 5, controller.top_pos, force=True)
        time.sleep(0.5)
        controller.move_to(original_pan, controller.top_pos, force=True)
        print("  ✓ Servo movement works")
        
        # Test ToF (may not be implemented yet)
        tof_range = controller.get_tof_range()
        if tof_range:
            print(f"  ✓ ToF sensor: {tof_range:.2f} inches")
        else:
            print("  ⚠ ToF sensor not available (implement readToFRange() in Arduino)")
        
        controller.disconnect()
        print("  ✓ Disconnected")
        return True
        
    except Exception as e:
        print(f"  ✗ Controller test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("TURRET SYSTEM TEST")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check dependencies.")
        return
    
    # Test geometry
    if not test_geometry():
        print("\n❌ Geometry test failed.")
        return
    
    # Test Arduino (optional)
    print("\n" + "=" * 60)
    response = input("Test Arduino connection? (y/N, requires hardware): ").strip().lower()
    
    if response == 'y':
        port = input("Enter Arduino port (e.g., COM3, /dev/ttyUSB0): ").strip()
        if port:
            if not test_controller(port):
                print("\n❌ Controller test failed.")
                return
        else:
            print("  Skipped - no port provided")
    else:
        print("  Skipped Arduino test")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Upload turret_debug/turret_debug.ino to Arduino")
    print("2. Configure ToF sensor in Arduino code")
    print("3. Run: python turret_demo.py")


if __name__ == '__main__':
    main()

