#!/usr/bin/env python3
"""
Standalone script to visualize YOLO-detected objects in 3D spherical representation.

This script initializes the vision system and displays detected objects as points
on a unit sphere using their x, y, z coordinates.

Usage:
    python visualize_spherical_3d.py
"""

import dill
import sys
from pathlib import Path

def load_config():
    """Load configuration from config.dill."""
    config_path = Path("config.dill")
    if not config_path.exists():
        print(f"❌ Error: {config_path} not found")
        sys.exit(1)
    
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        print("✅ Configuration loaded")
        return config.get('camera', {})
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def initialize_vision(camera_config):
    """Initialize the vision system."""
    import importlib.util
    
    # Get vision class path
    who_to_run = camera_config.get('who_to_run', 'src.hal.VISION.VISION_UPGRADE.VISION')
    module_path, class_name = who_to_run.rsplit(".", 1)
    
    # Import module
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        print(f"❌ Error: Cannot find module {module_path}")
        sys.exit(1)
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get vision class
    VisionClass = getattr(module, class_name)
    
    # Extract config
    left_config = camera_config.get('left', {})
    right_config = camera_config.get('right', {})
    yolo_config = camera_config.get('yolo', {})
    
    # Create vision instance
    vision = VisionClass(
        name="3D Visualization",
        left=left_config,
        right=right_config,
        yolo=yolo_config,
        baseline=camera_config.get('baseline', 0.12),
        focal_length_px=camera_config.get('focal_length_px', 800.0),
        buffer_size=camera_config.get('buffer_size', 2),
        fov_horizontal=camera_config.get('fov_horizontal', 60.0),
        fov_vertical=camera_config.get('fov_vertical', 45.0),
    )
    
    return vision

def main():
    """Main function."""
    print("=" * 60)
    print("3D Spherical Object Visualization")
    print("=" * 60)
    
    # Load configuration
    camera_config = load_config()
    
    # Initialize vision system
    vision = initialize_vision(camera_config)
    
    # Start vision system
    print("\n🚀 Starting vision system...")
    try:
        vision.start()
        print("✅ Vision system started")
        
        # Wait a moment for system to initialize
        import time
        time.sleep(2.0)
        
        # Start 3D visualization
        print("\n🎨 Starting 3D spherical visualization...")
        print("   - Objects are displayed as colored points on a unit sphere")
        print("   - Each object type has a unique color")
        print("   - Close the window to exit")
        print()
        
        vision.debug_spherical_3d()
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🛑 Stopping vision system...")
        vision.stop()
        print("✅ Vision system stopped")

if __name__ == "__main__":
    main()

