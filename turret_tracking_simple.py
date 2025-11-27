"""
Simplified Turret Tracking - Works reliably
"""

import dill
import time
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.VISION.VISION_UPGRADE import VISION
from hal.TurretControl import TurretControl
from hal.TurretTracker import TurretTracker


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
    print("Turret Tracking System (Simplified)")
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
    
    # Initialize systems
    print("\n📹 Initializing VISION system...")
    vision = VISION(name="TurretVision", **vision_config)
    
    try:
        vision.start()
        print("✅ VISION started")
        time.sleep(2)
        
        print("\n🎯 Initializing TurretControl...")
        turret_port = "COM5"
        if sys.platform.startswith('linux'):
            turret_port = "/dev/ttyUSB0"
        
        turret = TurretControl(
            port=turret_port,
            baudrate=115200,
            servo1_min=15, servo1_max=50, servo1_home=35,
            servo2_min=0, servo2_max=180, servo2_home=90,
            deadzone=0.5,  # Reduced deadzone for more sensitive tracking
            kp=0.8,  # Increased gain for faster response
            max_speed=10.0  # Increased max speed
        )
        
        turret.connect()
        if not turret.connected:
            print("❌ Failed to connect to turret")
            return
        
        print("✅ TurretControl connected")
        turret.go_home()
        time.sleep(0.5)
        
        # Initialize tracker
        print("\n🎯 Initializing TurretTracker...")
        tracker = TurretTracker(
            vision=vision,
            turret=turret,
            tracking_mode="largest",
            min_confidence=0.3,
            max_tracking_distance=60.0,
            camera_config=vision_config  # Pass config for camera intrinsics
        )
        
        # Start tracking
        print("\n🚀 Starting tracking...")
        tracker.start_tracking()
        print("✅ Tracking started")
        
        # Start visualization in separate thread
        print("\n📺 Starting camera preview...")
        print("   Press 'q' in preview window to quit")
        
        def run_visualization():
            try:
                vision.debug_visual()
            except Exception as e:
                print(f"Visualization error: {e}")
        
        viz_thread = threading.Thread(target=run_visualization, daemon=True)
        viz_thread.start()
        time.sleep(1)
        
        # Main loop - monitor and print status
        print("\n✅ System running!")
        print("   Objects will be tracked automatically\n")
        
        try:
            while True:
                stats = tracker.get_stats()
                vision_data = vision.read()
                objects = vision_data.get('objects', [])
                
                with tracker.lock:
                    tracked_id = tracker.tracked_object_id
                
                # Print status every 2 seconds
                print(f"📊 Frames: {stats['frames_processed']}, "
                      f"Objects: {len(objects)}, "
                      f"Tracked: {stats['objects_tracked']}, "
                      f"ID: {tracked_id if tracked_id else 'None'}")
                
                if objects:
                    for obj in objects[:2]:
                        print(f"   - {obj.get('type')} (ID:{obj.get('id')}, conf:{obj.get('confidence', 0):.2f})")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
            tracker.stop_tracking()
            turret.disconnect()
            vision.stop()
            print("✅ Done")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            if 'tracker' in locals():
                tracker.stop_tracking()
            if 'turret' in locals():
                turret.disconnect()
            if 'vision' in locals():
                vision.stop()
        except:
            pass


if __name__ == "__main__":
    main()

