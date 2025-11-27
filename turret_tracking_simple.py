"""
Simplified Turret Tracking - Works reliably
"""

import dill
import time
import sys
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.VISION.VISION_UPGRADE import VISION
from hal.TurretControl import TurretControl
from hal.TurretTracker import TurretTracker

# Visualization imports
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available - debug visualization disabled")


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


def turret_angles_to_direction(s1_angle: float, s2_angle: float, 
                                s1_home: float = 35, s2_home: float = 90) -> np.ndarray:
    """
    Convert turret servo angles to 3D direction vector.
    
    This accounts for pan-tilt geometry where:
    - S2 (horizontal/pan) rotates around vertical axis
    - S1 (vertical/tilt) rotates around horizontal axis that rotates with S2
    
    Args:
        s1_angle: Servo 1 angle (vertical/tilt, limited range)
        s2_angle: Servo 2 angle (horizontal/pan, full range)
        s1_home: Home position for servo 1
        s2_home: Home position for servo 2
    
    Returns:
        3D direction vector (normalized) pointing where turret is aimed
    """
    # Calculate relative angles from home position
    # S2 controls horizontal (yaw/pan), S1 controls vertical (pitch/tilt)
    yaw_deg = s2_angle - s2_home     # Horizontal angle from forward
    pitch_deg = s1_angle - s1_home  # Vertical angle from horizontal
    
    # Convert to radians
    yaw_rad = np.deg2rad(yaw_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    
    # Convert to 3D direction vector
    # Camera coordinate system: X=right, Y=down, Z=forward
    # Pan-tilt mount: first rotate around Y (yaw), then around rotated X (pitch)
    # This is equivalent to:
    # 1. Rotate around Y axis by yaw_rad
    # 2. Rotate around the rotated X axis by pitch_rad
    
    # Standard pan-tilt transformation:
    x = np.sin(yaw_rad) * np.cos(pitch_rad)
    y = np.sin(pitch_rad)  # Pitch rotates around horizontal axis
    z = np.cos(yaw_rad) * np.cos(pitch_rad)
    
    direction = np.array([x, y, z])
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        return direction / norm
    else:
        return np.array([0, 0, 1])  # Default forward direction


def object_angles_to_position(theta_deg: float, alpha_deg: float, distance: float = 2.0) -> np.ndarray:
    """
    Convert object angles (theta, alpha) to 3D position at fixed distance.
    
    Args:
        theta_deg: Horizontal angle in degrees (positive = right)
        alpha_deg: Vertical angle in degrees (positive = up)
        distance: Fixed distance in meters
    
    Returns:
        3D position vector relative to turret
    """
    theta_rad = np.deg2rad(theta_deg)
    alpha_rad = np.deg2rad(alpha_deg)
    
    # Camera coordinate system: X=right, Y=down, Z=forward
    # theta is horizontal (yaw), alpha is vertical (pitch)
    x = distance * np.sin(theta_rad) * np.cos(alpha_rad)
    y = -distance * np.sin(alpha_rad)  # Negative because Y points down
    z = distance * np.cos(theta_rad) * np.cos(alpha_rad)
    
    return np.array([x, y, z])


def run_turret_debug_visualization(tracker: TurretTracker, turret: TurretControl, 
                                   fixed_distance: float = 2.0, update_rate: float = 0.1):
    """
    Run debug visualization showing turret position and object location.
    
    Args:
        tracker: TurretTracker instance
        turret: TurretControl instance
        fixed_distance: Fixed distance for object visualization (meters)
        update_rate: Update rate in seconds
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib not available - skipping debug visualization")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Turret Tracking Debug Visualization', fontsize=16, fontweight='bold')
    
    # 3D view
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_3d.set_title('3D View: Turret Aim & Object Position')
    ax_3d.set_xlabel('X (Right)')
    ax_3d.set_ylabel('Y (Down)')
    ax_3d.set_zlabel('Z (Forward)')
    
    # Top view (X-Z plane)
    ax_top = fig.add_subplot(2, 2, 2)
    ax_top.set_title('Top View (X-Z plane)')
    ax_top.set_xlabel('X (Right)')
    ax_top.set_ylabel('Z (Forward)')
    ax_top.grid(True, alpha=0.3)
    ax_top.set_aspect('equal')
    
    # Side view (Y-Z plane)
    ax_side = fig.add_subplot(2, 2, 3)
    ax_side.set_title('Side View (Y-Z plane)')
    ax_side.set_xlabel('Y (Down)')
    ax_side.set_ylabel('Z (Forward)')
    ax_side.grid(True, alpha=0.3)
    ax_side.set_aspect('equal')
    
    # Info panel
    ax_info = fig.add_subplot(2, 2, 4)
    ax_info.axis('off')
    ax_info.set_title('Status Information', fontweight='bold')
    
    plt.tight_layout()
    
    # Initialize plot elements
    turret_line_3d = None
    object_point_3d = None
    turret_line_top = None
    object_point_top = None
    turret_line_side = None
    object_point_side = None
    info_text = None
    
    # Visualization range
    vis_range = fixed_distance * 1.5
    
    print("📊 Debug visualization started - Close window to stop")
    
    try:
        while plt.get_fignums():  # Continue while figure exists
            # Get current turret position
            s1, s2 = turret.get_position()
            
            # Get tracked object info
            vision_data = tracker.vision.read()
            objects = vision_data.get('objects', [])
            
            tracked_obj = None
            theta_deg = None
            alpha_deg = None
            
            with tracker.lock:
                tracked_id = tracker.tracked_object_id
            
            # Find tracked object
            if tracked_id is not None:
                for obj in objects:
                    if obj.get('id') == tracked_id:
                        tracked_obj = obj
                        theta_deg = obj.get('theta', 0.0)
                        alpha_deg = obj.get('alpha', 0.0)
                        break
            
            # Calculate turret direction
            turret_dir = turret_angles_to_direction(s1, s2, turret.servo1_home, turret.servo2_home)
            turret_end = turret_dir * fixed_distance
            
            # Calculate object position
            if theta_deg is not None and alpha_deg is not None:
                object_pos = object_angles_to_position(theta_deg, alpha_deg, fixed_distance)
            else:
                object_pos = None
            
            # Clear axes
            ax_3d.clear()
            ax_top.clear()
            ax_side.clear()
            ax_info.clear()
            
            # Set up 3D view
            ax_3d.set_title('3D View: Turret Aim & Object Position')
            ax_3d.set_xlabel('X (Right)')
            ax_3d.set_ylabel('Y (Down)')
            ax_3d.set_zlabel('Z (Forward)')
            
            # Draw turret origin
            ax_3d.scatter([0], [0], [0], c='blue', s=100, marker='o', label='Turret')
            
            # Draw turret direction
            ax_3d.plot([0, turret_end[0]], [0, turret_end[1]], [0, turret_end[2]], 
                       'b-', linewidth=3, label='Turret Aim')
            ax_3d.scatter([turret_end[0]], [turret_end[1]], [turret_end[2]], 
                         c='blue', s=50, marker='^')
            
            # Draw object position if available
            if object_pos is not None:
                ax_3d.scatter([object_pos[0]], [object_pos[1]], [object_pos[2]], 
                             c='red', s=100, marker='*', label='Object')
                # Draw line from turret to object
                ax_3d.plot([0, object_pos[0]], [0, object_pos[1]], [0, object_pos[2]], 
                          'r--', linewidth=2, alpha=0.5, label='To Object')
            
            # Set 3D view limits
            ax_3d.set_xlim([-vis_range, vis_range])
            ax_3d.set_ylim([-vis_range, vis_range])
            ax_3d.set_zlim([0, vis_range * 2])
            ax_3d.legend()
            
            # Top view (X-Z plane)
            ax_top.set_title('Top View (X-Z plane)')
            ax_top.set_xlabel('X (Right)')
            ax_top.set_ylabel('Z (Forward)')
            ax_top.grid(True, alpha=0.3)
            ax_top.set_aspect('equal')
            
            # Draw turret direction (top view)
            ax_top.plot([0, turret_end[0]], [0, turret_end[2]], 'b-', linewidth=3, label='Turret Aim')
            ax_top.scatter([turret_end[0]], [turret_end[2]], c='blue', s=50, marker='^')
            
            # Draw object position (top view)
            if object_pos is not None:
                ax_top.scatter([object_pos[0]], [object_pos[2]], c='red', s=100, marker='*', label='Object')
                ax_top.plot([0, object_pos[0]], [0, object_pos[2]], 'r--', linewidth=2, alpha=0.5)
            
            ax_top.set_xlim([-vis_range, vis_range])
            ax_top.set_ylim([0, vis_range * 2])
            ax_top.legend()
            
            # Side view (Y-Z plane)
            ax_side.set_title('Side View (Y-Z plane)')
            ax_side.set_xlabel('Y (Down)')
            ax_side.set_ylabel('Z (Forward)')
            ax_side.grid(True, alpha=0.3)
            ax_side.set_aspect('equal')
            
            # Draw turret direction (side view)
            ax_side.plot([0, turret_end[1]], [0, turret_end[2]], 'b-', linewidth=3, label='Turret Aim')
            ax_side.scatter([turret_end[1]], [turret_end[2]], c='blue', s=50, marker='^')
            
            # Draw object position (side view)
            if object_pos is not None:
                ax_side.scatter([object_pos[1]], [object_pos[2]], c='red', s=100, marker='*', label='Object')
                ax_side.plot([0, object_pos[1]], [0, object_pos[2]], 'r--', linewidth=2, alpha=0.5)
            
            ax_side.set_xlim([-vis_range, vis_range])
            ax_side.set_ylim([0, vis_range * 2])
            ax_side.legend()
            
            # Info panel
            ax_info.axis('off')
            ax_info.set_title('Status Information', fontweight='bold', pad=20)
            
            stats = tracker.get_stats()
            info_lines = [
                f"Turret Position:",
                f"  S1 (Vertical): {s1:.1f}° (home={turret.servo1_home}°)",
                f"  S2 (Horizontal): {s2:.1f}° (home={turret.servo2_home}°)",
                f"",
                f"Turret Direction Vector:",
                f"  X: {turret_dir[0]:.3f}",
                f"  Y: {turret_dir[1]:.3f}",
                f"  Z: {turret_dir[2]:.3f}",
                f"",
            ]
            
            if tracked_obj is not None:
                info_lines.extend([
                    f"Tracked Object:",
                    f"  ID: {tracked_id}",
                    f"  Type: {tracked_obj.get('type', 'unknown')}",
                    f"  Confidence: {tracked_obj.get('confidence', 0.0):.2f}",
                    f"  Theta: {theta_deg:.2f}°",
                    f"  Alpha: {alpha_deg:.2f}°",
                    f"",
                    f"Object Position (at {fixed_distance}m):",
                    f"  X: {object_pos[0]:.3f}m",
                    f"  Y: {object_pos[1]:.3f}m",
                    f"  Z: {object_pos[2]:.3f}m",
                ])
            else:
                info_lines.extend([
                    f"Tracked Object: None",
                    f"  Total Objects: {len(objects)}",
                ])
            
            info_lines.extend([
                f"",
                f"Tracking Stats:",
                f"  Frames Processed: {stats['frames_processed']}",
                f"  Objects Tracked: {stats['objects_tracked']}",
                f"  Track Lost Count: {stats['track_lost_count']}",
            ])
            
            ax_info.text(0.05, 0.95, '\n'.join(info_lines), 
                        transform=ax_info.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.draw()
            plt.pause(update_rate)
            
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')


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
            deadzone=0.1,  # Very small deadzone - allow fine adjustments
            kp=1.5,  # Moderate gain for smooth centering
            max_speed=8.0,  # Moderate speed
            angle_scale_s1=1.0,  # Vertical axis scaling
            angle_scale_s2=1.5  # Horizontal axis scaling - adjust if not moving enough
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
            max_tracking_distance=120.0,  # Increased to allow tracking wider angles
            camera_config=vision_config  # Pass config for camera intrinsics
        )
        
        # Start tracking
        print("\n🚀 Starting tracking...")
        tracker.start_tracking()
        print("✅ Tracking started")
        
        # Start visualization in separate threads
        print("\n📺 Starting visualizations...")
        print("   Camera preview: Press 'q' to quit")
        print("   Debug visualization: Close window to stop")
        
        def run_camera_visualization():
            try:
                vision.debug_visual()
            except Exception as e:
                print(f"Camera visualization error: {e}")
        
        def run_debug_visualization():
            try:
                run_turret_debug_visualization(tracker, turret, fixed_distance=2.0, update_rate=0.1)
            except Exception as e:
                print(f"Debug visualization error: {e}")
        
        # Start camera preview
        camera_viz_thread = threading.Thread(target=run_camera_visualization, daemon=True)
        camera_viz_thread.start()
        
        # Start debug visualization (only if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            debug_viz_thread = threading.Thread(target=run_debug_visualization, daemon=True)
            debug_viz_thread.start()
        else:
            print("⚠️ Skipping debug visualization (matplotlib not available)")
        
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

