# Turret System - Quick Start

## What Was Built

A complete 3-camera turret tracking system that:
1. Uses 2 fisheye cameras to scout for objects across a wide field of view
2. Points the turret at detected targets
3. Tracks targets with a center camera using PID control
4. Measures depth with ToF sensor when locked on target
5. Computes 3D positions in world coordinates (spherical + cartesian)
6. Returns data via simple `read()` API

## File Structure

```
src/hal/
├── Turret.py              # Main class - use this!
├── TurretGeometry.py      # Coordinate math & transformations
├── TurretController.py    # Arduino serial communication
├── MultiCameraYOLO.py     # 3-camera YOLO detection
├── TargetSelector.py      # Target prioritization logic
├── PositionCalculator.py  # 3D position from pixel + depth
└── turret.py              # Module imports

turret_debug/
└── turret_debug.ino       # Arduino firmware (MODIFIED - now has GET_RANGE)

turret_demo.py             # Example usage script
TURRET_README.md           # Full documentation
```

## Quick Start

### 1. Upload Arduino Firmware

```bash
# Upload turret_debug/turret_debug.ino to your Arduino
# Modify readToFRange() function for your ToF sensor
# Set tof_available = true when sensor configured
```

### 2. Run Demo

```python
# Edit turret_demo.py:
PORT = 'COM3'  # Your Arduino port
CAMERAS = {
    'left': 0,    # Your left fisheye camera
    'right': 1,   # Your right fisheye camera
    'center': 2   # Your center camera
}

# Run it:
python turret_demo.py
```

### 3. Use in Your Code

```python
from src.hal.Turret import Turret

turret = Turret(
    port='COM3',
    cameras={'left': 0, 'right': 1, 'center': 2}
)
turret.start()

# Read data in your loop
while True:
    output = turret.read()
    
    # 3D positions (with depth)
    for pos3d in output.detections_3d:
        azimuth, elevation, distance = pos3d.position_spherical
        x, y, z = pos3d.position_xyz
        print(f"Object at az={azimuth:.1f}° el={elevation:.1f}° dist={distance:.2f}in")
    
    # All detections (2D from all cameras)
    for det in output.all_detections:
        print(f"{det.class_name} from {det.camera_id} camera")
    
    # Turret state
    print(f"Pan={output.turret_pose.pan_angle}°, Locked={output.is_locked}")

turret.stop()
```

## Key Concepts

### Fisheye cameras are scouts only
- They just tell turret "hey there's something at ~60° left"
- No 3D reprojection or complex fisheye math
- Turret points that direction, center camera takes over

### Center camera is the tracker
- Fine tracking with PID control
- When locked (centered), ToF measures depth
- Only center camera gets 3D positions

### Rolling buffer
- Keeps last 2 seconds of 3D detections
- Smooths depth readings
- Use `read()` for latest or `read_buffer(max_age=1.0)` for history

## Next Steps

1. **Calibrate ToF sensor** - Modify Arduino `readToFRange()` function
2. **Test tracking** - Run demo, verify servos track objects smoothly
3. **Tune PID** - Adjust kp/ki/kd if tracking is jerky or sluggish
4. **Verify geometry** - Check if 3D positions match reality
5. **Integrate** - Import Turret class into your main application

## What Makes This Different

- **Simple architecture** - Fisheye cameras just scout, no complex reprojection
- **Automatic tracking** - System handles target selection and servo control
- **Clean API** - Just call `read()` to get 3D positions
- **Reuses existing code** - Built on your yolo_gimbal.py foundation
- **Unit sphere mapping** - Returns both cartesian (xyz) and spherical (az, el, dist)

See TURRET_README.md for complete documentation!

