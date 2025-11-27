# Turret Tracking System

This system automatically tracks objects detected by the VISION system and centers them using a two-servo turret.

## Hardware Setup

- **Camera**: Single camera mounted on top of the turret
- **Servo 1**: Vertical axis (limited range: 15-50 degrees)
- **Servo 2**: Horizontal axis (full range: 0-180 degrees)
- **Arduino/ESP32**: Controls servos via serial communication

## Architecture

```
VISION System → TurretTracker → TurretControl → Arduino → Servos
```

1. **VISION System** (`VISION_UPGRADE.py`): Detects objects using YOLO and returns their positions
2. **TurretTracker** (`TurretTracker.py`): Selects target object and calculates tracking angles
3. **TurretControl** (`TurretControl.py`): Sends servo commands to Arduino via serial
4. **Arduino** (`turret_control.ino`): Receives commands and controls servos

## Files

- `src/hal/TurretControl.py`: Python class for serial communication with Arduino
- `src/hal/TurretTracker.py`: Python class that integrates VISION with turret control
- `turret/turret_control/turret_control.ino`: Arduino code for servo control
- `turret_tracking_example.py`: Example script demonstrating usage

## Arduino Setup

1. Upload `turret/turret_control/turret_control.ino` to your ESP32/Arduino
2. Connect servos:
   - Servo 1 (vertical) → Pin 14
   - Servo 2 (horizontal) → Pin 33
3. Set serial baudrate to 115200

## Usage

### Basic Example

```python
from hal.VISION.VISION_UPGRADE import VISION
from hal.TurretControl import TurretControl
from hal.TurretTracker import TurretTracker

# Initialize VISION system
vision = VISION(name="TurretVision", **vision_config)
vision.start()

# Initialize turret control
turret = TurretControl(
    port="/dev/ttyUSB0",  # or "COM3" on Windows
    baudrate=115200,
    servo1_min=15, servo1_max=50, servo1_home=35,
    servo2_min=0, servo2_max=180, servo2_home=90
)
turret.connect()

# Initialize tracker
tracker = TurretTracker(
    vision=vision,
    turret=turret,
    tracking_mode="largest",  # or "highest_confidence" or "class"
    target_class=None,  # Set class name if mode="class"
    min_confidence=0.3
)

# Start tracking
tracker.start_tracking()

# ... run your code ...

# Stop tracking
tracker.stop_tracking()
turret.disconnect()
vision.stop()
```

### Running the Example

```bash
python turret_tracking_example.py
```

## Tracking Modes

1. **"largest"**: Track the object with the largest bounding box
2. **"highest_confidence"**: Track the object with highest confidence score
3. **"class"**: Track objects of a specific class (e.g., "person", "car")

## Configuration

### TurretControl Parameters

- `port`: Serial port (e.g., "/dev/ttyUSB0" or "COM3")
- `baudrate`: Serial baudrate (default: 115200)
- `servo1_min/max`: Vertical servo limits (default: 15-50)
- `servo2_min/max`: Horizontal servo limits (default: 0-180)
- `deadzone`: Deadzone in degrees (default: 2.0)
- `kp`: Proportional gain (default: 0.5)
- `max_speed`: Max movement speed in degrees/update (default: 5.0)

### TurretTracker Parameters

- `tracking_mode`: "largest", "highest_confidence", or "class"
- `target_class`: Class name for class-based tracking
- `min_confidence`: Minimum confidence threshold (default: 0.3)
- `max_tracking_distance`: Max angular distance to track (default: 30.0 degrees)

## How It Works

1. **Object Detection**: VISION system continuously detects objects using YOLO
2. **Object Selection**: TurretTracker selects target based on tracking mode
3. **Angle Calculation**: Object position (unit circle coordinates) converted to angles
4. **Servo Control**: TurretControl sends commands to Arduino to center object
5. **Smoothing**: Proportional control with deadzone prevents jitter

## Coordinate System

- **Horizontal (theta)**: Positive = right, Negative = left
- **Vertical (alpha)**: Positive = up, Negative = down
- **Servo 1**: Controls vertical axis (limited range)
- **Servo 2**: Controls horizontal axis (full range)

## Troubleshooting

### Turret not moving
- Check serial port connection
- Verify Arduino is running `turret_control.ino`
- Check servo connections and power

### Objects not being tracked
- Verify VISION system is detecting objects
- Check `min_confidence` threshold
- Ensure objects are within `max_tracking_distance`

### Jittery movement
- Increase `deadzone` parameter
- Decrease `kp` (proportional gain)
- Decrease `max_speed`

### Wrong tracking direction
- Check servo wiring (servo 1 = vertical, servo 2 = horizontal)
- Verify angle calculations match your coordinate system

## Notes

- The system runs at ~30 Hz update rate
- Tracking automatically stops if no valid target is found for 1 second
- Servo positions are clamped to safe limits
- Serial commands are rate-limited to prevent overwhelming Arduino

