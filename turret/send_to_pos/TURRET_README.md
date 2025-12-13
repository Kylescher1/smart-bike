# Turret 3D Tracking System

A complete 3-camera turret tracking system with YOLO object detection, servo control, and 3D position calculation.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TURRET SYSTEM                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Left Fish  │  │Right Fish  │  │  Center    │        │
│  │  (Scout)   │  │  (Scout)   │  │ (Tracker)  │        │
│  └─────┬──────┘  └──────┬─────┘  └──────┬─────┘        │
│        │                │               │               │
│        └────────┬───────┴───────┬───────┘               │
│                 │ YOLO Detect   │                       │
│                 └───────┬───────┘                       │
│                         │                               │
│                 ┌───────▼───────┐                       │
│                 │Target Selector│                       │
│                 └───────┬───────┘                       │
│                         │                               │
│              ┌──────────▼──────────┐                    │
│              │ Servo Controller    │◄──┐                │
│              │  (Pan/Tilt)         │   │                │
│              └──────────┬──────────┘   │                │
│                         │               │                │
│                 ┌───────▼────────┐      │                │
│                 │ ToF Sensor     │      │                │
│                 └───────┬────────┘      │                │
│                         │               │                │
│              ┌──────────▼──────────┐    │                │
│              │3D Position Calc     │    │                │
│              │(Spherical Coords)   │    │                │
│              └──────────┬──────────┘    │                │
│                         │               │                │
│                 ┌───────▼────────┐      │                │
│                 │Rolling Buffer   │      │                │
│                 └───────┬────────┘      │                │
│                         │               │                │
│                   read()│               │                │
│                         ▼               │                │
│                 User Application        │                │
│                                         │                │
└─────────────────────────────────────────┴────────────────┘
                                          │
                                    Arduino/Serial
```

## System Components

### Hardware

- **2 Fisheye Cameras** (left/right) - Static, mounted at z=0, pointing 30° outward
- **1 Center Camera** - Mounted on turret platform at z=5.14", actively tracks targets  
- **1 ToF Sensor** - Mounted at z=3.80", measures depth when locked on target
- **2 Servos** - Bottom (pan 0-180°), Top (tilt 60-120°)
- **Arduino** - Controls servos and reads ToF sensor

### Software Modules

1. **`TurretGeometry.py`** - Physical geometry, coordinate transformations, quaternion rotations
2. **`TurretController.py`** - Serial communication with Arduino, servo control
3. **`MultiCameraYOLO.py`** - 3 parallel YOLO detection streams
4. **`TargetSelector.py`** - Prioritizes targets, decides where to point turret
5. **`PositionCalculator.py`** - Computes 3D positions from pixel + depth
6. **`Turret.py`** - Main class integrating all subsystems

## How It Works

### Detection & Tracking Flow

1. **Scout Phase** - Fisheye cameras detect objects across wide field of view
2. **Target Selection** - System prioritizes largest/closest/priority-class objects
3. **Point Phase** - Turret points at estimated target direction
4. **Acquire Phase** - Center camera detects target, fine-tunes pointing with PID
5. **Lock Phase** - When centered (error < threshold), considered "locked"
6. **Measure Phase** - ToF sensor reads depth while locked
7. **Calculate Phase** - Compute 3D position in world coordinates
8. **Buffer Phase** - Store Position3D in rolling buffer (2 seconds)

### Coordinate Systems

**World Frame:**
- Origin: Turret rotation center
- +X: Forward
- +Y: Right
- +Z: Up
- Units: Inches

**Spherical Coordinates:**
- Azimuth: Angle in XY plane from +X (0-360°)
- Elevation: Angle from XY plane (-90 to +90°)
- Distance: Radial distance (inches)

## Usage

### Basic Example

```python
from src.hal.Turret import Turret

# Initialize turret
turret = Turret(
    port='COM3',  # Arduino serial port
    cameras={
        'left': 0,     # Left fisheye camera index
        'right': 1,    # Right fisheye camera index
        'center': 2    # Center tracking camera index
    },
    target_classes=['person', 'bottle']  # Optional: only track these
)

# Start system
turret.start()

# Read detections
output = turret.read()

print(f"Locked: {output.is_locked}")
print(f"Pan: {output.turret_pose.pan_angle}°")

# Show 3D positions
for pos3d in output.detections_3d:
    az, el, dist = pos3d.position_spherical
    print(f"{pos3d.detection.class_name}: "
          f"azimuth={az:.1f}°, elevation={el:.1f}°, distance={dist:.2f}in")

# Stop when done
turret.stop()
```

### Context Manager

```python
with Turret(port='COM3', cameras={...}) as turret:
    while True:
        output = turret.read()
        # Process output...
```

### Run Demo

```bash
python turret_demo.py
```

## Configuration

### Camera Indices

Find your camera indices:
```bash
python src/Debug_Tools/fuck_you_camerafinder.py
```

### Arduino Port

Find your Arduino port:
```bash
# Linux
ls /dev/ttyUSB* /dev/ttyACM*

# Windows
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
```

### ToF Sensor Setup

The Arduino firmware includes a placeholder `readToFRange()` function. Modify it based on your sensor:

**For VL53L0X (I2C):**
```cpp
#include <VL53L0X.h>
VL53L0X sensor;

float readToFRange() {
  uint16_t mm = sensor.readRangeSingleMillimeters();
  return mm / 25.4;  // Convert mm to inches
}
```

**For Sharp GP2Y0A21YK (Analog):**
```cpp
float readToFRange() {
  int raw = analogRead(PIN_TOF_ANALOG);
  float volts = raw * (5.0 / 1023.0);
  return 27.86 * pow(volts, -1.15);  // Calibration curve
}
```

Set `tof_available = true;` in Arduino code once configured.

## API Reference

### `Turret` Class

**Methods:**
- `start()` - Initialize and start all subsystems
- `read() -> TurretOutput` - Get latest state and detections
- `read_buffer(max_age=1.0) -> List[Position3D]` - Get time-windowed detections
- `stop()` - Cleanup and shutdown

### `TurretOutput` Dataclass

```python
@dataclass
class TurretOutput:
    detections_3d: List[Position3D]  # 3D positions with depth
    all_detections: List[Detection]  # All 2D detections
    current_target: Optional[Target]  # Active target
    turret_pose: TurretPose          # Current pan/tilt angles
    is_locked: bool                  # Locked on target?
    timestamp: float
```

### `Position3D` Dataclass

```python
@dataclass
class Position3D:
    detection: Detection                      # Original detection
    position_xyz: np.ndarray                  # Cartesian (x,y,z)
    position_spherical: Tuple[float,float,float]  # (az, el, dist)
    pan_angle: float                          # Servo angle when detected
    tilt_angle: float
    depth: float                              # ToF reading (inches)
    has_valid_depth: bool                     # True if real depth
    timestamp: float
```

## Dependencies

```bash
pip install numpy scipy opencv-python ultralytics pyserial
```

## Troubleshooting

**"No cameras found"**
- Check camera indices with camera finder tool
- Verify cameras are not in use by another program

**"Failed to connect to Arduino"**
- Check serial port name
- Ensure Arduino is running `turret_debug.ino`
- No other programs using serial port

**"No 3D positions"**
- Turret must be locked on target (error < threshold)
- ToF sensor must return valid readings
- Check `tof_available` flag in Arduino code

**Servos jittery**
- Adjust PID gains (kp, ki, kd) in Turret constructor
- Increase deadzone threshold
- Check power supply to servos

## Files

- `src/hal/Turret.py` - Main turret class
- `src/hal/TurretGeometry.py` - Geometry and coordinates
- `src/hal/TurretController.py` - Arduino communication
- `src/hal/MultiCameraYOLO.py` - Multi-camera detection
- `src/hal/TargetSelector.py` - Target prioritization
- `src/hal/PositionCalculator.py` - 3D position math
- `turret_debug/turret_debug.ino` - Arduino firmware
- `turret_demo.py` - Example usage

## License

Part of the smart-bike project.

