# Turret Complete Control - Arduino Sketch

## Overview

`turret_complete.ino` is a comprehensive Arduino sketch that combines **all turret functionality** into one system:

- ✅ **Servo Control** - Full compatibility with `yolo_gimbal.py` for automatic object tracking
- ✅ **TF03 LiDAR** - Distance measurement with checksum validation
- ✅ **MPU-6050 Gyro/Accelerometer** - Orientation and motion sensing
- ✅ **Motor Vibration** - PWM-controlled vibration motors
- ✅ **LED Indicators** - Status and feedback lighting
- ✅ **Serial Command Interface** - Full remote control at 115200 baud

This sketch is **production-ready** and designed to work seamlessly with the Python tracking system while also supporting manual control and sensor reading.

## Hardware Requirements

### Required Components

| Component | Pin | Connection | Notes |
|-----------|-----|------------|-------|
| Top Servo | D3 | Signal wire | Tilt servo (vertical) |
| Bottom Servo | D7 | Signal wire | Pan servo (horizontal) |
| Motor 1 | D5 | PWM output | Vibration motor 1 |
| Motor 2 | D6 | PWM output | Vibration motor 2 |
| LED 1 | D8 | Digital output | Status LED 1 |
| LED 2 | D10 | Digital output | Status LED 2 |
| MPU-6050 SDA | A4 | I2C data | Gyro/Accelerometer |
| MPU-6050 SCL | A5 | I2C clock | Gyro/Accelerometer |
| TF03 RX | D11 | SoftwareSerial | LiDAR Brown wire |
| TF03 TX | D12 | SoftwareSerial | LiDAR Blue wire |

### Power Requirements

- **Arduino**: 5V USB or 7-12V barrel jack
- **Servos**: External 5-6V power supply (1-2A recommended)
- **TF03 LiDAR**: 5V (from Arduino or external)
- **MPU-6050**: 3.3V-5V (from Arduino)
- **Motors**: 5V PWM from Arduino (or external if high power)

### Wiring Diagram

```
Arduino Nano/Uno
┌─────────────────┐
│                 │
│  D3  ───────────┼─── Top Servo (Signal)
│  D7  ───────────┼─── Bottom Servo (Signal)
│                 │
│  D5  ───────────┼─── Motor 1 (PWM)
│  D6  ───────────┼─── Motor 2 (PWM)
│                 │
│  D8  ───────────┼─── LED 1 (+ resistor)
│  D10 ───────────┼─── LED 2 (+ resistor)
│                 │
│  A4 (SDA) ──────┼─── MPU-6050 SDA
│  A5 (SCL) ──────┼─── MPU-6050 SCL
│                 │
│  D11 ───────────┼─── TF03 Brown (RX)
│  D12 ───────────┼─── TF03 Blue (TX)
│                 │
│  GND ───────────┼─── Common Ground (all devices)
│  5V  ───────────┼─── Power (MPU, TF03, LEDs)
│                 │
└─────────────────┘

Servos: Use external 5-6V power supply
  - Red wire: +5V (external)
  - Brown/Black: GND (common with Arduino)
  - Orange/Yellow: Signal (to Arduino pin)
```

## Installation

### 1. Upload to Arduino

1. Open `turret_complete.ino` in Arduino IDE
2. Select your board: **Tools → Board → Arduino Nano** (or Uno)
3. Select your port: **Tools → Port → COM3** (or /dev/ttyUSB0)
4. Click **Upload** (or Ctrl+U)

### 2. Verify Operation

Open Serial Monitor at **115200 baud**:

```
=== TURRET COMPLETE CONTROL ===
Initializing systems...
  ✓ MPU-6050 (Gyro/Accel) initialized
  ✓ TF03 LiDAR initialized

=== SYSTEM READY ===
Servo limits - Top: 60-120°, Bottom: 0-180°
Type HELP for available commands
```

If any sensor shows ✗, it means that sensor is not connected (the system will continue to work without it).

## Serial Commands

### Servo Control (for yolo_gimbal.py)

```
HOME                    Move both servos to 90° home position
TOP:90                  Set top servo to 90° (0-180)
BOTTOM:45               Set bottom servo to 45° (0-180)
BOTH:90                 Set both servos to same angle
STATUS                  Get current servo positions and limits
```

### Distance Reading (for yolo_gimbal.py)

```
DISTANCE                Get TF03 LiDAR distance in cm
                        Returns: "OK: DISTANCE:123.4 cm"

GET_RANGE               Get TF03 LiDAR distance in inches
                        Returns: "OK: Range: 48.58 in"
```

### Sensor Reading

```
GYRO                    Read MPU-6050 accelerometer/gyro
READ_SENSORS            Read all sensors (LiDAR + Gyro + Accel)
```

### Motor Control (Vibration)

```
MOTOR1:200              Set motor 1 to speed 200 (0-255)
MOTOR2:150              Set motor 2 to speed 150 (0-255)
VIBRATE:500             Quick vibration for 500ms
```

### LED Control

```
LED1:1                  Turn LED 1 ON
LED1:0                  Turn LED 1 OFF
LED2:1                  Turn LED 2 ON
BLINK:3                 Blink LEDs 3 times
```

### Servo Limit Configuration

```
SET_TOP_MIN:60          Set top servo minimum limit
SET_TOP_MAX:120         Set top servo maximum limit
SET_BOTTOM_MIN:0        Set bottom servo minimum limit
SET_BOTTOM_MAX:180      Set bottom servo maximum limit
GET_LIMITS              Display current limits
```

### Servo Testing (Safety Feature)

```
TEST_TOP_MIN            Gradually test top servo minimum
TEST_TOP_MAX            Gradually test top servo maximum
TEST_BOTTOM_MIN         Gradually test bottom servo minimum
TEST_BOTTOM_MAX         Gradually test bottom servo maximum
```

**Note:** Test commands will move servos gradually (1° at a time) and can be stopped by pressing any key.

## Usage with yolo_gimbal.py

### Basic Tracking

```bash
# Track person with distance display (default: enabled)
python yolo_gimbal.py --camera 0 --turret COM3 --class person

# Track any object
python yolo_gimbal.py --camera 0 --turret COM3

# Disable distance reading if TF03 not connected
python yolo_gimbal.py --camera 0 --turret COM3 --disable-distance
```

### With RKNN Acceleration (Rock Pi 5B)

```bash
python yolo_gimbal.py --camera 5 --turret /dev/ttyUSB0 --rknn --class person
```

### Advanced Options

```bash
# Maximum performance with distance
python yolo_gimbal.py --camera 0 --turret COM3 \
  --rknn --control-rate 60 --camera-fps 60 --class person

# With error plotting for debugging
python yolo_gimbal.py --camera 0 --turret COM3 \
  --class person --error-plot --timing
```

## Command Response Format

All commands return responses in this format:

### Success Response
```
OK: <message>
```

### Error Response
```
ERROR: <message>
```

### Data Response
```
OK: <parameter>:<value> <unit>
```

Examples:
```
> HOME
OK: Moved to home position

> DISTANCE
OK: DISTANCE:145.2 cm

> TOP:200
ERROR: Angle must be 0-180
```

## Sensor Details

### TF03 LiDAR

- **Range**: 0.1m - 180m (0.3ft - 590ft)
- **Accuracy**: ±1cm @ <10m
- **Update Rate**: 100Hz
- **Protocol**: UART 115200 baud with checksum
- **Output Format**: 
  - `DISTANCE` command returns cm (for Python compatibility)
  - `GET_RANGE` command returns inches (legacy)

### MPU-6050 (GY-521)

- **Accelerometer**: ±2g, ±4g, ±8g, ±16g (configured for ±2g)
- **Gyroscope**: ±250°/s, ±500°/s, ±1000°/s, ±2000°/s
- **Update Rate**: ~100Hz
- **Temperature**: Built-in sensor
- **Protocol**: I2C @ 400kHz

## Features

### Auto-Detach Servos

Servos automatically detach after 2 seconds of inactivity to reduce:
- **Buzzing noise**
- **Power consumption**  
- **Heat generation**

They re-attach automatically when a new command is received.

### Servo Smoothing

- Only updates servo if angle actually changed
- Reduces jitter and buzzing
- Improves tracking smoothness

### Rate Limiting Compatible

Works with `yolo_gimbal.py` rate limiting:
- Commands sent at max 20Hz
- Minimum angle change: 0.5°
- Prevents command flooding

### Robust Sensor Reading

- **TF03 LiDAR**: Checksum validation ensures data integrity
- **MPU-6050**: Error handling for I2C communication
- **Graceful degradation**: System works even if sensors fail

## Troubleshooting

### Servos Not Moving

1. Check power supply (servos need separate 5-6V power, 1-2A)
2. Verify signal wires connected to correct pins
3. Check limits: `GET_LIMITS` command
4. Try `HOME` command first
5. Use `STATUS` to see current positions

### LiDAR Not Working

1. Check wiring:
   - Brown (TX) → Arduino D11 (RX)
   - Blue (RX) → Arduino D12 (TX)
   - Red → 5V
   - Black → GND
2. Verify 115200 baud rate (factory default)
3. Use `DISTANCE` command to test
4. Check Serial Monitor for "✓ TF03 LiDAR initialized"

### MPU-6050 Not Responding

1. Check I2C connections:
   - SDA → A4
   - SCL → A5
   - VCC → 5V (or 3.3V)
   - GND → GND
2. Use `GYRO` command to test
3. Check Serial Monitor for "✓ MPU-6050 initialized"
4. Try different MPU-6050 address if needed (0x69)

### Python Script Can't Connect

1. Check COM port: `python yolo_gimbal.py --list-ports`
2. Close Serial Monitor (can't have two connections)
3. Check baud rate: 115200
4. Try reconnecting USB cable
5. On Linux: Add user to dialout group: `sudo usermod -a -G dialout $USER`

### Distance Reading Shows Wrong Values

1. Ensure nothing blocking LiDAR lens
2. Check target surface (works best on solid, non-reflective surfaces)
3. LiDAR range: 10cm - 180m
4. Use `READ_SENSORS` to see all sensor data
5. Verify checksum validation is working (LED blinks on successful reads)

## Performance

### Update Rates

- **Servo Commands**: 20Hz (50ms interval)
- **Distance Reading**: 10Hz (100ms interval)  
- **Gyro Reading**: 100Hz (on-demand)
- **Control Loop**: 30Hz default (configurable up to 60Hz)

### Latency

- **Command Response**: <10ms
- **Servo Movement**: ~15ms per command
- **LiDAR Read**: <50ms with timeout
- **Total Tracking Latency**: ~50-100ms

## Safety Features

- **Gradual Limit Testing**: `TEST_*` commands move 1° at a time
- **User-Stoppable**: Press any key to stop test commands
- **Angle Clamping**: Servos automatically limited to configured range
- **Timeout Protection**: Sensor reads have 50ms timeout
- **Graceful Degradation**: System works even if sensors fail

## Upgrading from turret_debug.ino

This sketch is **100% backward compatible** with `turret_debug.ino`. You can directly replace it:

### What's New:
- ✅ `DISTANCE` command for `yolo_gimbal.py`
- ✅ MPU-6050 gyro/accelerometer support
- ✅ `GYRO` and `READ_SENSORS` commands
- ✅ `VIBRATE` quick-test command
- ✅ LED control commands (`LED1`, `LED2`, `BLINK`)
- ✅ Better sensor initialization feedback

### What's the Same:
- ✅ All servo commands (HOME, TOP, BOTTOM, BOTH)
- ✅ All limit commands (SET_*, GET_LIMITS)
- ✅ All test commands (TEST_*)
- ✅ Motor commands (MOTOR1, MOTOR2)
- ✅ STATUS and HELP commands

## Example Python Integration

```python
import serial
import time

# Connect to Arduino
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino reset

# Read distance
ser.write(b'DISTANCE\n')
response = ser.readline().decode().strip()
print(response)  # "OK: DISTANCE:123.4 cm"

# Move servo
ser.write(b'BOTTOM:90\n')
response = ser.readline().decode().strip()
print(response)  # "OK: Bottom servo set to 90"

# Read all sensors
ser.write(b'READ_SENSORS\n')
while True:
    line = ser.readline().decode().strip()
    if not line:
        break
    print(line)

ser.close()
```

## License

This code is part of the smart-bike turret tracking system. Use freely for educational and personal projects.

## Support

For issues or questions:
1. Check `HELP` command output
2. Verify wiring matches pinout
3. Test each sensor individually
4. Check Serial Monitor at 115200 baud

Happy tracking! 🎯

