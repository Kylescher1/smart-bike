# Turret Debug Tools

This folder contains debugging tools for the turret system, allowing safe servo limit testing and manual control.

## Files

- `turret_debug.ino` - Arduino sketch for ESP32 that provides serial command interface
- `turret_control.py` - Python script to control the turret via serial commands
- `turret_game.py` - **🎮 VIDEO GAME STYLE** real-time keyboard control interface
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Setup

### Arduino Sketch

1. Open `turret_debug.ino` in Arduino IDE
2. Select your ESP32 board and port
3. Upload the sketch
4. Open Serial Monitor at 115200 baud

### Python Scripts

Install Python dependencies:

```bash
pip install -r requirements.txt
# or
pip install pyserial colorama
```

## Usage

### Direct Serial Commands

You can send commands directly via Serial Monitor or any serial terminal:

```
HOME                    - Move both servos to home (90)
TOP:<angle>            - Set top servo (0-180)
BOTTOM:<angle>         - Set bottom servo (0-180)
BOTH:<angle>           - Set both servos (0-180)
TEST_TOP_MIN           - Test top servo minimum limit
TEST_TOP_MAX           - Test top servo maximum limit
TEST_BOTTOM_MIN        - Test bottom servo minimum limit
TEST_BOTTOM_MAX        - Test bottom servo maximum limit
SET_MIN:<value>        - Set minimum limit (0-179)
SET_MAX:<value>        - Set maximum limit (1-180)
GET_LIMITS             - Print current limits
MOTOR1:<speed>         - Set motor 1 speed (0-255)
MOTOR2:<speed>         - Set motor 2 speed (0-255)
STATUS                 - Print current status
HELP                   - Show help
```

### Python Script Usage

#### 🎮 **GAME MODE** (Recommended - Most Fun!)

Real-time keyboard controls with visual feedback:

```bash
python turret_game.py COM3
# or on Linux:
python turret_game.py /dev/ttyUSB0
```

**Controls:**
- `W/S` or `↑/↓` - Move top servo up/down
- `A/D` or `←/→` - Move bottom servo left/right  
- `Q/E` - Fine adjust top servo (±1°)
- `Z/X` - Fine adjust bottom servo (±1°)
- `SPACE` - Home position
- `R` - Reset limits to 0-180
- `T/Y` - Test top min/max limits
- `G/H` - Test bottom min/max limits
- `M/N` - Motor 1 speed up/down
- `,/.` - Motor 2 speed up/down
- `1-9` - Quick position presets
- `ESC` - Exit

#### Command-line mode:

List available ports:
```bash
python turret_control.py --list-ports
```

Interactive mode:
```bash
python turret_control.py COM3 --interactive
# or on Linux:
python turret_control.py /dev/ttyUSB0 --interactive
```

One-shot commands:
```bash
python turret_control.py COM3
```

#### Interactive Commands:
```
home              - Move to home position
top <angle>       - Set top servo
bottom <angle>    - Set bottom servo
both <angle>      - Set both servos
test_top_min      - Test top min limit
test_top_max      - Test top max limit
test_bottom_min   - Test bottom min limit
test_bottom_max   - Test bottom max limit
set_min <value>   - Set minimum limit
set_max <value>   - Set maximum limit
limits            - Get current limits
motor1 <speed>    - Set motor 1 (0-255)
motor2 <speed>    - Set motor 2 (0-255)
status            - Get status
raw <command>     - Send raw command
quit              - Exit
```

## Finding Servo Limits Safely

1. **Start from home position:**
   ```
   HOME
   ```

2. **Test minimum limit gradually:**
   ```
   TEST_TOP_MIN
   ```
   Watch for physical interference. The servo will move slowly from current position to 0 degrees.
   Press any key in Serial Monitor to stop if you see interference.

3. **Note the safe minimum angle** where interference occurs, then set it:
   ```
   SET_MIN:15
   ```
   (Use a value slightly above where interference occurred)

4. **Repeat for maximum limit:**
   ```
   TEST_TOP_MAX
   SET_MAX:165
   ```

5. **Repeat for bottom servo:**
   ```
   TEST_BOTTOM_MIN
   TEST_BOTTOM_MAX
   ```

6. **Verify limits:**
   ```
   GET_LIMITS
   STATUS
   ```

## Example Workflow

```python
import sys
sys.path.append('.')  # Add current directory to path
from turret_debug.turret_control import TurretController

# Connect
controller = TurretController("COM3")
controller.connect()

# Move to home
controller.home()

# Test limits
controller.test_top_min()
# Watch for interference, then set limit
controller.set_min_limit(15)

controller.test_top_max()
controller.set_max_limit(165)

# Verify
print(controller.get_status())

# Disconnect
controller.disconnect()
```

Or use the script directly:
```python
# In Python script or interactive session
from turret_control import TurretController

controller = TurretController("COM3")
controller.connect()
controller.home()
controller.set_top(45)
controller.disconnect()
```

## Safety Features

- Servos are clamped to current limits when setting positions
- Test commands move gradually and can be stopped
- Home position is set to 90 degrees (middle)
- Motors start at 0 speed

## Notes

- Default limits are 0-180 (full range)
- Test commands move 1 degree at a time with delays
- LED 1 blinks during limit testing
- All commands return OK/ERROR responses

