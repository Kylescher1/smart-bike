# Servo Movement Logic Explanation

## How the Turret Decides to Move

### Step-by-Step Process:

1. **Object Detection (YOLO)**
   - YOLO detects objects and returns bounding boxes
   - We find the center of the target: `target_x, target_y` (in pixels)

2. **Calculate Error from Center**
   ```
   error_x = target_x - center_x
   error_y = target_y - center_y
   ```
   - `error_x > 0` means target is RIGHT of center
   - `error_x < 0` means target is LEFT of center
   - `error_y > 0` means target is BELOW center
   - `error_y < 0` means target is ABOVE center

3. **Normalize Error**
   ```
   error_x_norm = error_x / frame_width   # Range: -0.5 to +0.5
   error_y_norm = error_y / frame_height  # Range: -0.5 to +0.5
   ```
   - Converts pixel error to normalized units (-0.5 to +0.5)

4. **PID Controller Processing**
   ```
   output_x = kp * error_x_norm + ki * integral_x + kd * derivative_x
   output_y = kp * error_y_norm + ki * integral_y + kd * derivative_y
   ```
   - **Proportional (P)**: Direct response to error
   - **Integral (I)**: Accumulates error over time (removes steady-state error)
   - **Derivative (D)**: Responds to rate of change (reduces overshoot)

5. **Convert PID Output to Servo Movement**
   ```
   move_x = output_x * 2.0  # Horizontal movement (degrees)
   move_y = output_y * 2.0  # Vertical movement (degrees)
   ```
   - Multiplies PID output by sensitivity factor (2.0)
   - Result is in degrees of servo movement

6. **Apply Direction Inversions** (if needed)
   ```
   if invert_x: move_x = -move_x
   if invert_y: move_y = -move_y
   ```
   - Flips direction if servos are wired backwards

7. **Check Limits**
   ```
   new_bottom = current_bottom + move_x
   new_top = current_top + move_y
   ```
   - Clamps to servo limits (e.g., bottom: 0-180°, top: 60-120°)

8. **Move Servos**
   ```
   BOTTOM servo: new_bottom = current_bottom + move_x
   TOP servo:    new_top = current_top + move_y
   ```

## Current Assumptions:

**Horizontal Movement (X-axis):**
- Positive `error_x` (target RIGHT) → Positive `move_x` → **Increase** bottom servo angle
- Assumes: Increasing bottom servo angle moves turret **RIGHT**
- If your turret moves LEFT when bottom servo increases, use `--invert-x`

**Vertical Movement (Y-axis):**
- Positive `error_y` (target BELOW) → Positive `move_y` → **Increase** top servo angle  
- Assumes: Increasing top servo angle moves turret **DOWN**
- If your turret moves UP when top servo increases, use `--invert-y`

## Example Flow:

**Scenario: Target is 100 pixels RIGHT of center**

1. `error_x = 100` (target is RIGHT)
2. `error_x_norm = 100 / 640 = 0.156` (normalized)
3. PID calculates: `output_x = 0.5 * 0.156 + ... = ~0.08`
4. `move_x = 0.08 * 2.0 = 0.16 degrees`
5. `new_bottom = 90 + 0.16 = 90.16 degrees`
6. Send command: `BOTTOM:90` (rounded to int)

**If turret moves LEFT instead of RIGHT:**
- Use `--invert-x` flag
- This flips: `move_x = -move_x`
- Now: `new_bottom = 90 - 0.16 = 89.84 degrees` (moves LEFT to track RIGHT target)

## Debugging:

The script prints debug info when error is significant:
```
Target RIGHT of center (error=0.156), moving turret RIGHT (move_x=0.32deg, bottom_pos=90->90.32)
```

This shows:
- Which direction target is relative to center
- Which direction turret is trying to move
- Current and new servo positions

If these don't match what you expect, use the inversion flags!

