#!/usr/bin/env python3
"""
Rock Pi 5B GPIO PWM Haptic Control Test

This script controls a haptic output device using PWM on Rock Pi 5B GPIO pins.
The PWM signal pulses in 1-second intervals, alternating between 0V and 5V equivalent.

IMPORTANT NOTES:
- GPIO sysfs is deprecated (use libgpiod for GPIO control), but PWM sysfs (/sys/class/pwm/)
  is still the standard way to access PWM hardware on Linux.
- Rock Pi 5B GPIO pins operate at 3.3V logic levels. To achieve true 0-5V output,
  you may need a level shifter or transistor circuit. However, PWM duty cycle control
  works effectively for haptic devices even at 3.3V levels.

Installation:
    sudo apt-get install python3-periphery  # For PWM control
    sudo apt-get install python3-libgpiod    # For GPIO operations
    sudo apt-get install gpiod               # For gpiofind/gpiodetect tools (optional)

Finding PWM chip and channel:
    - Check Rock Pi 5B GPIO documentation for PWM-capable pins
    - Use: ls /sys/class/pwm/ to list available PWM chips
    - Use: cat /sys/kernel/debug/pwm to see PWM mappings
    - Use: gpiodetect to list GPIO chips (for reference)
    - Use: gpiofind PIN_X to find GPIO chip/line for a pin

Rockchip GPIO Naming:
    - 5 groups of banks (0-4), 32 GPIOs each
    - Each group has 4 subgroups: A, B, C, D (8 pins each: 0-7)
    - Formula: bank_idx = (group * 8) + pin
               where group: A=0, B=1, C=2, D=3
    - Example: GPIO4_B3 = gpiochip4, bank_idx 11 (group B=1, pin 3)
               bank_idx = (1 * 8) + 3 = 11

Usage:
    sudo python3 haptic_test.py
"""

from periphery import PWM
try:
    import gpiod
    from gpiod.line import Direction, Value
    LIBGPIOD_AVAILABLE = True
except ImportError:
    LIBGPIOD_AVAILABLE = False
    gpiod = None
    Direction = None
    Value = None
import time
import sys
import subprocess
import os

# PWM Configuration
# 
# RECOMMENDED PINS FOR ROCK PI 5B:
# 
# Option 1: PWM7_IR_M3
#   - Physical Pin: Pin 27 on 40-pin GPIO header (NOT Pin 3!)
#   - GPIO: GPIO4_B3
#   - PWM Chip: Usually pwmchip7, channel 0
#   - Requires PWM overlay: PWM7_IR_M3
#   - NOTE: Pin 27 is the correct physical pin for PWM7_IR_M3
#
# Option 2: Pin 12 (GPIO3_B5) - PWM12_M0  
#   - Physical Pin: Pin 12 on 40-pin GPIO header
#   - PWM Chip: Usually pwmchip12, channel 0
#   - Requires PWM overlay: PWM12_M0
#
# WIRING:
#   - Connect haptic device SIGNAL wire to the PWM pin (Pin 3 or Pin 12)
#   - Connect haptic device GROUND wire to any GND pin (e.g., Pin 6, 9, 14, 20, 25, 30, 34, or 39)
#   - If haptic device needs 5V power, use Pin 2 (5V) or Pin 4 (5V)
#     (Note: GPIO pins output 3.3V, but PWM duty cycle control works for haptic devices)
#
# TO FIND YOUR PWM CHIP/CHANNEL:
#   1. Enable PWM overlay in device tree (see Rock Pi 5B docs)
#   2. Run: ls /sys/class/pwm/ to see available chips
#   3. Check: cat /sys/kernel/debug/pwm to see PWM mappings
#   4. Use: gpiodetect to list GPIO chips (for reference)
#   5. Use: gpiofind PIN_X to find GPIO chip/line (for GPIO, not PWM)
#
# NOTE: PWM uses /sys/class/pwm/ (not deprecated), while GPIO sysfs is deprecated.
#       Use libgpiod (gpiofind, gpioset, etc.) for GPIO control, but PWM still
#       uses the sysfs interface via python3-periphery.
#
# CONFIGURED FOR: Pin 12 (PWM12_M0) - HARDCODED, NO FALLBACKS
PWM_CHIP = 12     # pwmchip12 for PWM12_M0 (Pin 12) - NO FALLBACK
PWM_CHANNEL = 0   # Channel 0
PWM_FREQUENCY = 1000  # Frequency in Hz (1 kHz is common for haptic devices)

# GPIO Configuration (using libgpiod)
# Pin 12 maps to gpiochip3, line 13 (GPIO3_B5)
GPIO_CHIP_PATH = "/dev/gpiochip3"  # gpiochip3 for Pin 12
GPIO_LINE_OFFSET = 13              # Line 13 for Pin 12 (GPIO3_B5)

# Control mode: "PWM" or "GPIO"
CONTROL_MODE = "GPIO"  # Use GPIO toggle mode (simpler) or "PWM" for PWM control

# Timing configuration
PULSE_INTERVAL = 1.0  # Seconds between pulses


def find_available_pwm_chips():
    """
    Helper function to list available PWM chips on the system.
    This can help identify which PWM chip and channel to use.
    
    Returns:
        List of available PWM chip numbers (integers)
    """
    import os
    pwm_path = "/sys/class/pwm"
    if os.path.exists(pwm_path):
        chips = [d for d in os.listdir(pwm_path) if d.startswith("pwmchip")]
        chip_numbers = [int(chip.replace("pwmchip", "")) for chip in chips]
        print(f"Available PWM chips: {chips} (numbers: {chip_numbers})")
        return chip_numbers
    else:
        print("PWM sysfs interface not found. PWM may not be enabled.")
        return []


def detect_gpio_chips():
    """
    Detect all available GPIO chips using libgpiod.
    
    Returns:
        List of gpiochip numbers (integers)
    """
    if not LIBGPIOD_AVAILABLE:
        return []
    
    chips = []
    try:
        # Try to find gpiochips (typically gpiochip0 through gpiochip4 on Rock Pi 5B)
        for i in range(10):  # Check up to gpiochip9
            try:
                chip = gpiod.Chip(f'gpiochip{i}')
                chips.append(i)
                chip.close()
            except (OSError, FileNotFoundError):
                continue
    except Exception as e:
        print(f"Error detecting GPIO chips: {e}")
    
    return chips


def get_gpio_info(chip_num):
    """
    Get information about a GPIO chip using libgpiod.
    
    Args:
        chip_num: GPIO chip number
    
    Returns:
        Dict with chip information or None
    """
    if not LIBGPIOD_AVAILABLE:
        return None
    
    try:
        chip = gpiod.Chip(f'gpiochip{chip_num}')
        info = {
            'name': chip.name(),
            'label': chip.label(),
            'num_lines': chip.num_lines()
        }
        chip.close()
        return info
    except Exception:
        return None


def find_gpio_pin_info(pin_name=None):
    """
    Find GPIO chip and line using gpiofind command (libgpiod tool).
    Useful for verifying pin mappings.
    
    Args:
        pin_name: Pin name like "PIN_3" or "PIN_12" (optional)
    
    Returns:
        Tuple of (gpiochip_num, line_num) or None if not found
    """
    if pin_name is None:
        return None
    
    try:
        result = subprocess.run(
            ['gpiofind', pin_name],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                gpiochip = int(parts[0].replace('gpiochip', ''))
                line = int(parts[1])
                return (gpiochip, line)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass  # gpiofind not available or pin not found
    return None


def rockchip_gpio_to_bank_idx(bank, group, pin):
    """
    Convert Rockchip GPIO naming to bank index.
    
    Args:
        bank: Bank number (0-4)
        group: Group letter (A=0, B=1, C=2, D=3)
        pin: Pin number within group (0-7)
    
    Returns:
        Bank index (0-31)
    """
    if isinstance(group, str):
        group_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        group = group_map.get(group.upper(), 0)
    
    bank_idx = (group * 8) + pin
    return bank_idx


def get_gpio_line_info(chip_num, line_num):
    """
    Get information about a specific GPIO line using libgpiod.
    
    Args:
        chip_num: GPIO chip number
        line_num: GPIO line number
    
    Returns:
        Dict with line information or None
    """
    if not LIBGPIOD_AVAILABLE:
        return None
    
    try:
        chip = gpiod.Chip(f'gpiochip{chip_num}')
        line = chip.get_line(line_num)
        info = {
            'name': line.name(),
            'consumer': line.consumer(),
            'direction': 'input' if line.direction() == gpiod.Line.DIRECTION_INPUT else 'output',
            'active_low': line.is_active_low(),
            'used': line.is_used()
        }
        chip.close()
        return info
    except Exception:
        return None


def toggle_gpio_value(value):
    """
    Toggle GPIO value between ACTIVE and INACTIVE.
    
    Args:
        value: Current Value (ACTIVE or INACTIVE)
    
    Returns:
        Toggled Value
    """
    if not LIBGPIOD_AVAILABLE:
        return None
    
    if value == Value.INACTIVE:
        return Value.ACTIVE
    return Value.INACTIVE


def haptic_gpio_toggle(chip_path, line_offset, interval=1.0):
    """
    Toggle GPIO line to control haptic device using libgpiod.
    Similar to the gpiod example but specifically for haptic control.
    
    Args:
        chip_path: Path to GPIO chip device (e.g., "/dev/gpiochip4")
        line_offset: GPIO line offset (e.g., 22 for Pin 27)
        interval: Time interval between toggles in seconds
    """
    if not LIBGPIOD_AVAILABLE:
        raise RuntimeError("libgpiod not available. Install python3-libgpiod")
    
    value_str = {Value.ACTIVE: "ON (3.3V)", Value.INACTIVE: "OFF (0V)"}
    value = Value.ACTIVE
    
    print(f"\nStarting GPIO toggle on {chip_path}, line {line_offset}")
    print(f"Interval: {interval} seconds")
    print("Press Ctrl+C to stop...\n")
    
    try:
        with gpiod.request_lines(
            chip_path,
            consumer="haptic-gpio-toggle",
            config={
                line_offset: gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    output_value=value
                )
            },
        ) as request:
            pulse_count = 0
            
            while True:
                pulse_count += 1
                print(f"Pulse {pulse_count}: {value_str[value]} - {time.strftime('%H:%M:%S')}")
                time.sleep(interval)
                
                value = toggle_gpio_value(value)
                request.set_value(line_offset, value)
                
    except KeyboardInterrupt:
        print("\n\nStopping GPIO toggle...")
        # Set to inactive before exiting
        try:
            with gpiod.request_lines(
                chip_path,
                consumer="haptic-gpio-cleanup",
                config={
                    line_offset: gpiod.LineSettings(
                        direction=Direction.OUTPUT,
                        output_value=Value.INACTIVE
                    )
                },
            ) as request:
                pass  # Line will be set to inactive and released
        except Exception:
            pass
        print("GPIO disabled and cleaned up.")
    except OSError as ex:
        print(f"\nError accessing GPIO: {ex}")
        print("\nTroubleshooting:")
        print("  1. Ensure python3-libgpiod is installed: sudo apt-get install python3-libgpiod")
        print("  2. Run with sudo privileges")
        print("  3. Check that the GPIO line is not already in use")
        print(f"  4. Verify chip path: {chip_path}")
        print(f"  5. Verify line offset: {line_offset}")
        raise


def main():
    """
    Main function to control haptic device via PWM or GPIO toggle.
    Can use either PWM (variable duty cycle) or GPIO toggle (simple on/off).
    """
    # Use local variable to avoid UnboundLocalError
    control_mode = CONTROL_MODE
    
    print("Rock Pi 5B Haptic Control Test")
    print("=" * 40)
    print(f"\nCONTROL MODE: {control_mode}")
    print("\nCONFIGURED FOR: Pin 12 (GPIO3_B5) - PWM12_M0")
    print("\nWIRING:")
    print("  - Haptic SIGNAL -> Pin 12 (gpiochip3, line 13)")
    print("  - Haptic GROUND -> Any GND pin (Pin 6, 9, 14, 20, 25, 30, 34, 39, etc.)")
    print("  - If needed, 5V power -> Pin 2 or Pin 4 (5V)")
    print("\nNote: GPIO pins output 3.3V")
    print("=" * 40)
    
    # Pin 12 is hardcoded - no detection needed
    print(f"\nUsing Pin 12 (gpiochip3, line 13) - HARDCODED")
    
    # Choose control mode
    if control_mode == "GPIO":
        # Use GPIO toggle mode (libgpiod)
        if not LIBGPIOD_AVAILABLE:
            print("\nError: GPIO mode requires python3-libgpiod")
            print("Install with: sudo apt-get install python3-libgpiod")
            print("\nFalling back to PWM mode...")
            control_mode = "PWM"
        else:
            print(f"\n⚠️  IMPORTANT: Connect haptic SIGNAL to Pin 12")
            print(f"   Using GPIO toggle mode: {GPIO_CHIP_PATH}, line {GPIO_LINE_OFFSET}")
            try:
                haptic_gpio_toggle(GPIO_CHIP_PATH, GPIO_LINE_OFFSET, PULSE_INTERVAL)
            except Exception as e:
                print(f"\nGPIO toggle error: {e}")
                sys.exit(1)
            return
    
    # PWM mode - Pin 12 hardcoded, no fallbacks
    print("\nUsing PWM mode...")
    print(f"\nUsing PWM chip {PWM_CHIP} (Pin 12, PWM12_M0) - HARDCODED")
    print(f"Channel: {PWM_CHANNEL}")
    print(f"Frequency: {PWM_FREQUENCY} Hz")
    print(f"Pulse interval: {PULSE_INTERVAL} seconds")
    print("\n⚠️  IMPORTANT: Connect haptic SIGNAL to Pin 12")
    print("\nStarting PWM output (Ctrl+C to stop)...")
    print("Signal will alternate between 0V and 5V equivalent every second.\n")
    
    pwm = None
    try:
        # Initialize PWM - Pin 12 only, no fallback
        pwm = PWM(PWM_CHIP, PWM_CHANNEL)
        
        # Configure PWM
        pwm.frequency = PWM_FREQUENCY
        pwm.duty_cycle = 0.0  # Start at 0V (0% duty cycle)
        pwm.enable()
        
        pulse_count = 0
        
        while True:
            # Set to 5V equivalent (100% duty cycle)
            pwm.duty_cycle = 1.0
            pulse_count += 1
            print(f"Pulse {pulse_count}: ON (5V equivalent) - {time.strftime('%H:%M:%S')}")
            time.sleep(PULSE_INTERVAL)
            
            # Set to 0V (0% duty cycle)
            pwm.duty_cycle = 0.0
            print(f"Pulse {pulse_count}: OFF (0V) - {time.strftime('%H:%M:%S')}")
            time.sleep(PULSE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopping PWM output...")
    except FileNotFoundError as e:
        print(f"\nError: PWM chip {PWM_CHIP} (Pin 12) or channel {PWM_CHANNEL} not found.")
        print("Please verify:")
        print("  1. PWM overlay PWM12_M0 is enabled in device tree")
        print("  2. Run with sudo privileges")
        print(f"\nDetails: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"\nError: Permission denied. Please run with sudo:")
        print("  sudo python3 haptic_test.py")
        print(f"\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure python3-periphery is installed: sudo apt-get install python3-periphery")
        print("  2. Ensure python3-libgpiod is installed: sudo apt-get install python3-libgpiod")
        print("  3. Verify PWM overlay PWM12_M0 is enabled in device tree for Pin 12")
        sys.exit(1)
    finally:
        # Clean up
        if pwm is not None:
            try:
                pwm.duty_cycle = 0.0  # Set to 0V before disabling
                pwm.disable()
                pwm.close()
                print("PWM disabled and cleaned up.")
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    main()

