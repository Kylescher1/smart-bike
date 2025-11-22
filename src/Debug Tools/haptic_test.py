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
    sudo apt-get install python3-periphery
    sudo apt-get install gpiod  # For gpiofind/gpiodetect tools (optional)

Finding PWM chip and channel:
    - Check Rock Pi 5B GPIO documentation for PWM-capable pins
    - Use: ls /sys/class/pwm/ to list available PWM chips
    - Use: cat /sys/kernel/debug/pwm to see PWM mappings
    - Use: gpiodetect to list GPIO chips (for reference)
    - Use: gpiofind PIN_X to find GPIO chip/line for a pin

Rockchip GPIO Naming:
    GPIO3_C5 = gpiochip3, line 21 (bank 3, bank_idx 21)
    Formula: bank_idx = (group * 8) + pin
             where group: A=0, B=1, C=2, D=3

Usage:
    sudo python3 haptic_test.py
"""

from periphery import PWM
import time
import sys

# PWM Configuration
# 
# RECOMMENDED PINS FOR ROCK PI 5B:
# 
# Option 1: Pin 3 (GPIO4_B3) - PWM7_IR_M3
#   - Physical Pin: Pin 3 on 40-pin GPIO header
#   - PWM Chip: Usually pwmchip7, channel 0
#   - Requires PWM overlay: PWM7_IR_M3
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
PWM_CHIP = 7      # Start with pwmchip7 for Pin 3, or try pwmchip12 for Pin 12
PWM_CHANNEL = 0   # Usually channel 0 for single PWM pinsgitgit push

PWM_FREQUENCY = 1000  # Frequency in Hz (1 kHz is common for haptic devices)

# Timing configuration
PULSE_INTERVAL = 1.0  # Seconds between pulses


def find_available_pwm_chips():
    """
    Helper function to list available PWM chips on the system.
    This can help identify which PWM chip and channel to use.
    """
    import os
    pwm_path = "/sys/class/pwm"
    if os.path.exists(pwm_path):
        chips = [d for d in os.listdir(pwm_path) if d.startswith("pwmchip")]
        print(f"Available PWM chips: {chips}")
        return chips
    else:
        print("PWM sysfs interface not found. PWM may not be enabled.")
        return []


def find_gpio_pin_info(pin_name=None):
    """
    Helper function to find GPIO chip and line using gpiofind command.
    Useful for verifying pin mappings.
    
    Args:
        pin_name: Pin name like "PIN_3" or "PIN_12" (optional)
    
    Returns:
        Tuple of (gpiochip, line) or None if not found
    """
    import subprocess
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
                gpiochip = parts[0].replace('gpiochip', '')
                line = parts[1]
                return (gpiochip, line)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass  # gpiofind not available or pin not found
    return None


def main():
    """
    Main function to control haptic device via PWM.
    Pulses between 0V (0% duty cycle) and 5V equivalent (100% duty cycle)
    in 1-second intervals.
    """
    print("Rock Pi 5B Haptic PWM Test")
    print("=" * 40)
    print("\nRECOMMENDED WIRING:")
    print("  - Haptic SIGNAL -> Pin 3 or Pin 12 (PWM output)")
    print("  - Haptic GROUND -> Any GND pin (Pin 6, 9, 14, 20, etc.)")
    print("  - If needed, 5V power -> Pin 2 or Pin 4 (5V)")
    print("\nNote: GPIO pins output 3.3V, but PWM duty cycle works for haptic control")
    print("=" * 40)
    
    # Optional: Try to find GPIO pin info using gpiofind
    print("\nChecking GPIO pin mappings (if gpiofind available)...")
    pin3_info = find_gpio_pin_info("PIN_3")
    pin12_info = find_gpio_pin_info("PIN_12")
    if pin3_info:
        print(f"  Pin 3: gpiochip{pin3_info[0]}, line {pin3_info[1]}")
    if pin12_info:
        print(f"  Pin 12: gpiochip{pin12_info[0]}, line {pin12_info[1]}")
    if not pin3_info and not pin12_info:
        print("  (gpiofind not available or pins not found - this is OK for PWM)")
    
    # Optional: List available PWM chips
    print("\nChecking available PWM chips...")
    find_available_pwm_chips()
    
    print(f"\nUsing PWM chip {PWM_CHIP}, channel {PWM_CHANNEL}")
    print(f"Frequency: {PWM_FREQUENCY} Hz")
    print(f"Pulse interval: {PULSE_INTERVAL} seconds")
    print("\nStarting PWM output (Ctrl+C to stop)...")
    print("Signal will alternate between 0V and 5V equivalent every second.\n")
    
    pwm = None
    try:
        # Initialize PWM
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
        print(f"\nError: PWM chip {PWM_CHIP} or channel {PWM_CHANNEL} not found.")
        print("Please verify:")
        print("  1. PWM overlay is enabled in device tree")
        print("  2. Correct PWM chip and channel numbers")
        print("  3. Run with sudo privileges")
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
        print("  2. Check PWM chip/channel numbers match your GPIO configuration")
        print("  3. Verify PWM overlay is enabled in device tree")
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

