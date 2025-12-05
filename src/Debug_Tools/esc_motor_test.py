#!/usr/bin/env python3
"""
Rock Pi 5B ESC Motor Control Test

This script controls a brushed motor ESC using PWM on Rock Pi 5B GPIO pins.
The motor alternates between forward and reverse rotation in a continuous pattern.

IMPORTANT NOTES:
- GPIO sysfs is deprecated (use libgpiod for GPIO control), but PWM sysfs (/sys/class/pwm/)
  is still the standard way to access PWM hardware on Linux.
- Rock Pi 5B GPIO pins operate at 3.3V logic levels. Most ESCs accept 3.3V PWM signals.
- Standard RC ESC PWM parameters: 50Hz frequency, 1-2ms pulse width

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
    GPIO1_B2 = gpiochip1, line 10 (bank 1, bank_idx 10)
    GPIO1_B3 = gpiochip1, line 11 (bank 1, bank_idx 11)
    Formula: bank_idx = (group * 8) + pin
             where group: A=0, B=1, C=2, D=3

Usage:
    sudo python3 esc_motor_test.py
"""

from periphery import PWM
import time
import sys

# PWM Configuration
# 
# GPIO PINS FOR ESC CONTROL:
# 
# GPIO1_B2 (Physical Pin 21) - First PWM input to ESC
# GPIO1_B3 (Physical Pin 23) - Second PWM input to ESC
#
# NOTE: You need to determine the PWM chip and channel numbers for these pins.
#       The chip/channel numbers below are placeholders - adjust based on your
#       PWM overlay configuration and system discovery.
#
# TO FIND YOUR PWM CHIP/CHANNEL:
#   1. Enable PWM overlay in device tree for GPIO1_B2 and GPIO1_B3 (see Rock Pi 5B docs)
#   2. Run: ls /sys/class/pwm/ to see available chips
#   3. Check: cat /sys/kernel/debug/pwm to see PWM mappings
#   4. Use: gpiodetect to list GPIO chips (for reference)
#   5. Use: gpiofind PIN_21 and gpiofind PIN_23 to find GPIO chip/line
#
# ESC CONFIGURATION:
#   - The ESC has a switch for "individual" vs "mixed" input mode
#   - In "mixed" mode: Both PWM inputs may be combined
#   - In "individual" mode: Each PWM input controls separately
#   - This script sends the same PWM signal to both pins (you can modify if needed)

# PWM Chip and Channel Configuration
# TODO: Update these values after discovering your PWM chip/channel mappings
PWM_CHIP_1 = 0      # PWM chip for GPIO1_B2 (pin 21) - UPDATE THIS
PWM_CHANNEL_1 = 0   # Channel for GPIO1_B2 (pin 21) - UPDATE THIS

PWM_CHIP_2 = 0      # PWM chip for GPIO1_B3 (pin 23) - UPDATE THIS
PWM_CHANNEL_2 = 0   # Channel for GPIO1_B3 (pin 23) - UPDATE THIS

# RC ESC Standard Parameters
PWM_FREQUENCY = 50  # 50Hz is standard for RC ESCs
PULSE_WIDTH_MIN = 0.001   # 1ms = reverse (full reverse)
PULSE_WIDTH_NEUTRAL = 0.0015  # 1.5ms = neutral (stop)
PULSE_WIDTH_MAX = 0.002   # 2ms = forward (full forward)
PERIOD = 1.0 / PWM_FREQUENCY  # 20ms period at 50Hz

# Motion Configuration
FORWARD_SPEED = 0.002   # Pulse width for forward (2ms = 10% duty cycle)
REVERSE_SPEED = 0.001   # Pulse width for reverse (1ms = 5% duty cycle)
NEUTRAL_PAUSE = 0.5     # Seconds to pause at neutral between direction changes
MOTION_DURATION = 2.0   # Seconds to run in each direction


def pulse_width_to_duty_cycle(pulse_width_ms, period_ms=None):
    """
    Convert pulse width in seconds to duty cycle percentage.
    
    Args:
        pulse_width_ms: Pulse width in seconds (e.g., 0.0015 for 1.5ms)
        period_ms: Period in seconds (default: calculated from PWM_FREQUENCY)
    
    Returns:
        Duty cycle as a float between 0.0 and 1.0
    """
    if period_ms is None:
        period_ms = PERIOD
    return pulse_width_ms / period_ms


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
        pin_name: Pin name like "PIN_21" or "PIN_23" (optional)
    
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
    Main function to control ESC motor via PWM.
    Alternates between forward and reverse rotation with neutral pauses.
    """
    print("Rock Pi 5B ESC Motor Control Test")
    print("=" * 50)
    print("\nESC CONFIGURATION:")
    print("  - GPIO1_B2 (Pin 21) -> ESC PWM Input 1")
    print("  - GPIO1_B3 (Pin 23) -> ESC PWM Input 2")
    print("  - ESC: RC Brushed Motor Speed Controller (2S-3S Lipo, 5A)")
    print("\nPWM PARAMETERS:")
    print(f"  - Frequency: {PWM_FREQUENCY} Hz")
    print(f"  - Period: {PERIOD*1000:.1f} ms")
    print(f"  - Reverse: {PULSE_WIDTH_MIN*1000:.1f} ms ({pulse_width_to_duty_cycle(PULSE_WIDTH_MIN)*100:.1f}% duty)")
    print(f"  - Neutral: {PULSE_WIDTH_NEUTRAL*1000:.1f} ms ({pulse_width_to_duty_cycle(PULSE_WIDTH_NEUTRAL)*100:.1f}% duty)")
    print(f"  - Forward: {PULSE_WIDTH_MAX*1000:.1f} ms ({pulse_width_to_duty_cycle(PULSE_WIDTH_MAX)*100:.1f}% duty)")
    print("=" * 50)
    
    # Optional: Try to find GPIO pin info using gpiofind
    print("\nChecking GPIO pin mappings (if gpiofind available)...")
    pin21_info = find_gpio_pin_info("PIN_21")
    pin23_info = find_gpio_pin_info("PIN_23")
    if pin21_info:
        print(f"  Pin 21 (GPIO1_B2): gpiochip{pin21_info[0]}, line {pin21_info[1]}")
    if pin23_info:
        print(f"  Pin 23 (GPIO1_B3): gpiochip{pin23_info[0]}, line {pin23_info[1]}")
    if not pin21_info and not pin23_info:
        print("  (gpiofind not available or pins not found - this is OK for PWM)")
    
    # Optional: List available PWM chips
    print("\nChecking available PWM chips...")
    find_available_pwm_chips()
    
    print(f"\nUsing PWM Configuration:")
    print(f"  Channel 1: PWM chip {PWM_CHIP_1}, channel {PWM_CHANNEL_1} (GPIO1_B2)")
    print(f"  Channel 2: PWM chip {PWM_CHIP_2}, channel {PWM_CHANNEL_2} (GPIO1_B3)")
    print(f"\nMotion Pattern:")
    print(f"  - Forward: {MOTION_DURATION} seconds")
    print(f"  - Neutral pause: {NEUTRAL_PAUSE} seconds")
    print(f"  - Reverse: {MOTION_DURATION} seconds")
    print(f"  - Neutral pause: {NEUTRAL_PAUSE} seconds")
    print(f"  - (repeats)")
    print("\nStarting motor control (Ctrl+C to stop)...")
    print("Motor will start at NEUTRAL position.\n")
    
    pwm1 = None
    pwm2 = None
    
    try:
        # Initialize PWM channels
        print("Initializing PWM channels...")
        pwm1 = PWM(PWM_CHIP_1, PWM_CHANNEL_1)
        pwm2 = PWM(PWM_CHIP_2, PWM_CHANNEL_2)
        
        # Configure PWM
        pwm1.frequency = PWM_FREQUENCY
        pwm2.frequency = PWM_FREQUENCY
        
        # Start at neutral position (motor stopped)
        neutral_duty = pulse_width_to_duty_cycle(PULSE_WIDTH_NEUTRAL)
        pwm1.duty_cycle = neutral_duty
        pwm2.duty_cycle = neutral_duty
        
        # Enable PWM outputs
        pwm1.enable()
        pwm2.enable()
        
        print(f"PWM initialized. Starting at neutral ({PULSE_WIDTH_NEUTRAL*1000:.1f}ms)...")
        time.sleep(1.0)  # Give ESC time to initialize
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            
            # Forward direction
            forward_duty = pulse_width_to_duty_cycle(FORWARD_SPEED)
            pwm1.duty_cycle = forward_duty
            pwm2.duty_cycle = forward_duty
            print(f"Cycle {cycle_count}: FORWARD - {time.strftime('%H:%M:%S')}")
            time.sleep(MOTION_DURATION)
            
            # Return to neutral
            pwm1.duty_cycle = neutral_duty
            pwm2.duty_cycle = neutral_duty
            print(f"         NEUTRAL (stopping)...")
            time.sleep(NEUTRAL_PAUSE)
            
            # Reverse direction
            reverse_duty = pulse_width_to_duty_cycle(REVERSE_SPEED)
            pwm1.duty_cycle = reverse_duty
            pwm2.duty_cycle = reverse_duty
            print(f"         REVERSE - {time.strftime('%H:%M:%S')}")
            time.sleep(MOTION_DURATION)
            
            # Return to neutral
            pwm1.duty_cycle = neutral_duty
            pwm2.duty_cycle = neutral_duty
            print(f"         NEUTRAL (stopping)...")
            time.sleep(NEUTRAL_PAUSE)
            
    except KeyboardInterrupt:
        print("\n\nEmergency stop! Stopping motor...")
    except FileNotFoundError as e:
        print(f"\nError: PWM chip or channel not found.")
        print("Please verify:")
        print("  1. PWM overlay is enabled in device tree for GPIO1_B2 and GPIO1_B3")
        print("  2. Correct PWM chip and channel numbers (update PWM_CHIP_1, PWM_CHANNEL_1, etc.)")
        print("  3. Run with sudo privileges")
        print(f"\nDetails: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"\nError: Permission denied. Please run with sudo:")
        print("  sudo python3 esc_motor_test.py")
        print(f"\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure python3-periphery is installed: sudo apt-get install python3-periphery")
        print("  2. Check PWM chip/channel numbers match your GPIO configuration")
        print("  3. Verify PWM overlay is enabled in device tree")
        print("  4. Check ESC wiring and power supply")
        sys.exit(1)
    finally:
        # Clean up - return to neutral and disable PWM
        print("\nCleaning up...")
        if pwm1 is not None:
            try:
                neutral_duty = pulse_width_to_duty_cycle(PULSE_WIDTH_NEUTRAL)
                pwm1.duty_cycle = neutral_duty
                pwm1.disable()
                pwm1.close()
                print("PWM channel 1 disabled and cleaned up.")
            except Exception as e:
                print(f"Warning: Error during cleanup of PWM channel 1: {e}")
        
        if pwm2 is not None:
            try:
                neutral_duty = pulse_width_to_duty_cycle(PULSE_WIDTH_NEUTRAL)
                pwm2.duty_cycle = neutral_duty
                pwm2.disable()
                pwm2.close()
                print("PWM channel 2 disabled and cleaned up.")
            except Exception as e:
                print(f"Warning: Error during cleanup of PWM channel 2: {e}")
        
        print("Motor stopped at neutral position.")


if __name__ == "__main__":
    main()

