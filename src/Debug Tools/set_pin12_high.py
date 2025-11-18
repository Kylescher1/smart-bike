#!/usr/bin/env python3
"""
Simple script to set PIN_12 high for a specified duration using libgpiod.

PIN_12 maps to gpiochip3, line 13 (GPIO3_B5)

Usage:
    sudo python3 set_pin12_high.py

Current Limits (Rock Pi 5B / RK3588):
    - Per GPIO pin: ~4-8mA typical, up to 12-16mA maximum
    - Total GPIO bank: ~50-100mA
    - Exceeding limits can cause voltage drops, instability, or damage

If current draw is too high, use set_pin12_high_pwm.py instead for PWM control
with reduced average current (adjustable duty cycle).
"""

import gpiod
import time
import sys

# PIN_12 configuration (from gpiofind output: gpiochip3 13)
GPIO_CHIP_NAME = "gpiochip3"
GPIO_LINE_OFFSET = 13
DURATION_SECONDS = 30.0

def main():
    print(f"Setting PIN_12 ({GPIO_CHIP_NAME}, line {GPIO_LINE_OFFSET}) HIGH for {DURATION_SECONDS} seconds...")
    
    chip = None
    line = None
    
    try:
        # Open the GPIO chip
        chip = gpiod.Chip(GPIO_CHIP_NAME)
        
        # Get the GPIO line
        line = chip.get_line(GPIO_LINE_OFFSET)
        
        # Request the line for output
        line.request(consumer="set-pin12-high", type=gpiod.LINE_REQ_DIR_OUT)
        
        # Set the line HIGH (1 = high, 0 = low)
        line.set_value(1)
        print(f"PIN_12 is now HIGH (3.3V)")
        print(f"Waiting {DURATION_SECONDS} seconds...")
        
        time.sleep(DURATION_SECONDS)
        
        # Set the line LOW before releasing
        line.set_value(0)
        print("PIN_12 is now LOW (0V)")
        
        print("Done!")
        
    except OSError as ex:
        print(f"\nError accessing GPIO: {ex}")
        print("\nTroubleshooting:")
        print("  1. Ensure python3-libgpiod is installed: sudo apt-get install python3-libgpiod")
        print("  2. Run with sudo privileges: sudo python3 set_pin12_high.py")
        print("  3. Check that the GPIO line is not already in use")
        print(f"  4. Verify chip name: {GPIO_CHIP_NAME}")
        print(f"  5. Verify line offset: {GPIO_LINE_OFFSET}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Setting PIN_12 LOW and exiting...")
        if line is not None:
            try:
                line.set_value(0)
                line.release()
            except Exception:
                pass
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up: release the line and close the chip
        if line is not None:
            try:
                line.set_value(0)  # Ensure it's LOW
                line.release()
            except Exception:
                pass
        if chip is not None:
            try:
                chip.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()

