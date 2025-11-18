#!/usr/bin/env python3
"""
PWM version to set PIN_12 high with reduced current draw using software PWM.

PIN_12 maps to gpiochip3, line 13 (GPIO3_B5)

This script uses software PWM (pulsing) to reduce average current draw.
By pulsing the pin instead of keeping it constantly high, the average current
is reduced proportionally to the duty cycle.

Usage:
    sudo python3 set_pin12_high_pwm.py

Current Limits (Rock Pi 5B / RK3588):
    - Per GPIO pin: ~4-8mA typical, up to 12-16mA maximum
    - Total GPIO bank: ~50-100mA
    - Exceeding limits can cause voltage drops, instability, or damage

Software Solutions:
    1. Reduce duty cycle (this script) - reduces average current
    2. Lower PWM frequency - may help with some loads
    3. Use hardware PWM if available (requires PWM overlay)
"""

import gpiod
import time
import sys
import signal

# PIN_12 configuration (from gpiofind output: gpiochip3 13)
GPIO_CHIP_NAME = "gpiochip3"
GPIO_LINE_OFFSET = 13
DURATION_SECONDS = 30.0

# PWM Configuration
PWM_FREQUENCY = 1000  # Hz - pulses per second (1000 Hz = 1ms period)
DUTY_CYCLE = 0.4      # 0.0 to 1.0 - 0.5 = 50% duty cycle = 50% average current
                      # Reduce this to lower current draw (e.g., 0.25 = 25% current)

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    running = False

def main():
    global running
    
    print("=" * 60)
    print("PIN_12 PWM Control - Reduced Current Draw")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Chip: {GPIO_CHIP_NAME}, Line: {GPIO_LINE_OFFSET}")
    print(f"  Duration: {DURATION_SECONDS} seconds")
    print(f"  PWM Frequency: {PWM_FREQUENCY} Hz")
    print(f"  Duty Cycle: {DUTY_CYCLE * 100:.1f}%")
    print(f"\nCurrent Reduction:")
    print(f"  Average current ≈ {DUTY_CYCLE * 100:.1f}% of full current")
    print(f"  Example: If full current is 10mA, PWM current ≈ {DUTY_CYCLE * 10:.1f}mA")
    print("\n" + "=" * 60)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    chip = None
    line = None
    
    try:
        # Open the GPIO chip
        chip = gpiod.Chip(GPIO_CHIP_NAME)
        
        # Get the GPIO line
        line = chip.get_line(GPIO_LINE_OFFSET)
        
        # Request the line for output
        line.request(consumer="set-pin12-pwm", type=gpiod.LINE_REQ_DIR_OUT)
        
        # Calculate timing
        period = 1.0 / PWM_FREQUENCY  # Period in seconds
        high_time = period * DUTY_CYCLE  # Time pin is HIGH
        low_time = period * (1.0 - DUTY_CYCLE)  # Time pin is LOW
        
        print(f"\nStarting PWM output...")
        print(f"  Period: {period*1000:.3f}ms (HIGH: {high_time*1000:.3f}ms, LOW: {low_time*1000:.3f}ms)")
        print(f"  Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        cycle_count = 0
        
        # PWM loop
        while running and (time.time() - start_time) < DURATION_SECONDS:
            # Set HIGH
            line.set_value(1)
            time.sleep(high_time)
            
            # Set LOW
            line.set_value(0)
            time.sleep(low_time)
            
            cycle_count += 1
            
            # Print status every second
            elapsed = time.time() - start_time
            if cycle_count % PWM_FREQUENCY == 0:  # Every second
                remaining = max(0, DURATION_SECONDS - elapsed)
                print(f"  Elapsed: {elapsed:.1f}s / {DURATION_SECONDS:.1f}s | "
                      f"Remaining: {remaining:.1f}s | Cycles: {cycle_count}")
        
        # Ensure pin is LOW before finishing
        line.set_value(0)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed: {elapsed:.2f} seconds")
        print(f"✓ Total PWM cycles: {cycle_count}")
        print(f"✓ PIN_12 is now LOW (0V)")
        print("Done!")
        
    except OSError as ex:
        print(f"\n✗ Error accessing GPIO: {ex}")
        print("\nTroubleshooting:")
        print("  1. Ensure python3-libgpiod is installed: sudo apt-get install python3-libgpiod")
        print("  2. Run with sudo privileges: sudo python3 set_pin12_high_pwm.py")
        print("  3. Check that the GPIO line is not already in use")
        print(f"  4. Verify chip name: {GPIO_CHIP_NAME}")
        print(f"  5. Verify line offset: {GPIO_LINE_OFFSET}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user!")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up: ensure pin is LOW and release resources
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

