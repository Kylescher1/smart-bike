#!/usr/bin/env python3
"""
ESP32 Servo Test Script

Tests the ESP32 servo control functionality.
Allows interactive control or runs automated test sequence.

Usage:
    python test_esp32_servos.py [port] [mode]
    
    Modes:
    - auto: Run automated test sequence (default)
    - interactive: Interactive mode for manual control
    
    If port is not specified, will prompt for it or use default "COM5"
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

try:
    import quaternion
    has_quaternion = True
except ImportError:
    has_quaternion = False

from hal.ESP32 import ESP32


def test_servo_movement(esp32):
    """Run automated servo test sequence."""
    print("\n" + "="*60)
    print("AUTOMATED SERVO TEST SEQUENCE")
    print("="*60)
    
    # Servo limits
    BOTTOM_MIN = 15
    BOTTOM_MAX = 75
    TOP_MIN = 0
    TOP_MAX = 180
    
    test_positions = [
        # (bottom_angle, top_angle, description, delay)
        (45, 90, "Center position", 2.0),
        (BOTTOM_MIN, TOP_MIN, "Minimum angles", 2.0),
        (BOTTOM_MAX, TOP_MAX, "Maximum angles", 2.0),
        (45, 90, "Return to center", 1.0),
        (30, 45, "Bottom left, Top forward", 1.5),
        (60, 135, "Bottom right, Top back", 1.5),
        (45, 0, "Bottom center, Top minimum", 1.5),
        (45, 180, "Bottom center, Top maximum", 1.5),
        (BOTTOM_MIN, 90, "Bottom minimum, Top center", 1.5),
        (BOTTOM_MAX, 90, "Bottom maximum, Top center", 1.5),
        (45, 90, "Final center position", 2.0),
    ]
    
    print(f"\nTesting {len(test_positions)} positions...")
    print("Servo limits:")
    print(f"  Bottom: {BOTTOM_MIN}° to {BOTTOM_MAX}°")
    print(f"  Top: {TOP_MIN}° to {TOP_MAX}°")
    print("\nStarting test sequence...\n")
    
    for i, (bottom_angle, top_angle, description, delay_time) in enumerate(test_positions, 1):
        print(f"Test {i}/{len(test_positions)}: {description}")
        print(f"  Moving to: Bottom={bottom_angle}°, Top={top_angle}°")
        
        try:
            result = esp32.move(0, bottom_angle, 0, top_angle)
            if result:
                print(f"  ✓ Position set successfully")
                time.sleep(delay_time)
            else:
                print(f"  ✗ Failed to set position")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print("\n" + "="*60)
    print("✓ Automated test sequence completed successfully!")
    print("="*60)
    return True


def interactive_mode(esp32):
    """Interactive mode for manual servo control."""
    print("\n" + "="*60)
    print("INTERACTIVE SERVO CONTROL MODE")
    print("="*60)
    print("\nCommands:")
    print("  move <bottom> <top>  - Move servos (e.g., 'move 45 90')")
    print("  sweep                - Sweep both servos through their range")
    print("  center               - Move both servos to center position")
    print("  limits               - Move to limit positions")
    print("  status               - Show current status")
    print("  quit                 - Exit interactive mode")
    print("\nServo limits:")
    print("  Bottom: 15° to 75°")
    print("  Top: 0° to 180°")
    print("="*60)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "quit" or cmd == "exit" or cmd == "q":
                break
            
            elif cmd == "center":
                print("Moving to center position...")
                result = esp32.move(0, 45, 0, 90)
                print("✓ Center position set" if result else "✗ Failed")
            
            elif cmd == "limits":
                print("Moving to limit positions...")
                print("  Minimum angles...")
                esp32.move(0, 15, 0, 0)
                time.sleep(2)
                print("  Maximum angles...")
                esp32.move(0, 75, 0, 180)
                time.sleep(2)
                print("  Returning to center...")
                esp32.move(0, 45, 0, 90)
                print("✓ Limit test complete")
            
            elif cmd == "sweep":
                print("Sweeping servos...")
                # Sweep bottom servo
                for angle in range(15, 76, 5):
                    esp32.move(0, angle, 0, 90)
                    time.sleep(0.2)
                # Sweep top servo
                for angle in range(0, 181, 10):
                    esp32.move(0, 45, 0, angle)
                    time.sleep(0.2)
                # Return to center
                esp32.move(0, 45, 0, 90)
                print("✓ Sweep complete")
            
            elif cmd.startswith("move"):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        bottom = int(parts[1])
                        top = int(parts[2])
                        
                        # Clamp to limits
                        bottom = max(15, min(75, bottom))
                        top = max(0, min(180, top))
                        
                        print(f"Moving: Bottom={bottom}°, Top={top}°")
                        result = esp32.move(0, bottom, 0, top)
                        print("✓ Servos moved" if result else "✗ Failed to move servos")
                    except ValueError:
                        print("Invalid angles. Use: move <bottom_angle> <top_angle>")
                else:
                    print("Usage: move <bottom_angle> <top_angle>")
                    print("Example: move 45 90")
            
            elif cmd == "status":
                esp32.print_status()
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    print("ESP32 Servo Test Script")
    print("="*60)
    
    # Get port from command line or prompt
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = input("Enter ESP32 serial port (e.g., COM5 or /dev/ttyUSB0) [default: COM5]: ").strip()
        if not port:
            port = "COM5"
    
    # Get mode (auto or interactive)
    mode = "auto"
    if len(sys.argv) > 2:
        mode = sys.argv[2].lower()
    else:
        mode_input = input("Mode [auto/interactive] [default: auto]: ").strip().lower()
        if mode_input in ["interactive", "i"]:
            mode = "interactive"
    
    print(f"\nUsing port: {port}")
    print(f"Mode: {mode}")
    print("Make sure the ESP32 is connected and the firmware is uploaded!")
    input("Press Enter to continue...")
    
    # Create ESP32 instance
    try:
        if has_quaternion:
            position = np.quaternion(1, 0, 0, 0)
            z_direction = np.quaternion(0, 0, 0, 1)
        else:
            position = None
            z_direction = None
        
        esp32 = ESP32(
            name="ServoTest",
            port=port,
            baudrate=115200,
            BUFFER_SIZE=200,
            position=position,
            z_direction=z_direction
        )
    except Exception as e:
        print(f"✗ Failed to create ESP32 instance: {e}")
        return
    
    # Connect
    try:
        print("\nConnecting to ESP32...")
        esp32.start()
        time.sleep(2)  # Give ESP32 time to initialize
        
        # Test connection
        print("Testing connection...")
        esp32.print_status()
        print("✓ Connection successful!")
        
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that the ESP32 is connected")
        print("  2. Verify the port name is correct")
        print("  3. Make sure no other program is using the serial port")
        print("  4. Check that the firmware is uploaded to the ESP32")
        return
    
    # Run test based on mode
    try:
        if mode == "interactive":
            interactive_mode(esp32)
        else:
            test_servo_movement(esp32)
            
            # Ask if user wants interactive mode
            print("\n" + "="*60)
            response = input("Enter interactive mode? (y/n) [n]: ").strip().lower()
            if response == 'y':
                interactive_mode(esp32)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nDisconnecting...")
        try:
            # Return servos to center position
            esp32.move(0, 45, 0, 90)
            time.sleep(0.5)
            esp32.stop()
            print("✓ Disconnected successfully")
        except:
            pass


if __name__ == "__main__":
    main()

