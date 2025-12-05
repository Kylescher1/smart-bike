"""
ESP32 Sensor Test Script

Tests the ESP32 sensor with MPU6050, servos, and vibration motors.
Run this script to verify all functionality is working correctly.

Usage:
    python test_esp32.py [port]
    
    If port is not specified, will prompt for it or use default "COM7"
"""

import sys
import time
import numpy as np
import quaternion  # Required for np.quaternion support
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from hal.ESP32 import ESP32


def test_connection(esp32):
    """Test basic connection and status."""
    print("\n" + "="*60)
    print("TEST 1: Connection Test")
    print("="*60)
    try:
        esp32.print_status()
        print("✓ Connection test passed")
        return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


def test_read_accelerometer(esp32, num_readings=5):
    """Test reading accelerometer and gyro data."""
    print("\n" + "="*60)
    print(f"TEST 2: Accelerometer/Gyro Reading Test ({num_readings} readings)")
    print("="*60)
    
    readings = []
    for i in range(num_readings):
        try:
            data = esp32.read()
            if data is not None:
                ax, ay, az, gx, gy, gz = data
                readings.append(data)
                print(f"Reading {i+1}: ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}, "
                      f"gx={gx:.3f}, gy={gy:.3f}, gz={gz:.3f}")
                time.sleep(0.5)
            else:
                print(f"✗ Reading {i+1} failed: No data returned")
                return False
        except Exception as e:
            print(f"✗ Reading {i+1} failed: {e}")
            return False
    
    if len(readings) == num_readings:
        # Calculate statistics
        readings_array = np.array(readings)
        mean = np.mean(readings_array, axis=0)
        std = np.std(readings_array, axis=0)
        print(f"\nStatistics over {num_readings} readings:")
        print(f"  Mean: ax={mean[0]:.3f}, ay={mean[1]:.3f}, az={mean[2]:.3f}, "
              f"gx={mean[3]:.3f}, gy={mean[4]:.3f}, gz={mean[5]:.3f}")
        print(f"  Std:  ax={std[0]:.3f}, ay={std[1]:.3f}, az={std[2]:.3f}, "
              f"gx={std[3]:.3f}, gy={std[4]:.3f}, gz={std[5]:.3f}")
        print("✓ Accelerometer reading test passed")
        return True
    else:
        print("✗ Accelerometer reading test failed")
        return False


def test_servo_movement(esp32):
    """Test servo movement commands."""
    print("\n" + "="*60)
    print("TEST 3: Servo Movement Test")
    print("="*60)
    
    test_positions = [
        (45, 90),   # Bottom: 45°, Top: 90°
        (30, 45),   # Bottom: 30°, Top: 45°
        (60, 135),  # Bottom: 60°, Top: 135°
        (45, 90),   # Return to center
    ]
    
    for i, (bottom_angle, top_angle) in enumerate(test_positions):
        try:
            print(f"Moving servos: Bottom={bottom_angle}°, Top={top_angle}°")
            result = esp32.move(0, bottom_angle, 0, top_angle)
            if result:
                print(f"  ✓ Position {i+1} set successfully")
                time.sleep(1)  # Wait for servo to move
            else:
                print(f"  ✗ Position {i+1} failed")
                return False
        except Exception as e:
            print(f"  ✗ Position {i+1} error: {e}")
            return False
    
    print("✓ Servo movement test passed")
    return True


def test_vibration_motors(esp32):
    """Test vibration motor control."""
    print("\n" + "="*60)
    print("TEST 4: Vibration Motor Test")
    print("="*60)
    
    test_patterns = [
        (128, 0),    # Left only, 50%
        (0, 128),     # Right only, 50%
        (128, 128),   # Both, 50%
        (255, 255),   # Both, 100%
        (64, 64),     # Both, 25%
        (0, 0),       # Stop
    ]
    
    for i, (left, right) in enumerate(test_patterns):
        try:
            print(f"Vibration pattern {i+1}: Left={left}, Right={right}")
            result = esp32.vibrate(left, right)
            if result:
                print(f"  ✓ Pattern {i+1} set successfully")
                time.sleep(1)  # Feel the vibration
            else:
                print(f"  ✗ Pattern {i+1} failed")
                return False
        except Exception as e:
            print(f"  ✗ Pattern {i+1} error: {e}")
            return False
    
    print("✓ Vibration motor test passed")
    return True


def interactive_mode(esp32):
    """Interactive mode for manual testing."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  read - Read accelerometer/gyro data")
    print("  move <bottom> <top> - Move servos (e.g., 'move 45 90')")
    print("  vibrate <left> <right> - Set vibration (e.g., 'vibrate 128 128')")
    print("  status - Print sensor status")
    print("  quit - Exit interactive mode")
    print("="*60)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "quit" or cmd == "exit":
                break
            elif cmd == "read":
                data = esp32.read()
                if data is not None:
                    ax, ay, az, gx, gy, gz = data
                    print(f"Data: ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}, "
                          f"gx={gx:.3f}, gy={gy:.3f}, gz={gz:.3f}")
                else:
                    print("Failed to read data")
            elif cmd.startswith("move"):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        bottom = int(parts[1])
                        top = int(parts[2])
                        result = esp32.move(0, bottom, 0, top)
                        print("✓ Servos moved" if result else "✗ Failed to move servos")
                    except ValueError:
                        print("Invalid angles. Use: move <bottom> <top>")
                else:
                    print("Usage: move <bottom_angle> <top_angle>")
            elif cmd.startswith("vibrate"):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        left = int(parts[1])
                        right = int(parts[2])
                        result = esp32.vibrate(left, right)
                        print("✓ Vibration set" if result else "✗ Failed to set vibration")
                    except ValueError:
                        print("Invalid intensities. Use: vibrate <left> <right>")
                else:
                    print("Usage: vibrate <left_intensity> <right_intensity>")
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
    """Main test function."""
    print("ESP32 Sensor Test Script")
    print("="*60)
    
    # Get port from command line or prompt
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = input("Enter ESP32 serial port (e.g., COM7 or /dev/ttyUSB0) [default: COM7]: ").strip()
        if not port:
            port = "COM5"
    
    print(f"\nUsing port: {port}")
    print("Make sure the ESP32 is connected and the firmware is uploaded!")
    input("Press Enter to continue...")
    
    # Create ESP32 instance
    try:
        esp32 = ESP32(
            name="TestESP32",
            port=port,
            baudrate=115200,
            BUFFER_SIZE=200,
            position=np.quaternion(1, 0, 0, 0),
            z_direction=np.quaternion(0, 0, 0, 1)
        )
    except Exception as e:
        print(f"✗ Failed to create ESP32 instance: {e}")
        return
    
    # Connect
    try:
        print("\nConnecting to ESP32...")
        esp32.start()
        time.sleep(2)  # Give ESP32 time to initialize
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that the ESP32 is connected")
        print("  2. Verify the port name is correct")
        print("  3. Make sure no other program is using the serial port")
        print("  4. Check that the firmware is uploaded to the ESP32")
        return
    
    # Run tests
    results = []
    try:
        results.append(("Connection", test_connection(esp32)))
        results.append(("Accelerometer Reading", test_read_accelerometer(esp32)))
        results.append(("Servo Movement", test_servo_movement(esp32)))
        results.append(("Vibration Motors", test_vibration_motors(esp32)))
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:30s} {status}")
        
        all_passed = all(result[1] for result in results)
        
        if all_passed:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
        
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
            esp32.vibrate(0, 0)  # Stop vibration motors
            time.sleep(0.5)
            esp32.stop()
            print("✓ Disconnected successfully")
        except:
            pass


if __name__ == "__main__":
    main()

