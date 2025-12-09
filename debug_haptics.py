#!/usr/bin/env python3
"""
Haptic Motors Debug Script

Systematic troubleshooting for haptic motor issues.
Tests communication, pin control, and hardware step by step.

Usage:
    python debug_haptics.py [port]
"""

import sys
import time
import serial
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

try:
    import quaternion
    has_quaternion = True
except ImportError:
    has_quaternion = False

from hal.ESP32 import ESP32


def test_serial_communication(port):
    """Test 1: Basic serial communication."""
    print("\n" + "="*60)
    print("TEST 1: Serial Communication")
    print("="*60)
    
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(2)  # Wait for ESP32 to initialize
        
        # Clear buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send READ command (should always work)
        print("Sending READ command...")
        ser.write(b"READ\n")
        ser.flush()
        time.sleep(0.1)
        
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"Response: {repr(response)}")
        
        if response and response != "ERROR" and ',' in response:
            print("✓ Serial communication working")
            ser.close()
            return True
        else:
            print("✗ Serial communication issue")
            print(f"  Got: {response}")
            ser.close()
            return False
            
    except Exception as e:
        print(f"✗ Serial communication failed: {e}")
        return False


def test_vibrate_command_response(esp32):
    """Test 2: VIBRATE command response."""
    print("\n" + "="*60)
    print("TEST 2: VIBRATE Command Response")
    print("="*60)
    
    test_cases = [
        (0, 0, "Stop"),
        (128, 128, "Both 50%"),
        (255, 0, "Left full, Right off"),
        (0, 255, "Left off, Right full"),
    ]
    
    all_passed = True
    for left, right, description in test_cases:
        print(f"\nTesting: {description} (L={left}, R={right})")
        
        try:
            # Send command directly to see raw response
            command = f"VIBRATE,{left},{right}"
            print(f"  Sending: {command}")
            
            response = esp32._send_command(command, wait_for_response=True)
            print(f"  Response: {repr(response)}")
            
            if response == "OK":
                print(f"  ✓ Command accepted")
            else:
                print(f"  ✗ Command failed - got: {response}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_passed = False
    
    return all_passed


def test_pin_control_direct(port):
    """Test 3: Direct pin control via serial."""
    print("\n" + "="*60)
    print("TEST 3: Direct Pin Control Test")
    print("="*60)
    print("This test sends VIBRATE commands and checks if pins are responding.")
    print("Use a multimeter or oscilloscope to check pins 25, 26, 27, 14")
    print("\nExpected behavior:")
    print("  - Pin 25 (IN1): Should show PWM when Left > 0")
    print("  - Pin 26 (IN2): Should stay LOW (forward mode)")
    print("  - Pin 27 (IN3): Should show PWM when Right > 0")
    print("  - Pin 14 (IN4): Should stay LOW (forward mode)")
    
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(2)
        
        test_sequence = [
            (0, 0, "Stop - all pins LOW", 2),
            (255, 0, "Left full - Pin 25 should pulse, Pin 26 LOW", 3),
            (0, 0, "Stop", 1),
            (0, 255, "Right full - Pin 27 should pulse, Pin 14 LOW", 3),
            (0, 0, "Stop", 1),
            (128, 128, "Both 50% - Pins 25 & 27 should pulse", 3),
            (0, 0, "Final stop", 1),
        ]
        
        for left, right, description, duration in test_sequence:
            print(f"\n{description}")
            print(f"  Command: VIBRATE,{left},{right}")
            
            ser.reset_input_buffer()
            ser.write(f"VIBRATE,{left},{right}\n".encode())
            ser.flush()
            time.sleep(0.1)
            
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  Response: {response}")
            
            if response == "OK":
                print(f"  ✓ Command accepted - check pins now!")
                print(f"  Waiting {duration} seconds...")
                time.sleep(duration)
            else:
                print(f"  ✗ Command failed")
        
        ser.close()
        print("\n✓ Pin control test complete")
        print("Did you see PWM signals on pins 25 and/or 27?")
        return True
        
    except Exception as e:
        print(f"\n✗ Pin control test failed: {e}")
        return False


def test_esp32_wrapper(esp32):
    """Test 4: ESP32 Python wrapper."""
    print("\n" + "="*60)
    print("TEST 4: ESP32 Python Wrapper")
    print("="*60)
    
    try:
        print("Testing esp32.vibrate() method...")
        
        # Test stop
        print("  Test 1: Stop (0, 0)")
        result = esp32.vibrate(0, 0)
        print(f"    Result: {result}")
        time.sleep(1)
        
        # Test left only
        print("  Test 2: Left only (128, 0)")
        result = esp32.vibrate(128, 0)
        print(f"    Result: {result}")
        time.sleep(2)
        
        # Test right only
        print("  Test 3: Right only (0, 128)")
        result = esp32.vibrate(0, 128)
        print(f"    Result: {result}")
        time.sleep(2)
        
        # Test both
        print("  Test 4: Both (128, 128)")
        result = esp32.vibrate(128, 128)
        print(f"    Result: {result}")
        time.sleep(2)
        
        # Stop
        print("  Test 5: Stop")
        esp32.vibrate(0, 0)
        
        print("\n✓ ESP32 wrapper test complete")
        print("Did the motors vibrate?")
        return True
        
    except Exception as e:
        print(f"\n✗ ESP32 wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_hardware_setup():
    """Test 5: Hardware checklist."""
    print("\n" + "="*60)
    print("TEST 5: Hardware Checklist")
    print("="*60)
    
    print("\nPlease verify the following:")
    print("\n1. Power Supply:")
    print("   [ ] Motors have separate power supply (not from ESP32)")
    print("   [ ] Power supply voltage matches motor requirements")
    print("   [ ] Power supply ground connected to ESP32 ground")
    
    print("\n2. Wiring:")
    print("   [ ] IN1 (GPIO 25) connected to motor driver")
    print("   [ ] IN2 (GPIO 26) connected to motor driver")
    print("   [ ] IN3 (GPIO 27) connected to motor driver")
    print("   [ ] IN4 (GPIO 14) connected to motor driver")
    print("   [ ] Motor driver outputs connected to motors")
    
    print("\n3. Motor Driver:")
    print("   [ ] Motor driver is powered")
    print("   [ ] Enable pins (if present) are set correctly")
    print("   [ ] Motor driver logic voltage matches ESP32 (3.3V)")
    
    print("\n4. Motors:")
    print("   [ ] Motors are connected to driver outputs")
    print("   [ ] Motors are not shorted")
    print("   [ ] Motors can spin freely")
    
    response = input("\nHave you checked all items? (y/n): ").strip().lower()
    return response == 'y'


def main():
    """Main troubleshooting function."""
    print("="*60)
    print("HAPTIC MOTORS TROUBLESHOOTING")
    print("="*60)
    
    # Get port
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = input("Enter ESP32 serial port (e.g., COM5) [default: COM5]: ").strip()
        if not port:
            port = "COM5"
    
    print(f"\nUsing port: {port}")
    
    results = {}
    
    # Test 1: Serial communication
    results['serial'] = test_serial_communication(port)
    if not results['serial']:
        print("\n⚠ Serial communication failed. Check:")
        print("  - Port name is correct")
        print("  - ESP32 is connected")
        print("  - No other program is using the port")
        print("  - Firmware is uploaded")
        return
    
    # Test 2: Create ESP32 instance
    print("\n" + "="*60)
    print("Creating ESP32 instance...")
    try:
        if has_quaternion:
            position = None  # Will be set if needed
            z_direction = None
        else:
            position = None
            z_direction = None
        
        esp32 = ESP32(
            name="HapticDebug",
            port=port,
            baudrate=115200,
            BUFFER_SIZE=200,
            position=position,
            z_direction=z_direction
        )
        esp32.start()
        time.sleep(2)
        print("✓ ESP32 connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    # Test 3: VIBRATE command response
    results['command'] = test_vibrate_command_response(esp32)
    
    # Test 4: Pin control
    results['pins'] = test_pin_control_direct(port)
    
    # Test 5: ESP32 wrapper
    results['wrapper'] = test_esp32_wrapper(esp32)
    
    # Test 6: Hardware checklist
    results['hardware'] = check_hardware_setup()
    
    # Summary
    print("\n" + "="*60)
    print("TROUBLESHOOTING SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not results.get('command', False):
        print("\n1. VIBRATE command not responding:")
        print("   - Check Serial Monitor to see if ESP32 receives commands")
        print("   - Verify firmware is uploaded correctly")
        print("   - Check for compilation errors")
    
    if not results.get('pins', False):
        print("\n2. Pins not responding:")
        print("   - Use multimeter to check if pins 25, 26, 27, 14 are outputting")
        print("   - Check for pin conflicts")
        print("   - Verify GPIO pins are not damaged")
    
    if results.get('command', False) and results.get('pins', False) and not results.get('wrapper', False):
        print("\n3. Commands work but motors don't vibrate:")
        print("   - Check motor power supply")
        print("   - Verify motor driver connections")
        print("   - Test motors directly with power supply")
        print("   - Check motor driver enable pins")
    
    # Cleanup
    try:
        esp32.vibrate(0, 0)
        esp32.stop()
    except:
        pass


if __name__ == "__main__":
    main()

