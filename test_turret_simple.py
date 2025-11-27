"""
Simple turret test - verify communication works
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hal.TurretControl import TurretControl

def main():
    print("=" * 60)
    print("Simple Turret Communication Test")
    print("=" * 60)
    
    # Initialize turret
    turret_port = "COM5"  # Change this to your port
    if sys.platform.startswith('linux'):
        turret_port = "/dev/ttyUSB0"
    
    print(f"\nConnecting to turret on {turret_port}...")
    turret = TurretControl(
        port=turret_port,
        baudrate=115200,
        servo1_min=15,
        servo1_max=50,
        servo1_home=35,
        servo2_min=0,
        servo2_max=180,
        servo2_home=90
    )
    
    try:
        turret.connect()
        if not turret.connected:
            print("❌ Failed to connect to turret")
            print("   Check:")
            print("   1. Arduino is connected and powered")
            print("   2. Port is correct (currently:", turret_port, ")")
            print("   3. Arduino is running turret_control.ino")
            return
        
        print("✅ Connected!")
        
        # Test serial communication first
        print("\n📡 Testing serial communication...")
        print("  Sending test command: S1:35,S2:90")
        turret.move_to_absolute(35, 90)
        time.sleep(0.5)
        
        # Check if we can read from Arduino
        print("  Checking for Arduino responses...")
        time.sleep(1.0)  # Give Arduino time to respond
        responses = []
        max_responses = 5
        while turret.ser and turret.ser.in_waiting > 0 and len(responses) < max_responses:
            try:
                response = turret.ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    responses.append(response)
                    print(f"  Arduino: {response}")
            except:
                break
        if not responses:
            print("  ⚠️  No response from Arduino - check:")
            print("     - Is Arduino Serial Monitor closed?")
            print("     - Is baudrate correct (115200)?")
            print("     - Try unplugging/replugging USB")
        
        # Test movements
        print("\nTesting servo movements...")
        
        print("  Moving to home (S1:35, S2:90)...")
        turret.move_to_absolute(35, 90)
        time.sleep(2)  # Longer delay to see movement
        
        print("  Moving servo 1 to MAX (50 degrees)...")
        turret.move_to_absolute(50, 90)
        time.sleep(3)  # Longer delay
        
        print("  Moving servo 1 to MIN (15 degrees)...")
        turret.move_to_absolute(15, 90)
        time.sleep(3)
        
        print("  Moving servo 1 back to center...")
        turret.move_to_absolute(35, 90)
        time.sleep(2)
        
        print("  Moving servo 2 to MAX (180 degrees)...")
        turret.move_to_absolute(35, 180)
        time.sleep(3)
        
        print("  Moving servo 2 to MIN (0 degrees)...")
        turret.move_to_absolute(35, 0)
        time.sleep(3)
        
        print("  Moving servo 2 back to center...")
        turret.move_to_absolute(35, 90)
        time.sleep(2)
        
        print("  Returning to home...")
        turret.move_to_absolute(35, 90)
        time.sleep(2)
        
        print("\n✅ Test complete! If servos moved, communication is working.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        turret.disconnect()

if __name__ == "__main__":
    main()

