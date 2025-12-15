#!/usr/bin/env python3
"""
Turret Range Test Script
Tests turret movement through its full range to verify limits and operation.

Usage:
    python test_turret_range.py --port COM3
    python test_turret_range.py --port /dev/ttyUSB0
    python test_turret_range.py --port COM3 --speed slow
"""

import serial
import time
import argparse
import sys


class TurretTester:
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        
        # Limits (will be read from Arduino)
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        self.top_pos = 90
        self.bottom_pos = 90
    
    def connect(self) -> bool:
        """Connect to turret"""
        try:
            print(f"Connecting to {self.port}...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            time.sleep(2)  # Wait for Arduino reset
            print("Connected!")
            return True
        except Exception as e:
            print(f"ERROR: Could not connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from turret"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected.")
    
    def send_command(self, cmd: str, wait_response: bool = True) -> str:
        """Send command and optionally wait for response"""
        if not self.ser or not self.ser.is_open:
            return ""
        
        try:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + '\n').encode())
            self.ser.flush()
            
            if not wait_response:
                return ""
            
            response = ""
            start = time.time()
            while time.time() - start < 1.0:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response += line + "\n"
                        if line.startswith("OK:") or line.startswith("ERROR:"):
                            break
                time.sleep(0.01)
            return response.strip()
        except Exception as e:
            print(f"Command error: {e}")
            return ""
    
    def update_status(self):
        """Read current status from Arduino"""
        resp = self.send_command("STATUS")
        if resp:
            for line in resp.split('\n'):
                if 'Top servo position:' in line:
                    try:
                        self.top_pos = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Bottom servo position:' in line:
                    try:
                        self.bottom_pos = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Top limits' in line:
                    try:
                        parts = line.split('MIN:')[1].split(',')
                        self.top_min = int(parts[0].strip())
                        self.top_max = int(parts[1].split('MAX:')[1].strip())
                    except:
                        pass
                elif 'Bottom limits' in line:
                    try:
                        parts = line.split('MIN:')[1].split(',')
                        self.bottom_min = int(parts[0].strip())
                        self.bottom_max = int(parts[1].split('MAX:')[1].strip())
                    except:
                        pass
    
    def move_to(self, bottom: int, top: int, wait: float = 0.5):
        """Move both servos to specified positions"""
        self.send_command(f"BOTTOM:{bottom}", wait_response=False)
        self.send_command(f"TOP:{top}", wait_response=False)
        time.sleep(wait)
        self.bottom_pos = bottom
        self.top_pos = top
    
    def home(self):
        """Move to home position"""
        print("Moving to HOME...")
        self.send_command("HOME")
        time.sleep(1)
        self.update_status()
        print(f"  At home: Bottom={self.bottom_pos}°, Top={self.top_pos}°")
    
    def test_range(self, speed: str = "normal"):
        """Test full range of motion"""
        # Speed settings
        if speed == "slow":
            step = 5
            wait = 0.3
        elif speed == "fast":
            step = 15
            wait = 0.1
        else:  # normal
            step = 10
            wait = 0.2
        
        print("\n" + "="*60)
        print("TURRET RANGE TEST")
        print("="*60)
        
        # Get current limits
        self.update_status()
        print(f"\nCurrent Limits:")
        print(f"  Bottom (pan):  {self.bottom_min}° to {self.bottom_max}° (range: {self.bottom_max - self.bottom_min}°)")
        print(f"  Top (tilt):    {self.top_min}° to {self.top_max}° (range: {self.top_max - self.top_min}°)")
        print(f"  Current pos:   Bottom={self.bottom_pos}°, Top={self.top_pos}°")
        
        # Calculate centers
        bottom_center = (self.bottom_min + self.bottom_max) // 2
        top_center = (self.top_min + self.top_max) // 2
        
        input("\nPress ENTER to start range test (or Ctrl+C to abort)...")
        
        # Test 1: Home position
        print("\n--- TEST 1: Home Position ---")
        self.home()
        
        # Test 2: Pan (bottom servo) sweep
        print(f"\n--- TEST 2: Pan Sweep (Bottom Servo: {self.bottom_min}° to {self.bottom_max}°) ---")
        print("Moving to pan minimum...")
        self.move_to(self.bottom_min, top_center, wait=1.0)
        print(f"  At pan MIN: {self.bottom_min}°")
        
        print("Sweeping pan from MIN to MAX...")
        for angle in range(self.bottom_min, self.bottom_max + 1, step):
            self.move_to(angle, top_center, wait=wait)
            print(f"  Pan: {angle}°", end='\r')
        print(f"\n  At pan MAX: {self.bottom_max}°")
        
        print("Sweeping pan from MAX to MIN...")
        for angle in range(self.bottom_max, self.bottom_min - 1, -step):
            self.move_to(angle, top_center, wait=wait)
            print(f"  Pan: {angle}°", end='\r')
        print(f"\n  At pan MIN: {self.bottom_min}°")
        
        # Return to center
        print("Returning to center...")
        self.move_to(bottom_center, top_center, wait=1.0)
        
        # Test 3: Tilt (top servo) sweep
        print(f"\n--- TEST 3: Tilt Sweep (Top Servo: {self.top_min}° to {self.top_max}°) ---")
        print("Moving to tilt minimum...")
        self.move_to(bottom_center, self.top_min, wait=1.0)
        print(f"  At tilt MIN: {self.top_min}°")
        
        print("Sweeping tilt from MIN to MAX...")
        for angle in range(self.top_min, self.top_max + 1, step):
            self.move_to(bottom_center, angle, wait=wait)
            print(f"  Tilt: {angle}°", end='\r')
        print(f"\n  At tilt MAX: {self.top_max}°")
        
        print("Sweeping tilt from MAX to MIN...")
        for angle in range(self.top_max, self.top_min - 1, -step):
            self.move_to(bottom_center, angle, wait=wait)
            print(f"  Tilt: {angle}°", end='\r')
        print(f"\n  At tilt MIN: {self.top_min}°")
        
        # Return to center
        print("Returning to center...")
        self.move_to(bottom_center, top_center, wait=1.0)
        
        # Test 4: Corner sweep
        print("\n--- TEST 4: Corner Sweep ---")
        corners = [
            ("Top-Left", self.bottom_min, self.top_min),
            ("Top-Right", self.bottom_max, self.top_min),
            ("Bottom-Right", self.bottom_max, self.top_max),
            ("Bottom-Left", self.bottom_min, self.top_max),
            ("Center", bottom_center, top_center),
        ]
        
        for name, pan, tilt in corners:
            print(f"  Moving to {name} (Pan={pan}°, Tilt={tilt}°)...")
            self.move_to(pan, tilt, wait=1.5)
        
        # Test 5: Figure-8 pattern
        print("\n--- TEST 5: Figure-8 Pattern ---")
        import math
        points = 36
        for i in range(points + 1):
            t = (i / points) * 2 * math.pi
            # Figure-8 parametric equations
            pan_range = (self.bottom_max - self.bottom_min) / 2 * 0.8
            tilt_range = (self.top_max - self.top_min) / 2 * 0.8
            pan = bottom_center + pan_range * math.sin(t)
            tilt = top_center + tilt_range * math.sin(2 * t)
            self.move_to(int(pan), int(tilt), wait=0.1)
            print(f"  Figure-8: {i}/{points}", end='\r')
        print("\n  Figure-8 complete!")
        
        # Final: Return home
        print("\n--- FINAL: Return to Home ---")
        self.home()
        
        print("\n" + "="*60)
        print("RANGE TEST COMPLETE!")
        print("="*60)
        print("\nIf the turret moved smoothly through all positions, it's working correctly.")
        print("If movement was jerky or didn't reach limits, check:")
        print("  1. Servo connections")
        print("  2. Power supply (servos need adequate current)")
        print("  3. Mechanical obstructions")
        print("  4. Arduino limit settings")
    
    def interactive_test(self):
        """Interactive manual control for testing"""
        print("\n" + "="*60)
        print("INTERACTIVE TURRET CONTROL")
        print("="*60)
        print("Commands:")
        print("  w/s - Tilt up/down")
        print("  a/d - Pan left/right")
        print("  h   - Home")
        print("  p   - Print current position")
        print("  l   - Print limits")
        print("  r   - Run full range test")
        print("  q   - Quit")
        print("="*60)
        
        self.update_status()
        step = 5
        
        try:
            while True:
                cmd = input(f"\n[Pan={self.bottom_pos:.0f}° Tilt={self.top_pos:.0f}°] > ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'w':
                    new_tilt = max(self.top_min, self.top_pos - step)
                    self.move_to(int(self.bottom_pos), int(new_tilt))
                    print(f"Tilt UP to {new_tilt}°")
                elif cmd == 's':
                    new_tilt = min(self.top_max, self.top_pos + step)
                    self.move_to(int(self.bottom_pos), int(new_tilt))
                    print(f"Tilt DOWN to {new_tilt}°")
                elif cmd == 'a':
                    new_pan = max(self.bottom_min, self.bottom_pos - step)
                    self.move_to(int(new_pan), int(self.top_pos))
                    print(f"Pan LEFT to {new_pan}°")
                elif cmd == 'd':
                    new_pan = min(self.bottom_max, self.bottom_pos + step)
                    self.move_to(int(new_pan), int(self.top_pos))
                    print(f"Pan RIGHT to {new_pan}°")
                elif cmd == 'h':
                    self.home()
                elif cmd == 'p':
                    self.update_status()
                    print(f"Position: Pan={self.bottom_pos}°, Tilt={self.top_pos}°")
                elif cmd == 'l':
                    self.update_status()
                    print(f"Limits: Pan={self.bottom_min}°-{self.bottom_max}°, Tilt={self.top_min}°-{self.top_max}°")
                elif cmd == 'r':
                    self.test_range()
                elif cmd.startswith('pan ') or cmd.startswith('p '):
                    try:
                        angle = int(cmd.split()[1])
                        self.move_to(angle, int(self.top_pos))
                        print(f"Pan to {angle}°")
                    except:
                        print("Usage: pan <angle>")
                elif cmd.startswith('tilt ') or cmd.startswith('t '):
                    try:
                        angle = int(cmd.split()[1])
                        self.move_to(int(self.bottom_pos), angle)
                        print(f"Tilt to {angle}°")
                    except:
                        print("Usage: tilt <angle>")
                elif cmd == '':
                    pass
                else:
                    print("Unknown command. Use w/a/s/d to move, h=home, p=position, l=limits, r=range test, q=quit")
        
        except KeyboardInterrupt:
            print("\nInterrupted.")
        
        # Return home before exit
        print("Returning home...")
        self.home()


def list_ports():
    """List available serial ports"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
    else:
        print("No serial ports found")


def main():
    parser = argparse.ArgumentParser(description="Test turret range of motion")
    parser.add_argument('--port', '-p', type=str, required=False,
                       help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--speed', '-s', type=str, default='normal',
                       choices=['slow', 'normal', 'fast'],
                       help='Test speed (slow/normal/fast)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode for manual control')
    parser.add_argument('--list-ports', '-l', action='store_true',
                       help='List available serial ports')
    
    args = parser.parse_args()
    
    if args.list_ports:
        list_ports()
        return
    
    if not args.port:
        list_ports()
        print("\nPlease specify a port with --port")
        return
    
    tester = TurretTester(args.port)
    
    if not tester.connect():
        sys.exit(1)
    
    try:
        if args.interactive:
            tester.interactive_test()
        else:
            tester.test_range(speed=args.speed)
    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        # Always try to home before disconnecting
        try:
            tester.home()
        except:
            pass
        tester.disconnect()


if __name__ == '__main__':
    main()

