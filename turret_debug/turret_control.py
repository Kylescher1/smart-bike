#!/usr/bin/env python3
"""
Turret Debug Control Script
Provides a Python interface to control the turret debug Arduino sketch.

Usage:
    python turret_control.py [port] [--interactive]
    
    port: Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
    --interactive: Start interactive command mode
"""

import serial
import sys
import time
import argparse
from typing import Optional


class TurretController:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        """Initialize connection to turret debug controller."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        
    def connect(self) -> bool:
        """Connect to the serial port."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected")
    
    def send_command(self, command: str) -> str:
        """Send a command and return the response."""
        if not self.ser or not self.ser.is_open:
            return "ERROR: Not connected"
        
        try:
            # Clear any pending input
            self.ser.reset_input_buffer()
            
            # Send command
            self.ser.write((command + '\n').encode())
            self.ser.flush()
            
            # Read response (with timeout)
            response = ""
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response += line + "\n"
                        # Stop if we get an OK or ERROR response
                        if line.startswith("OK:") or line.startswith("ERROR:"):
                            break
                time.sleep(0.01)
            
            return response.strip() if response else "No response"
        except Exception as e:
            return f"ERROR: {e}"
    
    def home(self) -> str:
        """Move servos to home position."""
        return self.send_command("HOME")
    
    def set_top(self, angle: int) -> str:
        """Set top servo angle."""
        return self.send_command(f"TOP:{angle}")
    
    def set_bottom(self, angle: int) -> str:
        """Set bottom servo angle."""
        return self.send_command(f"BOTTOM:{angle}")
    
    def set_both(self, angle: int) -> str:
        """Set both servos to same angle."""
        return self.send_command(f"BOTH:{angle}")
    
    def test_top_min(self) -> str:
        """Test top servo minimum limit."""
        return self.send_command("TEST_TOP_MIN")
    
    def test_top_max(self) -> str:
        """Test top servo maximum limit."""
        return self.send_command("TEST_TOP_MAX")
    
    def test_bottom_min(self) -> str:
        """Test bottom servo minimum limit."""
        return self.send_command("TEST_BOTTOM_MIN")
    
    def test_bottom_max(self) -> str:
        """Test bottom servo maximum limit."""
        return self.send_command("TEST_BOTTOM_MAX")
    
    def set_min_limit(self, value: int) -> str:
        """Set minimum servo limit."""
        return self.send_command(f"SET_MIN:{value}")
    
    def set_max_limit(self, value: int) -> str:
        """Set maximum servo limit."""
        return self.send_command(f"SET_MAX:{value}")
    
    def get_limits(self) -> str:
        """Get current servo limits."""
        return self.send_command("GET_LIMITS")
    
    def set_motor1(self, speed: int) -> str:
        """Set motor 1 speed (0-255)."""
        return self.send_command(f"MOTOR1:{speed}")
    
    def set_motor2(self, speed: int) -> str:
        """Set motor 2 speed (0-255)."""
        return self.send_command(f"MOTOR2:{speed}")
    
    def get_status(self) -> str:
        """Get current status."""
        return self.send_command("STATUS")
    
    def help(self) -> str:
        """Get help text."""
        return self.send_command("HELP")


def interactive_mode(controller: TurretController):
    """Run interactive command mode."""
    print("\n=== Interactive Turret Control ===")
    print("Type commands or 'quit' to exit")
    print("Type 'help' for available commands\n")
    
    while True:
        try:
            cmd = input("Turret> ").strip()
            
            if not cmd:
                continue
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                break
            
            if cmd.lower() == 'help':
                print("\nPython Commands:")
                print("  home              - Move to home position")
                print("  top <angle>       - Set top servo")
                print("  bottom <angle>    - Set bottom servo")
                print("  both <angle>      - Set both servos")
                print("  test_top_min      - Test top min limit")
                print("  test_top_max      - Test top max limit")
                print("  test_bottom_min   - Test bottom min limit")
                print("  test_bottom_max   - Test bottom max limit")
                print("  set_min <value>   - Set minimum limit")
                print("  set_max <value>   - Set maximum limit")
                print("  limits            - Get current limits")
                print("  motor1 <speed>    - Set motor 1 (0-255)")
                print("  motor2 <speed>    - Set motor 2 (0-255)")
                print("  status            - Get status")
                print("  raw <command>     - Send raw command")
                print("  quit              - Exit\n")
                continue
            
            # Parse commands
            parts = cmd.split()
            cmd_name = parts[0].lower()
            
            if cmd_name == 'home':
                print(controller.home())
            elif cmd_name == 'top' and len(parts) > 1:
                print(controller.set_top(int(parts[1])))
            elif cmd_name == 'bottom' and len(parts) > 1:
                print(controller.set_bottom(int(parts[1])))
            elif cmd_name == 'both' and len(parts) > 1:
                print(controller.set_both(int(parts[1])))
            elif cmd_name == 'test_top_min':
                print(controller.test_top_min())
            elif cmd_name == 'test_top_max':
                print(controller.test_top_max())
            elif cmd_name == 'test_bottom_min':
                print(controller.test_bottom_min())
            elif cmd_name == 'test_bottom_max':
                print(controller.test_bottom_max())
            elif cmd_name == 'set_min' and len(parts) > 1:
                print(controller.set_min_limit(int(parts[1])))
            elif cmd_name == 'set_max' and len(parts) > 1:
                print(controller.set_max_limit(int(parts[1])))
            elif cmd_name == 'limits':
                print(controller.get_limits())
            elif cmd_name == 'motor1' and len(parts) > 1:
                print(controller.set_motor1(int(parts[1])))
            elif cmd_name == 'motor2' and len(parts) > 1:
                print(controller.set_motor2(int(parts[1])))
            elif cmd_name == 'status':
                print(controller.get_status())
            elif cmd_name == 'raw' and len(parts) > 1:
                raw_cmd = ' '.join(parts[1:])
                print(controller.send_command(raw_cmd))
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def list_serial_ports():
    """List available serial ports."""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
    else:
        print("No serial ports found")


def main():
    parser = argparse.ArgumentParser(
        description='Turret Debug Control Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python turret_control.py COM3 --interactive
  python turret_control.py /dev/ttyUSB0
  python turret_control.py --list-ports
        """
    )
    parser.add_argument('port', nargs='?', help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive mode')
    parser.add_argument('--list-ports', '-l', action='store_true', help='List available serial ports')
    parser.add_argument('--baudrate', '-b', type=int, default=115200, help='Baud rate (default: 115200)')
    
    args = parser.parse_args()
    
    if args.list_ports:
        list_serial_ports()
        return
    
    if not args.port:
        print("Error: Serial port required")
        print("Use --list-ports to see available ports")
        parser.print_help()
        sys.exit(1)
    
    controller = TurretController(args.port, baudrate=args.baudrate)
    
    if not controller.connect():
        sys.exit(1)
    
    try:
        if args.interactive:
            interactive_mode(controller)
        else:
            # Non-interactive: just print status
            print(controller.get_status())
    finally:
        controller.disconnect()


if __name__ == '__main__':
    main()

