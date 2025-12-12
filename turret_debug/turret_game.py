#!/usr/bin/env python3
"""
Turret Control Game - Video Game Style Interface
Real-time keyboard controls for turret debugging

Controls:
  W/S or ↑/↓  - Move top servo up/down
  A/D or ←/→  - Move bottom servo left/right
  Q/E         - Fine adjust top servo
  Z/X         - Fine adjust bottom servo
  SPACE       - Home position
  R           - Reset limits
  T/Y         - Test top min/max
  G/H         - Test bottom min/max
  M/N         - Motor 1 speed up/down
  ,/.         - Motor 2 speed up/down
  1-9         - Quick position presets
  ESC         - Exit
"""

import serial
import sys
import time
import argparse
import threading
from typing import Optional
import os

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Fallback colors (empty strings)
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

# Cross-platform keyboard input
_is_windows = os.name == 'nt'

if _is_windows:
    import msvcrt
    def get_key():
        if msvcrt.kbhit():
            try:
                key = msvcrt.getch()
                if key == b'\xe0':  # Extended key (arrows)
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'H': return 'UP'
                        if key == b'P': return 'DOWN'
                        if key == b'K': return 'LEFT'
                        if key == b'M': return 'RIGHT'
                    return None
                elif key == b'\x1b': return 'ESC'
                elif key == b' ': return 'SPACE'
                elif key == b'\r': return 'ENTER'
                elif key == b'\x03': return 'ESC'  # Ctrl+C
                else:
                    try:
                        decoded = key.decode('utf-8', errors='ignore')
                        if decoded:
                            return decoded.upper()
                    except:
                        pass
            except:
                pass
        return None
else:
    import select
    import termios
    import tty
    
    # Set terminal to raw mode once at startup
    _fd = sys.stdin.fileno()
    _old_settings = termios.tcgetattr(_fd)
    tty.setraw(_fd)
    
    def get_key():
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                    ch = sys.stdin.read(2)
                    if ch == '[A': return 'UP'
                    if ch == '[B': return 'DOWN'
                    if ch == '[C': return 'RIGHT'
                    if ch == '[D': return 'LEFT'
                return 'ESC'
            elif ch == ' ': return 'SPACE'
            elif ch == '\r' or ch == '\n': return 'ENTER'
            elif ch == '\x03':  # Ctrl+C
                return 'ESC'
            return ch.upper()
        return None
    
    # Restore terminal on exit
    import atexit
    def restore_terminal():
        termios.tcsetattr(_fd, termios.TCSADRAIN, _old_settings)
    atexit.register(restore_terminal)


class TurretController:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.top_pos = 90
        self.bottom_pos = 90
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        self.motor1_speed = 0
        self.motor2_speed = 0
        
    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            time.sleep(2)
            return True
        except serial.SerialException as e:
            print(f"Error connecting: {e}")
            return False
    
    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def send_command(self, command: str, read_response: bool = True) -> Optional[str]:
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            self.ser.write((command + '\n').encode())
            self.ser.flush()
            
            if not read_response:
                return None
                
            response = ""
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response += line + "\n"
                        if line.startswith("OK:") or line.startswith("ERROR:"):
                            break
                time.sleep(0.01)
            return response.strip() if response else None
        except:
            return None
    
    def update_status(self):
        """Update internal status from Arduino"""
        resp = self.send_command("STATUS")
        if resp:
            # Parse status response
            for line in resp.split('\n'):
                if 'Top servo position:' in line:
                    try:
                        self.top_pos = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Bottom servo position:' in line:
                    try:
                        self.bottom_pos = int(line.split(':')[1].strip())
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


class TurretGame:
    def __init__(self, controller: TurretController):
        self.controller = controller
        self.running = True
        self.step_size = 5  # Coarse movement
        self.fine_step = 1  # Fine movement
        self.last_update = time.time()
        self.update_interval = 1.0  # Update status every 1 second (less frequent)
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_turret(self):
        """Draw ASCII art turret representation"""
        top_angle = self.controller.top_pos
        bottom_angle = self.controller.bottom_pos
        
        # Normalize angles for display (0-180 -> visual representation)
        top_bar = int((top_angle / 180.0) * 40)
        bottom_bar = int((bottom_angle / 180.0) * 40)
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}    TURRET CONTROL SYSTEM{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # Top Servo
        print(f"{Fore.GREEN}TOP SERVO:{Style.RESET_ALL}")
        bar = '█' * top_bar + '░' * (40 - top_bar)
        print(f"  [{bar}] {top_angle}°")
        
        # Visual turret representation
        print(f"\n{Fore.MAGENTA}     ┌─────┐")
        print(f"     │  ╱╲  │  ← Top: {top_angle}°")
        print(f"     │ ╱  ╲ │")
        print(f"     └───┬───┘")
        print(f"         │")
        print(f"     ┌───┴───┐")
        print(f"     │   ╱╲  │  ← Bottom: {bottom_angle}°")
        print(f"     │  ╱  ╲ │")
        print(f"     └───────┘{Style.RESET_ALL}")
        
        # Bottom Servo
        print(f"\n{Fore.GREEN}BOTTOM SERVO:{Style.RESET_ALL}")
        bar = '█' * bottom_bar + '░' * (40 - bottom_bar)
        print(f"  [{bar}] {bottom_angle}°")
        
        # Limits
        print(f"\n{Fore.BLUE}Top Limits: MIN={self.controller.top_min}°  MAX={self.controller.top_max}°{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Bottom Limits: MIN={self.controller.bottom_min}°  MAX={self.controller.bottom_max}°{Style.RESET_ALL}")
        
        # Motors
        m1_bar = int((self.controller.motor1_speed / 255.0) * 20)
        m2_bar = int((self.controller.motor2_speed / 255.0) * 20)
        print(f"\n{Fore.RED}MOTOR 1:{Style.RESET_ALL} {'█' * m1_bar}{'░' * (20 - m1_bar)} {self.controller.motor1_speed}/255")
        print(f"{Fore.RED}MOTOR 2:{Style.RESET_ALL} {'█' * m2_bar}{'░' * (20 - m2_bar)} {self.controller.motor2_speed}/255")
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def draw_controls(self):
        """Draw control instructions"""
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}CONTROLS:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}W/S{Style.RESET_ALL} or {Fore.GREEN}↑/↓{Style.RESET_ALL}  - Top servo up/down")
        print(f"  {Fore.GREEN}A/D{Style.RESET_ALL} or {Fore.GREEN}←/→{Style.RESET_ALL}  - Bottom servo left/right")
        print(f"  {Fore.GREEN}Q/E{Style.RESET_ALL}         - Fine adjust top servo")
        print(f"  {Fore.GREEN}Z/X{Style.RESET_ALL}         - Fine adjust bottom servo")
        print(f"  {Fore.GREEN}SPACE{Style.RESET_ALL}       - Home position")
        print(f"  {Fore.GREEN}R{Style.RESET_ALL}           - Reset limits")
        print(f"  {Fore.GREEN}T/Y{Style.RESET_ALL}         - Test top min/max")
        print(f"  {Fore.GREEN}G/H{Style.RESET_ALL}         - Test bottom min/max")
        print(f"  {Fore.GREEN}M/N{Style.RESET_ALL}         - Motor 1 speed +/-")
        print(f"  {Fore.GREEN},/.{Style.RESET_ALL}         - Motor 2 speed +/-")
        print(f"  {Fore.GREEN}1-9{Style.RESET_ALL}         - Quick presets")
        print(f"  {Fore.RED}ESC{Style.RESET_ALL}         - Exit")
        print()
    
    def move_top(self, delta: int):
        new_pos = self.controller.top_pos + delta
        new_pos = max(self.controller.top_min, min(self.controller.top_max, new_pos))
        # Only send command if position actually changed (prevents buzzing)
        if new_pos != self.controller.top_pos:
            self.controller.send_command(f"TOP:{new_pos}", read_response=False)
            self.controller.top_pos = new_pos
    
    def move_bottom(self, delta: int):
        new_pos = self.controller.bottom_pos + delta
        new_pos = max(self.controller.bottom_min, min(self.controller.bottom_max, new_pos))
        # Only send command if position actually changed (prevents buzzing)
        if new_pos != self.controller.bottom_pos:
            self.controller.send_command(f"BOTTOM:{new_pos}", read_response=False)
            self.controller.bottom_pos = new_pos
    
    def set_preset(self, num: int):
        """Set preset positions"""
        presets = {
            1: (45, 45),
            2: (90, 90),
            3: (135, 135),
            4: (45, 135),
            5: (135, 45),
            6: (0, 90),
            7: (180, 90),
            8: (90, 0),
            9: (90, 180),
        }
        if num in presets:
            top, bottom = presets[num]
            self.controller.send_command(f"TOP:{top}", read_response=False)
            self.controller.send_command(f"BOTTOM:{bottom}", read_response=False)
            self.controller.top_pos = top
            self.controller.bottom_pos = bottom
    
    def adjust_motor(self, motor: int, delta: int):
        if motor == 1:
            self.controller.motor1_speed = max(0, min(255, self.controller.motor1_speed + delta))
            self.controller.send_command(f"MOTOR1:{self.controller.motor1_speed}", read_response=False)
        else:
            self.controller.motor2_speed = max(0, min(255, self.controller.motor2_speed + delta))
            self.controller.send_command(f"MOTOR2:{self.controller.motor2_speed}", read_response=False)
    
    def run(self):
        self.clear_screen()
        print(f"{Fore.GREEN}Connecting to turret...{Style.RESET_ALL}")
        
        if not self.controller.connect():
            print(f"{Fore.RED}Failed to connect!{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}Connected! Initializing...{Style.RESET_ALL}")
        time.sleep(1)
        self.controller.update_status()
        
        print(f"\n{Fore.YELLOW}Starting game mode...{Style.RESET_ALL}")
        time.sleep(1)
        
        needs_redraw = True
        last_redraw = 0
        redraw_interval = 0.2  # Redraw every 200ms max
        
        # Main game loop
        while self.running:
            current_time = time.time()
            
            # Only redraw if needed or periodically
            if needs_redraw or (current_time - last_redraw) > redraw_interval:
                self.clear_screen()
                self.draw_turret()
                self.draw_controls()
                needs_redraw = False
                last_redraw = current_time
            
            # Update status periodically
            if current_time - self.last_update > self.update_interval:
                self.controller.update_status()
                self.last_update = current_time
                needs_redraw = True  # Status changed, need redraw
            
            # Get input (non-blocking)
            key = None
            try:
                key = get_key()
            except Exception as e:
                # Silently ignore errors
                pass
            
            if key:
                needs_redraw = True  # Key pressed, need redraw
                if key == 'ESC':
                    self.running = False
                    break
                elif key in ['W', 'UP']:
                    self.move_top(self.step_size)
                elif key in ['S', 'DOWN']:
                    self.move_top(-self.step_size)
                elif key in ['A', 'LEFT']:
                    self.move_bottom(-self.step_size)
                elif key in ['D', 'RIGHT']:
                    self.move_bottom(self.step_size)
                elif key == 'Q':
                    self.move_top(self.fine_step)
                elif key == 'E':
                    self.move_top(-self.fine_step)
                elif key == 'Z':
                    self.move_bottom(-self.fine_step)
                elif key == 'X':
                    self.move_bottom(self.fine_step)
                elif key == 'SPACE':
                    self.controller.send_command("HOME", read_response=False)
                    self.controller.top_pos = 90
                    self.controller.bottom_pos = 90
                elif key == 'R':
                    # Reset to default limits
                    self.controller.send_command("SET_TOP_MIN:60", read_response=False)
                    self.controller.send_command("SET_TOP_MAX:120", read_response=False)
                    self.controller.send_command("SET_BOTTOM_MIN:0", read_response=False)
                    self.controller.send_command("SET_BOTTOM_MAX:180", read_response=False)
                    self.controller.top_min = 60
                    self.controller.top_max = 120
                    self.controller.bottom_min = 0
                    self.controller.bottom_max = 180
                elif key == 'T':
                    print(f"\n{Fore.YELLOW}Testing top MIN...{Style.RESET_ALL}")
                    self.controller.send_command("TEST_TOP_MIN")
                elif key == 'Y':
                    print(f"\n{Fore.YELLOW}Testing top MAX...{Style.RESET_ALL}")
                    self.controller.send_command("TEST_TOP_MAX")
                elif key == 'G':
                    print(f"\n{Fore.YELLOW}Testing bottom MIN...{Style.RESET_ALL}")
                    self.controller.send_command("TEST_BOTTOM_MIN")
                elif key == 'H':
                    print(f"\n{Fore.YELLOW}Testing bottom MAX...{Style.RESET_ALL}")
                    self.controller.send_command("TEST_BOTTOM_MAX")
                elif key == 'M':
                    self.adjust_motor(1, 10)
                elif key == 'N':
                    self.adjust_motor(1, -10)
                elif key == ',':
                    self.adjust_motor(2, 10)
                elif key == '.':
                    self.adjust_motor(2, -10)
                elif key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    self.set_preset(int(key))
            
            # Small delay to prevent CPU spinning and reduce flashing
            time.sleep(0.1)
        
        self.clear_screen()
        print(f"{Fore.YELLOW}Shutting down turret...{Style.RESET_ALL}")
        try:
            self.controller.send_command("HOME", read_response=False)
            self.controller.send_command("MOTOR1:0", read_response=False)
            self.controller.send_command("MOTOR2:0", read_response=False)
        except:
            pass
        self.controller.disconnect()
        print(f"{Fore.GREEN}Disconnected. Goodbye!{Style.RESET_ALL}")
        
        # Restore terminal on Windows
        if _is_windows:
            pass  # Windows doesn't need restoration
        else:
            # Terminal will be restored by atexit handler
            pass


def list_serial_ports():
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
        description='Turret Control Game - Video Game Style Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('port', nargs='?', help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
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
    game = TurretGame(controller)
    
    try:
        game.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrupted!{Style.RESET_ALL}")
        controller.disconnect()
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        controller.disconnect()


if __name__ == '__main__':
    main()

