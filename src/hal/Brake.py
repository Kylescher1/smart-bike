"""
Brake ESC Control Module

Controls brake ESC using PWM signal via libgpiod API.
Also provides LED control on the same pin.
"""

import time
import threading
from typing import Optional
import math


class BrakeESC:
    """
    Manages brake ESC control using libgpiod software PWM
    
    ESC PWM Specifications:
    - Frequency: 50 Hz (lower frequency for cleaner software PWM signal)
    - Resolution: 16-bit (65536 steps)
    - Pulse widths:
      - 1500 us = Neutral (top)
      - 1650 us = Forward
      - 1350 us = Reverse
    
    Also provides LED control on the same pin.
    """
    
    # ESC PWM constants
    PWM_FREQUENCY = 50.0  # Hz (lower frequency for cleaner software PWM signal)
    PWM_RESOLUTION = 65536  # 16-bit (2^16)
    PWM_PERIOD_US = 20000  # 20ms for 50 Hz
    
    # ESC pulse width values (in microseconds)
    NEUTRAL_US = 1500  # Neutral position (top)
    FORWARD_US = 1650  # Forward
    REVERSE_US = 1350  # Reverse
    
    def __init__(self, name = "UnamedBrake", **kwargs):
        """
        Initialize brake ESC control
        """

        try:
            import gpiod
            self.gpiod = gpiod
        except ImportError:
            raise ImportError(
                "python3-libgpiod not installed. "
                "Run: sudo apt-get install python3-libgpiod"
            )
        #default parameters
        self.chip_num = 4
        self.line_num = 11
        # PWM parameters
        self.frequency = 50.0  # Hz (lower frequency for cleaner software PWM signal)
        self.resolution = 65536  # 16-bit (2^16)
        # Current pulse width in microseconds
        self.pulse_width_us = 1500  # Neutral position (top)

        for k,v in kwargs.items():#unpack config into self
            setattr(self, k, v)


        # PWM parameters
        self.period = 1.0 / self.frequency  # 0.01 seconds (10ms)
        self.step_duration = self.period / self.resolution

        # Initialize GPIO
        self.chip = None
        self.line = None
        self.running = False
        self.armed = False
        self.pwm_thread = None
        
        # Thread safety
        self._pulse_width_lock = threading.Lock()
        
        # Smoothing/filtering parameters
        self.smoothing_enabled = False  # Disabled by default for cleaner signal
        self.smoothing_rate = 0.1  # Rate of change per PWM cycle (0.0 to 1.0)
        self.min_change_threshold = 1.0  # Minimum change in microseconds to apply
        # Current pulse width in microseconds
        self.pulse_width_us = 1500  # Keep for backwards compatibility
        self._current_pulse_width_us = 1500.0  # Current smoothed pulse width (float for precision)
        self._target_pulse_width_us = 1500  # Target pulse width
        
        # LED control
        self.led_blink_thread = None
        self.led_blink_running = False
        
        self._init_gpio()



    
    def _init_gpio(self):
        """Initialize GPIO chip and line."""
        try:
            self.chip = self.gpiod.Chip(f'gpiochip{self.chip_num}')
            self.line = self.chip.get_line(self.line_num)
            self.line.request(consumer='brake_esc', type=self.gpiod.LINE_REQ_DIR_OUT)
            self.line.set_value(0)  # Start LOW
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GPIO chip {self.chip_num}, line {self.line_num}: {e}"
            )
    
    def _pwm_loop(self):
        """Internal PWM loop that generates servo-style PWM signal."""
        while self.running:
            cycle_start = time.perf_counter()
            
            # Thread-safe read of target pulse width
            with self._pulse_width_lock:
                target = self._target_pulse_width_us
            
            # Apply smoothing if enabled
            if self.smoothing_enabled:
                # Calculate difference
                diff = target - self._current_pulse_width_us
                
                # Only update if change is significant enough
                if abs(diff) > self.min_change_threshold:
                    # Apply exponential smoothing
                    self._current_pulse_width_us += diff * self.smoothing_rate
                else:
                    # Snap to target if very close
                    self._current_pulse_width_us = target
                
                pulse_width_us = int(round(self._current_pulse_width_us))
            else:
                # No smoothing - use exact target value
                pulse_width_us = int(target)
                self._current_pulse_width_us = float(pulse_width_us)
            
            # Clamp to safe range
            pulse_width_us = max(1000, min(2000, pulse_width_us))
            
            # Convert pulse width (microseconds) to seconds
            pulse_width_s = pulse_width_us / 1_000_000.0
            
            # Set HIGH for pulse width duration
            self.line.set_value(1)
            pulse_end_time = cycle_start + pulse_width_s
            
            # Use busy-wait for precise timing
            while time.perf_counter() < pulse_end_time:
                pass
            
            # Set LOW for the remainder of the period
            self.line.set_value(0)
            period_end_time = cycle_start + self.period
            
            # Wait for remainder of period using busy-wait for precision
            while time.perf_counter() < period_end_time:
                pass
    
    def _pulse_width_to_duty_cycle(self, pulse_width_us: int) -> float:
        """
        Convert pulse width in microseconds to duty cycle
        
        Args:
            pulse_width_us: Pulse width in microseconds
            
        Returns:
            Duty cycle (0.0 to 1.0)
        """
        return pulse_width_us / self.PWM_PERIOD_US
    
    def arm_esc(self, duration: float = 2.0):
        """
        Arm the ESC by sending neutral signal for specified duration
        
        Args:
            duration: Duration to hold neutral in seconds (default: 2.0)
        """
        print(f"Arming ESC (sending neutral for {duration} seconds)...")
        # Disable smoothing and set exact neutral value for clean signal
        was_smoothing_enabled = self.smoothing_enabled
        self.smoothing_enabled = False
        self.set_pulse_width(self.NEUTRAL_US, disable_smoothing=True)
        self.enable()
        time.sleep(duration)
        self.armed = True
        # Restore smoothing setting
        self.smoothing_enabled = was_smoothing_enabled
        print("ESC armed and ready!")
    
    def set_pulse_width(self, pulse_width_us: int, disable_smoothing: bool = False):
        """
        Set PWM pulse width in microseconds
        
        Args:
            pulse_width_us: Pulse width in microseconds (1350-1650 typical range)
            disable_smoothing: If True, temporarily disable smoothing for exact value
        """
        # Clamp to reasonable range
        pulse_width_us = max(1000, min(2000, pulse_width_us))
        
        # Thread-safe update
        with self._pulse_width_lock:
            self._target_pulse_width_us = pulse_width_us
            # If smoothing is disabled or requested, update current immediately
            if not self.smoothing_enabled or disable_smoothing:
                self._current_pulse_width_us = float(pulse_width_us)
        
        # Keep old attribute for backwards compatibility
        self.pulse_width_us = pulse_width_us
    
    def set_neutral(self):
        """Set ESC to neutral position"""
        # Disable smoothing for exact neutral value
        self.set_pulse_width(self.NEUTRAL_US, disable_smoothing=True)
    
    def set_forward(self):
        """Set ESC to forward (slight movement)"""
        self.set_pulse_width(self.FORWARD_US)
    
    def set_reverse(self):
        """Set ESC to reverse (slight movement)"""
        self.set_pulse_width(self.REVERSE_US)
    
    def set_smoothing(self, enabled: bool, rate: float = 0.1, min_threshold: float = 1.0):
        """
        Configure smoothing parameters for pulse width changes
        
        Args:
            enabled: Enable/disable smoothing
            rate: Smoothing rate (0.0 to 1.0). Lower = smoother but slower response
            min_threshold: Minimum change in microseconds to apply smoothing
        """
        with self._pulse_width_lock:
            self.smoothing_enabled = enabled
            self.smoothing_rate = max(0.0, min(1.0, rate))
            self.min_change_threshold = max(0.0, min_threshold)
            # If disabling smoothing, snap to target immediately
            if not enabled:
                self._current_pulse_width_us = float(self._target_pulse_width_us)
    
    def enable(self):
        """Enable PWM output"""
        if not self.running:
            self.running = True
            self.pwm_thread = threading.Thread(target=self._pwm_loop, daemon=True)
            self.pwm_thread.start()
    
    def disable(self):
        """Disable PWM output"""
        self.running = False
        if self.pwm_thread is not None:
            self.pwm_thread.join(timeout=1.0)
        if self.line is not None:
            try:
                self.line.set_value(0)  # Ensure LOW when disabled
            except Exception:
                pass
    
    def set_led(self, state: bool):
        """
        Set LED state directly (only works when PWM is disabled)
        
        Args:
            state: True for ON (HIGH), False for OFF (LOW)
        """
        if self.running:
            print("Warning: Cannot set LED while PWM is running. Disable PWM first.")
            return
        
        if self.line is None:
            raise RuntimeError("GPIO not initialized")
        
        try:
            self.line.set_value(1 if state else 0)
        except Exception as e:
            raise RuntimeError(f"Failed to set LED state: {e}")
    
    def blink_led(self, on_time: float = 0.5, off_time: float = 0.5, 
                  duration: Optional[float] = None, count: Optional[int] = None):
        """
        Blink LED (only works when PWM is disabled)
        
        Args:
            on_time: Time LED is ON in seconds (default: 0.5)
            off_time: Time LED is OFF in seconds (default: 0.5)
            duration: Total duration to blink in seconds (None = infinite)
            count: Number of blink cycles (None = infinite until stopped)
        """
        if self.running:
            print("Warning: Cannot blink LED while PWM is running. Disable PWM first.")
            return
        
        if self.led_blink_running:
            self.stop_led_blink()
        
        self.led_blink_running = True
        
        def blink_loop():
            cycles = 0
            start_time = time.time()
            
            while self.led_blink_running and not self.running:
                # Check duration limit
                if duration is not None:
                    if time.time() - start_time >= duration:
                        break
                
                # Check count limit
                if count is not None and cycles >= count:
                    break
                
                # Turn LED ON
                self.set_led(True)
                time.sleep(on_time)
                
                if not self.led_blink_running or self.running:
                    break
                
                # Turn LED OFF
                self.set_led(False)
                time.sleep(off_time)
                
                cycles += 1
            
            # Ensure LED is OFF when done
            self.set_led(False)
            self.led_blink_running = False
        
        self.led_blink_thread = threading.Thread(target=blink_loop, daemon=True)
        self.led_blink_thread.start()
    
    def stop_led_blink(self):
        """Stop blinking LED"""
        self.led_blink_running = False
        if self.led_blink_thread is not None:
            self.led_blink_thread.join(timeout=1.0)
        self.set_led(False)
    
    def cleanup(self):
        """Release GPIO resources"""
        self.disable()
        self.stop_led_blink()
        if self.line is not None:
            try:
                self.line.release()
            except Exception:
                pass
        if self.chip is not None:
            try:
                self.chip.close()
            except Exception:
                pass
    def start(self):
        # Arm the ESC (send neutral for 2 seconds)
        self.arm_esc(duration=2.0)

        time.sleep(1.0)
        self.set_neutral()
        time.sleep(0.5)
        self.disable()
    def dontdie(self):
        # print("\nTesting reverse direction (1200 us) for 0.5 seconds...")
        self.set_pulse_width(1200)
        time.sleep(0.5)

        # print("\nBacking off in forward direction (1650 us) for 1 second...")
        self.set_forward()
        time.sleep(1.0)

        # print("\nReturning to neutral and stopping...")
        self.set_neutral()
        time.sleep(0.5)
        self.disable()
        return
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


if __name__ == "__main__":
    import signal
    import argparse
    import sys
    import shlex
    
    brake_esc = None
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nStopping...")
        if brake_esc:
            brake_esc.set_neutral()
            time.sleep(0.1)
            brake_esc.cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    def create_parser():
        """Create argument parser for commands"""
        parser = argparse.ArgumentParser(
            description='Brake ESC Control - Interactive command line control',
            add_help=False
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Arm command
        arm_parser = subparsers.add_parser('arm', help='Arm the ESC by sending neutral signal')
        arm_parser.add_argument('--duration', type=float, default=2.0, 
                               help='Duration to hold neutral in seconds (default: 2.0)')
        
        # Set command
        set_parser = subparsers.add_parser('set', help='Set pulse width or position')
        set_group = set_parser.add_mutually_exclusive_group(required=True)
        set_group.add_argument('--pulse-width', type=int, metavar='US',
                              help='Set pulse width in microseconds (1000-2000)')
        set_group.add_argument('--neutral', action='store_true',
                              help='Set to neutral position (1500 us)')
        set_group.add_argument('--forward', action='store_true',
                              help='Set to forward position (1650 us)')
        set_group.add_argument('--reverse', action='store_true',
                              help='Set to reverse position (1350 us)')
        
        # Enable command
        enable_parser = subparsers.add_parser('enable', help='Enable PWM output')
        
        # Disable command
        disable_parser = subparsers.add_parser('disable', help='Disable PWM output')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show current status')
        
        # Help command
        help_parser = subparsers.add_parser('help', help='Show help message')
        
        # Quit command
        quit_parser = subparsers.add_parser('quit', help='Exit the program')
        quit_parser = subparsers.add_parser('exit', help='Exit the program')
        
        return parser
    
    def execute_command(brake_esc, cmd_line):
        """Execute a command line"""
        parser = create_parser()
        
        try:
            # Parse the command
            args = parser.parse_args(shlex.split(cmd_line))
            
            if not args.command:
                print("Unknown command. Type 'help' for available commands.")
                return False
            
            if args.command == 'arm':
                print(f"Arming ESC (sending neutral for {args.duration} seconds)...")
                brake_esc.arm_esc(duration=args.duration)
                print("ESC armed and ready!")
                
            elif args.command == 'set':
                if args.pulse_width:
                    print(f"Setting pulse width to {args.pulse_width} microseconds...")
                    brake_esc.set_pulse_width(args.pulse_width)
                    brake_esc.enable()
                    print(f"Pulse width set to {args.pulse_width} us")
                elif args.neutral:
                    print("Setting to neutral position...")
                    brake_esc.set_neutral()
                    brake_esc.enable()
                    print("Set to neutral (1500 us)")
                elif args.forward:
                    print("Setting to forward position...")
                    brake_esc.set_forward()
                    brake_esc.enable()
                    print("Set to forward (1650 us)")
                elif args.reverse:
                    print("Setting to reverse position...")
                    brake_esc.set_reverse()
                    brake_esc.enable()
                    print("Set to reverse (1350 us)")
                    
            elif args.command == 'enable':
                print("Enabling PWM output...")
                brake_esc.enable()
                print("PWM output enabled")
                
            elif args.command == 'disable':
                print("Disabling PWM output...")
                brake_esc.set_neutral()
                time.sleep(0.1)
                brake_esc.disable()
                print("PWM output disabled")
                
            elif args.command == 'status':
                print(f"ESC Status:")
                print(f"  Running: {brake_esc.running}")
                print(f"  Armed: {brake_esc.armed}")
                print(f"  Current pulse width: {brake_esc._target_pulse_width_us} us")
                print(f"  Smoothing enabled: {brake_esc.smoothing_enabled}")
                if brake_esc.running:
                    print(f"  PWM is ACTIVE")
                else:
                    print(f"  PWM is INACTIVE")
            
            elif args.command == 'help':
                print("\nAvailable commands:")
                print("  arm [--duration SECONDS]     - Arm the ESC (default: 2.0 seconds)")
                print("  set --pulse-width US         - Set pulse width in microseconds")
                print("  set --neutral                - Set to neutral position (1500 us)")
                print("  set --forward                - Set to forward position (1650 us)")
                print("  set --reverse                - Set to reverse position (1350 us)")
                print("  enable                       - Enable PWM output")
                print("  disable                      - Disable PWM output")
                print("  status                       - Show current ESC status")
                print("  help                         - Show this help message")
                print("  quit / exit                  - Exit the program")
                print("\nExamples:")
                print("  arm --duration 2.0")
                print("  set --neutral")
                print("  set --pulse-width 1600")
                print("  status")
                print()
            
            elif args.command in ('quit', 'exit'):
                return True  # Signal to exit
            
            return False
            
        except SystemExit:
            # argparse calls sys.exit() on error, catch it
            print("Invalid command. Type 'help' for available commands.")
            return False
        except Exception as e:
            print(f"Error executing command: {e}")
            return False
    
    # Parse initial arguments for GPIO configuration
    init_parser = argparse.ArgumentParser(description='Brake ESC Control - Interactive mode')
    init_parser.add_argument('--chip', type=int, default=4, help='GPIO chip number (default: 4)')
    init_parser.add_argument('--line', type=int, default=11, help='GPIO line number (default: 11)')
    init_args = init_parser.parse_args()
    
    try:
        print("Brake ESC Control - Interactive Mode")
        print("ESC PWM: 100 Hz, 16-bit resolution")
        print("Pulse widths: 1500=neutral, 1650=forward, 1350=reverse")
        print("Type 'help' for available commands, 'quit' or 'exit' to exit")
        print("-" * 60)
        
        brake_esc = BrakeESC(chip_num=init_args.chip, line_num=init_args.line)
        print(f"Brake ESC initialized (chip={init_args.chip}, line={init_args.line})")
        print()
        
        # Interactive command loop
        while True:
            try:
                cmd_line = input("brake> ").strip()
                
                if not cmd_line:
                    continue
                
                # Check for exit
                if execute_command(brake_esc, cmd_line):
                    break
                    
            except EOFError:
                # Handle Ctrl+D
                print("\n")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C (will be caught by signal handler, but just in case)
                print("\n")
                break
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install python3-libgpiod:")
        print("  sudo apt-get install python3-libgpiod")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if brake_esc:
            brake_esc.set_neutral()
            time.sleep(0.1)
            brake_esc.cleanup()
        sys.exit(1)
    finally:
        print("\nCleaning up...")
        if brake_esc:
            brake_esc.set_neutral()
            time.sleep(0.1)
            brake_esc.cleanup()
        print("Goodbye!")
