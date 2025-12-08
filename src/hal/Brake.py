"""
Brake ESC Control Module

Controls brake ESC using PWM signal via libgpiod API.
Also provides LED control on the same pin.
Now includes ESP32 encoder stall detection via serial port.
"""

import time
import threading
from typing import Optional, Dict
import math
import serial
from collections import deque


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
        self._cleaned_up = False  # Track if cleanup has been called
        
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
        
        # ESP32 encoder stall detection
        self.encoder_port = getattr(self, 'encoder_port', None)  # e.g., '/dev/ttyUSB1' or 'ttl1'
        self.encoder_baudrate = getattr(self, 'encoder_baudrate', 115200)
        self.encoder_timeout = getattr(self, 'encoder_timeout', 0.1)
        self.encoder_ser: Optional[serial.Serial] = None
        self.encoder_connected = False
        self.encoder_thread: Optional[threading.Thread] = None
        self.encoder_stop_event = threading.Event()
        
        # Encoder data storage
        self.encoder_data_lock = threading.Lock()
        self.latest_encoder_data: Optional[Dict] = None
        self.is_stalled = False
        self.stall_check_enabled = False
        
        # Brake calibration
        self.brake_actuated_position: Optional[int] = None  # Calibrated brake actuated position
        
        self._init_gpio()
        
        # Initialize encoder connection if port is specified
        if self.encoder_port:
            self._init_encoder()



    
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
    
    def _init_encoder(self):
        """Initialize ESP32 encoder serial connection."""
        if not self.encoder_port:
            return
        
        # Convert 'ttl1' to '/dev/ttyUSB1' if needed
        port = self.encoder_port
        if port.startswith('ttl'):
            port_num = port.replace('ttl', '')
            port = f'/dev/ttyUSB{port_num}'
        
        # Check if port exists before trying to open
        import os
        if not os.path.exists(port):
            print(f"BrakeESC: Warning - Encoder port {port} does not exist")
            print(f"BrakeESC: Available ports: {[p for p in os.listdir('/dev') if p.startswith('ttyUSB') or p.startswith('ttyACM')]}")
            self.encoder_connected = False
            return
        
        try:
            print(f"BrakeESC: Connecting to encoder on {port}...")
            self.encoder_ser = serial.Serial(
                port=port,
                baudrate=self.encoder_baudrate,
                timeout=self.encoder_timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)  # Allow ESP32 to initialize
            self.encoder_ser.reset_input_buffer()
            self.encoder_connected = True
            self.encoder_stop_event.clear()
            
            # Start encoder reading thread
            self.encoder_thread = threading.Thread(target=self._encoder_reader, daemon=True)
            self.encoder_thread.start()
            print(f"BrakeESC: Successfully connected to encoder on {port}")
        except serial.SerialException as e:
            print(f"BrakeESC: Warning - Serial error connecting to encoder on {port}: {e}")
            print(f"BrakeESC: Continuing without encoder (stall detection disabled)")
            self.encoder_connected = False
            self.encoder_ser = None
        except Exception as e:
            print(f"BrakeESC: Warning - Failed to connect to encoder on {port}: {e}")
            print(f"BrakeESC: Continuing without encoder (stall detection disabled)")
            self.encoder_connected = False
            self.encoder_ser = None
    
    def _parse_encoder_line(self, line: str) -> Optional[Dict]:
        """
        Parse a POS line from ESP32 encoder.
        
        Format: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>
        
        Returns:
            dict with keys: position, velocity, is_moving, is_stalled, pinA, pinB
            Returns None if line cannot be parsed
        """
        line = line.strip()
        if not line.startswith("POS,"):
            return None
        
        try:
            parts = line.split(",")
            if len(parts) < 5:
                return None
            
            result = {
                "position": int(parts[1]),
                "velocity": float(parts[2]),
                "is_moving": bool(int(parts[3])),
                "is_stalled": bool(int(parts[4])),
                "timestamp": time.time()
            }
            
            # Add pin states if present
            if len(parts) >= 7:
                result["pinA"] = int(parts[5])
                result["pinB"] = int(parts[6])
            
            return result
        except (ValueError, IndexError) as e:
            return None
    
    def _encoder_reader(self):
        """Background thread to continuously read encoder data."""
        while not self.encoder_stop_event.is_set():
            if not self.encoder_connected or not self.encoder_ser or not self.encoder_ser.is_open:
                time.sleep(0.1)
                continue
            
            try:
                if self.encoder_ser.in_waiting > 0:
                    line = self.encoder_ser.readline().decode('utf-8', errors='ignore')
                    encoder_data = self._parse_encoder_line(line)
                    
                    if encoder_data:
                        with self.encoder_data_lock:
                            self.latest_encoder_data = encoder_data
                            self.is_stalled = encoder_data.get('is_stalled', False)
                else:
                    time.sleep(0.01)  # Small delay when no data
            except Exception as e:
                print(f"BrakeESC: Encoder reading error: {e}")
                time.sleep(0.1)
    
    def get_encoder_data(self) -> Optional[Dict]:
        """
        Get the latest encoder data.
        
        Returns:
            dict with encoder data or None if no data available
        """
        with self.encoder_data_lock:
            return self.latest_encoder_data.copy() if self.latest_encoder_data else None
    
    def check_stall(self) -> bool:
        """
        Check if motor is currently stalled.
        
        Returns:
            True if stalled, False otherwise
        """
        if not self.stall_check_enabled:
            return False
        
        with self.encoder_data_lock:
            return self.is_stalled
    
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
    
    def set_pulse_width(self, pulse_width_us: int, disable_smoothing: bool = False, check_stall: bool = True):
        """
        Set PWM pulse width in microseconds
        
        Args:
            pulse_width_us: Pulse width in microseconds (1350-1650 typical range)
            disable_smoothing: If True, temporarily disable smoothing for exact value
            check_stall: If True, check for stall before setting (default: True)
        """
        # Check for stall if enabled
        if check_stall and self.stall_check_enabled and self.check_stall():
            print("BrakeESC: Stall detected - command ignored")
            return False
        
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
        return True
    
    def set_neutral(self, check_stall: bool = True):
        """Set ESC to neutral position"""
        # Disable smoothing for exact neutral value
        self.set_pulse_width(self.NEUTRAL_US, disable_smoothing=True, check_stall=check_stall)
    
    def set_forward(self, check_stall: bool = True):
        """Set ESC to forward (slight movement)"""
        self.set_pulse_width(self.FORWARD_US, check_stall=check_stall)
    
    def set_reverse(self, check_stall: bool = True):
        """Set ESC to reverse (slight movement)"""
        self.set_pulse_width(self.REVERSE_US, check_stall=check_stall)
    
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
        if self._cleaned_up:
            return  # Silently ignore if already cleaned up
        
        if self.running:
            print("Warning: Cannot set LED while PWM is running. Disable PWM first.")
            return
        
        if self.line is None:
            return  # Silently ignore if not initialized
        
        try:
            self.line.set_value(1 if state else 0)
        except (ValueError, RuntimeError, OSError) as e:
            # GPIO line is closed or invalid - silently ignore during cleanup
            if not self._cleaned_up:
                raise RuntimeError(f"Failed to set LED state: {e}")
            # Otherwise silently ignore
    
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
        try:
            self.set_led(False)
        except Exception:
            # Ignore errors during cleanup
            pass
    
    def cleanup(self):
        """Release GPIO resources and close encoder connection"""
        if self._cleaned_up:
            return  # Prevent double cleanup
        
        self._cleaned_up = True
        
        # Stop PWM first
        self.disable()
        
        # Stop LED blinking (before releasing GPIO)
        self.stop_led_blink()
        
        
        # Close encoder connection
        if self.encoder_connected:
            self.encoder_stop_event.set()
            if self.encoder_thread and self.encoder_thread.is_alive():
                self.encoder_thread.join(timeout=1.0)
            if self.encoder_ser and self.encoder_ser.is_open:
                try:
                    self.encoder_ser.close()
                except Exception:
                    pass
            self.encoder_connected = False
        
        # Release GPIO line (do this last, after stopping LED operations)
        if self.line is not None:
            try:
                self.line.release()
            except Exception:
                pass
            self.line = None
        
        if self.chip is not None:
            try:
                self.chip.close()
            except Exception:
                pass
            self.chip = None
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
    
    def brake_arm(self):
        """
        Initialize/arm brake with a simple routine.
        
        Process:
        1. Run in reverse for 3 seconds
        2. Set to neutral
        3. Done
        
        Returns:
            True if routine completed successfully, False otherwise
        """
        print("BrakeESC: Starting brake arm routine...")
        
        # Ensure PWM is enabled
        if not self.running:
            self.enable()
        
        # Run in reverse for 3 seconds
        print("BrakeESC: Running in reverse for 3 seconds")
        self.set_reverse(check_stall=False)
        
        # Monitor and print position during reverse
        start_time = time.time()
        last_print_time = start_time
        while time.time() - start_time < 3.0:
            elapsed = time.time() - start_time
            current_data = self.get_encoder_data()
            
            # Print updates every 0.5 seconds
            if elapsed - last_print_time >= 0.5:
                if current_data:
                    print(f"  Time: {elapsed:.1f}s | Position: {current_data['position']} | "
                          f"Velocity: {current_data.get('velocity', 0):+.2f}")
                else:
                    print(f"  Time: {elapsed:.1f}s | Waiting for encoder data...")
                last_print_time = elapsed
            
            time.sleep(0.05)
        
        # Set to neutral
        print("BrakeESC: Setting to neutral")
        self.set_neutral(check_stall=False)
        
        # Print final status
        final_data = self.get_encoder_data()
        if final_data:
            print(f"BrakeESC: Arm routine complete! Final position: {final_data['position']}")
        else:
            print("BrakeESC: Arm routine complete!")
        
        return True
    
    def brake(self, forward_time: float = 1.0, stop_time: float = 0.1, reverse_time: float = 0.25):
        """
        Execute brake routine: run in reverse for 3 seconds, then set to neutral.
        
        Args:
            forward_time: Not used (kept for compatibility)
            stop_time: Not used (kept for compatibility)
            reverse_time: Not used (kept for compatibility)
        
        Returns:
            True if routine completed successfully
        """
        print("BrakeESC: Starting brake routine...")
        
        # Ensure PWM is enabled
        if not self.running:
            self.enable()
        
        # Run in reverse for 3 seconds
        print("BrakeESC: Running in reverse for 3 seconds")
        self.set_reverse(check_stall=False)
        
        # Monitor and print position during reverse
        start_time = time.time()
        last_print_time = start_time
        while time.time() - start_time < 3.0:
            elapsed = time.time() - start_time
            current_data = self.get_encoder_data()
            
            # Print updates every 0.5 seconds
            if elapsed - last_print_time >= 0.5:
                if current_data:
                    print(f"  Time: {elapsed:.1f}s | Position: {current_data['position']} | "
                          f"Velocity: {current_data.get('velocity', 0):+.2f}")
                else:
                    print(f"  Time: {elapsed:.1f}s | Waiting for encoder data...")
                last_print_time = elapsed
            
            time.sleep(0.05)
        
        # Set to neutral
        print("BrakeESC: Setting to neutral")
        self.set_neutral(check_stall=False)

        # INSERT_YOUR_CODE
        # Command to disable PWM
        print("BrakeESC: Disabling PWM")
        self.disable()
        # Command to enable PWM
        print("BrakeESC: Enabling PWM")
        self.enable()

        self.set_forward(check_stall=False)
        time.sleep(0.075)

        self.disable()
        self.enable()
   
        
        # Print final status
        final_data = self.get_encoder_data()
        if final_data:
            print(f"BrakeESC: Brake routine complete! Final position: {final_data['position']}")
        else:
            print("BrakeESC: Brake routine complete!")
        
        return True
    
    def _brake_actuated(self):
        """
        Execute brake routine using calibrated actuated position.
        Moves motor in reverse until reaching brake_actuated_position or stall,
        then backs off forward.
        """
        print(f"BrakeESC: Activating brake (reverse) until position {self.brake_actuated_position} or stall...")
        
        # Enable stall checking
        self.stall_check_enabled = True
        
        # Ensure PWM is enabled
        if not self.running:
            self.enable()
        
        # Get initial position
        initial_data = self.get_encoder_data()
        if not initial_data:
            print("BrakeESC: ERROR - No encoder data available")
            return False
        
        initial_position = initial_data['position']
        print(f"BrakeESC: Starting position: {initial_position}, Target: {self.brake_actuated_position}")
        
        # Start moving in reverse at 1200 us
        self.set_pulse_width(1200, check_stall=False)
        
        check_interval = 0.05  # Check every 50ms
        timeout = 5.0  # Maximum 5 seconds
        start_time = time.time()
        last_print_time = start_time
        
        while time.time() - start_time < timeout:
            # Check for stall
            if self.check_stall():
                print("BrakeESC: STALL DETECTED - brake engaged")
                break
            
            # Get current position
            current_data = self.get_encoder_data()
            if not current_data:
                time.sleep(check_interval)
                continue
            
            current_position = current_data['position']
            
            # Check if we've reached the actuated position
            # Since we're moving in reverse, position should decrease
            if current_position <= self.brake_actuated_position:
                print(f"BrakeESC: Reached actuated position! Current: {current_position}, Target: {self.brake_actuated_position}")
                break
            
            # Print updates every 0.2 seconds
            current_time = time.time()
            if current_time - last_print_time >= 0.2:
                print(f"  Position: {current_position}, Target: {self.brake_actuated_position}, "
                      f"Remaining: {current_position - self.brake_actuated_position}, "
                      f"Velocity: {current_data.get('velocity', 0):+.2f}")
                last_print_time = current_time
            
            time.sleep(check_interval)
        
        # Stop reverse movement
        print("BrakeESC: Stopping reverse movement")
        self.set_neutral(check_stall=False)
        time.sleep(0.1)
        
        # Back off forward at 1650 us for 0.25 seconds
        print("BrakeESC: Backing off forward at 1650 us for 0.25 seconds")
        self.set_pulse_width(1650, check_stall=False)
        time.sleep(0.25)
        
        # Return to neutral
        print("BrakeESC: Returning to neutral")
        self.set_neutral(check_stall=False)
        self.stall_check_enabled = False
        
        final_data = self.get_encoder_data()
        final_position = final_data['position'] if final_data else None
        if final_position is not None:
            print(f"BrakeESC: Final position: {final_position}, Target: {self.brake_actuated_position}")
        
        return True
    
    def _brake_fallback(self, forward_time: float, stop_time: float, reverse_time: float):
        """
        Fallback brake routine: forward -> stop -> reverse.
        """
        print(f"BrakeESC: Starting brake routine (forward={forward_time}s, stop={stop_time}s, reverse={reverse_time}s)")
        
        # Enable stall checking
        self.stall_check_enabled = True
        
        # Ensure PWM is enabled
        if not self.running:
            self.enable()
        
        # Get initial position
        initial_data = self.get_encoder_data()
        initial_position = initial_data['position'] if initial_data else None
        print(f"BrakeESC: Initial encoder position: {initial_position}")
        
        # Phase 1: Forward (brake engagement)
        print(f"BrakeESC: Phase 1 - Forward for {forward_time}s")
        if self.check_stall():
            print("BrakeESC: STALL DETECTED before forward phase - aborting routine")
            self.set_neutral(check_stall=False)
            self.stall_check_enabled = False
            return False
        
        self.set_forward(check_stall=False)
        
        start_time = time.time()
        last_print_time = start_time
        while time.time() - start_time < forward_time:
            if self.check_stall():
                print("BrakeESC: STALL DETECTED during forward phase - stopping routine")
                self.set_neutral(check_stall=False)
                self.stall_check_enabled = False
                return False
            
            current_time = time.time()
            if current_time - last_print_time >= 0.2:
                data = self.get_encoder_data()
                if data:
                    print(f"  Position: {data['position']}, Velocity: {data['velocity']:.2f}, Moving: {data['is_moving']}")
                last_print_time = current_time
            
            time.sleep(0.05)
        
        # Phase 2: Stop/Neutral
        print(f"BrakeESC: Phase 2 - Stop/Neutral for {stop_time}s")
        self.set_neutral(check_stall=False)
        time.sleep(stop_time)
        
        # Phase 3: Reverse (backoff)
        print(f"BrakeESC: Phase 3 - Reverse (backoff) for {reverse_time}s")
        if self.check_stall():
            print("BrakeESC: STALL DETECTED before reverse phase - aborting routine")
            self.set_neutral(check_stall=False)
            self.stall_check_enabled = False
            return False
        
        self.set_reverse(check_stall=False)
        
        start_time = time.time()
        last_print_time = start_time
        while time.time() - start_time < reverse_time:
            if self.check_stall():
                print("BrakeESC: STALL DETECTED during reverse phase - stopping routine")
                self.set_neutral(check_stall=False)
                self.stall_check_enabled = False
                return False
            
            current_time = time.time()
            if current_time - last_print_time >= 0.2:
                data = self.get_encoder_data()
                if data:
                    print(f"  Position: {data['position']}, Velocity: {data['velocity']:.2f}, Moving: {data['is_moving']}")
                last_print_time = current_time
            
            time.sleep(0.05)
        
        # Return to neutral
        print("BrakeESC: Brake routine complete - returning to neutral")
        self.set_neutral(check_stall=False)
        self.stall_check_enabled = False
        
        final_data = self.get_encoder_data()
        final_position = final_data['position'] if final_data else None
        if initial_position is not None and final_position is not None:
            position_delta = final_position - initial_position
            print(f"BrakeESC: Final encoder position: {final_position} (delta: {position_delta:+d})")
        
        return True
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors during destruction
            pass


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
            try:
                brake_esc.set_neutral()
                time.sleep(0.1)
            except Exception:
                pass  # Ignore errors during shutdown
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
        
        # Brake command
        brake_parser = subparsers.add_parser('brake', help='Execute brake routine or initialize brake')
        brake_parser.add_argument('--arm', action='store_true',
                                 help='Initialize/calibrate brake: find actuated position')
        brake_parser.add_argument('--forward-time', type=float, default=1.0,
                                 help='Forward duration in seconds (default: 1.0)')
        brake_parser.add_argument('--stop-time', type=float, default=0.1,
                                 help='Stop/neutral duration in seconds (default: 0.1)')
        brake_parser.add_argument('--reverse-time', type=float, default=0.25,
                                 help='Reverse duration in seconds (default: 0.25)')
        
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
                
                # Show encoder status if connected
                if brake_esc.encoder_connected:
                    encoder_data = brake_esc.get_encoder_data()
                    if encoder_data:
                        print(f"\nEncoder Status:")
                        print(f"  Connected: {brake_esc.encoder_connected}")
                        print(f"  Position: {encoder_data['position']}")
                        print(f"  Velocity: {encoder_data['velocity']:.2f} pulses/s")
                        print(f"  Moving: {encoder_data['is_moving']}")
                        print(f"  Stalled: {encoder_data['is_stalled']}")
                    else:
                        print(f"\nEncoder Status:")
                        print(f"  Connected: {brake_esc.encoder_connected}")
                        print(f"  No data received yet")
                else:
                    print(f"\nEncoder Status: Not connected")
            
            elif args.command == 'brake':
                if args.arm:
                    print("Initializing brake calibration...")
                    success = brake_esc.brake_arm()
                    if success:
                        print(f"Brake calibrated! Actuated position: {brake_esc.brake_actuated_position}")
                    else:
                        print("Brake calibration failed")
                else:
                    print("Executing brake routine...")
                    success = brake_esc.brake(
                        forward_time=args.forward_time,
                        stop_time=args.stop_time,
                        reverse_time=args.reverse_time
                    )
                    if success:
                        print("Brake routine completed successfully")
                    else:
                        print("Brake routine stopped due to stall detection")
            
            elif args.command == 'help':
                print("\nAvailable commands:")
                print("  arm [--duration SECONDS]     - Arm the ESC (default: 2.0 seconds)")
                print("  set --pulse-width US         - Set pulse width in microseconds")
                print("  set --neutral                - Set to neutral position (1500 us)")
                print("  set --forward                - Set to forward position (1650 us)")
                print("  set --reverse                - Set to reverse position (1350 us)")
                print("  enable                       - Enable PWM output")
                print("  disable                      - Disable PWM output")
                print("  brake --arm                  - Initialize/calibrate brake (find actuated position)")
                print("  brake [--forward-time SEC]   - Execute brake routine")
                print("         [--stop-time SEC]      -   (uses calibrated position if available)")
                print("         [--reverse-time SEC]   -   with stall detection")
                print("  status                       - Show current ESC and encoder status")
                print("  help                         - Show this help message")
                print("  quit / exit                  - Exit the program")
                print("\nExamples:")
                print("  arm --duration 2.0")
                print("  set --neutral")
                print("  set --pulse-width 1600")
                print("  brake                         # Run default brake routine")
                print("  brake --forward-time 2.0      # Custom brake routine")
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
    init_parser.add_argument('--encoder-port', type=str, default=None, 
                           help='ESP32 encoder serial port (e.g., ttl1 or /dev/ttyUSB1)')
    init_parser.add_argument('--encoder-baudrate', type=int, default=115200,
                           help='ESP32 encoder baudrate (default: 115200)')
    init_args = init_parser.parse_args()
    
    try:
        print("Brake ESC Control - Interactive Mode")
        print("ESC PWM: 100 Hz, 16-bit resolution")
        print("Pulse widths: 1500=neutral, 1650=forward, 1350=reverse")
        print("Type 'help' for available commands, 'quit' or 'exit' to exit")
        print("-" * 60)
        
        # Build kwargs for BrakeESC
        brake_kwargs = {
            'chip_num': init_args.chip,
            'line_num': init_args.line
        }
        if init_args.encoder_port:
            brake_kwargs['encoder_port'] = init_args.encoder_port
            brake_kwargs['encoder_baudrate'] = init_args.encoder_baudrate
            print(f"Brake ESC will connect to encoder on {init_args.encoder_port}")
        
        try:
            brake_esc = BrakeESC(**brake_kwargs)
            print(f"Brake ESC initialized (chip={init_args.chip}, line={init_args.line})")
            if init_args.encoder_port:
                print(f"Encoder port: {init_args.encoder_port} (baudrate: {init_args.encoder_baudrate})")
                if brake_esc.encoder_connected:
                    print("Encoder status: CONNECTED")
                else:
                    print("Encoder status: NOT CONNECTED (stall detection disabled)")
            print()
            sys.stdout.flush()  # Ensure all output is flushed before prompt
        except Exception as e:
            print(f"\nERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Interactive command loop
        while True:
            try:
                # Check if stdin is still open
                if sys.stdin.closed:
                    print("\nERROR: stdin is closed. Exiting.")
                    break
                
                sys.stdout.flush()  # Flush before prompt
                sys.stderr.flush()  # Also flush stderr
                
                # Use readline if available for better input handling
                try:
                    cmd_line = input("brake> ").strip()
                except (EOFError, ValueError) as e:
                    if isinstance(e, EOFError):
                        print("\n")
                        break
                    else:
                        print(f"\nERROR reading input: {e}")
                        break
                
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
            except ValueError as e:
                if "I/O operation on closed file" in str(e):
                    print("\nERROR: stdin was closed.")
                    break
                raise
        
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
