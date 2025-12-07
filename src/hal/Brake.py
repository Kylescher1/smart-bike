"""
Brake ESC Control Module

Controls brake ESC using PWM signal via libgpiod API.
Also provides LED control on the same pin.
"""

import time
import threading
from typing import Optional


class BrakeESC:
    """
    Manages brake ESC control using libgpiod software PWM
    
    ESC PWM Specifications:
    - Frequency: 100 Hz
    - Resolution: 16-bit (65536 steps)
    - Pulse widths:
      - 1500 us = Neutral (top)
      - 1650 us = Forward
      - 1350 us = Reverse
    
    Also provides LED control on the same pin.
    """
    
    # ESC PWM constants
    PWM_FREQUENCY = 100.0  # Hz
    PWM_RESOLUTION = 65536  # 16-bit (2^16)
    PWM_PERIOD_US = 10000  # 10ms for 100 Hz
    
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
        self.frequency = 100.0  # Hz
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
            # Convert pulse width (microseconds) to duty cycle
            pulse_width_s = self.pulse_width_us / 1_000_000.0  # Convert to seconds
            duty_cycle = pulse_width_s / self.period  # Duty cycle (0.0 to 1.0)
            
            # Calculate number of steps to be HIGH (with 16-bit resolution)
            high_steps = int(duty_cycle * self.resolution)
            high_steps = max(0, min(self.resolution, high_steps))  # Clamp
            
            # Set HIGH for pulse width
            if high_steps > 0:
                self.line.set_value(1)
                time.sleep(high_steps * self.step_duration)
            
            # Set LOW for the remainder of the period
            low_steps = self.resolution - high_steps
            if low_steps > 0:
                self.line.set_value(0)
                time.sleep(low_steps * self.step_duration)
    
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
        self.set_pulse_width(self.NEUTRAL_US)
        self.enable()
        time.sleep(duration)
        self.armed = True
        print("ESC armed and ready!")
    
    def set_pulse_width(self, pulse_width_us: int):
        """
        Set PWM pulse width in microseconds
        
        Args:
            pulse_width_us: Pulse width in microseconds (1350-1650 typical range)
        """
        # Clamp to reasonable range
        pulse_width_us = max(1000, min(2000, pulse_width_us))
        self.pulse_width_us = pulse_width_us
    
    def set_neutral(self):
        """Set ESC to neutral position"""
        self.set_pulse_width(self.NEUTRAL_US)
    
    def set_forward(self):
        """Set ESC to forward (slight movement)"""
        self.set_pulse_width(self.FORWARD_US)
    
    def set_reverse(self):
        """Set ESC to reverse (slight movement)"""
        self.set_pulse_width(self.REVERSE_US)
    
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
    
    print("Brake ESC Test")
    print("ESC PWM: 100 Hz, 16-bit resolution")
    print("Pulse widths: 1500=neutral, 1650=forward, 1350=reverse")
    print("Press Ctrl+C to stop...")
    
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
    
    try:
        brake_esc = BrakeESC(chip_num=4, line_num=11)
        
        # Arm the ESC (send neutral for 2 seconds)
        brake_esc.arm_esc(duration=2.0)
        
        print("\nTesting reverse direction (1200 us) for 0.5 seconds...")
        brake_esc.set_pulse_width(1200)
        time.sleep(0.5)
        
        print("\nBacking off in forward direction (1650 us) for 1 second...")
        brake_esc.set_forward()
        time.sleep(1.0)
        
        print("\nReturning to neutral and stopping...")
        brake_esc.set_neutral()
        time.sleep(0.5)
        brake_esc.disable()
        
        print("\nTest complete!")
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install python3-libgpiod:")
        print("  sudo apt-get install python3-libgpiod")
    except KeyboardInterrupt:
        print("\n\nStopping...")
        if brake_esc:
            brake_esc.set_neutral()
            time.sleep(0.1)
            brake_esc.cleanup()
    except Exception as e:
        print(f"\nERROR: {e}")
        if brake_esc:
            brake_esc.set_neutral()
            time.sleep(0.1)
            brake_esc.cleanup()
