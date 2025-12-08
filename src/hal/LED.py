"""
Brake LED Control Module

Controls brake LED using libgpiod API.
"""

import time
import threading
from typing import Optional


class GPIO_LED:
    """
    Manages brake LED control using libgpiod
    
    Equivalent to:
        gpioset gpiochip4 11=1  # LED ON
        gpioset gpiochip4 11=0  # LED OFF
    """
    
    def __init__(self, chip_num: int = 4, line_num: int = 11):
        """
        Initialize brake LED GPIO control
        
        Args:sudo python3 src/hal/LED.py
            chip_num: GPIO chip number (default: 4)
            line_num: GPIO line number (default: 10)
        """
        try:
            import gpiod
            self.gpiod = gpiod
        except ImportError:
            raise ImportError(
                "python3-libgpiod not installed. "
                "Run: sudo apt-get install python3-libgpiod"
            )
        
        self.chip_num = chip_num
        self.line_num = line_num
        self.chip = None
        self.line = None
        self.blink_thread = None
        self.blink_running = False
        
        self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO chip and line."""
        try:
            self.chip = self.gpiod.Chip(f'gpiochip{self.chip_num}')
            self.line = self.chip.get_line(self.line_num)
            # Request line as output (may need sudo permissions)
            self.line.request(consumer='brake_led', type=self.gpiod.LINE_REQ_DIR_OUT)
            self.line.set_value(0)  # Start with LED OFF
            print(f"GPIO initialized: gpiochip{self.chip_num} line {self.line_num}")
        except PermissionError as e:
            raise RuntimeError(
                f"Permission denied accessing GPIO. Try running with sudo.\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GPIO chip {self.chip_num}, line {self.line_num}: {e}\n"
                f"Make sure the GPIO line is not already in use by another process."
            )
    
    def set_led(self, state: bool):
        """
        Set LED state directly
        
        Args:
            state: True for ON (HIGH), False for OFF (LOW)
        """
        if self.line is None:
            raise RuntimeError("GPIO not initialized")
        
        try:
            self.line.set_value(1 if state else 0)
        except Exception as e:
            raise RuntimeError(f"Failed to set LED state: {e}")
    
    def blink(self, on_time: float = 0.5, off_time: float = 0.5, 
              duration: Optional[float] = None, count: Optional[int] = None):
        """
        Blink LED with specified timing
        
        Args:
            on_time: Time LED is ON in seconds (default: 0.5)
            off_time: Time LED is OFF in seconds (default: 0.5)
            duration: Total duration to blink in seconds (None = infinite)
            count: Number of blink cycles (None = infinite until stopped)
        """
        if self.blink_running:
            self.stop_blink()
        
        self.blink_running = True
        
        def blink_loop():
            cycles = 0
            start_time = time.time()
            
            while self.blink_running:
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
                
                if not self.blink_running:
                    break
                
                # Turn LED OFF
                self.set_led(False)
                time.sleep(off_time)
                
                cycles += 1
            
            # Ensure LED is OFF when done
            self.set_led(False)
            self.blink_running = False
        
        self.blink_thread = threading.Thread(target=blink_loop, daemon=True)
        self.blink_thread.start()
    
    def stop_blink(self):
        """Stop blinking LED"""
        self.blink_running = False
        if self.blink_thread is not None:
            self.blink_thread.join(timeout=1.0)
        self.set_led(False)
    
    def cleanup(self):
        """Release GPIO resources"""
        self.stop_blink()
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
        #they go or they don't
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
    # Example usage
    print("Brake LED Blink Test")
    print("Equivalent to: gpioset gpiochip4 11=1 and gpioset gpiochip4 11=0")
    
    try:
        print(f"Initializing LED on gpiochip4 line 10...")
        brake_led = GPIO_LED(chip_num=1, line_num=9)
        
        print("Testing LED control...")
        brake_led.set_led(True)
        print("LED should be ON")
        time.sleep(1.0)
        brake_led.set_led(False)
        print("LED should be OFF")
        time.sleep(1.0)
        
        print("\n1. Blinking LED 5 times (0.5s on, 0.5s off)...")
        brake_led.blink(on_time=0.5, off_time=0.5, count=5)
        time.sleep(3.0)  # Wait for blinking to complete
        
        print("\n2. Blinking LED for 3 seconds (0.2s on, 0.2s off)...")
        brake_led.blink(on_time=0.2, off_time=0.2, duration=3.0)
        time.sleep(3.5)
        
        print("\n3. Continuous blink (will stop after 2 seconds)...")
        brake_led.blink(on_time=0.1, off_time=0.1)
        time.sleep(2.0)
        brake_led.stop_blink()
        
        print("\n4. Manual control...")
        brake_led.set_led(True)
        time.sleep(0.5)
        brake_led.set_led(False)
        time.sleep(0.5)
        brake_led.set_led(True)
        time.sleep(0.5)
        brake_led.set_led(False)
        
        print("\nTest complete!")
        brake_led.cleanup()
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install python3-libgpiod:")
        print("  sudo apt-get install python3-libgpiod")
    except Exception as e:
        print(f"\nERROR: {e}")

