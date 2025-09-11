"""
PWM and GPIO Interface Module

Manages servo actuation and status LED control.
"""

import time
from typing import Optional, List, Tuple

class PWMGPIOInterface:
    """
    Manages PWM servo control and GPIO status LEDs
    
    Attributes:
        - Servo command shaping
        - Rate limiting
        - Status LED control
        - Failsafe mechanisms
    """
    
    def __init__(self, servo_pin: int = 18, led_pins: Optional[List[int]] = None):
        """
        Initialize PWM and GPIO interfaces
        
        Args:
            servo_pin: GPIO pin for servo control
            led_pins: List of GPIO pins for status LEDs
        """
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            
            # Set GPIO mode
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            
            # Servo setup
            self.servo_pin = servo_pin
            self.GPIO.setup(servo_pin, self.GPIO.OUT)
            self.pwm = self.GPIO.PWM(servo_pin, 50)  # 50 Hz
            self.pwm.start(7.5)  # Neutral position
            
            # LED setup
            self.led_pins = led_pins or [23, 24, 25]  # Default status LED pins
            for pin in self.led_pins:
                self.GPIO.setup(pin, self.GPIO.OUT)
        
        except ImportError:
            print("GPIO library not found. Using mock interface.")
            self.GPIO = None
    
    def set_servo_angle(self, angle: float, min_angle: float = -45, max_angle: float = 45) -> bool:
        """
        Set servo angle with safety constraints
        
        Args:
            angle: Desired servo angle in degrees
            min_angle: Minimum allowed angle
            max_angle: Maximum allowed angle
        
        Returns:
            Success status of servo command
        """
        if not self.GPIO:
            print("Servo control unavailable")
            return False
        
        # Clamp angle to safe range
        clamped_angle = max(min_angle, min(max_angle, angle))
        
        # Convert angle to PWM duty cycle
        # Assumes 0 degrees = 2.5%, 90 degrees = 12.5%
        duty = 2.5 + (clamped_angle + 45) * (10 / 90)
        
        try:
            self.pwm.ChangeDutyCycle(duty)
            return True
        except Exception as e:
            print(f"Servo angle error: {e}")
            return False
    
    def set_status_led(self, led_index: int, state: bool) -> bool:
        """
        Control status LED
        
        Args:
            led_index: Index of LED in led_pins list
            state: LED on (True) or off (False)
        
        Returns:
            Success status of LED control
        """
        if not self.GPIO:
            print("LED control unavailable")
            return False
        
        try:
            if 0 <= led_index < len(self.led_pins):
                self.GPIO.output(self.led_pins[led_index], self.GPIO.HIGH if state else self.GPIO.LOW)
                return True
            else:
                print(f"Invalid LED index: {led_index}")
                return False
        except Exception as e:
            print(f"LED control error: {e}")
            return False
    
    def emergency_stop(self):
        """
        Perform emergency stop
        
        - Set servo to neutral
        - Turn on red status LED
        - Disable PWM
        """
        if not self.GPIO:
            print("Emergency stop unavailable")
            return
        
        # Neutral servo position
        self.set_servo_angle(0)
        
        # Red LED (assuming last LED is red)
        self.set_status_led(-1, True)
        
        # Optional: Disable PWM
        self.pwm.stop()
    
    def __del__(self):
        """
        Clean up GPIO resources
        """
        if self.GPIO:
            self.pwm.stop()
            self.GPIO.cleanup()
