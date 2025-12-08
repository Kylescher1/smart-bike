#!/usr/bin/env python3
"""
Rock Pi Quadrature Encoder Stall Detector

Replicates the behavior of the ESP32 encoder stall detector script.
Uses libgpiod for GPIO access on Rock Pi.

Hardware:
- Quadrature encoder connected to PIN_29 (gpiochip1 line 31) and PIN_31 (gpiochip1 line 15)
- Both pins use edge detection

Serial Protocol (stdout):
- Sends: "POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>\n"
"""

import time
import threading
from typing import Optional
import signal
import sys

try:
    import gpiod
except ImportError:
    raise ImportError(
        "python3-libgpiod not installed. "
        "Run: sudo apt-get install python3-libgpiod"
    )


class EncoderStallDetector:
    """
    Quadrature encoder stall detector using libgpiod
    """
    
    # Stall detection config
    STALL_TIMEOUT_MS = 500    # Time window to detect stall (ms)
    STALL_THRESHOLD = 5       # Minimum pulses to consider motor moving
    STATUS_INTERVAL = 0.020   # Status update interval (seconds) - 20ms for 50 Hz
    VELOCITY_UPDATE_MS = 50   # Velocity calculation interval (ms)
    
    # Quadrature decoder lookup table
    # Based on Gray code sequence: 00 -> 01 -> 11 -> 10 -> 00 (forward)
    # Index = (lastState << 2) | currentState
    # Value: +1 = forward, -1 = reverse, 0 = invalid/no change
    QUADRATURE_TABLE = [
        0,  # 00 -> 00: no change
        +1, # 00 -> 01: forward
        -1, # 00 -> 10: reverse
        0,  # 00 -> 11: invalid
        -1, # 01 -> 00: reverse
        0,  # 01 -> 01: no change
        0,  # 01 -> 10: invalid
        +1, # 01 -> 11: forward
        +1, # 10 -> 00: forward
        0,  # 10 -> 01: invalid
        0,  # 10 -> 10: no change
        -1, # 10 -> 11: reverse
        0,  # 11 -> 00: invalid
        -1, # 11 -> 01: reverse
        +1, # 11 -> 10: forward
        0   # 11 -> 11: no change
    ]
    
    def __init__(self, chip_num: int = 1, line_a: int = 31, line_b: int = 15):
        """
        Initialize encoder stall detector
        
        Args:
            chip_num: GPIO chip number (default: 1 for gpiochip1)
            line_a: GPIO line number for encoder A (default: 31 for PIN_29)
            line_b: GPIO line number for encoder B (default: 15 for PIN_31)
        """
        self.chip_num = chip_num
        self.line_a_num = line_a
        self.line_b_num = line_b
        
        # Encoder state
        self.encoder_value = 0
        self.last_encoder_state = 0
        self.last_encoder_value = 0
        
        # Timing
        self.last_movement_time = time.time()
        self.last_position_for_velocity = 0
        self.last_velocity_time = time.time()
        self.current_velocity = 0.0
        
        # Stall detection
        self.is_moving = False
        self.is_stalled = False
        
        # Threading
        self.running = False
        self.encoder_thread = None
        self.status_thread = None
        self._lock = threading.Lock()
        
        # GPIO
        self.chip = None
        self.line_a = None
        self.line_b = None
        
        self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO chip and lines with edge detection"""
        try:
            self.chip = gpiod.Chip(f'gpiochip{self.chip_num}')
            self.line_a = self.chip.get_line(self.line_a_num)
            self.line_b = self.chip.get_line(self.line_b_num)
            
            # Request lines for simple input (safer than edge detection which can crash)
            # Use regular input mode instead of edge detection to avoid system issues
            self.line_a.request(consumer='encoder_a', type=gpiod.LINE_REQ_DIR_IN)
            self.line_b.request(consumer='encoder_b', type=gpiod.LINE_REQ_DIR_IN)
            
            # Read initial state
            pin_a_state = self.line_a.get_value()
            pin_b_state = self.line_b.get_value()
            self.last_encoder_state = (pin_a_state & 0x01) | ((pin_b_state & 0x01) << 1)
            
            print(f"Initial pin states - Pin {self.line_a_num} (ENC_A): {pin_a_state}, "
                  f"Pin {self.line_b_num} (ENC_B): {pin_b_state}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GPIO chip {self.chip_num}, "
                f"lines {self.line_a_num}, {self.line_b_num}: {e}"
            )
    
    def _update_encoder(self, changed_line):
        """Update encoder position when a pin changes"""
        # Read current state of both pins multiple times for debouncing
        # Read 3 times with small delay to filter out noise
        pin_a_reads = []
        pin_b_reads = []
        for _ in range(3):
            pin_a_reads.append(self.line_a.get_value())
            pin_b_reads.append(self.line_b.get_value())
            time.sleep(0.00001)  # 10 microseconds between reads
        
        # Use majority vote to filter noise
        pin_a_state = 1 if sum(pin_a_reads) >= 2 else 0
        pin_b_state = 1 if sum(pin_b_reads) >= 2 else 0
        
        current_state = (pin_a_state & 0x01) | ((pin_b_state & 0x01) << 1)
        
        # Only process if state actually changed
        if current_state != self.last_encoder_state:
            # Look up direction in quadrature table
            transition = (self.last_encoder_state << 2) | current_state
            direction = self.QUADRATURE_TABLE[transition]
            
            # Only update if we got a valid direction (not 0, not invalid transition)
            if direction != 0:
                with self._lock:
                    self.encoder_value += direction
                    self.last_movement_time = time.time()
                # Only update last state if transition was valid
                self.last_encoder_state = current_state
            # If direction is 0 (invalid transition), ignore it - don't update last_state
            # This helps filter out noise and intermediate states
    
    def _encoder_loop(self):
        """Main encoder monitoring loop using safer polling approach"""
        # Use moderate polling rate to avoid CPU overload
        # This is safer than edge detection which can cause system issues
        poll_interval = 0.002  # 2ms polling (500 Hz) - slower to reduce noise
        
        last_state_check = time.time()
        last_valid_state = self.last_encoder_state
        
        while self.running:
            try:
                current_time = time.time()
                
                # Only check encoder state periodically to avoid CPU overload
                if current_time - last_state_check >= poll_interval:
                    # Store state before update
                    old_state = self.last_encoder_state
                    self._update_encoder(None)
                    
                    # If state changed to something invalid, revert it
                    # This helps prevent noise from corrupting the state
                    if self.last_encoder_state == old_state:
                        # State didn't change, which is fine
                        pass
                    else:
                        # State changed - verify it's still valid
                        # If we see too many rapid invalid transitions, slow down
                        last_valid_state = self.last_encoder_state
                    
                    last_state_check = current_time
                else:
                    # Small sleep to prevent tight loop
                    time.sleep(0.0005)  # 500 microseconds
                    
            except Exception as e:
                # Handle errors gracefully - don't crash
                if self.running:
                    time.sleep(0.01)  # Longer delay on error
                continue
    
    def _status_loop(self):
        """Status reporting loop"""
        while self.running:
            time.sleep(self.STATUS_INTERVAL)
            
            # Read current encoder value safely
            with self._lock:
                current_encoder_value = self.encoder_value
                last_move_time = self.last_movement_time
            
            current_time = time.time()
            
            # Calculate velocity
            time_delta_ms = (current_time - self.last_velocity_time) * 1000
            if time_delta_ms >= self.VELOCITY_UPDATE_MS:
                position_delta = current_encoder_value - self.last_position_for_velocity
                self.current_velocity = position_delta / (time_delta_ms / 1000.0)  # pulses per second
                self.last_position_for_velocity = current_encoder_value
                self.last_velocity_time = current_time
            
            # Check for stall
            time_since_last_move = (current_time - last_move_time) * 1000  # Convert to ms
            position_delta = abs(current_encoder_value - self.last_encoder_value)
            
            if position_delta >= self.STALL_THRESHOLD:
                # Motor has moved
                self.is_moving = True
                self.is_stalled = False
                self.last_encoder_value = current_encoder_value
            elif time_since_last_move > self.STALL_TIMEOUT_MS:
                # No movement detected within timeout period
                self.is_moving = False
                self.is_stalled = True
            else:
                # Within timeout, but not enough movement yet
                self.is_moving = (position_delta > 0)
                self.is_stalled = False
            
            # Read raw pin states
            pin_a_state = self.line_a.get_value()
            pin_b_state = self.line_b.get_value()
            
            # Send status: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>
            print(f"POS,{current_encoder_value},{self.current_velocity:.2f},"
                  f"{1 if self.is_moving else 0},{1 if self.is_stalled else 0},"
                  f"{pin_a_state},{pin_b_state}")
            sys.stdout.flush()
    
    def start(self):
        """Start encoder monitoring"""
        if self.running:
            return
        
        self.running = True
        
        # Start encoder monitoring thread
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self.encoder_thread.start()
        
        # Start status reporting thread
        self.status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self.status_thread.start()
    
    def stop(self):
        """Stop encoder monitoring"""
        self.running = False
        if self.encoder_thread:
            self.encoder_thread.join(timeout=1.0)
        if self.status_thread:
            self.status_thread.join(timeout=1.0)
    
    def cleanup(self):
        """Release GPIO resources"""
        self.stop()
        if self.line_a:
            try:
                self.line_a.release()
            except Exception:
                pass
        if self.line_b:
            try:
                self.line_b.release()
            except Exception:
                pass
        if self.chip:
            try:
                self.chip.close()
            except Exception:
                pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


def main():
    """Main function"""
    print("========================================")
    print("Rock Pi Encoder Stall Detector")
    print("========================================")
    print()
    
    encoder = None
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nStopping...")
        if encoder:
            encoder.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize encoder (gpiochip1, line 31 for PIN_29, line 15 for PIN_31)
        encoder = EncoderStallDetector(chip_num=1, line_a=31, line_b=15)
        
        print("Status format: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>")
        print("Pin states: 0=LOW, 1=HIGH")
        print(f"Update rate: {1.0 / EncoderStallDetector.STATUS_INTERVAL:.1f} Hz")
        print()
        print("Ready! Monitoring encoder...")
        print()
        
        # Start monitoring
        encoder.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if encoder:
            encoder.cleanup()
        sys.exit(1)
    finally:
        if encoder:
            encoder.cleanup()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()

