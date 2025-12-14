"""
Brake Routine Script

Provides start and dont_die methods for brake control.
"""

import time
import sys
import threading
from pathlib import Path

# Add src folder to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hal.Brake import BrakeESC


class BrakeRoutines:
    """
    Brake control routines for start and dont_die operations.
    """
    def __init__(self, name="Dawg the Brake Routines", brake=None, **kwargs):
        """
        Initialize with a BrakeESC instance.
        
        Args:
            name: Name for this BrakeRoutines instance
            brake: Optional BrakeESC instance. If not provided, one will be created from kwargs.
            **kwargs: Configuration parameters. Brake-specific params will be passed to BrakeESC.
                     Routine-specific params (like abs_enabled) will be used by BrakeRoutines.
        """
        # Track running threads for stop() method
        self._start_thread = None
        self._dont_die_thread = None
        self._stop_requested = threading.Event()
        
        # Check if ABS is enabled (if abs is in kwargs and is True or not None)
        # ABS is enabled if 'abs' is in kwargs and its value is True, or if set to any non-None value
        self.abs_enabled = kwargs.pop('abs_enabled', True)
        
        # If brake instance provided, use it; otherwise create one from kwargs
        if brake is not None:
            self.brake = brake
        else:
            # Filter out routine-specific kwargs before creating BrakeESC
            brake_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['abs_enabled']}
            self.brake = BrakeESC(name=f"{name}_BrakeESC", **brake_kwargs)
        
        # Unpack remaining config into self
        for k, v in kwargs.items():
            if k != 'abs_enabled':  # Already handled
                setattr(self, k, v)

        if self.abs_enabled:
            print(f"BrakeRoutines: ABS braking enabled")
    
    def _start_routine(self):
        """
        Start routine:
        - brake.set(1300)
        - sleep(2)
        - brake.disable
        - brake.enable
        - brake.set(1600)
        - sleep(0.1)
        - disable
        - enable
        """
        print("BrakeRoutines: Starting start routine...")
        
        if self._stop_requested.is_set():
            print("BrakeRoutines: Start routine cancelled")
            return


        #disable and enable brake
        print("BrakeRoutines: Disabling brake")
        self.brake.disable()
        time.sleep(0.01)
        print("BrakeRoutines: Enabling brake")
        self.brake.enable()
        time.sleep(0.01)
        
                   # stopping brake
        self.brake.set_pulse_width(1500, check_stall=False)
        time.sleep(0.1)

                #disable and enable brake
        print("BrakeRoutines: Disabling brake")
        self.brake.disable()
        time.sleep(0.01)
        print("BrakeRoutines: Enabling brake")
        self.brake.enable()
        time.sleep(0.01)

        #prime brake
        self.brake.set_pulse_width(1375, check_stall=False)
        time.sleep(.75)

        self.brake.set_pulse_width(1500, check_stall=False)
        time.sleep(0.5)

        #release a little bit
        self.brake.set_pulse_width(1600, check_stall=False)
        time.sleep(0.10)

        self.brake.set_pulse_width(1500, check_stall=False)
        time.sleep(0.5)


        self.brake.disable()
        print("BrakeRoutines: Start routine complete!")

    def start(self, blocking=False):
        """
        Start routine in a separate thread.
        
        Args:
            blocking: If True, wait for the routine to complete before returning.
                     If False (default), return immediately while routine runs in background.
        """
        print("BrakeRoutines: Creating thread for start routine...")
        self._stop_requested.clear()
        self._start_thread = threading.Thread(target=self._start_routine, name="start_thread", daemon=True)
        self._start_thread.start()
        if blocking:
            self._start_thread.join()
            self._start_thread = None
            print("BrakeRoutines: Start thread completed")

    
    def _dont_die_routine(self):
        print(f"BrakeRoutines: dont_die called, abs_enabled = {self.abs_enabled}")
        
        if self._stop_requested.is_set():
            print("BrakeRoutines: Dont_die routine cancelled")
            return

        if self.abs_enabled:
            self.brake.disable()
            self.brake.enable()
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.05)
            # ABS braking: rapid pulsing to prevent wheel lockup
            print("BrakeRoutines: Using ABS braking mode")
            abs_duration = 3  # Total ABS braking duration
            abs_cycle_time = 0.25  # Time for each pulse cycle (apply + release)
            abs_apply_time = 0.2  # Time to apply brake in each cycle
            abs_release_time = 0.05  # Time to release brake in each cycle
            
            start_time = time.time()
            cycle_count = 0
            
            while time.time() - start_time < abs_duration:
                if self._stop_requested.is_set():
                   
                    break
                    
                cycle_count += 1
                # Apply brake
               
                self.brake.set_pulse_width(1200, check_stall=False)
                time.sleep(abs_apply_time)
                
                # Release brake briefly
               
                self.brake.set_pulse_width(1500, check_stall=False)
                time.sleep(abs_release_time)

            
            
        else:

            self.brake.disable()
            self.brake.enable()
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.1)

            # brake.set(1200)
       
            self.brake.set_pulse_width(1200, check_stall=False)
            
            # sleep(2)
            
            time.sleep(2.5)
        

            # stopping brake
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.5)

            # brake.disable
            
            self.brake.disable()
            
            # brake.enable
            
            self.brake.enable()
            
            # release
            
            self.brake.set_pulse_width(1600, check_stall=False)
            
            # sleep(0.1)
            
            time.sleep(0.1)

            # stopping brake
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.5)
            
            # disable


            # brake.disable
            self.brake.disable()
            time.sleep(0.01)
            self.brake.enable()
            time.sleep(0.01)

        
        # disable
        
        self.brake.disable()
        self.brake.enable()
        

        
        print("BrakeRoutines: Dont_die routine complete!")

    def dont_die(self, blocking=False):
        """
        Dont_die routine in a separate thread.
        
        Args:
            blocking: If True, wait for the routine to complete before returning.
                     If False (default), return immediately while routine runs in background.
        """
        print("BrakeRoutines: Creating thread for dont_die routine...")
        self._stop_requested.clear()
        self._dont_die_thread = threading.Thread(target=self._dont_die_routine, name="dont_die_thread", daemon=True)
        self._dont_die_thread.start()
        if blocking:
            self._dont_die_thread.join()
            self._dont_die_thread = None
            print("BrakeRoutines: Dont_die thread completed")
    
    @property
    def is_running(self):
        """Check if any routine is currently running."""
        start_running = self._start_thread is not None and self._start_thread.is_alive()
        dont_die_running = self._dont_die_thread is not None and self._dont_die_thread.is_alive()
        return start_running or dont_die_running
    
    def wait(self, timeout=None):
        """
        Wait for any running routine to complete.
        
        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.
        
        Returns:
            True if routines completed, False if timeout occurred.
        """
        if self._start_thread is not None and self._start_thread.is_alive():
            self._start_thread.join(timeout=timeout)
            if self._start_thread.is_alive():
                return False
            self._start_thread = None
        
        if self._dont_die_thread is not None and self._dont_die_thread.is_alive():
            self._dont_die_thread.join(timeout=timeout)
            if self._dont_die_thread.is_alive():
                return False
            self._dont_die_thread = None
        
        return True
    
    def stop(self):
        """
        Stop any running brake routines and disable the brake.
        """
        print("BrakeRoutines: Stop requested")
        self._stop_requested.set()
        
        # Stop the brake
        if self.brake is not None:
            try:
                self.brake.set_pulse_width(1500, check_stall=False)  # Return to neutral
                time.sleep(0.1)
                self.brake.disable()
                print("BrakeRoutines: Brake stopped and disabled")
            except Exception as e:
                print(f"BrakeRoutines: Error stopping brake: {e}")
        
        # Wait for threads to finish (with timeout)
        if self._start_thread is not None and self._start_thread.is_alive():
            self._start_thread.join(timeout=1.0)
            if self._start_thread.is_alive():
                print("BrakeRoutines: Warning: start_thread did not stop in time")
        
        if self._dont_die_thread is not None and self._dont_die_thread.is_alive():
            self._dont_die_thread.join(timeout=1.0)
            if self._dont_die_thread.is_alive():
                print("BrakeRoutines: Warning: dont_die_thread did not stop in time")
        
        print("BrakeRoutines: Stop complete")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Brake routine control")
    parser.add_argument("--encoder-port", type=str, help="Encoder serial port (e.g., ttl1)")
    parser.add_argument("--encoder-baudrate", type=int, default=115200, help="Encoder baudrate")
    parser.add_argument("--abs", action="store_true", help="Enable ABS braking")
    parser.add_argument("routine", choices=["start", "dont_die"], help="Routine to execute")
    
    args = parser.parse_args()
    
    # Create brake instance
    brake_kwargs = {}
    if args.encoder_port:
        brake_kwargs["encoder_port"] = args.encoder_port
        brake_kwargs["encoder_baudrate"] = args.encoder_baudrate
    
    brake = BrakeESC(name="BrakeRoutine", **brake_kwargs)
    
    # Initialize encoder if port provided
    if args.encoder_port:
        brake._init_encoder()
    
    # Create routines instance with ABS option
    routines_kwargs = {}
    if args.abs:
        routines_kwargs["abs_enabled"] = True
    routines = BrakeRoutines(name="BrakeRoutine", brake=brake, **routines_kwargs)
    
    # Execute requested routine
    try:
        if args.routine == "start":
            routines.start()
        elif args.routine == "dont_die":
            routines.dont_die()
    finally:
        # Cleanup
        brake.cleanup()
