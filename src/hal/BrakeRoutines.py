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
    def __init__(self,brake: BrakeESC,name = "Dawg the Brake Routines", **kwargs):

        """
        Initialize with a BrakeESC instance.
        
        Args:
            brake: BrakeESC instance to control
        """
        self.brake = brake
        
        # Check if ABS is enabled (if abs is in kwargs and is True or not None)
        # ABS is enabled if 'abs' is in kwargs and its value is True, or if set to any non-None value
        self.abs_enabled = True  # Only True if explicitly set to True

        
        for k,v in kwargs.items():#unpack config into self
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

    def start(self):
        """
        Start routine in a separate thread.
        """
        print("BrakeRoutines: Creating thread for start routine...")
        start_thread = threading.Thread(target=self._start_routine, name="start_thread")
        start_thread.start()
        start_thread.join()
        print("BrakeRoutines: Start thread completed")

    
    def _dont_die_routine(self):
        print(f"BrakeRoutines: dont_die called, abs_enabled = {self.abs_enabled}")

        if self.abs_enabled:
            self.brake.disable()
            self.brake.enable()
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.05)
            # ABS braking: rapid pulsing to prevent wheel lockup
            print("BrakeRoutines: Using ABS braking mode")
            abs_duration = 3  # Total ABS braking duration
            abs_cycle_time = 0.3  # Time for each pulse cycle (apply + release)
            abs_apply_time = 0.075  # Time to apply brake in each cycle
            abs_release_time = 0.15  # Time to release brake in each cycle
            
            start_time = time.time()
            cycle_count = 0
            
            while time.time() - start_time < abs_duration:
                cycle_count += 1
                # Apply brake
                print(f"BrakeRoutines: ABS cycle {cycle_count} - Applying brake")
                self.brake.set_pulse_width(1200, check_stall=False)
                time.sleep(abs_apply_time)
                
                # Release brake briefly
                print(f"BrakeRoutines: ABS cycle {cycle_count} - Releasing brake")
                self.brake.set_pulse_width(1500, check_stall=False)
                time.sleep(abs_release_time)
            
            print(f"BrakeRoutines: ABS braking complete ({cycle_count} cycles)")
        else:
            print("BrakeRoutines: Using standard (non-ABS) braking mode")
            self.brake.disable()
            self.brake.enable()
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.1)

            # brake.set(1200)
            print("BrakeRoutines: Setting brake to 1200 us")
            self.brake.set_pulse_width(1200, check_stall=False)
            
            # sleep(2)
            print("BrakeRoutines: Brake Applied for 2.5 seconds")
            time.sleep(2.5)
        

            # stopping brake
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.5)

            # brake.disable
            print("BrakeRoutines: Disabling brake")
            self.brake.disable()
            
            # brake.enable
            print("BrakeRoutines: Enabling brake")
            self.brake.enable()
            
            # release
            print("BrakeRoutines: Setting brake to 1600 us")
            self.brake.set_pulse_width(1600, check_stall=False)
            
            # sleep(0.1)
            print("BrakeRoutines: Sleeping for 0.1 seconds")
            time.sleep(0.1)

            # stopping brake
            self.brake.set_pulse_width(1500, check_stall=False)
            time.sleep(0.5)
            
            # disable
            print("BrakeRoutines: Disabling brake")
            self.brake.disable()
            
            print("BrakeRoutines: Sleeping for 2 seconds")
            time.sleep(1.5)
        


            # brake.disable
            print("BrakeRoutines: Disabling brake")
            self.brake.disable()
            

        
        # disable
        print("BrakeRoutines: Disabling brake")
        self.brake.disable()
        

        
        print("BrakeRoutines: Dont_die routine complete!")

    def dont_die(self):
        """
        Dont_die routine in a separate thread.
        """
        print("BrakeRoutines: Creating thread for dont_die routine...")
        dont_die_thread = threading.Thread(target=self._dont_die_routine, name="dont_die_thread")
        dont_die_thread.start()
        dont_die_thread.join()
        print("BrakeRoutines: Dont_die thread completed")


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
        routines_kwargs["abs"] = True
    routines = BrakeRoutines(brake, **routines_kwargs)
    
    # Execute requested routine
    try:
        if args.routine == "start":
            routines.start()
        elif args.routine == "dont_die":
            routines.dont_die()
    finally:
        # Cleanup
        brake.cleanup()
