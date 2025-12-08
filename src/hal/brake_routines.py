"""
Brake Routine Script

Provides start and dont_die methods for brake control.
"""

import time
import sys
from pathlib import Path

# Add src folder to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hal.Brake import BrakeESC


class BrakeRoutines:
    """
    Brake control routines for start and dont_die operations.
    """
    
    def __init__(self, brake: BrakeESC):
        """
        Initialize with a BrakeESC instance.
        
        Args:
            brake: BrakeESC instance to control
        """
        self.brake = brake
    
    def start(self):
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

        

    
    def dont_die(self):
        """
        Don't die routine:
        - brake.set(1200)
        - sleep(2)
        - brake.disable
        - brake.enable
        - brake.set(1600)
        - sleep(0.1)
        - disable
        - enable
        """
        print("BrakeRoutines: Starting dont_die routine...")

        brake.enable()

        # brake.set(1200)
        print("BrakeRoutines: Setting brake to 1200 us")
        self.brake.set_pulse_width(1200, check_stall=False)
        
        # sleep(2)
        print("BrakeRoutines: Sleeping for 2 seconds")
        time.sleep(1.5)
    

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
        print("BrakeRoutines: Setting brake to 1700 us")
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
        

        
        print("BrakeRoutines: Dont_die routine complete!")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Brake routine control")
    parser.add_argument("--encoder-port", type=str, help="Encoder serial port (e.g., ttl1)")
    parser.add_argument("--encoder-baudrate", type=int, default=115200, help="Encoder baudrate")
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
    
    # Create routines instance
    routines = BrakeRoutines(brake)
    
    # Execute requested routine
    try:
        if args.routine == "start":
            routines.start()
        elif args.routine == "dont_die":
            routines.dont_die()
    finally:
        # Cleanup
        brake.cleanup()

