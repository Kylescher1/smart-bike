"""
Smart Bike Main Application Entry Point

Manages system initialization, mode transitions, and overall system lifecycle.
"""

import sys
import logging
from enum import Enum, auto
from typing import Optional

# Import local modules
from src.hal.camera_interface import CameraInterface
# from src.hal.lidar_interface import LidarInterface
# from src.hal.imu_interface import IMUInterface
# from src.system.health_monitor import HealthMonitor

class SystemMode(Enum):
    """
    Defines the operational modes for the Smart Bike system
    """
    INIT = auto()
    STANDBY = auto()
    RUN = auto()
    DEGRADED = auto()
    SAFE = auto()
    SHUTDOWN = auto()

class SmartBikeSystem:
    """
    Primary system controller for Smart Bike platform
    
    Manages system initialization, sensor bring-up, and mode transitions
    """
    
    def __init__(self):
        """
        Initialize system components and prepare for operation
        """
        self.logger = logging.getLogger('SmartBikeSystem')
        self.logger.setLevel(logging.INFO)
        
        # Configure logging
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # System state
        self.current_mode = SystemMode.INIT
        
        # Sensor interfaces
        self.camera_interface: Optional[CameraInterface] = None
        # self.lidar_h_interface: Optional[LidarInterface] = None
        # self.lidar_g_interface: Optional[LidarInterface] = None
        # self.imu_interface: Optional[IMUInterface] = None
        
    def initialize(self):
        """
        Perform system initialization:
        1. Bring up hardware interfaces
        2. Load configurations
        3. Perform self-tests
        """
        self.logger.info("Initializing Smart Bike System")
        
        try:
            # Initialize camera interface
            self.camera_interface = CameraInterface()
            self.logger.info("Camera interface initialized successfully")
            
            # TODO: Initialize other sensor interfaces
            # self.lidar_h_interface = LidarInterface(type='horizontal')
            # self.lidar_g_interface = LidarInterface(type='ground')
            # self.imu_interface = IMUInterface()
            
            # Transition to standby mode
            self.transition_to(SystemMode.STANDBY)
        
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.transition_to(SystemMode.SAFE)
    
    def transition_to(self, new_mode: SystemMode):
        """
        Handle system mode transitions with appropriate logging and checks
        
        Args:
            new_mode: Target system mode to transition to
        """
        self.logger.info(f"Transitioning from {self.current_mode} to {new_mode}")
        
        # TODO: Implement mode-specific transition logic
        # - Validate guard conditions
        # - Perform necessary setup/teardown
        # - Log transition details
        
        self.current_mode = new_mode
    
    def run(self):
        """
        Main system run loop
        Manages system lifecycle and mode-specific behaviors
        """
        self.initialize()
        
        try:
            while self.current_mode not in {SystemMode.SAFE, SystemMode.SHUTDOWN}:
                # TODO: Implement main system loop
                # - Sensor data acquisition
                # - Perception processing
                # - Decision making
                # - Mode management
                
                # Example: Grab stereo frames
                frames = self.camera_interface.grab_stereo_frame()
                if frames is None:
                    self.logger.warning("Failed to grab stereo frames")
                
                # Placeholder: Add processing logic
        
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
            self.transition_to(SystemMode.SAFE)
        
        except Exception as e:
            self.logger.error(f"Unhandled system error: {e}")
            self.transition_to(SystemMode.SAFE)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Perform orderly system shutdown and resource cleanup
        """
        self.logger.info("Performing system cleanup")
        
        # Close sensor interfaces
        if self.camera_interface:
            del self.camera_interface
        
        # TODO: Close other interfaces
        
        self.transition_to(SystemMode.SHUTDOWN)

def main():
    """
    Entry point for Smart Bike application
    """
    system = SmartBikeSystem()
    system.run()

if __name__ == "__main__":
    main()
