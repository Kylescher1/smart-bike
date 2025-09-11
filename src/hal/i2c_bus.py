"""
I2C Bus Interface Module

Provides abstraction for I2C communication with sensors (IMU, TOF).
"""

import smbus2
import time
from typing import Optional, List, Union

class I2CBus:
    """
    Manages I2C bus communication for sensor interfaces
    
    Supports:
    - Multiple device communication
    - Read/write operations
    - Error handling
    """
    
    def __init__(self, bus_number: int = 1):
        """
        Initialize I2C bus
        
        Args:
            bus_number: I2C bus number (default: 1 for most single-board computers)
        """
        try:
            self.bus = smbus2.SMBus(bus_number)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize I2C bus: {e}")
    
    def read_byte(self, device_address: int, register: int) -> Optional[int]:
        """
        Read a single byte from a specific register
        
        Args:
            device_address: I2C device address
            register: Register to read from
        
        Returns:
            Byte value or None if read fails
        """
        try:
            return self.bus.read_byte_data(device_address, register)
        except IOError as e:
            print(f"I2C read error (device {hex(device_address)}, reg {hex(register)}): {e}")
            return None
    
    def read_word(self, device_address: int, register: int) -> Optional[int]:
        """
        Read a 16-bit word from a specific register
        
        Args:
            device_address: I2C device address
            register: Starting register to read from
        
        Returns:
            Word value or None if read fails
        """
        try:
            return self.bus.read_word_data(device_address, register)
        except IOError as e:
            print(f"I2C word read error (device {hex(device_address)}, reg {hex(register)}): {e}")
            return None
    
    def write_byte(self, device_address: int, register: int, value: int) -> bool:
        """
        Write a single byte to a specific register
        
        Args:
            device_address: I2C device address
            register: Register to write to
            value: Byte value to write
        
        Returns:
            Success status
        """
        try:
            self.bus.write_byte_data(device_address, register, value)
            return True
        except IOError as e:
            print(f"I2C write error (device {hex(device_address)}, reg {hex(register)}): {e}")
            return False
    
    def read_block(self, device_address: int, register: int, length: int) -> Optional[List[int]]:
        """
        Read a block of data from a specific register
        
        Args:
            device_address: I2C device address
            register: Starting register to read from
            length: Number of bytes to read
        
        Returns:
            List of bytes or None if read fails
        """
        try:
            return self.bus.read_i2c_block_data(device_address, register, length)
        except IOError as e:
            print(f"I2C block read error (device {hex(device_address)}, reg {hex(register)}): {e}")
            return None
    
    def device_scan(self) -> List[int]:
        """
        Scan the I2C bus for connected devices
        
        Returns:
            List of detected device addresses
        """
        devices = []
        for address in range(0x03, 0x78):
            try:
                self.bus.read_byte(address)
                devices.append(address)
            except IOError:
                pass
        return devices
    
    def __del__(self):
        """
        Close I2C bus connection
        """
        if hasattr(self, 'bus'):
            self.bus.close()
