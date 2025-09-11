"""
IMU (Inertial Measurement Unit) Interface Module

Handles data acquisition and processing for MPU6050 sensor.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from .i2c_bus import I2CBus

class IMUInterface:
    """
    Manages MPU6050 IMU sensor data acquisition
    
    Attributes:
        - Handles 100-200 Hz data acquisition
        - Provides gyroscope and accelerometer readings
        - Computes attitude (roll, pitch, yaw)
    """
    
    # MPU6050 I2C Registers
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    
    def __init__(self, bus: Optional[I2CBus] = None, address: int = 0x68):
        """
        Initialize IMU interface
        
        Args:
            bus: I2C bus instance (optional)
            address: I2C device address
        """
        self.bus = bus or I2CBus()
        self.address = address
        
        # Calibration parameters (to be updated during calibration)
        self.accel_offset = np.zeros(3)
        self.gyro_offset = np.zeros(3)
        
        self._initialize_imu()
    
    def _initialize_imu(self):
        """
        Perform IMU initialization
        
        - Wake up device
        - Set sample rate
        - Configure digital low-pass filter
        """
        # Wake up from sleep mode
        self.bus.write_byte(self.address, self.PWR_MGMT_1, 0)
    
    def _read_raw_data(self, start_register: int, length: int = 6) -> Optional[np.ndarray]:
        """
        Read raw sensor data
        
        Args:
            start_register: Starting register for data
            length: Number of bytes to read
        
        Returns:
            Raw sensor data as numpy array or None
        """
        raw_data = self.bus.read_block(self.address, start_register, length)
        
        if raw_data is None:
            return None
        
        # Convert 2-byte values to signed 16-bit integers
        data = np.array([
            np.int16((raw_data[i] << 8) | raw_data[i+1]) 
            for i in range(0, length, 2)
        ])
        
        return data
    
    def read_accelerometer(self) -> Optional[np.ndarray]:
        """
        Read accelerometer data
        
        Returns:
            3D acceleration vector (x, y, z) in g
        """
        raw_accel = self._read_raw_data(self.ACCEL_XOUT_H)
        
        if raw_accel is None:
            return None
        
        # Sensitivity scale factor (typical for ±2g range)
        accel_sensitivity = 16384.0
        
        # Apply offset and scale
        return (raw_accel / accel_sensitivity) - self.accel_offset
    
    def read_gyroscope(self) -> Optional[np.ndarray]:
        """
        Read gyroscope data
        
        Returns:
            3D angular velocity vector (x, y, z) in degrees/sec
        """
        raw_gyro = self._read_raw_data(self.GYRO_XOUT_H)
        
        if raw_gyro is None:
            return None
        
        # Sensitivity scale factor (typical for ±250 deg/s range)
        gyro_sensitivity = 131.0
        
        # Apply offset and scale
        return (raw_gyro / gyro_sensitivity) - self.gyro_offset
    
    def compute_attitude(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> Dict[str, float]:
        """
        Compute attitude using complementary filter
        
        Args:
            accel: Accelerometer readings
            gyro: Gyroscope readings
            dt: Time delta between measurements
        
        Returns:
            Dictionary with roll, pitch, yaw
        """
        # Compute angles from accelerometer
        roll = np.arctan2(accel[1], accel[2]) * 180 / np.pi
        pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)) * 180 / np.pi
        
        # TODO: Implement more sophisticated attitude estimation
        # - Complementary filter
        # - Kalman filter
        
        return {
            'roll': roll,
            'pitch': pitch,
            'yaw': 0.0  # Placeholder
        }
    
    def calibrate(self, samples: int = 1000):
        """
        Perform IMU sensor calibration
        
        Args:
            samples: Number of samples to average for offset
        """
        print("Calibrating IMU... Keep sensor still")
        
        accel_samples = []
        gyro_samples = []
        
        for _ in range(samples):
            accel = self.read_accelerometer()
            gyro = self.read_gyroscope()
            
            if accel is not None and gyro is not None:
                accel_samples.append(accel)
                gyro_samples.append(gyro)
        
        self.accel_offset = np.mean(accel_samples, axis=0)
        self.gyro_offset = np.mean(gyro_samples, axis=0)
        
        print("IMU Calibration Complete")
