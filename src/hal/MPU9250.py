#!/usr/bin/env python3
import time
from smbus2 import SMBus

class MPU9250:
    # MPU9250 default I2C addresses
    MPU_ADDR = 0x68
    MAG_ADDR = 0x0C

    # MPU9250 Register addresses
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    # Magnetometer registers
    AK8963_ST1 = 0x02
    AK8963_XOUT_L = 0x03
    AK8963_CNTL = 0x0A

    def __init__(self, bus_num=1):
        self.bus = SMBus(bus_num)
        self.initialize_sensor()

    def initialize_sensor(self):
        # Wake up MPU9250
        self.bus.write_byte_data(self.MPU_ADDR, self.PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Configure magnetometer for continuous measurement mode 2 (16-bit)
        self.bus.write_byte_data(self.MAG_ADDR, self.AK8963_CNTL, 0x16)
        time.sleep(0.1)

    def read_i2c_word(self, addr, reg):
        high = self.bus.read_byte_data(addr, reg)
        low = self.bus.read_byte_data(addr, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:  # convert to signed
            value = -((65535 - value) + 1)
        return value

    def read_accel(self):
        accel_x = self.read_i2c_word(self.MPU_ADDR, self.ACCEL_XOUT_H)
        accel_y = self.read_i2c_word(self.MPU_ADDR, self.ACCEL_XOUT_H + 2)
        accel_z = self.read_i2c_word(self.MPU_ADDR, self.ACCEL_XOUT_H + 4)
        # Convert to g’s (assuming ±2g)
        accel_scale = 16384.0
        return {
            'x': accel_x / accel_scale,
            'y': accel_y / accel_scale,
            'z': accel_z / accel_scale,
        }

    def read_gyro(self):
        gyro_x = self.read_i2c_word(self.MPU_ADDR, self.GYRO_XOUT_H)
        gyro_y = self.read_i2c_word(self.MPU_ADDR, self.GYRO_XOUT_H + 2)
        gyro_z = self.read_i2c_word(self.MPU_ADDR, self.GYRO_XOUT_H + 4)
        # Convert to deg/s (assuming ±250°/s)
        gyro_scale = 131.0
        return {
            'x': gyro_x / gyro_scale,
            'y': gyro_y / gyro_scale,
            'z': gyro_z / gyro_scale,
        }

    def read_mag(self):
        # Check if data ready
        if not (self.bus.read_byte_data(self.MAG_ADDR, self.AK8963_ST1) & 0x01):
            return None

        data = self.bus.read_i2c_block_data(self.MAG_ADDR, self.AK8963_XOUT_L, 7)
        x = (data[1] << 8) | data[0]
        y = (data[3] << 8) | data[2]
        z = (data[5] << 8) | data[4]
        # Convert to signed 16-bit
        for axis in (x, y, z):
            if axis >= 0x8000:
                axis = -((65535 - axis) + 1)
        return {'x': x, 'y': y, 'z': z}

    def read(self):
        """Return a combined reading of accelerometer, gyroscope, and magnetometer."""
        return {
            'accel': self.read_accel(),
            'gyro': self.read_gyro(),
            'mag': self.read_mag(),
        }


if __name__ == "__main__":
    mpu = MPU9250()
    try:
        while True:
            data = mpu.read()
            print(f"Accel: {data['accel']} | Gyro: {data['gyro']} | Mag: {data['mag']}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting...")
