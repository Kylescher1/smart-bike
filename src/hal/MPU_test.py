#!/usr/bin/env python3
import time
from smbus2 import SMBus

# I2C addresses
MPU_ADDR = 0x68
AK8963_ADDR = 0x0C

# Registers
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
INT_PIN_CFG = 0x37  # Needed to access magnetometer
AK8963_ST1 = 0x02
AK8963_XOUT_L = 0x03

# Select your Rock Pi I2C bus (usually /dev/i2c-7)
I2C_BUS = 7

def read_word(bus, addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    val = (high << 8) | low
    if val & 0x8000:
        val = -((65535 - val) + 1)
    return val

def init_mpu(bus):
    # Wake up MPU
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)

    # Configure accelerometer ±2g
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x00)

    # Configure gyro ±250 °/s
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)

    # Enable bypass mode so master can access magnetometer
    bus.write_byte_data(MPU_ADDR, INT_PIN_CFG, 0x02)
    time.sleep(0.1)

    # Configure AK8963 (magnetometer)
    try:
        bus.write_byte_data(AK8963_ADDR, 0x0A, 0x16)  # 16-bit, continuous measurement mode 2
    except OSError:
        print("⚠️  Magnetometer not responding — continuing without it.")

def read_mpu(bus):
    # Accelerometer
    ax = read_word(bus, MPU_ADDR, 0x3B)
    ay = read_word(bus, MPU_ADDR, 0x3D)
    az = read_word(bus, MPU_ADDR, 0x3F)

    # Gyroscope
    gx = read_word(bus, MPU_ADDR, 0x43)
    gy = read_word(bus, MPU_ADDR, 0x45)
    gz = read_word(bus, MPU_ADDR, 0x47)

    # Magnetometer
    mx = my = mz = 0
    try:
        st1 = bus.read_byte_data(AK8963_ADDR, AK8963_ST1)
        if st1 & 0x01:
            data = bus.read_i2c_block_data(AK8963_ADDR, AK8963_XOUT_L, 7)
            mx = (data[1] << 8) | data[0]
            my = (data[3] << 8) | data[2]
            mz = (data[5] << 8) | data[4]
            # Convert to signed
            if mx >= 32768: mx -= 65536
            if my >= 32768: my -= 65536
            if mz >= 32768: mz -= 65536
    except OSError:
        pass

    return (ax, ay, az, gx, gy, gz, mx, my, mz)

def main():
    with SMBus(I2C_BUS) as bus:
        print(f"Initializing MPU-9250 on I2C bus {I2C_BUS}...")
        init_mpu(bus)
        print("Ready! Press Ctrl+C to stop.\n")

        while True:
            ax, ay, az, gx, gy, gz, mx, my, mz = read_mpu(bus)
            print(f"ACCEL[g]: {ax:6d} {ay:6d} {az:6d} | "
                  f"GYRO[dps]: {gx:6d} {gy:6d} {gz:6d} | "
                  f"MAG: {mx:6d} {my:6d} {mz:6d}")
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
