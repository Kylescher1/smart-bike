#!/usr/bin/env python3
"""
Test script to read serial data from Arduino Nano encoder stall detector

This script connects to the Arduino Nano via serial and displays
the encoder position, velocity, and stall status in real-time.

Usage:
    python test_nano_encoder.py [--port PORT] [--baudrate BAUDRATE]

Example:
    python test_nano_encoder.py --port /dev/ttyUSB0 --baudrate 115200
"""

import serial
import argparse
import sys
import time
from datetime import datetime


def parse_status_line(line):
    """
    Parse a status line from the Arduino Nano
    
    Format: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>
    
    Returns:
        dict with keys: position, velocity, is_moving, is_stalled, pinA, pinB
        Returns None if line cannot be parsed
    """
    line = line.strip()
    if not line.startswith("POS,"):
        return None
    
    try:
        parts = line.split(",")
        if len(parts) < 5:
            return None
        
        result = {
            "position": int(parts[1]),
            "velocity": float(parts[2]),
            "is_moving": bool(int(parts[3])),
            "is_stalled": bool(int(parts[4]))
        }
        
        # Add pin states if present
        if len(parts) >= 7:
            result["pinA"] = int(parts[5])
            result["pinB"] = int(parts[6])
        
        return result
    except (ValueError, IndexError) as e:
        print(f"Error parsing line: {line}, error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test Arduino Nano encoder stall detector"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Serial timeout in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Arduino Nano Encoder Stall Detector Test")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    try:
        # Open serial connection
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        # Wait a moment for Arduino to initialize
        time.sleep(2)
        
        # Clear any buffered data
        ser.reset_input_buffer()
        
        print("Connected! Reading encoder data...")
        print()
        print(f"{'Time':<12} {'Position':<12} {'Velocity':<12} {'Moving':<8} {'Stalled':<8} {'Pin3':<6} {'Pin9':<6}")
        print("-" * 80)
        
        last_position = None
        stall_count = 0
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore')
                status = parse_status_line(line)
                
                if status:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    # Display status
                    pinA_str = str(status.get('pinA', 'N/A'))
                    pinB_str = str(status.get('pinB', 'N/A'))
                    
                    print(
                        f"{timestamp:<12} "
                        f"{status['position']:<12} "
                        f"{status['velocity']:<12.2f} "
                        f"{'YES' if status['is_moving'] else 'NO':<8} "
                        f"{'YES' if status['is_stalled'] else 'NO':<8} "
                        f"{pinA_str:<6} "
                        f"{pinB_str:<6}"
                    )
                    
                    # Warn if pins are stuck or not changing
                    if 'pinA' in status and 'pinB' in status:
                        if status['pinA'] == 1 and status['pinB'] == 1:
                            print("  >>> Both pins HIGH - check encoder connection <<<")
                        elif status['pinA'] == 0 and status['pinB'] == 0:
                            print("  >>> Both pins LOW - check encoder connection <<<")
                    
                    # Track stall events
                    if status['is_stalled']:
                        stall_count += 1
                        if stall_count == 1:
                            print("  >>> STALL DETECTED! Motor is not moving. <<<")
                    else:
                        if stall_count > 0:
                            print(f"  >>> Stall cleared after {stall_count} detections. <<<")
                            stall_count = 0
                    
                    # Track position changes
                    if last_position is not None:
                        position_delta = status['position'] - last_position
                        if abs(position_delta) > 0:
                            direction = "forward" if position_delta > 0 else "reverse"
                            print(f"  Position change: {position_delta:+d} pulses ({direction})")
                    
                    last_position = status['position']
                else:
                    # Print non-status lines (like initialization messages)
                    if line.strip():
                        print(f"[Arduino] {line.strip()}")
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
            
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("\nCommon issues:")
        print("  - Port doesn't exist (check with: ls /dev/ttyUSB* or ls /dev/ttyACM*)")
        print("  - Permission denied (try: sudo chmod 666 /dev/ttyUSB0)")
        print("  - Port already in use")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")


if __name__ == "__main__":
    main()

