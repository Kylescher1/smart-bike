import serial.tools.list_ports
import subprocess
import platform

def list_serial_devices():
    print("=== Serial Devices ===")
    for port in serial.tools.list_ports.comports():
        print(f"{port.device}: {port.description}")

def list_usb_devices():
    print("\n=== USB Devices ===")
    if platform.system() == "Windows":
        subprocess.run(["wmic", "path", "Win32_USBHub", "get", "Name"], text=True)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["system_profiler", "SPUSBDataType"], text=True)
    else:  # Linux
        subprocess.run(["lsusb"], text=True)

def list_video_devices():
    print("\n=== Video / Camera Devices ===")
    if platform.system() == "Windows":
        subprocess.run(["wmic", "path", "Win32_PnPEntity", "where", "Description like '%Camera%'", "get", "Name"], text=True)
    elif platform.system() == "Darwin":
        subprocess.run(["system_profiler", "SPCameraDataType"], text=True)
    else:
        subprocess.run(["v4l2-ctl", "--list-devices"], text=True)

def list_audio_devices():
    print("\n=== Audio Devices ===")
    if platform.system() == "Windows":
        subprocess.run(["wmic", "path", "Win32_SoundDevice", "get", "Name"], text=True)
    elif platform.system() == "Darwin":
        subprocess.run(["system_profiler", "SPAudioDataType"], text=True)
    else:
        subprocess.run(["pactl", "list", "short", "sinks"], text=True)

if __name__ == "__main__":
    list_serial_devices()
    list_usb_devices()
    list_video_devices()
    list_audio_devices()
