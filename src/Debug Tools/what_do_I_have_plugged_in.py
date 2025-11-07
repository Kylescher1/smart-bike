import serial.tools.list_ports
import subprocess
import platform
import shutil


def run_command(cmd, label):
    """Run a system command safely and print the output."""
    print(f"\n=== {label} ===")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        print(result.stdout.strip() or "(No output)")
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
    except Exception as e:
        print(f"Error running {cmd}: {e}")


def list_serial_devices():
    print("=== Serial Devices ===")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("(No serial ports found)")
    for port in ports:
        print(f"{port.device}: {port.description}")


def list_usb_devices():
    system = platform.system()
    if system == "Windows":
        # Use PowerShell instead of deprecated WMIC
        cmd = [
            "powershell",
            "-Command",
            "Get-PnpDevice -Class USB | Select-Object -Property Name, Status",
        ]
    elif system == "Darwin":  # macOS
        cmd = ["system_profiler", "SPUSBDataType"]
    else:  # Linux
        cmd = ["lsusb"]
    run_command(cmd, "USB Devices")


def list_video_devices():
    system = platform.system()
    if system == "Windows":
        cmd = [
            "powershell",
            "-Command",
            "Get-PnpDevice | Where-Object {$_.FriendlyName -like '*Camera*'} | Select-Object -Property FriendlyName, Status",
        ]
    elif system == "Darwin":
        cmd = ["system_profiler", "SPCameraDataType"]
    else:
        cmd = ["v4l2-ctl", "--list-devices"]
    run_command(cmd, "Video / Camera Devices")


def list_audio_devices():
    system = platform.system()
    if system == "Windows":
        cmd = [
            "powershell",
            "-Command",
            "Get-PnpDevice -Class Media | Select-Object -Property Name, Status",
        ]
    elif system == "Darwin":
        cmd = ["system_profiler", "SPAudioDataType"]
    else:
        cmd = ["pactl", "list", "short", "sinks"]
    run_command(cmd, "Audio Devices")


if __name__ == "__main__":
    list_serial_devices()
    list_usb_devices()
    list_video_devices()
    list_audio_devices()
