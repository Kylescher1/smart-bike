import serial.tools.list_ports
import time
import platform
import signal
import subprocess
import cv2


stop_flag = False


def handle_sigint(sig, frame):
    global stop_flag
    print("\n🛑 Stopping detection...")
    stop_flag = True


signal.signal(signal.SIGINT, handle_sigint)


def get_serial_ports():
    """Return a set of serial port names."""
    return {p.device for p in serial.tools.list_ports.comports()}


def get_video_devices():
    """Return a set of available video device indices/names."""
    system = platform.system()
    devices = set()

    if system == "Windows":
        # Try PowerShell query
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-PnpDevice | Where-Object {$_.FriendlyName -like '*Camera*'} | Select-Object -ExpandProperty FriendlyName",
                ],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if line.strip():
                    devices.add(line.strip())
        except Exception:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if "Model ID:" in line:
                    devices.add(line.split(":", 1)[1].strip())
        except Exception:
            pass
    else:  # Linux
        try:
            result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if "/dev/video" in line:
                    devices.add(line.strip())
        except Exception:
            pass

    # Fallback: probe with OpenCV
    if not devices:
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.add(f"Camera {i}")
                cap.release()
    return devices


def detect_changes(get_devices_fn, label):
    """Monitor a category of devices and report changes."""
    global stop_flag
    before = get_devices_fn()
    print(f"Initial {label}: {', '.join(before) if before else '(none)'}")

    while not stop_flag:
        time.sleep(1)
        after = get_devices_fn()
        added = after - before
        removed = before - after

        if added:
            print(f"\n✅ New {label}: {', '.join(added)}")
        if removed:
            print(f"\n⚠️ {label} removed: {', '.join(removed)}")

        before = after


if __name__ == "__main__":
    print("=== USB Serial & Camera Device Detector ===")
    print("Press Ctrl+C to stop.\n")

    try:
        while not stop_flag:
            print("\n--- Checking Serial Ports ---")
            detect_changes(get_serial_ports, "Serial Devices")

            if stop_flag:
                break

            print("\n--- Checking Cameras ---")
            detect_changes(get_video_devices, "Camera Devices")

    except KeyboardInterrupt:
        handle_sigint(None, None)

    print("\n✅ Detection stopped cleanly.")
