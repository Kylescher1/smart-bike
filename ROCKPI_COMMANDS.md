# Rock Pi Useful Commands

## Smart Bike Service Management

```bash
# Start the service
sudo systemctl start smart-bike.service

# Stop the service
sudo systemctl stop smart-bike.service

# Restart the service (after code changes)
sudo systemctl restart smart-bike.service

# Check service status
sudo systemctl status smart-bike.service

# Enable auto-start on boot
sudo systemctl enable smart-bike.service

# Disable auto-start on boot
sudo systemctl disable smart-bike.service
```

## Viewing Logs

```bash
# View smart-bike logs (live)
tail -f /home/radxa/smart-bike/logs/smart-bike.log

# View last 100 lines of logs
tail -n 100 /home/radxa/smart-bike/logs/smart-bike.log

# View systemd logs for the service
journalctl -u smart-bike.service -f

# View systemd logs (last 50 lines)
journalctl -u smart-bike.service -n 50
```

## Running Manually (for debugging)

```bash
# Activate venv and run main.py
cd /home/radxa/smart-bike
source venv/bin/activate
python main.py

# Run turret directly (standalone)
python turret_debug/yolo_gimbal.py --camera 1 --turret /dev/ttyUSB0 --invert-y --rknn
```

## Git Commands (updating code)

```bash
cd /home/radxa/smart-bike

# Pull latest changes
git pull

# Check status
git status

# Discard local changes and pull fresh
git reset --hard HEAD
git pull
```

## Serial Port Commands

```bash
# List available serial ports
ls /dev/ttyUSB*
ls /dev/ttyACM*

# Check which device is on which port
dmesg | grep tty

# Give user permission to serial ports (run once)
sudo usermod -a -G dialout radxa
```

## Camera Commands

```bash
# List available cameras
ls /dev/video*

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(1); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
```

## System Commands

```bash
# Check CPU temperature
cat /sys/class/thermal/thermal_zone0/temp

# Check memory usage
free -h

# Check disk usage
df -h

# Reboot
sudo reboot

# Shutdown
sudo shutdown now
```

## First-Time Setup

```bash
# 1. Create logs directory
mkdir -p /home/radxa/smart-bike/logs

# 2. Copy service file to systemd
sudo cp /home/radxa/smart-bike/smart-bike.service /etc/systemd/system/

# 3. Reload systemd
sudo systemctl daemon-reload

# 4. Enable service
sudo systemctl enable smart-bike.service

# 5. Start service
sudo systemctl start smart-bike.service

# 6. Check it's running
sudo systemctl status smart-bike.service
```

## Troubleshooting

```bash
# If service won't start, check logs
journalctl -u smart-bike.service -n 50 --no-pager

# If serial port permission denied
sudo chmod 666 /dev/ttyUSB0

# If camera not found, check index
for i in 0 1 2 3 4 5; do
  python -c "import cv2; cap = cv2.VideoCapture($i); print(f'Camera $i:', 'OK' if cap.isOpened() else 'FAIL'); cap.release()"
done

# Kill any stuck Python processes
pkill -f "python.*main.py"
pkill -f "python.*yolo_gimbal.py"
```

