# ESP32 Smart Bike Bridge

This directory contains the ESP32 firmware and setup instructions for creating a WiFi bridge that receives sensor data from the ROCK Pi and serves it to clients for visualization.

## Overview

The ESP32 acts as:
- **WiFi Access Point**: Creates a local network (`SmartBike_AP`) for devices to connect
- **Web Server**: Serves a visualization dashboard via HTTP
- **WebSocket Server**: Streams real-time sensor data to connected clients
- **Data Bridge**: Receives camera, depth map, and LIDAR data from ROCK Pi

## Hardware Requirements

- ESP32 development board (e.g., ESP32-DevKitC, ESP32-WROOM)
- USB cable for programming
- Optional: Status LED on GPIO 2 (most boards have built-in LED)

## Software Requirements

### Arduino IDE Setup

1. **Install Arduino IDE** (1.8.x or 2.x)

2. **Add ESP32 Board Support**:
   - Open Arduino IDE → File → Preferences
   - Add this URL to "Additional Board Manager URLs":
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Go to Tools → Board → Boards Manager
   - Search for "ESP32" and install "esp32 by Espressif Systems"

3. **Install Required Libraries**:
   - Open Tools → Manage Libraries
   - Install the following:
     - **WebSockets** by Markus Sattler (version 2.3.6 or later)
     - **ArduinoJson** by Benoit Blanchon (version 6.x)

4. **Board Configuration**:
   - Tools → Board → ESP32 Dev Module (or your specific board)
   - Tools → Upload Speed → 115200
   - Tools → Port → Select your ESP32's COM port

## Uploading Firmware

1. Open `smart_bike_bridge.ino` in Arduino IDE
2. Connect ESP32 to your computer via USB
3. Select the correct board and port (Tools menu)
4. Click Upload (or press Ctrl+U)
5. Wait for upload to complete
6. Open Serial Monitor (Tools → Serial Monitor) at 115200 baud to see status

## Configuration

Edit these constants in `smart_bike_bridge.ino` to customize:

```cpp
const char* AP_SSID = "SmartBike_AP";        // WiFi network name
const char* AP_PASSWORD = "smartbike123";     // WiFi password (change this!)
const int AP_CHANNEL = 1;                     // WiFi channel (1-11)
const int WEB_PORT = 80;                      // HTTP server port
const int WS_PORT = 81;                       // WebSocket server port
```

## Usage

1. **Power on ESP32** and wait for it to initialize
2. **Connect to WiFi**: Look for network "SmartBike_AP" and connect with password "smartbike123"
3. **Open Dashboard**: Navigate to `http://192.168.4.1` in your web browser
4. **Start ROCK Pi Bridge**: On the ROCK Pi, run:
   ```bash
   python src/esp32_bridge.py
   ```

## Network Information

- **AP IP Address**: `192.168.4.1` (default)
- **HTTP Endpoint**: `http://192.168.4.1`
- **WebSocket Endpoint**: `ws://192.168.4.1:81`

## API Endpoints

The ESP32 exposes these HTTP endpoints:

- `GET /` - Web visualization dashboard
- `POST /camera` - Send camera frame (base64 encoded JPEG)
- `POST /depth` - Send depth map (base64 encoded JPEG)
- `POST /lidar` - Send LIDAR scan data (JSON array)
- `GET /status` - Get bridge status (JSON)

### Example: Sending Camera Frame

```python
import requests
import base64

# Encode your image
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Send to ESP32
response = requests.post(
    "http://192.168.4.1/camera",
    data={"frame": img_b64}
)
```

### Example: Sending LIDAR Data

```python
import requests
import json

lidar_points = [
    {"d_mm": 1500, "a_deg": 45, "q": 100},
    {"d_mm": 2000, "a_deg": 90, "q": 95},
    # ... more points
]

response = requests.post(
    "http://192.168.4.1/lidar",
    data={"data": json.dumps(lidar_points)}
)
```

## Troubleshooting

### ESP32 won't upload
- Check COM port selection
- Try pressing BOOT button during upload
- Lower upload speed to 115200 or lower
- Check USB cable (data-capable, not charge-only)

### Can't connect to WiFi
- Check SSID and password in code
- Ensure you're connecting to "SmartBike_AP"
- Try resetting ESP32
- Check Serial Monitor for error messages

### No data showing on dashboard
- Verify ROCK Pi bridge is running: `python src/esp32_bridge.py`
- Check ESP32 IP address matches bridge configuration
- Open browser console (F12) for JavaScript errors
- Check Serial Monitor for HTTP request logs

### WebSocket connection fails
- Ensure firewall isn't blocking port 81
- Try different browser
- Check browser console for WebSocket errors

### Performance Issues
- Reduce FPS in `esp32_bridge.py` (lower `CAMERA_FPS`, `DEPTH_FPS`, `LIDAR_FPS`)
- Reduce image quality (`IMAGE_QUALITY` in bridge script)
- Reduce image resolution (`CAMERA_WIDTH/HEIGHT`, `DEPTH_WIDTH/HEIGHT`)
- Limit LIDAR points (`MAX_LIDAR_POINTS`)

## Serial Monitor Output

When running, you should see:
```
=== Smart Bike ESP32 Bridge ===
Starting Access Point... Done!
AP IP address: 192.168.4.1
SSID: SmartBike_AP
Password: smartbike123
HTTP Server started on port 80
WebSocket Server started on port 81

=== Ready! Connect to the WiFi network ===
Open http://192.168.4.1 in your browser
```

## Next Steps

- Customize the web dashboard HTML/CSS/JS
- Add authentication for WiFi network
- Implement data logging/storage
- Add command/control endpoints for ROCK Pi

## License

See main project LICENSE file.

