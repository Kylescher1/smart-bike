# Smart Bike: Intelligent Robotic Assistance Platform

## Project Overview

Smart Bike is a high-performance, sensor-fusion robotic assistance platform designed for advanced navigation and environmental perception. Leveraging dual stereo cameras, LIDARs, IMU, and advanced perception algorithms, the system provides real-time hazard detection and navigation assistance.

### System Architecture

The project follows a NASA-inspired multi-layered architecture with clear separation of concerns:

1. **Hardware Abstraction Layer (HAL)**
2. **Sensor Acquisition Layer**
3. **Calibration/Rectification Layer**
4. **Perception Layer**
5. **Fusion & World Model**
6. **Decision Layer**
7. **Control Layer**
8. **Systems Layer**

### Key Components

- **Sensors**:
  - Dual USB AR0234 Stereo Cameras (30 Hz, 1280×720)
  - Two RPLIDARs (Horizontal + 30° Ground)
  - Time-of-Flight (TOF) Rangefinder
  - Inertial Measurement Unit (IMU)

- **Compute Platform**: ROCK 5B (RK3588)

### Prerequisites

- Python 3.9+ (tested with Python 3.11)
- OpenCV 4.7+
- NumPy 1.24+
- pyserial (for LIDAR and serial devices)
- smbus2 (for I2C communication)
- ZeroMQ (optional, for pub/sub)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-bike.git
cd smart-bike
```

2. Create and activate a virtual environment

   **On Linux/Mac:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   **On Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   You should see `(venv)` prefix in your terminal prompt when the virtual environment is active.

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Verify installation
```bash
python --version  # Should show Python 3.9 or higher
pip list           # Verify packages from requirements.txt are installed
```

### Development Setup

#### Virtual Environment Management

**Activating the virtual environment:**
- **Linux/Mac**: `source venv/bin/activate`
- **Windows**: `venv\Scripts\activate`

**Deactivating the virtual environment:**
```bash
deactivate
```

**Important:** Always activate the virtual environment before running any Python scripts or installing packages.

#### Development Notes

- Python cache files (`.pyc` and `__pycache__/`) are automatically ignored via `.gitignore`
- The project structure follows standard Python conventions with hardware abstraction layers and sensor interfaces
- All Python scripts should be run with the virtual environment activated to ensure correct dependency versions

### Running the System

#### Using main.py (Vision Pipeline)

The `main.py` script is the primary entry point for the Smart Bike vision pipeline. It initializes the stereo camera system and provides an interactive disparity tuning interface.

**Before running, ensure:**
- Virtual environment is activated (see Installation step 2)
- Dependencies are installed (see Installation step 3)
- Stereo cameras are connected and calibrated

**To run:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the vision pipeline
python main.py
```

**Features:**
- Live disparity map visualization
- Interactive parameter tuning via trackbars
- Real-time stereo depth processing
- Press `Ctrl+C` to stop the system

**Tuning Windows:**
- **Disparity Tuner**: Core stereo matching parameters (blockSize, numDisparities, etc.)
- **Viz/Seg Tuner**: Visualization and segmentation parameters

#### Alternative Entry Points

```bash
# Full Smart Bike system
python src/apps/run_smart_bike.py

# Camera development/testing
python src/apps/camera_dev.py
```

#### ESP32 WiFi Bridge

The ESP32 bridge allows remote visualization of camera, depth map, and LIDAR data via a web browser. The ESP32 acts as a WiFi Access Point and web server.

**Setup:**

1. **Flash ESP32 firmware** (see `esp32/README.md` for detailed instructions):
   - Install Arduino IDE with ESP32 board support
   - Install required libraries (WebSockets, ArduinoJson)
   - Upload `esp32/smart_bike_bridge.ino` to your ESP32

2. **Connect to ESP32 WiFi network**:
   - SSID: `SmartBike_AP`
   - Password: `smartbike123` (default, change in firmware)

3. **Start the bridge on ROCK Pi**:
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Run the ESP32 bridge
   python src/esp32_bridge.py
   ```

4. **Open visualization dashboard**:
   - Navigate to `http://192.168.4.1` in your web browser
   - Dashboard displays:
     - Live camera stream
     - Real-time depth map
     - LIDAR scan visualization

**Bridge Options:**
```bash
# Customize ESP32 IP and ports
python src/esp32_bridge.py --esp32-ip 192.168.4.1 --esp32-port 80

# Adjust frame rates
python src/esp32_bridge.py --camera-fps 5 --depth-fps 5 --lidar-fps 2

# Run without LIDAR
python src/esp32_bridge.py --no-lidar

# Custom LIDAR port
python src/esp32_bridge.py --lidar-port /dev/ttyUSB0 --lidar-baudrate 256000
```

See `esp32/README.md` for complete setup instructions and troubleshooting.

### Project Structure

```
smart-bike/
├── esp32/                 # ESP32 firmware and documentation
│   ├── smart_bike_bridge.ino  # ESP32 Arduino code
│   └── README.md          # ESP32 setup instructions
├── src/
│   ├── apps/              # Application entry points
│   │   ├── run_smart_bike.py
│   │   └── camera_dev.py
│   ├── esp32_bridge.py    # ROCK Pi to ESP32 bridge script
│   ├── hal/               # Hardware Abstraction Layer
│   │   ├── cam/           # Camera interfaces and tools
│   │   ├── MPU6250.py     # IMU interface
│   │   └── SpinningLidar.py # LIDAR interface
│   └── Debug Tools/       # Development utilities
├── data/                  # Calibration data and depth maps
├── requirements.txt       # Python dependencies
└── README.md
```

### Development Status

- [x] Project Structure
- [ ] HAL Implementations
- [ ] Sensor Acquisition
- [ ] Calibration Routines
- [ ] Perception Algorithms
- [ ] Fusion Engine
- [ ] Decision & Control Logic

### Safety & Modes

- **INIT**: Hardware bring-up
- **STANDBY**: Sensors streaming
- **RUN**: Full pipeline active
- **DEGRADED**: Reduced sensor input
- **SAFE**: Emergency halt
- **SHUTDOWN**: Orderly system stop

### Development Notes

- Python cache files (`.pyc` and `__pycache__/`) are ignored by git
- The project uses a layered architecture for maintainability and testability
- Sensor calibration data is stored in `data/` directory
- Debug tools are available in `src/Debug Tools/` for hardware diagnostics

### Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

- NASA JPL Robotics Architecture Inspiration
- OpenCV Community
- ROS2 Ecosystem
