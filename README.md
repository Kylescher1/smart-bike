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

- Python 3.9+
- OpenCV
- NumPy
- ZeroMQ (optional, for pub/sub)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-bike.git
cd smart-bike
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running the System

```bash
python src/apps/run_smart_bike.py
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

### Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

- NASA JPL Robotics Architecture Inspiration
- OpenCV Community
- ROS2 Ecosystem
