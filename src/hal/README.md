# Camera Streaming Module (ELP AR0234)

## Overview
This module provides a robust utility for streaming video from USB cameras, specifically designed for the ELP AR0234 cameras on the Radxa Rock 5B.

## Prerequisites
- OpenCV (`opencv-python`)
- NumPy
- v4l2-python3

## Installation
```bash
pip install opencv-python numpy v4l2-python3
```

## Usage Scenarios

### 1. Basic Single Camera Capture
```python
from camera_stream import CameraStreamer

# Initialize with first camera (index 0)
streamer = CameraStreamer(camera_indices=[0])

if streamer.init_cameras():
    # Capture a single frame
    frame = streamer.capture_frame()
    
    # Display the frame
    streamer.show_frame(frame)
    
    # Save the frame
    streamer.save_frame(frame)
    
    # Clean up
    streamer.release_cameras()
```

### 2. Stereo Camera Capture
```python
# Initialize with two cameras (indices 0 and 1)
streamer = CameraStreamer(
    camera_indices=[0, 1],  # Specify camera indices
    resolution=(1280, 720),  # Set resolution
    fps=30                   # Set frame rate
)

if streamer.init_cameras():
    # Capture stereo frames
    stereo_frames = streamer.capture_stereo_frames()
    
    if stereo_frames:
        # Display and save both frames
        for i, frame in enumerate(stereo_frames):
            streamer.show_frame(frame, f"Camera {i}")
            streamer.save_frame(frame, prefix=f"stereo_cam_{i}")
    
    streamer.release_cameras()
```

### 3. Continuous Streaming
```python
streamer = CameraStreamer()

if streamer.init_cameras():
    try:
        while True:
            frame = streamer.capture_frame()
            if frame is not None:
                streamer.show_frame(frame)
                
                # Optional: Save frames periodically
                # streamer.save_frame(frame)
                
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        streamer.release_cameras()
```

## Troubleshooting
- Verify camera connections
- Check camera indices with `v4l2-ctl --list-devices`
- Ensure proper camera permissions
- Confirm OpenCV and dependencies are installed

## Performance Notes
- Tested resolution: 1280x720
- Target FPS: 30
- Supports multiple camera configurations

## Known Limitations
- Requires OpenCV and v4l2 support
- Camera performance may vary based on hardware
- Stereo synchronization depends on camera hardware
