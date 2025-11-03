"""
ESP32 Data Bridge for Smart Bike
=================================

This script collects data from camera, depth map, and LIDAR sensors on the ROCK Pi
and sends it to the ESP32 via WiFi HTTP POST requests. The ESP32 acts as a WiFi
access point and web server for visualization.

Usage:
    python src/esp32_bridge.py

Requirements:
    - Vision system calibrated and running
    - LIDAR connected and accessible
    - ESP32 WiFi AP active (default: SmartBike_AP)
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import requests
from rplidarc1 import RPLidar

from src.hal.Vision import VisionSystem, default_calibration_file

# ========== Configuration ==========
ESP32_IP = "192.168.4.1"  # Default ESP32 AP IP
ESP32_HTTP_PORT = 80
CAMERA_FPS = 10  # Frames per second to send camera data
DEPTH_FPS = 10   # Frames per second to send depth data
LIDAR_FPS = 5    # Frames per second to send LIDAR data
IMAGE_QUALITY = 85  # JPEG compression quality (1-100)
MAX_LIDAR_POINTS = 500  # Maximum LIDAR points to send per update

# Resize images for transmission (reduce bandwidth)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480


class ESP32Bridge:
    """Bridge class to send Smart Bike sensor data to ESP32."""

    def __init__(
        self,
        esp32_ip: str = ESP32_IP,
        esp32_port: int = ESP32_HTTP_PORT,
        camera_fps: float = CAMERA_FPS,
        depth_fps: float = DEPTH_FPS,
        lidar_fps: float = LIDAR_FPS,
    ):
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.base_url = f"http://{esp32_ip}:{esp32_port}"
        
        self.camera_fps = camera_fps
        self.depth_fps = depth_fps
        self.lidar_fps = lidar_fps
        
        self.camera_interval = 1.0 / camera_fps
        self.depth_interval = 1.0 / depth_fps
        self.lidar_interval = 1.0 / lidar_fps
        
        self.vision_system: Optional[VisionSystem] = None
        self.lidar: Optional[RPLidar] = None
        self.lidar_thread: Optional[threading.Thread] = None
        
        self.running = False
        self.camera_last_sent = 0
        self.depth_last_sent = 0
        self.lidar_last_sent = 0
        self.lidar_buffer = deque(maxlen=600)  # Local LIDAR buffer

    def initialize_vision(self) -> None:
        """Initialize the vision system with stereo cameras."""
        print("Initializing vision system...")
        calibration_path = default_calibration_file()
        self.vision_system = VisionSystem(calibration_file=calibration_path)
        self.vision_system.open()
        print("✅ Vision system initialized")

    def initialize_lidar(self, port: str = "/dev/ttyUSB1", baudrate: int = 460800) -> None:
        """Initialize the LIDAR sensor."""
        print(f"Initializing LIDAR on {port}...")
        try:
            self.lidar = RPLidar(port, baudrate, timeout=3)
            print("✅ LIDAR initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize LIDAR: {e}")
            self.lidar = None

    def encode_image(self, image: np.ndarray, quality: int = IMAGE_QUALITY) -> str:
        """Encode numpy image to base64 JPEG."""
        # Resize if needed
        if image.shape[1] != CAMERA_WIDTH or image.shape[0] != CAMERA_HEIGHT:
            image = cv2.resize(image, (CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Encode as JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        return base64.b64encode(buffer).decode("utf-8")

    def encode_depth_map(self, depth_map: np.ndarray, quality: int = IMAGE_QUALITY) -> str:
        """Encode depth map to base64 JPEG (normalized for visualization)."""
        # Normalize depth map to 0-255
        if depth_map.size == 0:
            depth_map = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint8)
        else:
            # Handle invalid values
            valid_depth = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
            if valid_depth.size > 0:
                depth_min = np.min(valid_depth)
                depth_max = np.max(valid_depth)
                if depth_max > depth_min:
                    # Normalize to 0-255
                    normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                    # Apply colormap for better visualization
                    depth_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                else:
                    depth_map = np.zeros_like(depth_map, dtype=np.uint8)
            else:
                depth_map = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint8)
        
        # Resize if needed
        if isinstance(depth_map, np.ndarray) and len(depth_map.shape) == 3:
            if depth_map.shape[1] != DEPTH_WIDTH or depth_map.shape[0] != DEPTH_HEIGHT:
                depth_map = cv2.resize(depth_map, (DEPTH_WIDTH, DEPTH_HEIGHT))
        elif depth_map.shape[1] != DEPTH_WIDTH or depth_map.shape[0] != DEPTH_HEIGHT:
            depth_map = cv2.resize(depth_map, (DEPTH_WIDTH, DEPTH_HEIGHT))
        
        # Encode as JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", depth_map, encode_params)
        
        if not success:
            raise ValueError("Failed to encode depth map")
        
        return base64.b64encode(buffer).decode("utf-8")

    def send_camera_frame(self, frame: np.ndarray) -> bool:
        """Send camera frame to ESP32."""
        try:
            frame_b64 = self.encode_image(frame)
            response = requests.post(
                f"{self.base_url}/camera",
                data={"frame": frame_b64},
                timeout=1.0,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to send camera frame: {e}")
            return False

    def send_depth_map(self, depth_map: np.ndarray) -> bool:
        """Send depth map to ESP32."""
        try:
            depth_b64 = self.encode_depth_map(depth_map)
            response = requests.post(
                f"{self.base_url}/depth",
                data={"frame": depth_b64},
                timeout=1.0,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to send depth map: {e}")
            return False

    def send_lidar_data(self, lidar_points: list) -> bool:
        """Send LIDAR data to ESP32."""
        try:
            # Limit number of points to reduce bandwidth
            if len(lidar_points) > MAX_LIDAR_POINTS:
                step = len(lidar_points) // MAX_LIDAR_POINTS
                lidar_points = lidar_points[::step][:MAX_LIDAR_POINTS]
            
            # Convert to JSON-serializable format
            data = [
                {
                    "d_mm": int(point.get("d_mm", 0)),
                    "a_deg": float(point.get("a_deg", 0)),
                    "q": int(point.get("q", 0)),
                }
                for point in lidar_points
            ]
            
            json_data = json.dumps(data)
            response = requests.post(
                f"{self.base_url}/lidar",
                data={"data": json_data},
                timeout=1.0,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to send LIDAR data: {e}")
            return False

    def lidar_collector_thread(self) -> None:
        """Background thread to collect LIDAR data."""
        if self.lidar is None:
            return
        
        import asyncio

        async def run_scan(lidar_obj):
            await lidar_obj.simple_scan(make_return_dict=True)

        async def process_queue(queue, stop_event):
            while self.running:
                try:
                    measurement_dict = await asyncio.wait_for(queue.get(), timeout=1.0)
                    # Update instance buffer
                    self.lidar_buffer.append(measurement_dict)
                except asyncio.TimeoutError:
                    continue
            stop_event.set()

        async def main_loop():
            async with asyncio.TaskGroup() as tg:
                tg.create_task(run_scan(self.lidar))
                tg.create_task(process_queue(self.lidar.output_queue, self.lidar.stop_event))

        try:
            asyncio.run(main_loop())
        except ExceptionGroup as eg:
            print(f"LIDAR ERROR: TaskGroup failed")
            for error in eg.exceptions:
                print(f"  - {error}")
        except Exception as e:
            print(f"LIDAR thread error: {e}")
        finally:
            if self.lidar:
                self.lidar.reset()

    def run(self) -> None:
        """Main run loop."""
        if self.vision_system is None:
            raise RuntimeError("Vision system not initialized. Call initialize_vision() first.")
        
        self.running = True
        
        # Start LIDAR thread if available
        if self.lidar is not None:
            global lidar_is_running
            lidar_is_running = True
            self.lidar_thread = threading.Thread(target=self.lidar_collector_thread, daemon=True)
            self.lidar_thread.start()
            print("✅ LIDAR collector thread started")
        
        print("\n🚀 Starting ESP32 bridge...")
        print(f"📡 Sending to: {self.base_url}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                current_time = time.time()
                
                # Capture frames
                frames = self.vision_system.capture_frames()
                
                if frames is not None:
                    left_frame, right_frame = frames
                    
                    # Send camera frame (use left camera)
                    if current_time - self.camera_last_sent >= self.camera_interval:
                        if self.send_camera_frame(left_frame):
                            self.camera_last_sent = current_time
                    
                    # Compute depth map
                    depth_result = self.vision_system.compute_depth(left_frame, right_frame)
                    
                    # Send depth map
                    if current_time - self.depth_last_sent >= self.depth_interval:
                        if self.send_depth_map(depth_result.depth_map):
                            self.depth_last_sent = current_time
                
                # Send LIDAR data
                if self.lidar is not None and current_time - self.lidar_last_sent >= self.lidar_interval:
                    lidar_data = list(self.lidar_buffer) if hasattr(self, 'lidar_buffer') else []
                    if len(lidar_data) > 0:
                        if self.send_lidar_data(lidar_data):
                            self.lidar_last_sent = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the bridge and cleanup resources."""
        print("Cleaning up...")
        self.running = False
        
        if self.vision_system:
            self.vision_system.close()
        
        if self.lidar_thread:
            self.lidar_thread.join(timeout=2.0)
        
        if self.lidar:
            try:
                self.lidar.reset()
                self.lidar.close()
            except:
                pass
        
        print("✅ Cleanup complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ESP32 Bridge for Smart Bike")
    parser.add_argument("--esp32-ip", default=ESP32_IP, help=f"ESP32 IP address (default: {ESP32_IP})")
    parser.add_argument("--esp32-port", type=int, default=ESP32_HTTP_PORT, help=f"ESP32 HTTP port (default: {ESP32_HTTP_PORT})")
    parser.add_argument("--camera-fps", type=float, default=CAMERA_FPS, help=f"Camera FPS (default: {CAMERA_FPS})")
    parser.add_argument("--depth-fps", type=float, default=DEPTH_FPS, help=f"Depth map FPS (default: {DEPTH_FPS})")
    parser.add_argument("--lidar-fps", type=float, default=LIDAR_FPS, help=f"LIDAR FPS (default: {LIDAR_FPS})")
    parser.add_argument("--lidar-port", default="/dev/ttyUSB1", help="LIDAR serial port (default: /dev/ttyUSB1)")
    parser.add_argument("--lidar-baudrate", type=int, default=460800, help="LIDAR baudrate (default: 460800)")
    parser.add_argument("--no-lidar", action="store_true", help="Disable LIDAR")
    
    args = parser.parse_args()
    
    # Create bridge
    bridge = ESP32Bridge(
        esp32_ip=args.esp32_ip,
        esp32_port=args.esp32_port,
        camera_fps=args.camera_fps,
        depth_fps=args.depth_fps,
        lidar_fps=args.lidar_fps,
    )
    
    # Initialize systems
    try:
        bridge.initialize_vision()
        
        if not args.no_lidar:
            try:
                bridge.initialize_lidar(port=args.lidar_port, baudrate=args.lidar_baudrate)
            except Exception as e:
                print(f"⚠️  LIDAR initialization failed: {e}")
                print("   Continuing without LIDAR...")
        
        # Run bridge
        bridge.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exception(e)
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()

