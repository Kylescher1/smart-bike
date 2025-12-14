"""
YOLO11n detection class for integration with main system.

Follows sensor pattern with start(), stop(), and read() methods.
Can be used standalone or integrated via config.dill and main.py.

Example config entry (using VISION system):
    "yolo_detector": {
        "weights": "yolo/models/yolo11n.pt",
        "vision_system": "camera",  # Name of VISION sensor, or pass object directly
        "camera_side": "left",  # "left" or "right" (default: "left")
        "imgsz": 640,
        "conf": 0.25,
        "device": None,
        "frame_size": None,
        "who_to_run": "yolo.live_demo.YOLODetector",
    }

Example config entry (using direct Camera):
    "yolo_detector": {
        "weights": "yolo/models/yolo11n.pt",
        "camera": camera_object,  # Direct Camera object reference
        "imgsz": 640,
        "conf": 0.25,
        "who_to_run": "yolo.live_demo.YOLODetector",
    }

Example config entry (fallback to direct source):
    "yolo_detector": {
        "weights": "yolo/models/yolo11n.pt",
        "source": "0",  # camera index or video file path
        "imgsz": 640,
        "conf": 0.25,
        "who_to_run": "yolo.live_demo.YOLODetector",
    }
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11n.pt"


@dataclass
class Detection:
    """Represents a single detection."""
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class DetectionResult:
    """Complete detection result for a frame."""
    frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Detection]
    fps: float
    inference_time_ms: float
    timestamp: float


class YOLODetector:
    """
    YOLO object detector following sensor pattern.
    Supports start(), stop(), and read() methods for integration.
    """
    
    def __init__(self, name="Unnamed YOLODetector", **kwargs):
        """
        Initialize YOLO detector.
        
        Args:
            name: Sensor name
            **kwargs: Configuration parameters:
                - weights: Path to weights file (default: yolo/models/yolo11n.pt)
                - vision_system: VISION system object or sensor name (optional)
                - camera: Camera object (optional, alternative to vision_system)
                - camera_side: "left" or "right" when using vision_system (default: "left")
                - source: Camera index or video file path (fallback if vision_system/camera not provided)
                - imgsz: Inference image size (default: 640)
                - conf: Confidence threshold (default: 0.25)
                - device: Computation device (default: None/auto)
                - frame_size: Optional "WIDTHxHEIGHT" for center cropping
                - half: Use FP16 half precision (default: False, GPU only)
        """
        self.name = name
        self.debug_mode = True
        
        # Load configuration from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Set defaults
        self.weights = getattr(self, "weights", DEFAULT_WEIGHTS)
        self.vision_system = getattr(self, "vision_system", None)
        self.camera = getattr(self, "camera", None)
        self.camera_side = getattr(self, "camera_side", "left")
        self.source = getattr(self, "source", "0")
        self.imgsz = getattr(self, "imgsz", 640)
        self.conf = getattr(self, "conf", 0.25)
        self.device = getattr(self, "device", None)
        self.frame_size = getattr(self, "frame_size", None)
        self.half = getattr(self, "half", False)  # FP16 half precision
        
        # Convert weights path to Path if string
        if isinstance(self.weights, str):
            self.weights = Path(self.weights)
        if not self.weights.is_absolute():
            self.weights = ROOT / self.weights
        
        # Parse frame_size if provided
        self.frame_width = None
        self.frame_height = None
        if self.frame_size:
            try:
                width_str, height_str = str(self.frame_size).lower().split("x")
                self.frame_width = int(width_str)
                self.frame_height = int(height_str)
            except (ValueError, AttributeError):
                self.frame_width = None
                self.frame_height = None
        
        # Runtime state
        self.connected = False
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.results_iter = None
        self.data_buffer = deque(maxlen=10)  # Keep last 10 results
        self.data_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.save_dir: Optional[Path] = None
        self.frame_idx = 0
        
        # Frame source: VISION system, Camera, or direct source
        self.frame_source_type = None  # "vision", "camera", or "direct"
        self.frame_source_camera: Optional[object] = None  # Camera object if using vision/camera
        
        # Normalize source (only used if not using vision_system or camera)
        self.source_normalized = self._normalize_source(self.source) if not (self.vision_system or self.camera) else None
    
    def set_vision_system(self, vision_system, camera_side="left"):
        """
        Set the VISION system to use for frame capture (can be called after instantiation).
        
        Args:
            vision_system: VISION system object
            camera_side: "left" or "right" (default: "left")
        """
        if self.connected:
            raise RuntimeError(f"{self.name}: Cannot change vision_system while connected. Call stop() first.")
        self.vision_system = vision_system
        self.camera_side = camera_side
        self.source_normalized = None  # Clear direct source since we're using VISION
    
    def _normalize_source(self, source: str) -> Union[int, str]:
        """Convert string source to int if it's a digit."""
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source
    
    def _center_crop(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Crop the frame to the requested size, centered."""
        height, width = frame.shape[:2]
        crop_width = min(target_width, width)
        crop_height = min(target_height, height)
        x0 = max((width - crop_width) // 2, 0)
        y0 = max((height - crop_height) // 2, 0)
        return frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
    
    def _extract_detections(self, result) -> List[Detection]:
        """Extract detections from YOLO result."""
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        names = None
        if hasattr(result, "names") and result.names is not None:
            names = result.names
        elif hasattr(result, "model") and hasattr(result.model, "names"):
            names = result.model.names
        
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        for idx, bbox in enumerate(boxes_xyxy):
            class_id = classes[idx]
            label = str(class_id)
            if isinstance(names, dict):
                label = names.get(class_id, label)
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                label = names[class_id]
            
            detections.append(Detection(
                label=label,
                confidence=float(confs[idx]),
                bbox=tuple(map(float, bbox))
            ))
        
        return detections
    
    def _process_frame(self, result) -> DetectionResult:
        """Process a YOLO result into DetectionResult."""
        annotated_frame = result.plot()
        inference_time_ms = (
            result.speed.get("inference", 0.0) if hasattr(result, "speed") else 0.0
        )
        
        # Get original frame
        orig_img = getattr(result, "orig_img", None)
        if orig_img is None:
            orig_img = annotated_frame.copy()
        
        detections = self._extract_detections(result)
        
        return DetectionResult(
            frame=orig_img,
            annotated_frame=annotated_frame,
            detections=detections,
            fps=0.0,  # Will be updated by caller
            inference_time_ms=inference_time_ms,
            timestamp=time.time()
        )
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the configured source (VISION, Camera, or direct)."""
        if self.frame_source_type == "vision":
            # Get frame from VISION system's camera
            if self.frame_source_camera is None:
                return None
            try:
                frame = self.frame_source_camera.read_frame()
                return frame
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Error reading frame from VISION camera: {e}")
                return None
        elif self.frame_source_type == "camera":
            # Get frame from direct Camera object
            if self.frame_source_camera is None:
                return None
            try:
                frame = self.frame_source_camera.read_frame()
                return frame
            except Exception as e:
                if self.debug_mode:
                    print(f"{self.name}: Error reading frame from Camera: {e}")
                return None
        else:
            # Direct source (VideoCapture)
            if self.cap is None:
                return None
            ret, frame = self.cap.read()
            return frame if ret else None
    
    def _data_collector(self):
        """Background thread to continuously process frames."""
        print(f"{self.name}: Data collector started.")
        prev_time = time.time()
        
        # Determine if we're using manual frame capture or YOLO streaming
        use_manual_capture = (
            self.frame_source_type in ("vision", "camera") or
            (self.frame_width is not None and self.frame_height is not None)
        )
        
        if use_manual_capture:
            # Manual frame capture (from VISION/Camera or with cropping)
            while not self.stop_event.is_set():
                frame = self._get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Apply cropping if specified
                if self.frame_width is not None and self.frame_height is not None:
                    frame = self._center_crop(frame, self.frame_width, self.frame_height)
                    imgsz_override = (self.frame_height, self.frame_width)
                else:
                    imgsz_override = self.imgsz
                
                # Run inference
                inference_results = self.model.predict(
                    source=[frame],
                    imgsz=imgsz_override,
                    conf=self.conf,
                    device=self.device,
                    half=self.half,  # FP16 half precision
                    verbose=False,
                )
                result = inference_results[0]
                
                now = time.time()
                delta = max(now - prev_time, 1e-6)
                prev_time = now
                
                detection_result = self._process_frame(result)
                detection_result.fps = 1.0 / delta
                
                self.data_buffer.append(detection_result)
        else:
            # Use YOLO's streaming prediction (for direct video sources)
            results_iter = self.model.predict(
                stream=True,
                source=self.source_normalized,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                half=self.half,  # FP16 half precision
                show=False,
            )
            
            for result in results_iter:
                if self.stop_event.is_set():
                    break
                
                now = time.time()
                delta = max(now - prev_time, 1e-6)
                prev_time = now
                
                detection_result = self._process_frame(result)
                detection_result.fps = 1.0 / delta
                
                self.data_buffer.append(detection_result)
        
        print(f"{self.name}: Data collector stopped.")
    
    def start(self):
        """Initialize model and start data collection."""
        if self.connected:
            return
        
        print(f"{self.name}: Starting YOLO detector...")
        
        # Load weights
        weights_path = Path(self.weights).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"{self.name}: Weights file not found: {weights_path}")
        
        # Initialize model
        try:
            self.model = YOLO(str(weights_path))
            print(f"{self.name}: Model loaded from {weights_path}")
        except Exception as e:
            raise RuntimeError(f"{self.name}: Failed to load model: {e}")
        
        # Setup frame source: VISION system, Camera, or direct source
        if self.vision_system is not None:
            # Using VISION system
            self.frame_source_type = "vision"
            
            # If vision_system is a string, try to resolve it (for future enhancement)
            # For now, assume it's passed as an object reference
            if isinstance(self.vision_system, str):
                raise ValueError(
                    f"{self.name}: vision_system as string name not yet supported. "
                    "Pass the VISION object directly or use 'camera' parameter."
                )
            
            # Get the appropriate camera from VISION system
            if not hasattr(self.vision_system, 'left_camera') or not hasattr(self.vision_system, 'right_camera'):
                raise ValueError(f"{self.name}: vision_system must have left_camera and right_camera attributes")
            
            if not self.vision_system.connected:
                raise RuntimeError(f"{self.name}: vision_system must be started before YOLODetector")
            
            if self.camera_side.lower() == "left":
                self.frame_source_camera = self.vision_system.left_camera
            elif self.camera_side.lower() == "right":
                self.frame_source_camera = self.vision_system.right_camera
            else:
                raise ValueError(f"{self.name}: camera_side must be 'left' or 'right', got '{self.camera_side}'")
            
            print(f"{self.name}: Using {self.camera_side} camera from VISION system")
        
        elif self.camera is not None:
            # Using direct Camera object
            self.frame_source_type = "camera"
            self.frame_source_camera = self.camera
            
            # Verify camera is open
            if not hasattr(self.camera, 'read_frame'):
                raise ValueError(f"{self.name}: camera object must have read_frame() method")
            
            if not hasattr(self.camera, 'cap') or self.camera.cap is None or not self.camera.cap.isOpened():
                raise RuntimeError(f"{self.name}: camera must be opened before YOLODetector starts")
            
            print(f"{self.name}: Using direct Camera object")
        
        else:
            # Fallback to direct source (VideoCapture)
            self.frame_source_type = "direct"
            
            # Open video source if using manual capture with cropping
            if self.frame_width is not None and self.frame_height is not None:
                self.cap = cv2.VideoCapture(self.source_normalized)
                if not self.cap.isOpened():
                    raise RuntimeError(f"{self.name}: Unable to open source {self.source_normalized}")
                print(f"{self.name}: Video source opened: {self.source_normalized}")
            else:
                # Will use YOLO's streaming prediction
                print(f"{self.name}: Using YOLO streaming prediction with source: {self.source_normalized}")
        
        # Start data collection thread
        self.stop_event.clear()
        self.data_thread = threading.Thread(target=self._data_collector, daemon=True)
        self.data_thread.start()
        
        self.connected = True
        print(f"{self.name}: Started successfully.")
    
    def stop(self):
        """Stop data collection and clean up resources."""
        if not self.connected:
            return
        
        print(f"{self.name}: Stopping...")
        self.connected = False
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=2)
        
        # Close video capture (only if we opened it ourselves)
        if self.frame_source_type == "direct" and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Note: We don't close VISION system or Camera objects - they're managed elsewhere
        
        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Close results iterator if it has close method
        if self.results_iter and hasattr(self.results_iter, "close"):
            try:
                self.results_iter.close()
            except Exception:
                pass
        
        # Clear frame source references
        self.frame_source_camera = None
        
        print(f"{self.name}: Stopped.")
    
    def read(self) -> Optional[DetectionResult]:
        """
        Return the most recent detection result.
        
        Returns:
            DetectionResult with frame, detections, fps, etc., or None if no data available
        """
        if not self.connected or len(self.data_buffer) == 0:
            return None
        
        return self.data_buffer[-1]
    
    def read_all(self) -> List[DetectionResult]:
        """Return all buffered detection results."""
        return list(self.data_buffer)


# -------------------------------------------------------------------------
# Standalone demo functionality
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone usage."""
    parser = argparse.ArgumentParser(description="Run a YOLO11n live detection demo.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the YOLO weights file (.pt). Defaults to yolo/models/yolo11n.pt.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Inference source: camera index, video file, image directory, RTSP/HTTP stream, etc.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference image size (pixels). Ignored when --frame-size is provided.",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default=None,
        help="Center-crop before inference to WIDTHxHEIGHT (e.g., 640x480). Overrides --imgsz.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device. Examples: 'cpu', '0' (CUDA GPU 0), '0,1'. Defaults to auto.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Optional path to save an annotated video.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to dump annotated frames as images.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the preview window (useful for headless recording).",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=None,
        help="Resize annotated frames for display to this width while preserving aspect ratio.",
    )
    return parser.parse_args()


def main() -> int:
    """Standalone demo main function."""
    args = parse_args()
    
    # Create detector instance
    detector = YOLODetector(
        name="YOLODemo",
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        frame_size=args.frame_size,
    )
    
    # Setup recording/saving if requested
    if args.record:
        detector.record_path = args.record.expanduser().resolve()
        detector.record_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.save_dir:
        detector.save_dir = args.save_dir.expanduser().resolve()
        detector.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Start detector
    try:
        detector.start()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    
    window_name = f"YOLO11n Live Demo ({detector.source_normalized})"
    video_writer = None
    
    if not args.no_show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            pass
    
    try:
        while True:
            result = detector.read()
            if result is None:
                time.sleep(0.01)
                continue
            
            annotated_frame = result.annotated_frame.copy()
            
            # Overlay FPS and model info
            cv2.putText(
                annotated_frame,
                f"FPS: {result.fps:.1f} | Inference: {result.inference_time_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            # Initialize video writer if needed
            if args.record and video_writer is None:
                height, width = annotated_frame.shape[:2]
                fourcc = (
                    cv2.VideoWriter_fourcc(*"mp4v")
                    if args.record.suffix.lower() == ".mp4"
                    else cv2.VideoWriter_fourcc(*"XVID")
                )
                video_writer = cv2.VideoWriter(str(args.record), fourcc, 30.0, (width, height))
                if not video_writer.isOpened():
                    print(f"[ERROR] Failed to open video writer for {args.record}", file=sys.stderr)
                    video_writer = None
            
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            if detector.save_dir is not None:
                frame_file = detector.save_dir / f"frame_{detector.frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_file), annotated_frame)
                detector.frame_idx += 1
            
            display_frame = annotated_frame
            if args.display_width:
                height, width = display_frame.shape[:2]
                if width != args.display_width and width > 0:
                    scale = args.display_width / width
                    new_height = max(int(round(height * scale)), 1)
                    display_frame = cv2.resize(
                        display_frame, (args.display_width, new_height), interpolation=cv2.INTER_AREA
                    )
            
            if not args.no_show:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            
            time.sleep(0.01)  # Small delay to prevent busy loop
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, shutting down gracefully.")
    finally:
        detector.stop()
        if video_writer is not None:
            video_writer.release()
        if not args.no_show:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
