#!/usr/bin/env python3
"""
Web streaming server for YOLO object detection and tracking.
Access from your phone or browser to view real-time object detection with tracking.

Usage:
    python yolo_web_stream.py --model yolo/models/yolo11n.rknn --source 1
    
Then open http://<your-ip>:5001 on your phone or browser
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from threading import Lock, Thread
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import YOLO inference functions
from yolo.rknn_inference import (
    RKNNLite, Camera, CAMERA_CONFIG, ByteTrackerWrapper,
    process_output, draw_detections, letterbox, DEFAULT_MODEL
)

app = Flask(__name__)

# Global state
class YOLOStreamState:
    def __init__(self):
        self.rknn = None
        self.camera = None
        self.tracker = None
        self.lock = Lock()
        self.latest_frame = None
        self.latest_frame_bytes = None
        self.frame_ready = False
        self.capturing = False
        
        # Detection settings
        self.conf_threshold = 0.25
        self.track_enabled = True
        self.imgsz = 640
        
        # Stats
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self.num_detections = 0
        self.num_tracks = 0
        self.frame_count = 0
        
        # Pre-allocated buffers
        self.img_input_buffer = None
        
state = YOLOStreamState()

def initialize_rknn(model_path):
    """Initialize RKNN model."""
    print(f"📦 Loading RKNN model: {model_path}")
    
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {ret}")
    
    ret = rknn.init_runtime(target=None, core_mask=None)
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Failed to initialize runtime: {ret}")
    
    print("✅ RKNN model loaded successfully")
    return rknn

def initialize_camera(source_index):
    """Initialize camera."""
    print(f"📹 Initializing camera {source_index}")
    
    camera_config = CAMERA_CONFIG.copy()
    camera_config.update({
        "width": 1280,
        "height": 720,
        "fps": 30,
        "fourcc": "MJPG"
    })
    
    camera = Camera(index=source_index, config=camera_config)
    camera.open()
    
    # Test frame read
    test_frame = camera.read_frame()
    if test_frame is None:
        camera.close()
        raise RuntimeError(f"Camera {source_index} opened but cannot read frames")
    
    print(f"✅ Camera {source_index} initialized ({camera.width}x{camera.height})")
    return camera

def generate_yolo_frame():
    """Generate a frame with YOLO detections and tracking."""
    if state.rknn is None or state.camera is None:
        return None
    
    try:
        # Capture frame
        frame = state.camera.read_frame()
        if frame is None:
            return None
        
        h_orig, w_orig = frame.shape[:2]
        
        # Preprocess
        img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(state.imgsz, state.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Pre-allocate buffer
        if state.img_input_buffer is None or state.img_input_buffer.shape != (1, state.imgsz, state.imgsz, 3):
            state.img_input_buffer = np.zeros((1, state.imgsz, state.imgsz, 3), dtype=np.uint8)
        state.img_input_buffer[0] = img_rgb.astype(np.uint8)
        img_input = state.img_input_buffer
        
        # Run inference
        inference_start = time.time()
        try:
            outputs = state.rknn.inference([img_input])
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
            return frame  # Return original frame on error
        
        inference_time_ms = (time.time() - inference_start) * 1000
        state.inference_time_ms = inference_time_ms
        
        # Process output
        detections = []
        if outputs is not None:
            try:
                detections = process_output(outputs, conf_threshold=state.conf_threshold, img_shape=(state.imgsz, state.imgsz))
            except Exception as e:
                print(f"[ERROR] Failed to process output: {e}", file=sys.stderr)
                detections = []
        
        # Scale boxes back to original image size
        if detections:
            scale = min(state.imgsz / w_orig, state.imgsz / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            pad_x = (state.imgsz - new_w) / 2
            pad_y = (state.imgsz - new_h) / 2
            
            boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
            
            for i, det in enumerate(detections):
                det['bbox'] = [int(float(boxes[i, 0])), int(float(boxes[i, 1])), 
                              int(float(boxes[i, 2])), int(float(boxes[i, 3]))]
        
        # Update tracker if enabled
        if state.tracker is not None and state.track_enabled:
            tracked_detections = state.tracker.update(detections)
            detections = tracked_detections
            state.num_tracks = len(state.tracker.tracked_tracks) if state.tracker else 0
        
        state.num_detections = len(detections)
        
        # Draw detections
        annotated = draw_detections(frame.copy(), detections, tracker=state.tracker if state.track_enabled else None)
        
        # Add info overlay
        fps_text = f"FPS: {state.fps:.1f}"
        inf_text = f"Inference: {state.inference_time_ms:.1f}ms"
        det_text = f"Detections: {state.num_detections}"
        track_text = f"Tracks: {state.num_tracks}" if state.track_enabled else ""
        
        y_offset = 30
        cv2.putText(annotated, fps_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(annotated, inf_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(annotated, det_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if track_text:
            y_offset += 25
            cv2.putText(annotated, track_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated
        
    except Exception as e:
        print(f"[ERROR] Error generating frame: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def capture_frames_continuously():
    """Background thread that continuously captures and encodes frames."""
    print("📹 Frame capture thread started")
    prev_time = time.time()
    
    while state.capturing:
        try:
            frame = generate_yolo_frame()
            
            if frame is not None:
                # Update FPS
                now = time.time()
                state.fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                state.frame_count += 1
                
                with state.lock:
                    state.latest_frame = frame
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        state.latest_frame_bytes = buffer.tobytes()
                        state.frame_ready = True
        except Exception as e:
            print(f"[ERROR] Error in frame capture: {e}", file=sys.stderr)
        
        time.sleep(0.033)  # ~30 FPS
    
    print("📹 Frame capture thread stopped")

def generate_frames():
    """Generator function that yields frames from the shared buffer."""
    # Wait for first frame to be ready
    while not state.frame_ready:
        time.sleep(0.1)
    
    while True:
        # Get the latest frame from shared buffer
        with state.lock:
            if state.latest_frame_bytes is not None:
                frame_bytes = state.latest_frame_bytes
        
        # Yield the frame to this client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS per client

@app.route('/')
def index():
    """Render the main web interface."""
    response = app.make_response(render_template('yolo_index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Returns MJPEG stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get current detection statistics."""
    with state.lock:
        return jsonify({
            'status': 'success',
            'fps': round(state.fps, 2),
            'inference_time_ms': round(state.inference_time_ms, 2),
            'num_detections': state.num_detections,
            'num_tracks': state.num_tracks,
            'frame_count': state.frame_count,
            'track_enabled': state.track_enabled,
            'conf_threshold': state.conf_threshold
        })

@app.route('/set_conf_threshold', methods=['POST'])
def set_conf_threshold():
    """Set confidence threshold."""
    data = request.get_json()
    threshold = float(data.get('threshold', 0.25))
    threshold = max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
    
    with state.lock:
        state.conf_threshold = threshold
    
    print(f"🎯 Confidence threshold set to {threshold}")
    return jsonify({'status': 'success', 'threshold': threshold})

@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    """Toggle object tracking on/off."""
    with state.lock:
        state.track_enabled = not state.track_enabled
    
    status = "enabled" if state.track_enabled else "disabled"
    print(f"🎯 Tracking {status}")
    return jsonify({'status': 'success', 'track_enabled': state.track_enabled})

@app.route('/restart_camera', methods=['POST'])
def restart_camera():
    """Restart the camera."""
    try:
        print("🔄 Restarting camera...")
        
        with state.lock:
            if state.camera:
                state.camera.close()
            
            # Reinitialize camera (assuming same source index)
            # You might want to store the source index in state
            # For now, we'll just try to reopen
            if state.camera:
                state.camera.open()
        
        print("  ✅ Camera restarted")
        return jsonify({'status': 'success', 'message': 'Camera restarted'})
    except Exception as e:
        print(f"  ❌ Error restarting camera: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def main():
    """Initialize and run the Flask web server."""
    parser = argparse.ArgumentParser(description="YOLO Web Streaming Server")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to RKNN model file. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="1",
        help="Camera index (e.g., 0, 1). Default: 1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Web server port. Default: 5001",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable object tracking",
    )
    
    args = parser.parse_args()
    
    capture_thread = None
    try:
        # Validate model file
        model_path = args.model.expanduser().resolve()
        if not model_path.exists():
            print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
            return 1
        
        # Initialize RKNN
        state.rknn = initialize_rknn(model_path)
        
        # Initialize camera
        source_index = int(args.source) if args.source.isdigit() else 1
        state.camera = initialize_camera(source_index)
        
        # Initialize tracker if enabled
        if not args.no_track:
            state.tracker = ByteTrackerWrapper(
                track_thresh=0.5,
                high_thresh=0.6,
                match_thresh=0.8,
                frame_rate=30,
                track_buffer=30
            )
            state.track_enabled = True
            print("✅ ByteTrack tracking enabled")
        else:
            state.track_enabled = False
            print("ℹ️  Tracking disabled")
        
        # Set confidence threshold
        state.conf_threshold = args.conf
        
        # Start background frame capture thread
        state.capturing = True
        capture_thread = Thread(target=capture_frames_continuously, daemon=True)
        capture_thread.start()
        
        # Get local IP for display
        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "localhost"
        
        print("\n" + "="*60)
        print("🌐 YOLO WEB STREAMING SERVER READY")
        print("="*60)
        print(f"📱 Open on your phone: http://{local_ip}:{args.port}")
        print(f"💻 Or locally: http://localhost:{args.port}")
        print(f"🎯 Model: {model_path.name}")
        print(f"📹 Camera: {source_index}")
        print(f"🔍 Confidence: {state.conf_threshold}")
        print(f"🎯 Tracking: {'Enabled' if state.track_enabled else 'Disabled'}")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n⚠️ Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Stop frame capture
        state.capturing = False
        if capture_thread:
            capture_thread.join(timeout=2)
        
        # Clean up
        if state.camera:
            print("Stopping camera...")
            state.camera.close()
        if state.rknn:
            print("Releasing RKNN resources...")
            state.rknn.release()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

