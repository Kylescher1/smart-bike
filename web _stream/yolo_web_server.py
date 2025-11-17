#!/usr/bin/env python3
"""
Internet-Accessible Web Server for YOLO RKNN Live Stream

This script creates a web server that streams live YOLO object detection
from your camera to the internet. Access it from any device with a browser.

Usage:
    python yolo_web_server.py --model yolo/models/yolo11n.rknn --source 0 --port 8080
    
Then access from:
    - Local network: http://<your-local-ip>:8080
    - Internet: http://<your-public-ip>:8080 (requires port forwarding)
    - Or use ngrok/tunneling service for easy internet access

Features:
    - Real-time MJPEG video streaming
    - Live object detection with tracking
    - FPS and detection statistics
    - Mobile-friendly responsive interface
    - Optional password protection
"""

import cv2
import numpy as np
import sys
import time
import threading
import argparse
import base64
from pathlib import Path
from flask import Flask, render_template_string, Response, request, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import from rknn_inference
from yolo.rknn_inference import (
    RKNNLite, FastCamera, Camera, CAMERA_CONFIG,
    process_output, draw_detections, letterbox, DEFAULT_MODEL,
    ByteTrackerWrapper
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this!

# Global state
class StreamState:
    def __init__(self):
        self.rknn = None
        self.camera = None
        self.tracker = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_jpeg = None
        self.running = False
        self.capture_thread = None
        
        # Settings
        self.conf_threshold = 0.25
        self.track_enabled = False
        self.imgsz = 640
        self.fast_capture = True
        
        # Stats
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self.num_detections = 0
        self.frame_count = 0
        self.last_stats_time = time.time()
        
        # Pre-allocated buffers
        self.img_input_buffer = None
        self.frame_timestamp = 0
        
        # Streaming settings
        self.jpeg_quality = 65
        self.max_web_width = 1280
        self.stream_fps = 30

state = StreamState()

# Optional password protection
PASSWORD_PROTECTED = False
PASSWORD_HASH = None  # Set this if you want password protection

def check_auth():
    """Check if request is authenticated."""
    if not PASSWORD_PROTECTED:
        return True
    auth = request.authorization
    if not auth:
        return False
    # Check password (username is ignored - can be anything)
    try:
        return check_password_hash(PASSWORD_HASH, auth.password)
    except Exception as e:
        print(f"[ERROR] Auth check failed: {e}", file=sys.stderr)
        return False

def requires_auth(f):
    """Decorator for routes that require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if PASSWORD_PROTECTED and not check_auth():
            return Response(
                'Could not verify your access level for that URL.\n'
                'You have to login with proper credentials', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

def initialize_rknn(model_path):
    """Initialize RKNN model."""
    print(f"[INFO] Loading RKNN model: {model_path}")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {ret}")
    ret = rknn.init_runtime(target=None, core_mask=0)
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Failed to initialize runtime: {ret}")
    print("[INFO] ✅ RKNN model loaded successfully")
    return rknn

def initialize_camera(source_index, fast_capture=True):
    """Initialize camera."""
    print(f"[INFO] 📹 Initializing camera {source_index}")
    camera_config = CAMERA_CONFIG.copy()
    camera_config.update({
        "width": 800,
        "height": 600,
        "fps": 30,
        "fourcc": "MJPG"
    })
    
    if fast_capture:
        camera = FastCamera(index=source_index, config=camera_config)
    else:
        camera = Camera(index=source_index, config=camera_config)
    camera.open()
    
    test_frame = camera.read_frame()
    if test_frame is None:
        camera.close()
        raise RuntimeError(f"Camera {source_index} opened but cannot read frames")
    
    print(f"[INFO] ✅ Camera {source_index} initialized ({camera.width}x{camera.height})")
    return camera

def capture_and_process_loop():
    """Main loop that captures frames and runs inference - optimized for streaming."""
    global state
    
    prev_time = time.time()
    frame_count = 0
    skip_frames = 0  # Skip every N frames for faster processing (0 = process all)
    
    while state.running:
        try:
            # Capture frame
            frame = state.camera.read_frame()
            if frame is None:
                time.sleep(0.005)  # Reduced sleep for faster recovery
                continue
            
            # Skip frames if processing is too slow (helps maintain frame rate)
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Preprocess
            img_resized, ratio, (dw, dh) = letterbox(frame, new_shape=(state.imgsz, state.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Pre-allocate buffer
            if state.img_input_buffer is None or state.img_input_buffer.shape != (1, state.imgsz, state.imgsz, 3):
                state.img_input_buffer = np.zeros((1, state.imgsz, state.imgsz, 3), dtype=np.uint8)
            state.img_input_buffer[0] = img_rgb.astype(np.uint8)
            
            # Run inference
            inference_start = time.time()
            try:
                outputs = state.rknn.inference([state.img_input_buffer])
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
                continue
            
            inference_time_ms = (time.time() - inference_start) * 1000
            state.inference_time_ms = inference_time_ms
            
            # Process output
            if outputs is None:
                continue
            
            detections = []
            try:
                detections = process_output(outputs, conf_threshold=state.conf_threshold, img_shape=(state.imgsz, state.imgsz))
            except Exception as e:
                print(f"[ERROR] Failed to process output: {e}", file=sys.stderr)
                detections = []
            
            # Scale boxes back to original image size
            if detections:
                h_orig, w_orig = frame.shape[:2]
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
            if state.tracker is not None:
                tracked_detections = state.tracker.update(detections)
                detections = tracked_detections
            
            # Draw detections
            annotated = draw_detections(frame.copy(), detections, tracker=state.tracker)
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            state.fps = fps
            state.num_detections = len(detections)
            frame_count += 1
            state.frame_count = frame_count
            
            # Add stats overlay
            track_info = f" | Tracks: {len(state.tracker.tracked_tracks)}" if state.tracker else ""
            info_text = f"FPS: {fps:.1f} | Inf: {inference_time_ms:.1f}ms | Det: {len(detections)}{track_info}"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode to JPEG with optimized quality for faster transfer
            # Quality 60-70 is good balance: smaller files = faster network = better update rate
            # Also resize for web if frame is too large (reduces transfer time)
            web_frame = annotated
            h, w = web_frame.shape[:2]
            if w > state.max_web_width:  # Resize if too large for web streaming
                scale = state.max_web_width / w
                new_w, new_h = int(w * scale), int(h * scale)
                web_frame = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            ret, jpeg = cv2.imencode('.jpg', web_frame, [cv2.IMWRITE_JPEG_QUALITY, state.jpeg_quality])
            if ret:
                with state.lock:
                    state.latest_frame = annotated
                    state.latest_frame_jpeg = jpeg.tobytes()
                    state.frame_timestamp = time.time()  # Track when frame was updated
            
        except Exception as e:
            print(f"[ERROR] Error in capture loop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Live Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            max-width: 1200px;
            width: 100%;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 20px;
            font-size: 2em;
        }
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        img {
            width: 100%;
            height: auto;
            display: block;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }
        button:hover {
            background: #5568d3;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .status.connected {
            background: #d4edda;
            color: #155724;
        }
        .status.disconnected {
            background: #f8d7da;
            color: #721c24;
        }
        @media (max-width: 768px) {
            h1 {
                font-size: 1.5em;
            }
            .stats {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 YOLO Live Stream</h1>
        <div class="video-container">
            <img id="stream" src="/video_feed" alt="Live Stream">
        </div>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0.0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Inference Time</div>
                <div class="stat-value" id="inference">0.0ms</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Detections</div>
                <div class="stat-value" id="detections">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Frames</div>
                <div class="stat-value" id="frames">0</div>
            </div>
        </div>
        <div class="controls">
            <button onclick="updateStats()">Refresh Stats</button>
            <button onclick="location.reload()">Reload Stream</button>
        </div>
        <div class="status connected" id="status">
            ✅ Connected
        </div>
    </div>
    
    <script>
        let errorCount = 0;
        const maxErrors = 5;
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('inference').textContent = data.inference_time_ms.toFixed(1) + 'ms';
                    document.getElementById('detections').textContent = data.num_detections;
                    document.getElementById('frames').textContent = data.frame_count;
                })
                .catch(err => console.error('Error fetching stats:', err));
        }
        
        const img = document.getElementById('stream');
        img.onerror = function() {
            errorCount++;
            if (errorCount >= maxErrors) {
                document.getElementById('status').textContent = '❌ Connection Lost';
                document.getElementById('status').className = 'status disconnected';
            }
            setTimeout(() => {
                img.src = '/video_feed?t=' + new Date().getTime();
            }, 1000);
        };
        
        img.onload = function() {
            errorCount = 0;
            document.getElementById('status').textContent = '✅ Connected';
            document.getElementById('status').className = 'status connected';
        };
        
        // Update stats every 2 seconds
        setInterval(updateStats, 2000);
        updateStats();
    </script>
</body>
</html>
"""

@app.route('/')
@requires_auth
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
@requires_auth
def video_feed():
    """MJPEG video stream endpoint - optimized for high frame rate."""
    def generate():
        last_frame_time = 0
        last_frame_timestamp = 0
        frame_interval = 1.0 / state.stream_fps
        
        while True:
            current_time = time.time()
            with state.lock:
                frame_jpeg = state.latest_frame_jpeg
                frame_timestamp = getattr(state, 'frame_timestamp', 0)
            
            if frame_jpeg:
                # Send frame if:
                # 1. Enough time has passed (rate limiting), OR
                # 2. This is a new frame (timestamp changed - always send latest)
                is_new_frame = frame_timestamp > last_frame_timestamp
                time_elapsed = (current_time - last_frame_time) >= frame_interval
                
                if is_new_frame or time_elapsed:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
                    last_frame_time = current_time
                    last_frame_timestamp = frame_timestamp
                else:
                    # Small sleep to prevent CPU spinning when waiting
                    time.sleep(0.001)
            else:
                # No frame available yet, wait a bit
                time.sleep(0.01)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
@requires_auth
def stats():
    """Get current statistics."""
    return jsonify({
        'fps': state.fps,
        'inference_time_ms': state.inference_time_ms,
        'num_detections': state.num_detections,
        'frame_count': state.frame_count,
        'track_enabled': state.track_enabled
    })

@app.route('/update_settings', methods=['POST'])
@requires_auth
def update_settings():
    """Update detection settings."""
    data = request.json
    if 'conf_threshold' in data:
        state.conf_threshold = float(data['conf_threshold'])
    if 'track_enabled' in data:
        state.track_enabled = bool(data['track_enabled'])
        if state.track_enabled and state.tracker is None:
            state.tracker = ByteTrackerWrapper(
                track_thresh=0.5,
                high_thresh=0.6,
                match_thresh=0.8,
                frame_rate=30,
                track_buffer=30
            )
        elif not state.track_enabled:
            state.tracker = None
    return jsonify({'status': 'ok'})

def main():
    parser = argparse.ArgumentParser(description="Internet-accessible YOLO web stream server")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to RKNN model file. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g., 0, 1). Default: 0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Web server port. Default: 8080",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to. Use 0.0.0.0 for internet access. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable object tracking",
    )
    parser.add_argument(
        "--no-fast-capture",
        dest="fast_capture",
        action="store_false",
        help="Disable fast threaded capture",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Set password for web access (optional)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=65,
        help="JPEG quality for web streaming (1-100, lower = faster). Default: 65",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum width for web streaming (reduces bandwidth). Default: 1280",
    )
    parser.add_argument(
        "--stream-fps",
        type=int,
        default=30,
        help="Target FPS for web streaming. Default: 30",
    )
    
    args = parser.parse_args()
    
    # Set password if provided
    global PASSWORD_PROTECTED, PASSWORD_HASH
    if args.password:
        PASSWORD_PROTECTED = True
        PASSWORD_HASH = generate_password_hash(args.password)
        print(f"[INFO] 🔒 Password protection enabled (password: '{args.password}')")
        print(f"[INFO] 💡 Username can be anything, password must be: '{args.password}'")
    
    # Initialize RKNN
    try:
        state.rknn = initialize_rknn(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to initialize RKNN: {e}", file=sys.stderr)
        return 1
    
    # Initialize camera
    source = int(args.source) if args.source.isdigit() else 0
    try:
        state.camera = initialize_camera(source, fast_capture=args.fast_capture)
        state.fast_capture = args.fast_capture
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}", file=sys.stderr)
        state.rknn.release()
        return 1
    
    # Initialize tracker if enabled
    if args.track:
        state.tracker = ByteTrackerWrapper(
            track_thresh=0.5,
            high_thresh=0.6,
            match_thresh=0.8,
            frame_rate=30,
            track_buffer=30
        )
        state.track_enabled = True
        print("[INFO] ✅ Tracking enabled")
    
    state.conf_threshold = args.conf
    state.jpeg_quality = args.jpeg_quality
    state.max_web_width = args.max_width
    state.stream_fps = args.stream_fps
    state.running = True
    
    # Start capture thread
    state.capture_thread = threading.Thread(target=capture_and_process_loop, daemon=True)
    state.capture_thread.start()
    
    # Get local IP address
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print("\n" + "="*60)
    print("🌐 YOLO Web Server Started!")
    print("="*60)
    print(f"📡 Local access:  http://{local_ip}:{args.port}")
    print(f"📡 Localhost:     http://127.0.0.1:{args.port}")
    print(f"📡 Internet:      http://<your-public-ip>:{args.port}")
    print("\n💡 To access from internet:")
    print("   1. Configure port forwarding on your router (port {})".format(args.port))
    print("   2. Or use ngrok: ngrok http {}".format(args.port))
    print("   3. Or use Cloudflare Tunnel, localtunnel, etc.")
    print("="*60 + "\n")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        state.running = False
        if state.capture_thread:
            state.capture_thread.join(timeout=2.0)
        if state.camera:
            state.camera.close()
        if state.rknn:
            state.rknn.release()
        print("[INFO] ✅ Cleanup complete")

if __name__ == "__main__":
    sys.exit(main())

