#!/usr/bin/env python3
"""
Web streaming server for depth camera visualization.
Access from your phone to view depth maps and tune calibration parameters.

Usage:
    python web_stream.py
    
Then open http://<your-laptop-ip>:5000 on your phone
"""

import cv2
import numpy as np
import dill
import sys
import os
import importlib.util
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from threading import Lock, Thread
import time
from datetime import datetime

# Make sure src folder is on sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

app = Flask(__name__)

# Global state
class StreamState:
    def __init__(self):
        self.vision = None
        self.camera_config = None
        self.mode = "debug"  # "debug", "calibrate", or "calibrate_maps"
        self.view = "depth"  # "depth", "disparity", "left", "right", or "left_overlay"
        self.lock = Lock()
        self.latest_frame = None
        self.latest_frame_bytes = None
        self.parameters = {}
        self.frame_ready = False
        self.capturing = False
        # Recording state
        self.recording = False
        self.recording_writer = None
        self.recording_path = None
        self.recording_start_time = None
        self.recording_view = None
        # Calibration state
        self.calibrating_maps = False
        self.captured_pairs = []
        self.checkerboard_size = (7, 10)
        self.square_size_mm = 20.0
        self.min_pairs = 5
        # Colormap
        self.colormap = cv2.COLORMAP_JET  # Default colormap
        self.invert_colormap = False  # Invert colormap colors
        # Auto-calibration state
        self.autocal_active = False
        self.autocal_baseline_params = None
        self.autocal_current_params = None
        self.autocal_profile_a = None
        self.autocal_profile_b = None
        self.autocal_current_param_name = None
        self.autocal_param_list = []
        self.autocal_param_index = 0
        self.autocal_iteration = 0
        self.autocal_best_params = None
        self.autocal_waiting_for_choice = False
        
state = StreamState()

def load_config():
    """Load config.dill and initialize camera configuration."""
    config_path = "config.dill"
    print("Loading Config...")
    try:
        with open(config_path, "rb") as f:
            config = dill.load(f)
        print("✅ Loaded whole Dill")
        camera_config = config['camera']
        print("✅ Loaded Camera Config")
        return camera_config
    except Exception as e:
        raise KeyError(f"An unexpected error occurred loading config.dill: {e}")

def initialize_vision(camera_config):
    """Initialize the vision system."""
    print("\n" + "="*60)
    print("Initializing Vision System...")
    print("="*60)
    
    # Import and load the vision class
    module_path, class_name = camera_config['who_to_run'].rsplit(".", 1)
    spec = importlib.util.spec_from_file_location(
        module_path, 
        os.path.join(os.path.dirname(__file__), *module_path.split(".")) + ".py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    VisionClass = getattr(module, class_name)
    
    # Create vision instance
    vision = VisionClass(name="camera", **camera_config)
    
    # Start the vision system
    vision.start()
    print("✅ Vision system started")
    
    return vision

def generate_camera_frame(camera_side):
    """Generate a frame from left or right camera."""
    if state.vision is None or not state.vision.connected:
        return None
    
    try:
        # Get raw frame from the appropriate camera
        if camera_side == "left":
            frame = state.vision.left_camera.read_frame()
            label = "LEFT CAMERA"
        else:  # right
            frame = state.vision.right_camera.read_frame()
            label = "RIGHT CAMERA"
        
        if frame is None:
            return None
        
        # Convert to color if grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Add label
        cv2.putText(frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
        
    except Exception as e:
        print(f"Error generating {camera_side} camera frame: {e}")
        return None

def generate_stereo_calibration_frame():
    """Generate side-by-side camera view for calibration with checkerboard detection."""
    if state.vision is None or not state.vision.connected:
        return None
    
    try:
        # Get frames from both cameras
        left_frame = state.vision.left_camera.read_frame()
        right_frame = state.vision.right_camera.read_frame()
        
        if left_frame is None or right_frame is None:
            return None
        
        # Convert to grayscale for checkerboard detection
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Try to detect checkerboard
        retL, cornersL = cv2.findChessboardCorners(gray_left, state.checkerboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(gray_right, state.checkerboard_size, None)
        
        # Draw checkerboard if detected
        if retL:
            cv2.drawChessboardCorners(left_frame, state.checkerboard_size, cornersL, retL)
            cv2.putText(left_frame, "DETECTED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(left_frame, "NO PATTERN", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if retR:
            cv2.drawChessboardCorners(right_frame, state.checkerboard_size, cornersR, retR)
            cv2.putText(right_frame, "DETECTED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(right_frame, "NO PATTERN", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(left_frame, "LEFT CAMERA", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right_frame, "RIGHT CAMERA", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize to half width each
        h, w = left_frame.shape[:2]
        left_half = cv2.resize(left_frame, (w // 2, h // 2))
        right_half = cv2.resize(right_frame, (w // 2, h // 2))
        
        # Combine side by side
        combined = np.hstack([left_half, right_half])
        
        # Add progress overlay
        pairs_count = len(state.captured_pairs)
        progress_text = f"Captured: {pairs_count}/{state.min_pairs}"
        cv2.putText(combined, progress_text, (combined.shape[1]//2 - 100, combined.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return combined
        
    except Exception as e:
        print(f"Error generating stereo calibration frame: {e}")
        return None

def generate_debug_frame():
    """Generate a single debug frame (simple depth visualization)."""
    if state.vision is None or not state.vision.connected:
        return None
    
    # Check if in calibration mode
    if state.calibrating_maps:
        return generate_stereo_calibration_frame()
    
    # Check which view to show
    if state.view == "left":
        return generate_camera_frame("left")
    elif state.view == "right":
        return generate_camera_frame("right")
    
    # Otherwise show depth map
    try:
        # Get depth data
        result = state.vision.read()
        depth_map = result.get('depth_map')
        disparity_map = result.get('disparity_map')
        metadata = result.get('metadata', {})
        
        # Select data based on view
        if state.view == "disparity":
            data_map = disparity_map
            label = "DISPARITY MAP"
        elif state.view == "left_overlay":
            data_map = depth_map
            label = "LEFT + DEPTH OVERLAY"
        else:
            data_map = depth_map
            label = "DEPTH MAP"
        
        # Check for errors
        if ('error' in metadata 
            or data_map is None 
            or not hasattr(data_map, 'size') 
            or data_map.size == 0):
            return None
        
        data_map = np.nan_to_num(data_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize depth map for visualization (0-255 range)
        normalized = cv2.normalize(data_map, None, 0, 255, cv2.NORM_MINMAX)
        display = normalized.astype(np.uint8)
        
        # Invert if needed
        if state.invert_colormap:
            display = 255 - display
        
        # Apply colormap for better visualization
        colored = cv2.applyColorMap(display, state.colormap)

        # Prepare final frame depending on view
        frame_to_show = colored
        if state.view == "left_overlay":
            left_frame = state.vision.left_camera.read_frame()
            if left_frame is None:
                return None
            if len(left_frame.shape) == 2:
                left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
            h, w = left_frame.shape[:2]
            if colored.shape[0] != h or colored.shape[1] != w:
                colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)
            frame_to_show = cv2.addWeighted(left_frame, 0.6, colored, 0.4, 0)

        # Add metadata text overlay
        timestamp_raw = metadata.get('timestamp', 'N/A')
        timestamp_str = str(timestamp_raw)
        timestamp = timestamp_str[-8:] if len(timestamp_str) >= 8 else timestamp_str
        num_disp = metadata.get('num_disparities', 'N/A')
        
        cv2.putText(frame_to_show, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_to_show, f"Time: {timestamp}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_to_show, f"Disparities: {num_disp}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if state.view == "disparity":
            nonzero = data_map[data_map > 0]
            if nonzero.size > 0:
                disp_min = float(np.min(nonzero))
                disp_max = float(np.max(nonzero))
                cv2.putText(frame_to_show, f"Range: {disp_min:.2f} - {disp_max:.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_to_show
        
    except Exception as e:
        print(f"Error generating debug frame: {e}")
        return None

def generate_calibrate_frame():
    """Generate a calibration frame with current parameters applied."""
    if state.vision is None or not state.vision.connected:
        return None
    
    # Check if in calibration mode
    if state.calibrating_maps:
        return generate_stereo_calibration_frame()
    
    # Check which view to show
    if state.view == "left":
        return generate_camera_frame("left")
    elif state.view == "right":
        return generate_camera_frame("right")
    
    # Otherwise show depth map
    try:
        # Get depth data with current parameters
        result = state.vision.read()
        depth_map = result.get('depth_map')
        disparity_map = result.get('disparity_map')
        metadata = result.get('metadata', {})
        
        if state.view == "disparity":
            data_map = disparity_map
            label = "DISPARITY MAP - CALIBRATION"
        elif state.view == "left_overlay":
            data_map = depth_map
            label = "LEFT + DEPTH OVERLAY - CALIBRATION"
        else:
            data_map = depth_map
            label = "DEPTH MAP - CALIBRATION"
        
        # Check for errors
        if ('error' in metadata 
            or data_map is None 
            or not hasattr(data_map, 'size') 
            or data_map.size == 0):
            return None
        
        data_map = np.nan_to_num(data_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize depth map for visualization
        normalized = cv2.normalize(data_map, None, 0, 255, cv2.NORM_MINMAX)
        display = normalized.astype(np.uint8)
        
        # Invert if needed
        if state.invert_colormap:
            display = 255 - display
        
        display_colored = cv2.applyColorMap(display, state.colormap)

        # Prepare final frame depending on view
        frame_to_show = display_colored
        if state.view == "left_overlay":
            left_frame = state.vision.left_camera.read_frame()
            if left_frame is None:
                return None
            if len(left_frame.shape) == 2:
                left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
            h, w = left_frame.shape[:2]
            if display_colored.shape[0] != h or display_colored.shape[1] != w:
                display_colored = cv2.resize(display_colored, (w, h), interpolation=cv2.INTER_LINEAR)
            frame_to_show = cv2.addWeighted(left_frame, 0.6, display_colored, 0.4, 0)
        
        # Add calibration mode overlay
        cv2.putText(frame_to_show, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if state.view != "left_overlay" else (255, 255, 255), 2)
        
        # Show key parameters
        y_offset = 60
        params_to_show = [
            ('blockSize', state.vision.blockSize),
            ('numDisparities', state.vision.numDisparities),
            ('uniqueness', state.vision.uniquenessRatio),
            ('WLS', 'ON' if state.vision.useWLS else 'OFF'),
            ('speckleWindowSize', state.vision.speckleWindowSize),
            ('speckleRange', state.vision.speckleRange),
        ]
        
        for param_name, param_value in params_to_show:
            cv2.putText(frame_to_show, f"{param_name}: {param_value}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        if state.view == "disparity":
            nonzero = data_map[data_map > 0]
            if nonzero.size > 0:
                cv2.putText(frame_to_show, f"Range: {np.min(nonzero):.2f} - {np.max(nonzero):.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        return frame_to_show
        
    except Exception as e:
        print(f"Error generating calibrate frame: {e}")
        return None

def capture_frames_continuously():
    """Background thread that continuously captures and encodes frames."""
    print("📹 Frame capture thread started")
    while state.capturing:
        try:
            with state.lock:
                if state.mode == "debug":
                    frame = generate_debug_frame()
                else:  # calibrate mode
                    frame = generate_calibrate_frame()
            
            if frame is not None:
                with state.lock:
                    state.latest_frame = frame
                    if state.recording:
                        if state.recording_writer is None:
                            try:
                                Path("recordings").mkdir(parents=True, exist_ok=True)
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                writer = cv2.VideoWriter(
                                    state.recording_path,
                                    fourcc,
                                    30.0,
                                    (w, h)
                                )
                                if not writer.isOpened():
                                    print(f"❌ Failed to open video writer for {state.recording_path}")
                                    writer.release()
                                    state.recording = False
                                    state.recording_path = None
                                    state.recording_start_time = None
                                else:
                                    state.recording_writer = writer
                                    print(f"🎥 Recording initialized: {state.recording_path}")
                            except Exception as err:
                                print(f"❌ Error initializing recording: {err}")
                                state.recording = False
                                state.recording_path = None
                                state.recording_start_time = None
                        if state.recording_writer is not None:
                            state.recording_writer.write(frame)
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    with state.lock:
                        state.latest_frame_bytes = buffer.tobytes()
                        state.frame_ready = True
        except Exception as e:
            print(f"Error in frame capture: {e}")
        
        time.sleep(0.033)  # ~30 FPS
    # Cleanup recording writer if still open
    with state.lock:
        writer = state.recording_writer
        state.recording_writer = None
        state.recording = False
        state.recording_path = None
        state.recording_start_time = None
        state.recording_view = None
    if writer is not None:
        writer.release()
        print("🎥 Recording writer released on capture stop")
    print("📹 Frame capture thread stopped")

def generate_frames():
    """Generator function that yields frames from the shared buffer.
    Multiple clients can call this simultaneously."""
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
    response = app.make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Returns MJPEG stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Switch between debug, calibrate, and autocal modes."""
    data = request.get_json()
    new_mode = data.get('mode', 'debug')
    
    if new_mode in ['debug', 'calibrate', 'autocal']:
        with state.lock:
            state.mode = new_mode
        return jsonify({'status': 'success', 'mode': state.mode})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400

@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    """Cycle through camera views: depth -> disparity -> left -> right -> left overlay -> depth."""
    with state.lock:
        views = ["depth", "disparity", "left", "right", "left_overlay"]
        try:
            current_index = views.index(state.view)
        except ValueError:
            current_index = 0
        state.view = views[(current_index + 1) % len(views)]
    
    return jsonify({'status': 'success', 'view': state.view})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start saving the active view to disk."""
    with state.lock:
        if not state.capturing:
            return jsonify({'status': 'error', 'message': 'Stream not running'}), 400
        if state.recording:
            return jsonify({'status': 'error', 'message': 'Recording already in progress'}), 400
        
        Path("recordings").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{state.view}.mp4"
        recording_path = str(Path("recordings") / filename)
        
        state.recording = True
        state.recording_writer = None  # initialized on next frame
        state.recording_path = recording_path
        state.recording_start_time = time.time()
        state.recording_view = state.view
    
    print(f"🎥 Recording requested: {recording_path}")
    return jsonify({'status': 'success', 'path': recording_path})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording and finalize the file."""
    with state.lock:
        if not state.recording:
            return jsonify({'status': 'error', 'message': 'No active recording'}), 400
        
        writer = state.recording_writer
        recording_path = state.recording_path
        duration = 0.0
        if state.recording_start_time is not None:
            duration = time.time() - state.recording_start_time
        
        state.recording = False
        state.recording_writer = None
        state.recording_path = None
        state.recording_start_time = None
        state.recording_view = None
    
    if writer is not None:
        writer.release()
    print(f"💾 Recording saved: {recording_path}")
    
    return jsonify({'status': 'success', 'path': recording_path, 'duration': duration})

@app.route('/restart_vision', methods=['POST'])
def restart_vision():
    """Restart the vision system (cameras and depth processing)."""
    try:
        print("🔄 Restarting vision system...")
        
        with state.lock:
            if state.vision and state.vision.connected:
                state.vision.stop()
                print("  Stopped cameras")
            
            # Restart
            state.vision.start()
            print("  ✅ Vision system restarted")
        
        return jsonify({'status': 'success', 'message': 'Vision system restarted'})
    except Exception as e:
        print(f"  ❌ Error restarting: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/start_map_calibration', methods=['POST'])
def start_map_calibration():
    """Start the map calibration process."""
    with state.lock:
        state.mode = "calibrate_maps"
        state.calibrating_maps = True
        state.captured_pairs = []
    
    print("📐 Starting map calibration mode")
    return jsonify({
        'status': 'success', 
        'pairs_captured': 0, 
        'min_pairs': state.min_pairs,
        'checkerboard_size': f"{state.checkerboard_size[0]}x{state.checkerboard_size[1]}"
    })

@app.route('/capture_calibration_pair', methods=['POST'])
def capture_calibration_pair():
    """Capture a stereo pair with checkerboard detection."""
    if not state.calibrating_maps:
        return jsonify({'status': 'error', 'message': 'Not in calibration mode'}), 400
    
    try:
        # Get frames from cameras
        left_frame = state.vision.left_camera.read_frame()
        right_frame = state.vision.right_camera.read_frame()
        
        if left_frame is None or right_frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to grab frames'}), 500
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find checkerboard
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-6)
        retL, cornersL = cv2.findChessboardCorners(gray_left, state.checkerboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(gray_right, state.checkerboard_size, None)
        
        if retL and retR:
            # Refine corners
            cornersL_refined = cv2.cornerSubPix(gray_left, cornersL, (11, 11), (-1, -1), criteria)
            cornersR_refined = cv2.cornerSubPix(gray_right, cornersR, (11, 11), (-1, -1), criteria)
            
            with state.lock:
                state.captured_pairs.append((gray_left.copy(), gray_right.copy(), cornersL_refined, cornersR_refined))
            
            pairs_count = len(state.captured_pairs)
            print(f"✅ Captured pair {pairs_count}: Checkerboard detected in both cameras")
            
            return jsonify({
                'status': 'success',
                'pairs_captured': pairs_count,
                'min_pairs': state.min_pairs,
                'ready': pairs_count >= state.min_pairs
            })
        else:
            missing = []
            if not retL:
                missing.append('left')
            if not retR:
                missing.append('right')
            
            return jsonify({
                'status': 'error',
                'message': f'Checkerboard not detected in {" and ".join(missing)} camera(s)',
                'pairs_captured': len(state.captured_pairs)
            }), 400
            
    except Exception as e:
        print(f"❌ Error capturing pair: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def process_stereo_calibration(captured_pairs, checkerboard_size, square_size_mm, img_shape):
    """Process captured stereo pairs and return calibration data."""
    print(f"\n📸 Processing {len(captured_pairs)} captured stereo pairs...")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-6)
    
    # Prepare object points
    objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    
    objpoints, imgpointsL, imgpointsR = [], [], []
    
    for i, (imgL, imgR, cornersL, cornersR) in enumerate(captured_pairs):
        objpoints.append(objp)
        imgpointsL.append(cornersL.reshape(1, -1, 2))
        imgpointsR.append(cornersR.reshape(1, -1, 2))
        print(f"  ✓ Processed pair {i + 1}/{len(captured_pairs)}")
    
    N_OK = len(objpoints)
    print(f"✅ Using {N_OK} valid pairs for calibration")
    
    # Initialize camera matrices
    K1 = np.eye(3)
    D1 = np.zeros((4, 1))
    K2 = np.eye(3)
    D2 = np.zeros((4, 1))
    
    print("\n--- Stereo Calibration (Fisheye) ---")
    
    # Perform stereo calibration
    rms, K1, D1, K2, D2, R, T, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1,
        D1,
        K2,
        D2,
        img_shape,
        criteria=criteria,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    )
    
    print(f"RMS reprojection error: {rms:.4f}")
    
    # Stereo rectification
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        img_shape,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0,
        fov_scale=1.2,
    )
    
    # Generate undistort/rectify maps
    leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, img_shape, cv2.CV_32FC1
    )
    rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, img_shape, cv2.CV_32FC1
    )
    
    print(f"\n💾 Calibration complete")
    print(f"   Maps shape: {leftMapX.shape}")
    print(f"   Image size: {img_shape}")
    
    return {
        'leftMapX': leftMapX,
        'leftMapY': leftMapY,
        'rightMapX': rightMapX,
        'rightMapY': rightMapY,
        'Q': Q,
        'imageSize': img_shape
    }

@app.route('/finish_map_calibration', methods=['POST'])
def finish_map_calibration():
    """Process captured pairs and perform stereo calibration."""
    if not state.calibrating_maps:
        return jsonify({'status': 'error', 'message': 'Not in calibration mode'}), 400
    
    if len(state.captured_pairs) < state.min_pairs:
        return jsonify({
            'status': 'error',
            'message': f'Need at least {state.min_pairs} pairs, only have {len(state.captured_pairs)}'
        }), 400
    
    try:
        print(f"\n📐 Processing {len(state.captured_pairs)} captured pairs...")
        
        # Get image shape from first captured pair
        img_shape = state.captured_pairs[0][0].shape[::-1]  # (width, height)
        
        # Run calibration processing
        result = process_stereo_calibration(
            state.captured_pairs,
            state.checkerboard_size,
            state.square_size_mm,
            img_shape
        )
        
        # Update config with new calibration
        with state.lock:
            state.camera_config['left']['map_x'] = result['leftMapX']
            state.camera_config['left']['map_y'] = result['leftMapY']
            state.camera_config['right']['map_x'] = result['rightMapX']
            state.camera_config['right']['map_y'] = result['rightMapY']
            state.camera_config['imageSize'] = result['imageSize']
            state.camera_config['Q'] = result['Q']
            
            # Save to config.dill
            config_path = "config.dill"
            with open(config_path, "rb") as f:
                full_config = dill.load(f)
            
            full_config['camera'] = state.camera_config
            
            with open(config_path, "wb") as f:
                dill.dump(full_config, f)
            
            # Reset calibration state
            state.calibrating_maps = False
            state.captured_pairs = []
            state.mode = "debug"
            state.view = "depth"
        
        # Restart vision with new calibration
        state.vision.stop()
        state.vision = initialize_vision(state.camera_config)
        
        print("✅ Calibration complete and saved!")
        
        return jsonify({'status': 'success', 'message': 'Calibration complete!'})
        
    except Exception as e:
        print(f"❌ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/cancel_map_calibration', methods=['POST'])
def cancel_map_calibration():
    """Cancel the map calibration process."""
    with state.lock:
        state.calibrating_maps = False
        state.captured_pairs = []
        state.mode = "debug"
        state.view = "depth"
    
    print("❌ Map calibration cancelled")
    return jsonify({'status': 'success'})

@app.route('/set_colormap', methods=['POST'])
def set_colormap():
    """Change the depth map colormap."""
    data = request.get_json()
    colormap_name = data.get('colormap', 'JET')
    
    # Map colormap names to OpenCV constants
    colormap_dict = {
        'JET': cv2.COLORMAP_JET,
        'BONE': cv2.COLORMAP_BONE,
        'HOT': cv2.COLORMAP_HOT,
        'RAINBOW': cv2.COLORMAP_RAINBOW,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'MAGMA': cv2.COLORMAP_MAGMA,
        'COOL': cv2.COLORMAP_COOL,
        'SPRING': cv2.COLORMAP_SPRING,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'AUTUMN': cv2.COLORMAP_AUTUMN,
        'WINTER': cv2.COLORMAP_WINTER,
        'OCEAN': cv2.COLORMAP_OCEAN,
        'PINK': cv2.COLORMAP_PINK,
        'HSV': cv2.COLORMAP_HSV,
        'PARULA': cv2.COLORMAP_PARULA,
        'TURBO': cv2.COLORMAP_TURBO,
    }
    
    if colormap_name in colormap_dict:
        with state.lock:
            state.colormap = colormap_dict[colormap_name]
        print(f"🎨 Colormap changed to {colormap_name}")
        return jsonify({'status': 'success', 'colormap': colormap_name})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid colormap'}), 400

@app.route('/toggle_colormap_invert', methods=['POST'])
def toggle_colormap_invert():
    """Toggle colormap inversion."""
    data = request.get_json()
    invert = data.get('invert', False)
    
    with state.lock:
        state.invert_colormap = invert
    
    print(f"🎨 Colormap invert: {'ON' if invert else 'OFF'}")
    return jsonify({'status': 'success', 'invert': state.invert_colormap})

@app.route('/get_parameters', methods=['GET'])
def get_parameters():
    """Get current calibration parameters."""
    if state.vision is None:
        return jsonify({'status': 'error', 'message': 'Vision not initialized'}), 500
    
    params = {
        # Stereo block matcher core parameters
        'minDisparity': state.vision.minDisparity,
        'numDisparitiesK': state.vision.numDisparitiesK,
        'numDisparities': state.vision.numDisparities,
        'blockSize': state.vision.blockSize,
        'P1': getattr(state.vision, 'P1', 968),
        'P2': getattr(state.vision, 'P2', 3872),
        'preFilterCap': state.vision.preFilterCap,
        'uniquenessRatio': state.vision.uniquenessRatio,
        'speckleWindowSize': state.vision.speckleWindowSize,
        'speckleRange': state.vision.speckleRange,
        'disp12MaxDiff': state.vision.disp12MaxDiff,
        'sgbmMode': getattr(state.vision, 'sgbmMode', 2),
        
        # Pre-processing & scaling
        'medianBlurK': state.vision.medianBlurK,
        'downSample': state.vision.downSample,
        'crop': state.vision.crop,
        'farEnhance': state.vision.farEnhance,
        'nearCutoff': state.vision.nearCutoff,
        'farCutoff': state.vision.farCutoff,
        
        # Filtering
        'useMorph': state.vision.useMorph,
        'morphIter': state.vision.morphIter,
        'useBilateral': state.vision.useBilateral,
        'bilateralStrength': state.vision.bilateralStrength,
        'useWLS': state.vision.useWLS,
        'wlsLambda': state.vision.wlsLambda,
        'wlsSigma': state.vision.wlsSigma,
        'smoothingKernel': getattr(state.vision, 'smoothingKernel', 0),
        'confidenceWindow': getattr(state.vision, 'confidenceWindow', 5),
        'confidenceThreshold': getattr(state.vision, 'confidenceThreshold', 0.0),
    }
    
    return jsonify({'status': 'success', 'parameters': params})

@app.route('/update_parameter', methods=['POST'])
def update_parameter():
    """Update a single calibration parameter."""
    data = request.get_json()
    param_name = data.get('name')
    param_value = data.get('value')
    
    if state.vision is None:
        return jsonify({'status': 'error', 'message': 'Vision not initialized'}), 500
    
    try:
        with state.lock:
            # Handle boolean parameters
            if param_name in ['useMorph', 'useBilateral', 'useWLS']:
                param_value = bool(param_value)
            # Handle float parameters
            elif param_name in ['wlsSigma', 'confidenceThreshold']:
                param_value = float(param_value)
            else:
                param_value = int(param_value)
            
            # Update the parameter
            setattr(state.vision, param_name, param_value)
            
            # If core stereo parameters changed, recreate stereo matcher
            stereo_params = ['minDisparity', 'numDisparitiesK', 'numDisparities', 
                           'blockSize', 'P1', 'P2', 'preFilterCap', 'uniquenessRatio', 
                           'speckleWindowSize', 'speckleRange', 'disp12MaxDiff', 'sgbmMode']
            
            if param_name in stereo_params:
                block_size = state.vision.blockSize
                block_size = block_size if block_size % 2 == 1 else block_size + 1
                num_disparities = max(16, 16 * state.vision.numDisparitiesK)
                
                # Get P1 and P2 with defaults
                P1 = getattr(state.vision, 'P1', 8 * 1 * block_size * block_size)
                P2 = getattr(state.vision, 'P2', 32 * 1 * block_size * block_size)
                
                # Map mode integer to OpenCV enum
                sgbm_mode = getattr(state.vision, 'sgbmMode', 2)
                mode_map = {
                    0: cv2.STEREO_SGBM_MODE_SGBM,
                    1: cv2.STEREO_SGBM_MODE_HH,
                    2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                }
                mode = mode_map.get(sgbm_mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                
                state.vision.stereo = cv2.StereoSGBM_create(
                    minDisparity=state.vision.minDisparity,
                    numDisparities=num_disparities,
                    blockSize=max(3, block_size),
                    P1=P1,
                    P2=P2,
                    preFilterCap=state.vision.preFilterCap,
                    uniquenessRatio=state.vision.uniquenessRatio,
                    speckleWindowSize=state.vision.speckleWindowSize,
                    speckleRange=state.vision.speckleRange,
                    disp12MaxDiff=state.vision.disp12MaxDiff,
                    mode=mode,
                )
            
            # Refresh depth processor with updated parameters
            state.vision._refresh_depth_processor()
        
        return jsonify({'status': 'success', 'name': param_name, 'value': param_value})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/save_parameters', methods=['POST'])
def save_parameters():
    """Save current parameters to config.dill."""
    if state.vision is None or state.camera_config is None:
        return jsonify({'status': 'error', 'message': 'Vision not initialized'}), 500
    
    try:
        # Update camera_config with current parameters
        params_to_save = [
            'minDisparity', 'numDisparitiesK', 'numDisparities', 'blockSize',
            'P1', 'P2', 'preFilterCap', 'uniquenessRatio', 'speckleWindowSize', 'speckleRange',
            'disp12MaxDiff', 'sgbmMode', 'medianBlurK', 'downSample', 'crop', 'farEnhance',
            'nearCutoff', 'farCutoff', 'useMorph', 'morphIter', 'useBilateral',
            'bilateralStrength', 'useWLS', 'wlsLambda', 'wlsSigma',
            'smoothingKernel', 'confidenceWindow', 'confidenceThreshold',
        ]
        
        for param_name in params_to_save:
            if hasattr(state.vision, param_name):
                state.camera_config[param_name] = getattr(state.vision, param_name)
        
        # Load full config, update camera section, and save
        config_path = "config.dill"
        with open(config_path, "rb") as f:
            config = dill.load(f)
        
        config['camera'] = state.camera_config
        
        with open(config_path, "wb") as f:
            dill.dump(config, f)
        
        print("✅ Parameters saved to config.dill")
        return jsonify({'status': 'success', 'message': 'Parameters saved successfully'})
        
    except Exception as e:
        print(f"❌ Error saving parameters: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def _get_param_ranges():
    """Get parameter ranges and step sizes for auto-calibration."""
    return {
        'minDisparity': {'min': 0, 'max': 100, 'step': 5, 'default': 0},
        'blockSize': {'min': 5, 'max': 51, 'step': 2, 'default': 11},
        'numDisparitiesK': {'min': 1, 'max': 16, 'step': 1, 'default': 2},
        'P1': {'min': 0, 'max': 10000, 'step': 200, 'default': 968},
        'P2': {'min': 0, 'max': 50000, 'step': 1000, 'default': 3872},
        'preFilterCap': {'min': 1, 'max': 100, 'step': 5, 'default': 43},
        'uniquenessRatio': {'min': 0, 'max': 100, 'step': 5, 'default': 1},
        'speckleWindowSize': {'min': 0, 'max': 200, 'step': 10, 'default': 196},
        'speckleRange': {'min': 0, 'max': 64, 'step': 2, 'default': 34},
        'disp12MaxDiff': {'min': 0, 'max': 100, 'step': 5, 'default': 18},
        'sgbmMode': {'min': 0, 'max': 2, 'step': 1, 'default': 2},
        'downSample': {'min': 0, 'max': 100, 'step': 5, 'default': 57},
        'crop': {'min': 0, 'max': 300, 'step': 10, 'default': 128},
        'nearCutoff': {'min': 0, 'max': 100, 'step': 5, 'default': 72},
        'farCutoff': {'min': 0, 'max': 100, 'step': 5, 'default': 5},
        'wlsLambda': {'min': 100, 'max': 10000, 'step': 500, 'default': 2389},
        'wlsSigma': {'min': 0.1, 'max': 5.0, 'step': 0.2, 'default': 2.1},
        'morphIter': {'min': 1, 'max': 20, 'step': 1, 'default': 5},
    }

def _apply_params_to_vision(params):
    """Apply a parameter set to the vision system."""
    print(f"[AUTOCAL] _apply_params_to_vision called with {len(params)} parameters")
    
    if state.vision is None:
        print("[AUTOCAL] ERROR: Vision is None")
        return False
    
    try:
        print("[AUTOCAL] Acquiring lock for _apply_params_to_vision...")
        with state.lock:
            print("[AUTOCAL] Lock acquired in _apply_params_to_vision")
            for param_name, param_value in params.items():
                setattr(state.vision, param_name, param_value)
            
            print("[AUTOCAL] All parameters set, checking if stereo matcher needs recreation...")
            # Recreate stereo matcher if core params changed
            stereo_params = ['minDisparity', 'numDisparitiesK', 'numDisparities', 
                           'blockSize', 'P1', 'P2', 'preFilterCap', 'uniquenessRatio', 
                           'speckleWindowSize', 'speckleRange', 'disp12MaxDiff', 'sgbmMode']
            
            if any(p in params for p in stereo_params):
                print("[AUTOCAL] Recreating stereo matcher...")
                block_size = int(getattr(state.vision, 'blockSize', 11))
                block_size = block_size if block_size % 2 == 1 else block_size + 1
                num_disparities = int(max(16, 16 * int(getattr(state.vision, 'numDisparitiesK', 2))))
                
                P1 = int(getattr(state.vision, 'P1', 8 * 1 * block_size * block_size))
                P2 = int(getattr(state.vision, 'P2', 32 * 1 * block_size * block_size))
                
                sgbm_mode = int(getattr(state.vision, 'sgbmMode', 2))
                mode_map = {
                    0: cv2.STEREO_SGBM_MODE_SGBM,
                    1: cv2.STEREO_SGBM_MODE_HH,
                    2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                }
                mode = mode_map.get(sgbm_mode, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                
                min_disparity = int(getattr(state.vision, 'minDisparity', 0))
                pre_filter_cap = int(getattr(state.vision, 'preFilterCap', 43))
                uniqueness_ratio = int(getattr(state.vision, 'uniquenessRatio', 1))
                speckle_window_size = int(getattr(state.vision, 'speckleWindowSize', 196))
                speckle_range = int(getattr(state.vision, 'speckleRange', 34))
                disp12_max_diff = int(getattr(state.vision, 'disp12MaxDiff', 18))
                
                print(f"[AUTOCAL] Creating stereo matcher with blockSize={block_size}, numDisparities={num_disparities}, P1={P1}, P2={P2}, mode={mode}")
                state.vision.stereo = cv2.StereoSGBM_create(
                    minDisparity=min_disparity,
                    numDisparities=num_disparities,
                    blockSize=max(3, block_size),
                    P1=P1,
                    P2=P2,
                    preFilterCap=pre_filter_cap,
                    uniquenessRatio=uniqueness_ratio,
                    speckleWindowSize=speckle_window_size,
                    speckleRange=speckle_range,
                    disp12MaxDiff=disp12_max_diff,
                    mode=mode,
                )
                print("[AUTOCAL] Stereo matcher created")
            
            print("[AUTOCAL] Refreshing depth processor...")
            state.vision._refresh_depth_processor()
            print("[AUTOCAL] Depth processor refreshed")
        
        print("[AUTOCAL] Lock released, _apply_params_to_vision complete")
        return True
    except Exception as e:
        print(f"[AUTOCAL] ERROR in _apply_params_to_vision: {e}")
        import traceback
        traceback.print_exc()
        return False

def _generate_next_comparison():
    """Generate next A/B comparison for current parameter."""
    print(f"[AUTOCAL] _generate_next_comparison called, param_index={state.autocal_param_index}, list_len={len(state.autocal_param_list)}")
    
    if state.autocal_param_index >= len(state.autocal_param_list):
        print("[AUTOCAL] All parameters processed, finishing")
        state.autocal_waiting_for_choice = False
        return
    
    param_name = state.autocal_param_list[state.autocal_param_index]
    print(f"[AUTOCAL] Processing parameter: {param_name}")
    param_ranges = _get_param_ranges()
    
    if param_name not in param_ranges:
        print(f"[AUTOCAL] Warning: Parameter {param_name} not in ranges, skipping")
        state.autocal_param_index += 1
        _generate_next_comparison()
        return
    
    param_info = param_ranges[param_name]
    
    current_value = state.autocal_best_params.get(param_name, param_info['default'])
    print(f"[AUTOCAL] Current value for {param_name}: {current_value}")
    
    # Generate test values: lower, current, higher
    step = param_info['step']
    min_val = param_info['min']
    max_val = param_info['max']
    
    # Use larger step for more noticeable differences (2x step)
    effective_step = max(step, (max_val - min_val) * 0.1)  # At least 10% of range
    
    # Profile A: lower value (use larger step)
    value_a = max(min_val, current_value - effective_step)
    # Profile B: higher value (use larger step)
    value_b = min(max_val, current_value + effective_step)
    
    # If at boundary, try opposite direction
    if value_a == current_value:
        value_a = min(max_val, current_value + effective_step)
    if value_b == current_value:
        value_b = max(min_val, current_value - effective_step)
    
    # Ensure we have different values and they're meaningfully different
    if value_a == value_b:
        # Make them more different
        if current_value < (min_val + max_val) / 2:
            value_a = max(min_val, current_value - effective_step)
            value_b = min(max_val, current_value + effective_step * 2)
        else:
            value_a = max(min_val, current_value - effective_step * 2)
            value_b = min(max_val, current_value + effective_step)
    
    # Ensure minimum difference (at least 5% of range or 1 unit, whichever is larger)
    min_diff = max(1, (max_val - min_val) * 0.05)
    if abs(value_b - value_a) < min_diff:
        mid = (value_a + value_b) / 2
        value_a = max(min_val, mid - min_diff / 2)
        value_b = min(max_val, mid + min_diff / 2)
    
    print(f"[AUTOCAL] Generated values - Profile A: {value_a}, Profile B: {value_b} (current: {current_value}, diff: {abs(value_b - value_a)})")
    
    # Create profile copies
    state.autocal_profile_a = state.autocal_best_params.copy()
    state.autocal_profile_a[param_name] = value_a
    
    state.autocal_profile_b = state.autocal_best_params.copy()
    state.autocal_profile_b[param_name] = value_b
    
    state.autocal_current_param_name = param_name
    state.autocal_waiting_for_choice = True
    
    # Apply profile A first (without sleep - let it happen async)
    print(f"[AUTOCAL] Applying profile A to vision system...")
    try:
        _apply_params_to_vision(state.autocal_profile_a)
        print(f"[AUTOCAL] Profile A applied successfully")
    except Exception as e:
        print(f"[AUTOCAL] Error applying profile A: {e}")
        import traceback
        traceback.print_exc()
        state.autocal_waiting_for_choice = False

@app.route('/start_autocal', methods=['POST'])
def start_autocal():
    """Start auto-calibration process."""
    print("[AUTOCAL] Start autocal requested")
    
    if state.vision is None:
        print("[AUTOCAL] ERROR: Vision not initialized")
        return jsonify({'status': 'error', 'message': 'Vision not initialized'}), 500
    
    if not state.vision.connected:
        print("[AUTOCAL] ERROR: Vision system not connected")
        return jsonify({'status': 'error', 'message': 'Vision system not connected'}), 500
    
    try:
        print("[AUTOCAL] Acquiring lock...")
        with state.lock:
            print("[AUTOCAL] Lock acquired, saving baseline parameters...")
            # Save current parameters as baseline
            param_ranges = _get_param_ranges()
            print(f"[AUTOCAL] Got {len(param_ranges)} parameter ranges")
            state.autocal_baseline_params = {}
            state.autocal_best_params = {}
            
            # Read current parameters safely
            for param_name in param_ranges.keys():
                try:
                    if hasattr(state.vision, param_name):
                        value = getattr(state.vision, param_name)
                        state.autocal_baseline_params[param_name] = value
                        state.autocal_best_params[param_name] = value
                    else:
                        # Use default if attribute doesn't exist
                        default_val = param_ranges[param_name]['default']
                        state.autocal_baseline_params[param_name] = default_val
                        state.autocal_best_params[param_name] = default_val
                except Exception as e:
                    print(f"[AUTOCAL] Warning: Could not read parameter {param_name}: {e}")
                    # Use default on error
                    default_val = param_ranges[param_name]['default']
                    state.autocal_baseline_params[param_name] = default_val
                    state.autocal_best_params[param_name] = default_val
            
            print(f"[AUTOCAL] Saved {len(state.autocal_baseline_params)} baseline parameters")
            
            # Create parameter list (focus on core SGBM params first)
            priority_params = [
                'blockSize', 'numDisparitiesK', 'P1', 'P2', 'preFilterCap',
                'uniquenessRatio', 'speckleWindowSize', 'speckleRange', 'disp12MaxDiff',
                'sgbmMode', 'minDisparity', 'downSample', 'crop', 'nearCutoff',
                'farCutoff', 'wlsLambda', 'wlsSigma', 'morphIter'
            ]
            
            state.autocal_param_list = [p for p in priority_params if p in param_ranges]
            print(f"[AUTOCAL] Created parameter list with {len(state.autocal_param_list)} parameters: {state.autocal_param_list}")
            state.autocal_param_index = 0
            state.autocal_iteration = 0
            state.autocal_active = True
            state.autocal_waiting_for_choice = False
        
        print("[AUTOCAL] Lock released, generating first comparison...")
        # Generate first comparison OUTSIDE the lock to avoid deadlock
        _generate_next_comparison()
        print("[AUTOCAL] First comparison generated successfully")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        import traceback
        print(f"[AUTOCAL] EXCEPTION: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_autocal_status', methods=['GET'])
def get_autocal_status():
    """Get current auto-calibration status."""
    if not state.autocal_active:
        return jsonify({
            'status': 'success',
            'state': 'ready',
            'iteration': 0,
            'total_iterations': 0,
            'current_parameter': ''
        })
    
    total_iterations = len(state.autocal_param_list) * 2  # Rough estimate
    
    if state.autocal_waiting_for_choice:
        state_str = 'comparing'
    elif state.autocal_param_index >= len(state.autocal_param_list):
        state_str = 'finished'
    else:
        state_str = 'comparing'
    
    return jsonify({
        'status': 'success',
        'state': state_str,
        'iteration': state.autocal_iteration,
        'total_iterations': total_iterations,
        'current_parameter': state.autocal_current_param_name or '',
        'progress': state.autocal_param_index / len(state.autocal_param_list) if state.autocal_param_list else 0
    })

@app.route('/autocal_choice', methods=['POST'])
def autocal_choice():
    """Submit user's choice for A/B comparison."""
    print(f"[AUTOCAL] autocal_choice called")
    
    if not state.autocal_active:
        print("[AUTOCAL] ERROR: Auto-calibration not active")
        return jsonify({'status': 'error', 'message': 'Auto-calibration not active'}), 400
    
    data = request.get_json()
    choice = data.get('choice', '').upper()
    print(f"[AUTOCAL] User selected: {choice}")
    
    if choice not in ['A', 'B', 'KEEP']:
        print(f"[AUTOCAL] ERROR: Invalid choice: {choice}")
        return jsonify({'status': 'error', 'message': 'Invalid choice'}), 400
    
    if not state.autocal_waiting_for_choice:
        print("[AUTOCAL] ERROR: Not waiting for choice")
        return jsonify({'status': 'error', 'message': 'Not waiting for choice'}), 400
    
    try:
        # Get the selected profile and param info while holding lock briefly
        selected_profile = None
        param_name = None
        with state.lock:
            param_name = state.autocal_current_param_name
            if choice == 'A':
                selected_profile = state.autocal_profile_a.copy()
                state.autocal_best_params[param_name] = state.autocal_profile_a[param_name]
                print(f"[AUTOCAL] Selected Profile A, value: {state.autocal_profile_a[param_name]}")
            elif choice == 'B':
                selected_profile = state.autocal_profile_b.copy()
                state.autocal_best_params[param_name] = state.autocal_profile_b[param_name]
                print(f"[AUTOCAL] Selected Profile B, value: {state.autocal_profile_b[param_name]}")
            else:  # choice == 'KEEP'
                # Keep current value - don't change best_params, just skip to next
                current_value = state.autocal_best_params.get(param_name)
                print(f"[AUTOCAL] Selected No Change (KEEP), keeping current value: {current_value}")
                selected_profile = None  # Don't apply any changes
            
            state.autocal_iteration += 1
            state.autocal_param_index += 1
            state.autocal_waiting_for_choice = False
        
        # Apply the selected profile OUTSIDE the lock (only if not KEEP)
        if selected_profile is not None:
            print(f"[AUTOCAL] Applying selected profile {choice}...")
            _apply_params_to_vision(selected_profile)
        else:
            print(f"[AUTOCAL] Skipping parameter change, keeping current value")
        
        # Generate next comparison OUTSIDE the lock
        if state.autocal_param_index < len(state.autocal_param_list):
            print(f"[AUTOCAL] Generating next comparison for parameter {state.autocal_param_index}...")
            _generate_next_comparison()
        else:
            print("[AUTOCAL] All parameters processed!")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"[AUTOCAL] EXCEPTION in autocal_choice: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/finish_autocal', methods=['POST'])
def finish_autocal():
    """Finish auto-calibration and save best parameters."""
    if not state.autocal_active:
        return jsonify({'status': 'error', 'message': 'Auto-calibration not active'}), 400
    
    try:
        with state.lock:
            # Apply best parameters
            _apply_params_to_vision(state.autocal_best_params)
            
            # Save to config
            if state.camera_config:
                for param_name, param_value in state.autocal_best_params.items():
                    state.camera_config[param_name] = param_value
                
                config_path = "config.dill"
                with open(config_path, "rb") as f:
                    config = dill.load(f)
                config['camera'] = state.camera_config
                with open(config_path, "wb") as f:
                    dill.dump(config, f)
            
            # Reset state
            state.autocal_active = False
            state.autocal_waiting_for_choice = False
        
        return jsonify({'status': 'success', 'message': 'Best parameters saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/switch_autocal_profile', methods=['POST'])
def switch_autocal_profile():
    """Switch between Profile A and B for comparison."""
    print(f"[AUTOCAL] switch_autocal_profile called")
    
    if not state.autocal_active:
        print("[AUTOCAL] ERROR: Auto-calibration not active")
        return jsonify({'status': 'error', 'message': 'Auto-calibration not active'}), 400
    
    data = request.get_json()
    profile = data.get('profile', '').upper()
    print(f"[AUTOCAL] Switching to profile: {profile}")
    
    if profile not in ['A', 'B']:
        print(f"[AUTOCAL] ERROR: Invalid profile: {profile}")
        return jsonify({'status': 'error', 'message': 'Invalid profile'}), 400
    
    try:
        # Get the profile to apply
        profile_to_apply = None
        with state.lock:
            if profile == 'A' and state.autocal_profile_a:
                profile_to_apply = state.autocal_profile_a.copy()
                print(f"[AUTOCAL] Profile A found with {len(profile_to_apply)} parameters")
            elif profile == 'B' and state.autocal_profile_b:
                profile_to_apply = state.autocal_profile_b.copy()
                print(f"[AUTOCAL] Profile B found with {len(profile_to_apply)} parameters")
            else:
                print(f"[AUTOCAL] ERROR: Profile {profile} not available")
                return jsonify({'status': 'error', 'message': f'Profile {profile} not available'}), 400
        
        # Apply profile OUTSIDE the lock to avoid deadlock
        if profile_to_apply:
            print(f"[AUTOCAL] Applying profile {profile}...")
            success = _apply_params_to_vision(profile_to_apply)
            if success:
                print(f"[AUTOCAL] Profile {profile} applied successfully")
                return jsonify({'status': 'success'})
            else:
                print(f"[AUTOCAL] ERROR: Failed to apply profile {profile}")
                return jsonify({'status': 'error', 'message': 'Failed to apply profile'}), 500
        
        return jsonify({'status': 'error', 'message': 'Profile not found'}), 500
    except Exception as e:
        print(f"[AUTOCAL] EXCEPTION in switch_autocal_profile: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/cancel_autocal', methods=['POST'])
def cancel_autocal():
    """Cancel auto-calibration and restore baseline parameters."""
    try:
        with state.lock:
            # Restore baseline if available
            if state.autocal_baseline_params and state.vision is not None:
                try:
                    _apply_params_to_vision(state.autocal_baseline_params)
                except Exception as e:
                    print(f"Error restoring baseline params: {e}")
            
            # Reset state
            state.autocal_active = False
            state.autocal_waiting_for_choice = False
            state.autocal_param_index = 0
            state.autocal_iteration = 0
        
        return jsonify({'status': 'success', 'message': 'Auto-calibration cancelled'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def main():
    """Initialize and run the Flask web server."""
    capture_thread = None
    try:
        # Load configuration
        state.camera_config = load_config()
        
        # Initialize vision system
        state.vision = initialize_vision(state.camera_config)
        
        # Start background frame capture thread
        state.capturing = True
        capture_thread = Thread(target=capture_frames_continuously, daemon=True)
        capture_thread.start()
        
        # Get local IP for display
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("\n" + "="*60)
        print("🌐 WEB STREAMING SERVER READY")
        print("="*60)
        print(f"📱 Open on your phone: http://{local_ip}:5000")
        print(f"💻 Or locally: http://localhost:5000")
        print(f"👥 Multiple users supported!")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n⚠️ Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop frame capture
        state.capturing = False
        if capture_thread:
            capture_thread.join(timeout=2)
        
        # Clean up
        if state.vision and state.vision.connected:
            print("Stopping vision system...")
            state.vision.stop()

if __name__ == "__main__":
    main()

