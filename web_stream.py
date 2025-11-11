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

# Make sure src folder is on sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

app = Flask(__name__)

# Global state
class StreamState:
    def __init__(self):
        self.vision = None
        self.camera_config = None
        self.mode = "debug"  # "debug", "calibrate", or "calibrate_maps"
        self.view = "depth"  # "left", "right", or "depth"
        self.lock = Lock()
        self.latest_frame = None
        self.latest_frame_bytes = None
        self.parameters = {}
        self.frame_ready = False
        self.capturing = False
        # Calibration state
        self.calibrating_maps = False
        self.captured_pairs = []
        self.checkerboard_size = (7, 10)
        self.square_size_mm = 20.0
        self.min_pairs = 5
        # Colormap
        self.colormap = cv2.COLORMAP_JET  # Default colormap
        self.invert_colormap = False  # Invert colormap colors
        
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
        metadata = result.get('metadata', {})
        
        # Check for errors
        if 'error' in metadata or depth_map is None or depth_map.size == 0:
            return None
        
        # Normalize depth map for visualization (0-255 range)
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_normalized.astype(np.uint8)
        
        # Invert if needed
        if state.invert_colormap:
            depth_display = 255 - depth_display
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_display, state.colormap)
        
        # Add metadata text overlay
        timestamp = metadata.get('timestamp', 'N/A')[-8:]  # Just show time part
        num_disp = metadata.get('num_disparities', 'N/A')
        
        cv2.putText(depth_colored, f"DEPTH MAP", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"Time: {timestamp}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_colored, f"Disparities: {num_disp}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return depth_colored
        
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
        metadata = result.get('metadata', {})
        
        # Check for errors
        if 'error' in metadata or depth_map is None or depth_map.size == 0:
            return None
        
        # Normalize depth map for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_normalized.astype(np.uint8)
        
        # Invert if needed
        if state.invert_colormap:
            depth_display = 255 - depth_display
        
        depth_colored = cv2.applyColorMap(depth_display, state.colormap)
        
        # Add calibration mode overlay
        cv2.putText(depth_colored, "DEPTH MAP - CALIBRATION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show key parameters
        y_offset = 60
        params_to_show = [
            ('blockSize', state.vision.blockSize),
            ('numDisparities', state.vision.numDisparities),
            ('uniqueness', state.vision.uniquenessRatio),
            ('WLS', 'ON' if state.vision.useWLS else 'OFF'),
        ]
        
        for param_name, param_value in params_to_show:
            cv2.putText(depth_colored, f"{param_name}: {param_value}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        return depth_colored
        
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
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    with state.lock:
                        state.latest_frame_bytes = buffer.tobytes()
                        state.frame_ready = True
        except Exception as e:
            print(f"Error in frame capture: {e}")
        
        time.sleep(0.033)  # ~30 FPS
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
    """Switch between debug and calibrate modes."""
    data = request.get_json()
    new_mode = data.get('mode', 'debug')
    
    if new_mode in ['debug', 'calibrate']:
        with state.lock:
            state.mode = new_mode
        return jsonify({'status': 'success', 'mode': state.mode})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400

@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    """Cycle through camera views: depth -> left -> right -> depth."""
    with state.lock:
        if state.view == "depth":
            state.view = "left"
        elif state.view == "left":
            state.view = "right"
        else:  # right
            state.view = "depth"
    
    return jsonify({'status': 'success', 'view': state.view})

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
        'preFilterCap': state.vision.preFilterCap,
        'uniquenessRatio': state.vision.uniquenessRatio,
        'speckleWindowSize': state.vision.speckleWindowSize,
        'speckleRange': state.vision.speckleRange,
        'disp12MaxDiff': state.vision.disp12MaxDiff,
        
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
                           'blockSize', 'preFilterCap', 'uniquenessRatio', 
                           'speckleWindowSize', 'speckleRange', 'disp12MaxDiff']
            
            if param_name in stereo_params:
                block_size = state.vision.blockSize
                block_size = block_size if block_size % 2 == 1 else block_size + 1
                num_disparities = max(16, 16 * state.vision.numDisparitiesK)
                
                state.vision.stereo = cv2.StereoSGBM_create(
                    minDisparity=state.vision.minDisparity,
                    numDisparities=num_disparities,
                    blockSize=max(3, block_size),
                    preFilterCap=state.vision.preFilterCap,
                    uniquenessRatio=state.vision.uniquenessRatio,
                    speckleWindowSize=state.vision.speckleWindowSize,
                    speckleRange=state.vision.speckleRange,
                    disp12MaxDiff=state.vision.disp12MaxDiff,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
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
            'preFilterCap', 'uniquenessRatio', 'speckleWindowSize', 'speckleRange',
            'disp12MaxDiff', 'medianBlurK', 'downSample', 'crop', 'farEnhance',
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

