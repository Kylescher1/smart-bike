import cv2
import numpy as np
import dill
import sys
import os
import importlib.util
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

config_path = r"config.dill"

def transform_points_for_view(points, center_point, camera_radius, azimuth_deg, elevation_deg, roll_deg=0):
    """
    Transform points for viewing using a camera orbiting around a center point.
    Uses spherical coordinates: radius (distance), azimuth (horizontal angle), elevation (vertical angle).
    
    Args:
        points: Nx3 array of 3D points
        center_point: [x, y, z] center point to orbit around
        camera_radius: Distance from center point (positive = away, negative = inside)
        azimuth_deg: Horizontal rotation angle (0-360 degrees)
        elevation_deg: Vertical angle (-90 to +90 degrees, 0 = horizontal)
        roll_deg: Roll angle around viewing axis (optional)
    
    Returns:
        Transformed points in camera space (Z is forward, Y is up)
    """
    # Convert to radians
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    roll = np.radians(roll_deg)
    
    # Calculate camera position in spherical coordinates
    # X = radius * cos(elevation) * cos(azimuth)
    # Y = radius * cos(elevation) * sin(azimuth)
    # Z = radius * sin(elevation)
    camera_x = camera_radius * np.cos(elevation) * np.cos(azimuth)
    camera_y = camera_radius * np.cos(elevation) * np.sin(azimuth)
    camera_z = camera_radius * np.sin(elevation)
    camera_pos = np.array([camera_x, camera_y, camera_z])
    
    # Translate points relative to center
    points_centered = points - center_point
    
    # Build view matrix: camera looks at center point
    # Forward vector: from camera to center (normalized)
    forward = -camera_pos / (np.linalg.norm(camera_pos) + 1e-10)
    
    # Right vector: cross product of forward and world up (0,0,1)
    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-10)
    
    # Up vector: cross product of right and forward
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-10)
    
    # Apply roll rotation around forward axis
    if roll != 0:
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        right_rolled = right * cos_r + up * sin_r
        up_rolled = -right * sin_r + up * cos_r
        right, up = right_rolled, up_rolled
    
    # Build rotation matrix (camera to world)
    R = np.column_stack([right, up, -forward])
    
    # Transform points: translate to camera space, then rotate
    # Points are relative to center, so we rotate them to camera's coordinate system
    points_camera_space = points_centered @ R.T
    
    return points_camera_space

def project_points_to_2d(points, view_axis='xy', canvas_size=800, offset_x=0, offset_y=0):
    """Project 3D points to 2D based on view axis with optional offset."""
    if len(points) == 0:
        return np.array([]), np.array([]), 0, 0, 0, 0
    
    try:
        # Select axes based on view
        if view_axis == 'xy':  # Top view (X-Y plane, looking down Z)
            coords = points[:, [0, 1]]
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            if (x_max - x_min) > 1e-6 and (y_max - y_min) > 1e-6:
                scale = min(canvas_size / (x_max - x_min), canvas_size / (y_max - y_min)) * 0.8
                center_x = canvas_size // 2 - (x_min + x_max) / 2 * scale
                center_y = canvas_size // 2 - (y_min + y_max) / 2 * scale
            else:
                scale = 1.0
                center_x = center_y = canvas_size // 2
            x_coords = ((coords[:, 0] - x_min) * scale + center_x + offset_x).astype(int)
            y_coords = ((coords[:, 1] - y_min) * scale + center_y + offset_y).astype(int)
            return x_coords, y_coords, x_min, x_max, y_min, y_max
        
        elif view_axis == 'xz':  # Front view (X-Z plane, looking along Y)
            coords = points[:, [0, 2]]
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            if (x_max - x_min) > 1e-6 and (z_max - z_min) > 1e-6:
                scale = min(canvas_size / (x_max - x_min), canvas_size / (z_max - z_min)) * 0.8
                center_x = canvas_size // 2 - (x_min + x_max) / 2 * scale
                center_z = canvas_size // 2 - (z_min + z_max) / 2 * scale
            else:
                scale = 1.0
                center_x = center_z = canvas_size // 2
            x_coords = ((coords[:, 0] - x_min) * scale + center_x + offset_x).astype(int)
            y_coords = ((coords[:, 1] - z_min) * scale + center_z + offset_y).astype(int)
            return x_coords, y_coords, x_min, x_max, z_min, z_max
        
        elif view_axis == 'yz':  # Side view (Y-Z plane, looking along X)
            coords = points[:, [1, 2]]
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            if (y_max - y_min) > 1e-6 and (z_max - z_min) > 1e-6:
                scale = min(canvas_size / (y_max - y_min), canvas_size / (z_max - z_min)) * 0.8
                center_y = canvas_size // 2 - (y_min + y_max) / 2 * scale
                center_z = canvas_size // 2 - (z_min + z_max) / 2 * scale
            else:
                scale = 1.0
                center_y = center_z = canvas_size // 2
            x_coords = ((coords[:, 0] - y_min) * scale + center_y + offset_x).astype(int)
            y_coords = ((coords[:, 1] - z_min) * scale + center_z + offset_y).astype(int)
            return x_coords, y_coords, y_min, y_max, z_min, z_max
        
        elif view_axis == 'rotated':  # Isometric/rotated view
            # Use rotated points and project to X-Y plane
            coords = points[:, [0, 1]]
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            if (x_max - x_min) > 1e-6 and (y_max - y_min) > 1e-6:
                scale = min(canvas_size / (x_max - x_min), canvas_size / (y_max - y_min)) * 0.8
                center_x = canvas_size // 2 - (x_min + x_max) / 2 * scale
                center_y = canvas_size // 2 - (y_min + y_max) / 2 * scale
            else:
                scale = 1.0
                center_x = center_y = canvas_size // 2
            x_coords = ((coords[:, 0] - x_min) * scale + center_x + offset_x).astype(int)
            y_coords = ((coords[:, 1] - y_min) * scale + center_y + offset_y).astype(int)
            return x_coords, y_coords, x_min, x_max, y_min, y_max
        
        return np.array([]), np.array([]), 0, 0, 0, 0
    except Exception as e:
        print(f"Error in project_points_to_2d: {e}")
        return np.array([]), np.array([]), 0, 0, 0, 0

def draw_points_fast(view, x_coords, y_coords, colors_array, canvas_size):
    """Fast point drawing using numpy indexing."""
    try:
        if len(x_coords) == 0 or len(y_coords) == 0:
            return
        
        # Filter valid coordinates
        valid = (x_coords >= 0) & (x_coords < canvas_size) & (y_coords >= 0) & (y_coords < canvas_size)
        if not np.any(valid):
            return
        
        x_valid = x_coords[valid]
        y_valid = y_coords[valid]
        colors_valid = colors_array[valid]
        
        # Use numpy advanced indexing for fast drawing
        # Clamp coordinates to ensure they're within bounds
        x_valid = np.clip(x_valid, 0, canvas_size - 1)
        y_valid = np.clip(y_valid, 0, canvas_size - 1)
        
        # Draw points directly using numpy indexing (much faster than cv2.circle)
        view[y_valid, x_valid] = colors_valid
    except Exception as e:
        print(f"Error in draw_points_fast: {e}")

def generate_pointcloud_frame(disparity_map, depth_map, left_frame, q_matrix, center_x, center_y, center_z, camera_radius, azimuth, elevation, roll, zoom, pan_x, pan_y, 
                              filter_min_x=None, filter_max_x=None, filter_min_y=None, filter_max_y=None, 
                              filter_min_z=None, filter_max_z=None, filter_min_dist=None, filter_max_dist=None):
    """Generate a point cloud visualization using spherical camera coordinates."""
    try:
        if depth_map is None or disparity_map is None or q_matrix is None:
            return None
        
        # Convert disparity to 3D points
        points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) * 16.0, q_matrix)
        
        # Filter out invalid points (zero depth, infinite, NaN)
        valid_mask = (depth_map > 0) & np.isfinite(points_3d[:, :, 2])
        valid_mask = valid_mask & (points_3d[:, :, 2] > 0)
        
        # Get valid points and colors
        # Resize left_frame to match disparity_map size if needed
        h, w = disparity_map.shape[:2]
        if left_frame.shape[:2] != (h, w):
            left_frame = cv2.resize(left_frame, (w, h))
        
        points = points_3d[valid_mask]
        colors = left_frame[valid_mask]
        
        if len(points) == 0:
            return None
        
        # Apply spatial filters
        filter_mask = np.ones(len(points), dtype=bool)
        
        if filter_min_x is not None:
            filter_mask = filter_mask & (points[:, 0] >= filter_min_x)
        if filter_max_x is not None:
            filter_mask = filter_mask & (points[:, 0] <= filter_max_x)
        if filter_min_y is not None:
            filter_mask = filter_mask & (points[:, 1] >= filter_min_y)
        if filter_max_y is not None:
            filter_mask = filter_mask & (points[:, 1] <= filter_max_y)
        if filter_min_z is not None:
            filter_mask = filter_mask & (points[:, 2] >= filter_min_z)
        if filter_max_z is not None:
            filter_mask = filter_mask & (points[:, 2] <= filter_max_z)
        
        # Distance filter (distance from center point)
        if filter_min_dist is not None or filter_max_dist is not None:
            center_point = np.array([center_x, center_y, center_z])
            distances = np.linalg.norm(points - center_point, axis=1)
            if filter_min_dist is not None:
                filter_mask = filter_mask & (distances >= filter_min_dist)
            if filter_max_dist is not None:
                filter_mask = filter_mask & (distances <= filter_max_dist)
        
        # Apply filters
        points = points[filter_mask]
        colors = colors[filter_mask]
        
        if len(points) == 0:
            print(f"[POINTCLOUD] WARNING: All points filtered out!")
            return None
        
        # Aggressive subsampling for performance (limit to ~50k points for single view)
        max_points = 50000
        subsample = max(1, len(points) // max_points)
        points = points[::subsample]
        colors = colors[::subsample]
        
        print(f"[POINTCLOUD] Processing {len(points)} points")
        
        # Center point for camera to orbit around
        center_point = np.array([center_x, center_y, center_z])
        
        # Transform points to camera space using spherical coordinates
        points_camera_space = transform_points_for_view(
            points, center_point, camera_radius, azimuth, elevation, roll
        )
        
        # Apply zoom (scale around origin in camera space)
        if zoom != 1.0:
            points_camera_space = points_camera_space * zoom
        
        # Project to 2D (X-Y plane in camera space, Z is depth/forward)
        # Use X and Y coordinates for screen position, filter by Z depth
        coords_2d = points_camera_space[:, [0, 1]]  # X, Y in camera space
        z_depth = points_camera_space[:, 2]  # Z depth (forward)
        
        # Filter points that are behind the camera (negative Z) or too far
        # Only show points in front of camera
        valid_view_mask = z_depth > 0  # Points in front of camera
        if not np.any(valid_view_mask):
            print(f"[POINTCLOUD] WARNING: No points in front of camera! Z range: {z_depth.min():.2f} to {z_depth.max():.2f}")
            # Try showing all points anyway
            valid_view_mask = np.ones(len(z_depth), dtype=bool)
        
        coords_2d_valid = coords_2d[valid_view_mask]
        colors_valid = colors[valid_view_mask]
        
        if len(coords_2d_valid) == 0:
            print(f"[POINTCLOUD] ERROR: No valid points to display!")
            return None
        
        # Create single large canvas for the view
        canvas_size = 1200
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Calculate bounds for scaling
        x_min, x_max = coords_2d_valid[:, 0].min(), coords_2d_valid[:, 0].max()
        y_min, y_max = coords_2d_valid[:, 1].min(), coords_2d_valid[:, 1].max()
        
        print(f"[POINTCLOUD] 2D bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z depth range: [{z_depth[valid_view_mask].min():.2f}, {z_depth[valid_view_mask].max():.2f}]")
        
        # Scale and center the projection
        if (x_max - x_min) > 1e-6 and (y_max - y_min) > 1e-6:
            scale = min(canvas_size / (x_max - x_min), canvas_size / (y_max - y_min)) * 0.8
            center_x_screen = canvas_size // 2 - (x_min + x_max) / 2 * scale
            center_y_screen = canvas_size // 2 - (y_min + y_max) / 2 * scale
        else:
            scale = 1.0
            center_x_screen = center_y_screen = canvas_size // 2
        
        # Apply panning offset
        center_x_screen += pan_x
        center_y_screen += pan_y
        
        # Project to screen coordinates
        x_coords = ((coords_2d_valid[:, 0] - x_min) * scale + center_x_screen).astype(int)
        y_coords = ((coords_2d_valid[:, 1] - y_min) * scale + center_y_screen).astype(int)
        
        print(f"[POINTCLOUD] Projecting {len(coords_2d_valid)} points, scale={scale:.2f}")
        draw_points_fast(canvas, x_coords, y_coords, colors_valid, canvas_size)
        
        # Add title and camera info
        cv2.putText(canvas, f"Point Cloud Viewer", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Azimuth: {azimuth:.0f}°  Elevation: {elevation:.0f}°  Distance: {camera_radius:.2f}m", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Center: [{center_x:.2f}, {center_y:.2f}, {center_z:.2f}]  Zoom: {zoom:.2f}x", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        print(f"[POINTCLOUD] Frame generated successfully")
        
        # Add stats at bottom
        num_points = len(points)
        num_visible = len(coords_2d_valid)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        x_min_world, x_max_world = points[:, 0].min(), points[:, 0].max()
        y_min_world, y_max_world = points[:, 1].min(), points[:, 1].max()
        z_depth_min, z_depth_max = z_depth[valid_view_mask].min(), z_depth[valid_view_mask].max() if np.any(valid_view_mask) else (0, 0)
        
        # Show filter status
        filter_active = any([filter_min_x is not None, filter_max_x is not None, 
                            filter_min_y is not None, filter_max_y is not None,
                            filter_min_z is not None, filter_max_z is not None,
                            filter_min_dist is not None, filter_max_dist is not None])
        
        cv2.putText(canvas, f"Points: {num_points:,} (visible: {num_visible:,}) {'[FILTERED]' if filter_active else ''}", (10, canvas_size - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"World X: [{x_min_world:.2f}, {x_max_world:.2f}] Y: [{y_min_world:.2f}, {y_max_world:.2f}]", (10, canvas_size - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, f"World Depth: {z_min:.2f}m - {z_max:.2f}m", (10, canvas_size - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"Camera Z: {z_depth_min:.2f}m - {z_depth_max:.2f}m", (10, canvas_size - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return canvas
        
    except Exception as e:
        print(f"Error generating point cloud frame: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_rotation_trackbars(window_name, default_values=None):
    """Create trackbars for camera controls using spherical coordinates."""
    # Default values
    defaults = {
        'center_x': 500,  # 0.0m (500 = 0.0, range -10.0 to +10.0m)
        'center_y': 500,  # 0.0m
        'center_z': 500,  # 0.0m
        'camera_radius': 500,  # 0.0m (500 = 0.0, range -10.0 to +10.0m)
        'azimuth': 0,
        'elevation': 90,  # 0 degrees (90 in trackbar, -90 to +90 range)
        'roll': 0,
        'zoom': 100,
        'pan_x': 600,  # Center position
        'pan_y': 600,  # Center position
        # Filter defaults (0 = disabled, 1000 = max range)
        'filter_min_x': 0,  # Disabled
        'filter_max_x': 1000,  # Disabled
        'filter_min_y': 0,  # Disabled
        'filter_max_y': 1000,  # Disabled
        'filter_min_z': 0,  # Disabled
        'filter_max_z': 1000,  # Disabled
        'filter_min_dist': 0,  # Disabled
        'filter_max_dist': 1000,  # Disabled
    }
    
    # Merge with provided values, using defaults for missing keys
    if default_values is not None:
        defaults.update(default_values)
    
    # Center point controls (where camera orbits around)
    cv2.createTrackbar("Center X", window_name, defaults.get('center_x', 500), 1000, lambda x: None)  # -10m to +10m
    cv2.createTrackbar("Center Y", window_name, defaults.get('center_y', 500), 1000, lambda x: None)
    cv2.createTrackbar("Center Z", window_name, defaults.get('center_z', 500), 1000, lambda x: None)
    
    # Camera distance (radius in spherical coordinates)
    cv2.createTrackbar("Distance", window_name, defaults.get('camera_radius', 500), 1000, lambda x: None)  # -10m to +10m
    
    # Camera angles
    cv2.createTrackbar("Azimuth", window_name, defaults.get('azimuth', 0), 360, lambda x: None)
    cv2.createTrackbar("Elevation", window_name, defaults.get('elevation', 90), 180, lambda x: None)  # -90 to +90 degrees
    cv2.createTrackbar("Roll", window_name, defaults.get('roll', 0), 360, lambda x: None)
    
    # Zoom and pan
    cv2.createTrackbar("Zoom", window_name, defaults.get('zoom', 100), 500, lambda x: None)  # 100 = 1.0x, 500 = 5.0x
    cv2.createTrackbar("Pan X", window_name, defaults.get('pan_x', 600), 1200, lambda x: None)
    cv2.createTrackbar("Pan Y", window_name, defaults.get('pan_y', 600), 1200, lambda x: None)
    
    # Filters (0 = disabled, 1000 = max range -10m to +10m)
    cv2.createTrackbar("Filt Min X", window_name, defaults.get('filter_min_x', 0), 1000, lambda x: None)
    cv2.createTrackbar("Filt Max X", window_name, defaults.get('filter_max_x', 1000), 1000, lambda x: None)
    cv2.createTrackbar("Filt Min Y", window_name, defaults.get('filter_min_y', 0), 1000, lambda x: None)
    cv2.createTrackbar("Filt Max Y", window_name, defaults.get('filter_max_y', 1000), 1000, lambda x: None)
    cv2.createTrackbar("Filt Min Z", window_name, defaults.get('filter_min_z', 0), 1000, lambda x: None)
    cv2.createTrackbar("Filt Max Z", window_name, defaults.get('filter_max_z', 1000), 1000, lambda x: None)
    cv2.createTrackbar("Filt Min Dist", window_name, defaults.get('filter_min_dist', 0), 1000, lambda x: None)
    cv2.createTrackbar("Filt Max Dist", window_name, defaults.get('filter_max_dist', 1000), 1000, lambda x: None)

def get_camera_values(window_name):
    """Get current camera values from trackbars."""
    # Center point: convert from 0-1000 range to -10.0 to +10.0 meters
    center_x = (cv2.getTrackbarPos("Center X", window_name) - 500) / 50.0
    center_y = (cv2.getTrackbarPos("Center Y", window_name) - 500) / 50.0
    center_z = (cv2.getTrackbarPos("Center Z", window_name) - 500) / 50.0
    
    # Camera distance: convert from 0-1000 range to -10.0 to +10.0 meters
    camera_radius = (cv2.getTrackbarPos("Distance", window_name) - 500) / 50.0
    
    # Angles
    azimuth = cv2.getTrackbarPos("Azimuth", window_name)
    elevation = cv2.getTrackbarPos("Elevation", window_name) - 90  # -90 to +90 degrees
    roll = cv2.getTrackbarPos("Roll", window_name)
    
    # Zoom and pan
    zoom = cv2.getTrackbarPos("Zoom", window_name) / 100.0
    pan_x = cv2.getTrackbarPos("Pan X", window_name) - 600
    pan_y = cv2.getTrackbarPos("Pan Y", window_name) - 600
    
    # Filters: convert from 0-1000 range to -10.0 to +10.0 meters
    # 0 = disabled (None), otherwise convert to actual value
    def get_filter_value(pos, default_disabled=0):
        if pos == default_disabled:
            return None
        return (pos - 500) / 50.0
    
    filter_min_x = get_filter_value(cv2.getTrackbarPos("Filt Min X", window_name))
    filter_max_x = get_filter_value(cv2.getTrackbarPos("Filt Max X", window_name), default_disabled=1000)
    filter_min_y = get_filter_value(cv2.getTrackbarPos("Filt Min Y", window_name))
    filter_max_y = get_filter_value(cv2.getTrackbarPos("Filt Max Y", window_name), default_disabled=1000)
    filter_min_z = get_filter_value(cv2.getTrackbarPos("Filt Min Z", window_name))
    filter_max_z = get_filter_value(cv2.getTrackbarPos("Filt Max Z", window_name), default_disabled=1000)
    filter_min_dist = get_filter_value(cv2.getTrackbarPos("Filt Min Dist", window_name))
    filter_max_dist = get_filter_value(cv2.getTrackbarPos("Filt Max Dist", window_name), default_disabled=1000)
    
    return (center_x, center_y, center_z, camera_radius, azimuth, elevation, roll, zoom, pan_x, pan_y,
            filter_min_x, filter_max_x, filter_min_y, filter_max_y, 
            filter_min_z, filter_max_z, filter_min_dist, filter_max_dist)

def save_settings(window_name, settings_file="pointcloud_settings.json"):
    """Save current trackbar settings to a JSON file."""
    try:
        settings = {
            'center_x': cv2.getTrackbarPos("Center X", window_name),
            'center_y': cv2.getTrackbarPos("Center Y", window_name),
            'center_z': cv2.getTrackbarPos("Center Z", window_name),
            'camera_radius': cv2.getTrackbarPos("Distance", window_name),
            'azimuth': cv2.getTrackbarPos("Azimuth", window_name),
            'elevation': cv2.getTrackbarPos("Elevation", window_name),
            'roll': cv2.getTrackbarPos("Roll", window_name),
            'zoom': cv2.getTrackbarPos("Zoom", window_name),
            'pan_x': cv2.getTrackbarPos("Pan X", window_name),
            'pan_y': cv2.getTrackbarPos("Pan Y", window_name),
            'filter_min_x': cv2.getTrackbarPos("Filt Min X", window_name),
            'filter_max_x': cv2.getTrackbarPos("Filt Max X", window_name),
            'filter_min_y': cv2.getTrackbarPos("Filt Min Y", window_name),
            'filter_max_y': cv2.getTrackbarPos("Filt Max Y", window_name),
            'filter_min_z': cv2.getTrackbarPos("Filt Min Z", window_name),
            'filter_max_z': cv2.getTrackbarPos("Filt Max Z", window_name),
            'filter_min_dist': cv2.getTrackbarPos("Filt Min Dist", window_name),
            'filter_max_dist': cv2.getTrackbarPos("Filt Max Dist", window_name),
        }
        
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"✅ Settings saved to {settings_file}")
        return True
    except Exception as e:
        print(f"⚠️  Error saving settings: {e}")
        return False

def load_settings(settings_file="pointcloud_settings.json"):
    """Load trackbar settings from a JSON file."""
    try:
        if not os.path.exists(settings_file):
            print(f"ℹ️  No settings file found ({settings_file}), using defaults")
            return None
        
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        print(f"✅ Settings loaded from {settings_file}")
        return settings
    except Exception as e:
        print(f"⚠️  Error loading settings: {e}")
        return None

def load_config():
    """Load config.dill and initialize camera configuration."""
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
    
    # Construct file path correctly (module_path already includes 'src')
    file_path = os.path.join(os.path.dirname(__file__), *module_path.split(".")) + ".py"
    spec = importlib.util.spec_from_file_location(
        module_path, 
        file_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    VisionClass = getattr(module, class_name)
    
    # Create vision instance with config
    vision = VisionClass(name="camera", **camera_config)
    vision.start()
    
    print("✅ Vision system initialized")
    return vision

def main():
    """Main point cloud viewer loop."""
    print("="*60)
    print("Point Cloud Viewer")
    print("="*60)
    
    # Load config
    camera_config = load_config()
    
    # Initialize vision system
    vision = initialize_vision(camera_config)
    
    # Load saved settings
    saved_settings = load_settings()
    
    # Auto-detect center point from first frame (optional - can be overridden)
    print("Waiting for first frame to auto-detect center point...")
    result = vision.read()
    depth_map = result.get('depth_map')
    disparity_map = result.get('disparity_map')
    
    auto_center = None
    if depth_map is not None and disparity_map is not None:
        q_matrix = getattr(vision, 'Q', None)
        if q_matrix is not None:
            points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) * 16.0, q_matrix)
            valid_mask = (depth_map > 0) & np.isfinite(points_3d[:, :, 2]) & (points_3d[:, :, 2] > 0)
            if np.any(valid_mask):
                valid_points = points_3d[valid_mask]
                # Use median as center (more robust than mean)
                auto_center = np.median(valid_points, axis=0)
                print(f"✅ Auto-detected center point: [{auto_center[0]:.2f}, {auto_center[1]:.2f}, {auto_center[2]:.2f}]")
                # Convert to trackbar values
                if saved_settings is None:
                    saved_settings = {}
                saved_settings['center_x'] = int(auto_center[0] * 50.0 + 500)
                saved_settings['center_y'] = int(auto_center[1] * 50.0 + 500)
                saved_settings['center_z'] = int(auto_center[2] * 50.0 + 500)
                # Set initial distance to be slightly away from center
                if 'camera_radius' not in saved_settings:
                    saved_settings['camera_radius'] = 550  # 1.0m away
    
    # Create window and trackbars
    window_name = "Point Cloud Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 1200)
    
    # Create a separate control window for trackbars
    control_window = "Point Cloud Controls"
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 400, 600)  # Taller window for filters
    create_rotation_trackbars(control_window, saved_settings)
    
    print("\nControls:")
    print("  - Center X/Y/Z: Set the point to orbit around (in meters)")
    print("  - Distance: Move camera closer/farther from center (negative = inside, positive = outside)")
    print("  - Azimuth: Rotate horizontally around center (0-360°)")
    print("  - Elevation: Rotate vertically (-90° to +90°, 0 = horizontal)")
    print("  - Pan X/Y: Pan the view on screen")
    print("  - Filt Min/Max X/Y/Z: Filter points by world coordinates (0 = disabled)")
    print("  - Filt Min/Max Dist: Filter points by distance from center (0 = disabled)")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 'r' to reset all settings")
    print("  - Press 's' to save current settings")
    print("\nStarting point cloud visualization...\n")
    
    try:
        while True:
            # Read depth data
            result = vision.read()
            depth_map = result.get('depth_map')
            disparity_map = result.get('disparity_map')
            left_frame = vision.left_camera.read_frame()
            q_matrix = getattr(vision, 'Q', None)
            
            if depth_map is None or disparity_map is None or left_frame is None or q_matrix is None:
                print("⚠️  Waiting for depth data...")
                cv2.waitKey(100)
                continue
            
            # Get camera values from trackbars (use control window)
            (center_x, center_y, center_z, camera_radius, azimuth, elevation, roll, zoom, pan_x, pan_y,
             filter_min_x, filter_max_x, filter_min_y, filter_max_y,
             filter_min_z, filter_max_z, filter_min_dist, filter_max_dist) = get_camera_values(control_window)
            
            # Generate point cloud visualization
            pointcloud_frame = generate_pointcloud_frame(
                disparity_map, depth_map, left_frame, q_matrix,
                center_x, center_y, center_z, camera_radius, azimuth, elevation, roll, zoom, pan_x, pan_y,
                filter_min_x, filter_max_x, filter_min_y, filter_max_y,
                filter_min_z, filter_max_z, filter_min_dist, filter_max_dist
            )
            
            if pointcloud_frame is not None:
                cv2.imshow(window_name, pointcloud_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                # Save settings before quitting
                save_settings(control_window)
                break
            elif key == ord('r'):  # Reset all settings
                cv2.setTrackbarPos("Center X", control_window, 500)  # 0.0m
                cv2.setTrackbarPos("Center Y", control_window, 500)  # 0.0m
                cv2.setTrackbarPos("Center Z", control_window, 500)  # 0.0m
                cv2.setTrackbarPos("Distance", control_window, 500)  # 0.0m
                cv2.setTrackbarPos("Azimuth", control_window, 0)
                cv2.setTrackbarPos("Elevation", control_window, 90)  # 0 degrees (90 in trackbar)
                cv2.setTrackbarPos("Roll", control_window, 0)
                cv2.setTrackbarPos("Zoom", control_window, 100)
                cv2.setTrackbarPos("Pan X", control_window, 600)  # Center position
                cv2.setTrackbarPos("Pan Y", control_window, 600)  # Center position
                # Reset filters to disabled
                cv2.setTrackbarPos("Filt Min X", control_window, 0)
                cv2.setTrackbarPos("Filt Max X", control_window, 1000)
                cv2.setTrackbarPos("Filt Min Y", control_window, 0)
                cv2.setTrackbarPos("Filt Max Y", control_window, 1000)
                cv2.setTrackbarPos("Filt Min Z", control_window, 0)
                cv2.setTrackbarPos("Filt Max Z", control_window, 1000)
                cv2.setTrackbarPos("Filt Min Dist", control_window, 0)
                cv2.setTrackbarPos("Filt Max Dist", control_window, 1000)
                print("All settings reset")
            elif key == ord('s'):  # Save settings
                save_settings(control_window)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        vision.stop()
        cv2.destroyAllWindows()
        print("✅ Done")

if __name__ == "__main__":
    main()

