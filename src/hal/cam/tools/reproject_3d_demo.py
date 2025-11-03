"""
Live demonstration of cv2.reprojectImageTo3D()

This script demonstrates the cv2.reprojectImageTo3D() function by:
1. Capturing live stereo frames
2. Computing disparity maps
3. Converting disparity to 3D point cloud using reprojectImageTo3D
4. Displaying multiple visualizations:
   - Disparity map (colorized)
   - Depth map (Z channel)
   - 3D point cloud visualization
   - X, Y, Z channels separately

Usage:
    python -m src.hal.cam.tools.reproject_3d_demo

Controls:
    Press 'q' to quit
    Press 's' to save current point cloud to file
"""

import cv2
import numpy as np
from pathlib import Path

from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Camera import open_stereo_pair
from src.hal.config import LEFT_INDEX, RIGHT_INDEX, SWAP_LR


def rectify_pair(left_frame, right_frame, calib):
    """Rectify a stereo pair using calibration maps."""
    left_map_x, left_map_y, right_map_x, right_map_y, _image_size, _Q = calib
    rectL = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)
    return rectL, rectR


def compute_disparity(grayL, grayR):
    """Compute disparity map from rectified stereo pair."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 6,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    return disp


def visualize_3d_points(points_3d, disparity, mask=None):
    """
    Create visualizations of the 3D point cloud.
    
    Returns:
        depth_map: Z channel visualization (grayscale)
        x_map: X channel visualization
        y_map: Y channel visualization
        point_cloud_view: Bird's-eye view of point cloud
    """
    # Extract X, Y, Z channels
    X = points_3d[:, :, 0]
    Y = points_3d[:, :, 1]
    Z = points_3d[:, :, 2]
    
    # Create mask for valid points (disparity > 0 and finite values)
    if mask is None:
        valid_mask = (disparity > 0) & np.isfinite(Z) & (Z > 0)
    else:
        valid_mask = mask & (disparity > 0) & np.isfinite(Z) & (Z > 0)
    
    # Depth map (Z channel) - normalized for visualization
    Z_valid = Z.copy()
    Z_valid[~valid_mask] = 0
    Z_max = np.percentile(Z_valid[valid_mask], 95) if np.any(valid_mask) else 1.0
    Z_normalized = np.clip(Z_valid / Z_max, 0, 1)
    depth_map = (Z_normalized * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    # X channel visualization (horizontal offset)
    X_valid = X.copy()
    X_valid[~valid_mask] = 0
    X_abs = np.abs(X_valid)
    X_max = np.percentile(X_abs[valid_mask], 95) if np.any(valid_mask) else 1.0
    if X_max > 0:
        X_normalized = np.clip(X_abs / X_max, 0, 1)
        x_map = (X_normalized * 255).astype(np.uint8)
        # Color: red for positive X, blue for negative X
        x_map_colored = np.zeros((*x_map.shape, 3), dtype=np.uint8)
        x_map_colored[:, :, 2] = np.where(X_valid >= 0, x_map, 0)  # Red for positive
        x_map_colored[:, :, 0] = np.where(X_valid < 0, x_map, 0)   # Blue for negative
    else:
        x_map_colored = np.zeros((*X.shape, 3), dtype=np.uint8)
    
    # Y channel visualization (vertical offset)
    Y_valid = Y.copy()
    Y_valid[~valid_mask] = 0
    Y_abs = np.abs(Y_valid)
    Y_max = np.percentile(Y_abs[valid_mask], 95) if np.any(valid_mask) else 1.0
    if Y_max > 0:
        Y_normalized = np.clip(Y_abs / Y_max, 0, 1)
        y_map = (Y_normalized * 255).astype(np.uint8)
        # Color: green for positive Y, magenta for negative Y
        y_map_colored = np.zeros((*y_map.shape, 3), dtype=np.uint8)
        y_map_colored[:, :, 1] = np.where(Y_valid >= 0, y_map, 0)  # Green for positive
        y_map_colored[:, :, 0] = np.where(Y_valid < 0, y_map // 2, 0)  # Blue for negative
        y_map_colored[:, :, 2] = np.where(Y_valid < 0, y_map // 2, 0)  # Red for negative (makes magenta)
    else:
        y_map_colored = np.zeros((*Y.shape, 3), dtype=np.uint8)
    
    # Point cloud bird's-eye view (XY projection colored by Z)
    h, w = Z.shape
    point_cloud_view = np.zeros((h, w, 3), dtype=np.uint8)
    if np.any(valid_mask):
        # Sample points for visualization (too many points to display all)
        step = max(1, int(np.sqrt(np.sum(valid_mask)) / 200))
        y_coords, x_coords = np.where(valid_mask[::step, ::step])
        y_coords *= step
        x_coords *= step
        
        if len(y_coords) > 0:
            # Normalize Z values for coloring
            z_sample = Z[y_coords, x_coords]
            z_normalized = (z_sample - z_sample.min()) / (z_sample.max() - z_sample.min() + 1e-6)
            
            # Create color based on Z (depth)
            colors = cv2.applyColorMap((z_normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            
            # Draw points
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                color = tuple(map(int, colors[i, 0]))
                cv2.circle(point_cloud_view, (x, y), 2, color, -1)
    
    return depth_map, x_map_colored, y_map_colored, point_cloud_view


def save_point_cloud(points_3d, disparity, filename="pointcloud.xyz"):
    """Save 3D point cloud to file (XYZ format)."""
    valid_mask = (disparity > 0) & np.isfinite(points_3d[:, :, 2]) & (points_3d[:, :, 2] > 0)
    points = points_3d[valid_mask]
    
    if len(points) > 0:
        output_path = Path(filename)
        # Save as XYZ format
        with open(output_path, 'w') as f:
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"Saved {len(points)} points to {output_path}")
    else:
        print("No valid points to save")


def main():
    """Main function to demonstrate cv2.reprojectImageTo3D() with live output."""
    print("Loading calibration...")
    try:
        calib = load_calibration()
        *_, Q = calib
        print(f"Q matrix shape: {Q.shape}")
        print(f"Q matrix:\n{Q}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure stereo calibration is completed.")
        return
    
    print("Opening stereo cameras...")
    left_cam, right_cam = open_stereo_pair(LEFT_INDEX, RIGHT_INDEX)
    
    # Create windows
    cv2.namedWindow("Disparity Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Map (Z)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("X Channel", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Y Channel", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3D Point Cloud View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original Left", cv2.WINDOW_NORMAL)
    
    print("\n" + "="*60)
    print("cv2.reprojectImageTo3D() Live Demonstration")
    print("="*60)
    print("\nHow it works:")
    print("  For each pixel (x, y) with disparity d:")
    print("    [X, Y, Z, W]^T = Q * [x, y, d, 1]^T")
    print("  Then divide by W:")
    print("    X = X / W, Y = Y / W, Z = Z / W")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current point cloud")
    print("="*60 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frames
            left_frame = left_cam.get_frame()
            right_frame = right_cam.get_frame()
            
            if left_frame is None or right_frame is None:
                continue
            
            if SWAP_LR:
                left_frame, right_frame = right_frame, left_frame
            
            # Rectify stereo pair
            rectL, rectR = rectify_pair(left_frame, right_frame, calib)
            
            # Convert to grayscale
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY) if rectL.ndim == 3 else rectL
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY) if rectR.ndim == 3 else rectR
            
            # Compute disparity
            disparity = compute_disparity(grayL, grayR)
            
            # Apply cv2.reprojectImageTo3D() - THE MAIN DEMONSTRATION
            # This converts disparity map to 3D point cloud using Q matrix
            points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False, ddepth=cv2.CV_32F)
            
            # Handle invalid points (disparity <= 0)
            mask = disparity > 0
            
            # Visualize results
            depth_map, x_map, y_map, point_cloud_view = visualize_3d_points(points_3d, disparity, mask)
            
            # Display disparity map (colorized)
            disp_color = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_color = cv2.applyColorMap(disp_color.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.putText(disp_color, "Disparity Map", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Disparity Map", disp_color)
            
            # Display depth map (Z channel)
            cv2.putText(depth_map, "Depth Map (Z channel)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Add stats
            valid_z = points_3d[mask, 2]
            if len(valid_z) > 0:
                mean_z = np.mean(valid_z)
                min_z = np.min(valid_z)
                max_z = np.max(valid_z)
                cv2.putText(depth_map, f"Mean Z: {mean_z:.2f}mm", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(depth_map, f"Range: [{min_z:.1f}, {max_z:.1f}]mm", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Depth Map (Z)", depth_map)
            
            # Display X channel
            cv2.putText(x_map, "X Channel (Horizontal offset)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("X Channel", x_map)
            
            # Display Y channel
            cv2.putText(y_map, "Y Channel (Vertical offset)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Y Channel", y_map)
            
            # Display point cloud view
            cv2.putText(point_cloud_view, "3D Point Cloud (XY projection)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("3D Point Cloud View", point_cloud_view)
            
            # Display original left frame
            cv2.putText(rectL, "Original Left (Rectified)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Original Left", rectL)
            
            # Print statistics every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                valid_points = np.sum(mask)
                total_points = mask.size
                print(f"Frame {frame_count}: {valid_points}/{total_points} valid points "
                      f"({100*valid_points/total_points:.1f}%)")
            
            # Handle keyboard input
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_point_cloud(points_3d, disparity, f"pointcloud_{timestamp}.xyz")
                
    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()
        print("\nDemonstration ended.")


if __name__ == "__main__":
    main()

