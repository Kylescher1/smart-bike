#!/usr/bin/env python3
"""
Radar View Visualization for Vision System

Displays detected objects in two radar views:
1. Top-down view: Shows horizontal angles (theta) - bird's eye view
2. Front view: Shows both horizontal (theta) and vertical (alpha) angles

Usage:
    Can be called from debug.py or standalone with a vision system instance
"""

import cv2
import numpy as np
import time
import math
from typing import Optional, Dict, List


class RadarView:
    """Radar visualization for displaying object detections."""
    
    def __init__(self, vision_system, canvas_size: int = 600, max_range: float = 90.0):
        """
        Initialize radar view.
        
        Args:
            vision_system: Vision system instance with read() method
            canvas_size: Size of radar canvas in pixels (square)
            max_range: Maximum angle range to display (degrees)
        """
        self.vision = vision_system
        self.canvas_size = canvas_size
        self.max_range = max_range  # Maximum angle in degrees
        
        # Window names
        self.window_top_down = "Radar - Top Down View"
        self.window_front = "Radar - Front View"
        self.window_3d = "Radar - 3D Depth View"
        
        # Colors for different object types
        self.color_map = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'bicycle': (0, 255, 255),   # Yellow
            'motorcycle': (255, 0, 255), # Magenta
            'truck': (0, 0, 255),       # Red
            'bus': (128, 0, 128),       # Purple
            'default': (255, 255, 255)  # White
        }
        
        # Camera colors
        self.camera_colors = {
            'left': (0, 255, 0),   # Green
            'right': (255, 0, 0)   # Blue
        }
    
    def angle_to_rad(self, angle_deg: float) -> float:
        """Convert degrees to radians."""
        return math.radians(angle_deg)
    
    def draw_top_down_view(self, objects: List[Dict]) -> np.ndarray:
        """
        Draw top-down radar view showing horizontal angles (theta).
        Top of screen is forward (0 degrees).
        
        Args:
            objects: List of object dictionaries with 'theta' and 'camera' keys
            
        Returns:
            Canvas image with radar visualization
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        center = self.canvas_size // 2
        
        # Draw radar circles (concentric circles for angle reference)
        for i in range(1, 6):
            radius = int((self.canvas_size // 2) * (i / 5))
            cv2.circle(canvas, (center, center), radius, (50, 50, 50), 1)
        
        # Draw cardinal directions
        # Forward (top)
        cv2.line(canvas, (center, center), (center, 0), (100, 100, 100), 2)
        cv2.putText(canvas, "F", (center - 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Right (right side)
        cv2.line(canvas, (center, center), (self.canvas_size, center), (100, 100, 100), 2)
        cv2.putText(canvas, "R", (self.canvas_size - 25, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Left (left side)
        cv2.line(canvas, (center, center), (0, center), (100, 100, 100), 2)
        cv2.putText(canvas, "L", (5, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Back (bottom)
        cv2.line(canvas, (center, center), (center, self.canvas_size), (100, 100, 100), 2)
        cv2.putText(canvas, "B", (center - 10, self.canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw center point (vehicle position)
        cv2.circle(canvas, (center, center), 5, (255, 255, 255), -1)
        
        # Draw objects
        for obj in objects:
            theta = obj.get('theta', 0.0)  # Horizontal angle in degrees
            # Note: camera info not available in new format, using default
            obj_type = obj.get('name', 'default')
            obj_id = obj.get('ID', 0)
            confidence = obj.get('confidence', 0.0)
            
            # Convert theta to screen coordinates
            # Theta: 0 = forward, positive = right, negative = left
            # Screen: top = forward, right = right side
            theta_rad = self.angle_to_rad(theta)
            
            # Calculate position on radar (distance from center based on angle magnitude)
            # Normalize theta to max_range
            normalized_theta = max(-self.max_range, min(self.max_range, theta))
            distance_ratio = abs(normalized_theta) / self.max_range
            max_radius = self.canvas_size // 2 - 20
            radius = int(max_radius * distance_ratio)
            
            # Calculate x, y position
            # Forward is up (negative y in screen coords)
            x = center + int(radius * math.sin(theta_rad))
            y = center - int(radius * math.cos(theta_rad))
            
            # Use default color (camera info not in new format)
            color = (255, 255, 255)  # White for all objects
            
            # Draw object as circle with size based on confidence
            radius_obj = max(5, int(10 * confidence))
            cv2.circle(canvas, (x, y), radius_obj, color, -1)
            cv2.circle(canvas, (x, y), radius_obj, (255, 255, 255), 1)
            
            # Draw line from center to object
            cv2.line(canvas, (center, center), (x, y), color, 1)
            
            # Draw label
            label = f"{obj_type[:3]} {obj_id}"
            if confidence < 1.0:
                label += f" {confidence:.1f}"
            
            # Position label above object
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = x - label_w // 2
            label_y = y - radius_obj - 5
            
            # Draw label background
            cv2.rectangle(canvas, 
                         (label_x - 2, label_y - label_h - 2),
                         (label_x + label_w + 2, label_y + 2),
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(canvas, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title and info
        title = "Top-Down View (Theta)"
        cv2.putText(canvas, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add legend
        legend_y = self.canvas_size - 60
        cv2.putText(canvas, "Left Camera", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.camera_colors['left'], 1)
        cv2.putText(canvas, "Right Camera", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.camera_colors['right'], 1)
        
        return canvas
    
    def draw_front_view(self, objects: List[Dict]) -> np.ndarray:
        """
        Draw front view radar showing both horizontal (theta) and vertical (alpha) angles.
        Top of screen is up, center is forward.
        
        Args:
            objects: List of object dictionaries with 'theta', 'alpha', and 'camera' keys
            
        Returns:
            Canvas image with front view radar visualization
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        center_x = self.canvas_size // 2
        center_y = self.canvas_size // 2
        
        # Draw grid lines
        # Horizontal lines (for alpha/vertical angles)
        for i in range(-2, 3):
            y = center_y + int((i / 2) * (self.canvas_size // 2))
            if 0 <= y < self.canvas_size:
                cv2.line(canvas, (0, y), (self.canvas_size, y), (50, 50, 50), 1)
        
        # Vertical lines (for theta/horizontal angles)
        for i in range(-2, 3):
            x = center_x + int((i / 2) * (self.canvas_size // 2))
            if 0 <= x < self.canvas_size:
                cv2.line(canvas, (x, 0), (x, self.canvas_size), (50, 50, 50), 1)
        
        # Draw center crosshair
        cv2.line(canvas, (center_x, 0), (center_x, self.canvas_size), (100, 100, 100), 2)
        cv2.line(canvas, (0, center_y), (self.canvas_size, center_y), (100, 100, 100), 2)
        
        # Draw labels
        cv2.putText(canvas, "UP", (center_x - 15, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(canvas, "LEFT", (10, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(canvas, "RIGHT", (self.canvas_size - 70, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(canvas, "DOWN", (center_x - 30, self.canvas_size - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw center point
        cv2.circle(canvas, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Draw objects
        for obj in objects:
            theta = obj.get('theta', 0.0)  # Horizontal angle
            alpha = obj.get('alpha', 0.0)   # Vertical angle
            obj_type = obj.get('name', 'default')
            obj_id = obj.get('ID', 0)
            confidence = obj.get('confidence', 0.0)
            
            # Normalize angles to max_range
            normalized_theta = max(-self.max_range, min(self.max_range, theta))
            normalized_alpha = max(-self.max_range, min(self.max_range, alpha))
            
            # Convert to screen coordinates
            # X axis: theta (left/right), center = 0
            # Y axis: alpha (up/down), center = 0, but screen Y increases downward
            x_ratio = normalized_theta / self.max_range
            y_ratio = -normalized_alpha / self.max_range  # Negative because screen Y increases downward
            
            x = center_x + int(x_ratio * (self.canvas_size // 2 - 20))
            y = center_y + int(y_ratio * (self.canvas_size // 2 - 20))
            
            # Clamp to canvas bounds
            x = max(10, min(self.canvas_size - 10, x))
            y = max(10, min(self.canvas_size - 10, y))
            
            # Use default color (camera info not in new format)
            color = (255, 255, 255)  # White for all objects
            
            # Draw object as circle with size based on confidence
            radius_obj = max(5, int(10 * confidence))
            cv2.circle(canvas, (x, y), radius_obj, color, -1)
            cv2.circle(canvas, (x, y), radius_obj, (255, 255, 255), 1)
            
            # Draw crosshair lines for object position
            cv2.line(canvas, (x - radius_obj - 3, y), (x + radius_obj + 3, y), color, 1)
            cv2.line(canvas, (x, y - radius_obj - 3), (x, y + radius_obj + 3), color, 1)
            
            # Draw label
            label = f"{obj_type[:3]} {obj_id}"
            if confidence < 1.0:
                label += f" {confidence:.1f}"
            
            # Position label above object
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = x - label_w // 2
            label_y = y - radius_obj - 5
            
            # Draw label background
            cv2.rectangle(canvas, 
                         (label_x - 2, label_y - label_h - 2),
                         (label_x + label_w + 2, label_y + 2),
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(canvas, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title
        title = "Front View (Theta & Alpha)"
        cv2.putText(canvas, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add angle info
        info_text = f"X: Theta (L/R), Y: Alpha (U/D)"
        cv2.putText(canvas, info_text, (10, self.canvas_size - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Add legend
        legend_y = self.canvas_size - 60
        cv2.putText(canvas, "Left Camera", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.camera_colors['left'], 1)
        cv2.putText(canvas, "Right Camera", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.camera_colors['right'], 1)
        
        return canvas
    
    def draw_3d_depth_view(self, objects: List[Dict], max_depth: float = 50.0) -> np.ndarray:
        """
        Draw 3D depth view showing objects with depth information.
        Top-down view with depth as distance from center.
        
        Args:
            objects: List of object dictionaries with 'theta', 'depth' keys
            max_depth: Maximum depth to display in meters
            
        Returns:
            Canvas image with 3D depth radar visualization
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        center = self.canvas_size // 2
        
        # Draw depth circles (concentric circles for depth reference)
        for i in range(1, 6):
            radius = int((self.canvas_size // 2) * (i / 5))
            depth_value = (i / 5) * max_depth
            cv2.circle(canvas, (center, center), radius, (50, 50, 50), 1)
            # Label depth circles
            cv2.putText(canvas, f"{depth_value:.0f}m", (center + radius - 20, center - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Draw cardinal directions
        cv2.line(canvas, (center, center), (center, 0), (100, 100, 100), 2)
        cv2.putText(canvas, "F", (center - 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.line(canvas, (center, center), (self.canvas_size, center), (100, 100, 100), 2)
        cv2.putText(canvas, "R", (self.canvas_size - 25, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.line(canvas, (center, center), (0, center), (100, 100, 100), 2)
        cv2.putText(canvas, "L", (5, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.line(canvas, (center, center), (center, self.canvas_size), (100, 100, 100), 2)
        cv2.putText(canvas, "B", (center - 10, self.canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw center point (vehicle position)
        cv2.circle(canvas, (center, center), 5, (255, 255, 255), -1)
        
        # Draw objects with depth
        for obj in objects:
            theta = obj.get('theta', 0.0)  # Horizontal angle
            depth = obj.get('depth', 0.0)  # Depth in meters
            obj_type = obj.get('name', 'default')
            obj_id = obj.get('ID', 0)
            confidence = obj.get('confidence', 0.0)
            
            if depth <= 0 or depth > max_depth:
                continue  # Skip invalid or out-of-range depths
            
            # Convert theta to screen coordinates
            theta_rad = self.angle_to_rad(theta)
            
            # Calculate position based on depth and angle
            # Depth determines distance from center
            depth_ratio = depth / max_depth
            max_radius = self.canvas_size // 2 - 20
            radius = int(max_radius * depth_ratio)
            
            # Calculate x, y position
            x = center + int(radius * math.sin(theta_rad))
            y = center - int(radius * math.cos(theta_rad))
            
            # Color based on depth (closer = brighter/more red, farther = darker/more blue)
            depth_normalized = depth / max_depth
            color_r = int(255 * (1.0 - depth_normalized))
            color_b = int(255 * depth_normalized)
            color_g = int(128)
            color = (color_b, color_g, color_r)  # BGR format
            
            # Draw object as circle with size based on confidence
            radius_obj = max(5, int(10 * confidence))
            cv2.circle(canvas, (x, y), radius_obj, color, -1)
            cv2.circle(canvas, (x, y), radius_obj, (255, 255, 255), 1)
            
            # Draw line from center to object
            cv2.line(canvas, (center, center), (x, y), color, 1)
            
            # Draw label
            label = f"{obj_type[:3]} {obj_id}"
            label += f" {depth:.1f}m"
            
            # Position label above object
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = x - label_w // 2
            label_y = y - radius_obj - 5
            
            # Draw label background
            cv2.rectangle(canvas, 
                         (label_x - 2, label_y - label_h - 2),
                         (label_x + label_w + 2, label_y + 2),
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(canvas, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title
        title = "3D Depth View (Theta & Depth)"
        cv2.putText(canvas, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add depth legend
        legend_y = self.canvas_size - 80
        cv2.putText(canvas, "Close (red)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        cv2.putText(canvas, "Far (blue)", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        cv2.putText(canvas, f"Max Depth: {max_depth}m", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return canvas
    
    def run(self):
        """Run radar visualization loop."""
        print("Starting radar visualization...")
        print("Press 'q' to quit")
        
        try:
            cv2.namedWindow(self.window_top_down, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.window_front, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.window_3d, cv2.WINDOW_NORMAL)
            
            # Position windows
            cv2.moveWindow(self.window_top_down, 100, 100)
            cv2.moveWindow(self.window_front, 750, 100)
            cv2.moveWindow(self.window_3d, 100, 750)
            
            while True:
                # Get latest objects from vision system
                objects = self.vision.read()
                
                # Draw radar views
                top_down_canvas = self.draw_top_down_view(objects)
                front_canvas = self.draw_front_view(objects)
                depth_3d_canvas = self.draw_3d_depth_view(objects, max_depth=50.0)
                
                # Display
                cv2.imshow(self.window_top_down, top_down_canvas)
                cv2.imshow(self.window_front, front_canvas)
                cv2.imshow(self.window_3d, depth_3d_canvas)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\nRadar visualization interrupted")
        except Exception as e:
            print(f"Error in radar visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cv2.destroyWindow(self.window_top_down)
                cv2.destroyWindow(self.window_front)
                cv2.destroyWindow(self.window_3d)
            except:
                pass
            print("Radar visualization ended")


def show_radar_view(vision_system, canvas_size: int = 600, max_range: float = 90.0):
    """
    Convenience function to show radar view.
    
    Args:
        vision_system: Vision system instance with read() method
        canvas_size: Size of radar canvas in pixels
        max_range: Maximum angle range to display (degrees)
    """
    radar = RadarView(vision_system, canvas_size, max_range)
    radar.run()


if __name__ == "__main__":
    # Example usage - would need to be called with a vision system instance
    print("This module should be imported and used with a vision system instance")
    print("Example: from radar_view import show_radar_view; show_radar_view(vision_system)")

