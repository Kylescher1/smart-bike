#!/usr/bin/env python3
"""
Real-time Encoder Data Plotter

Reads encoder data from EncoderStallDetector and plots it in real-time.
"""

import sys
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

class EncoderPlotter:
    def __init__(self, max_points=500):
        """
        Initialize encoder plotter
        
        Args:
            max_points: Maximum number of data points to keep in history
        """
        self.max_points = max_points
        
        # Data buffers
        self.time_data = deque(maxlen=max_points)
        self.position_data = deque(maxlen=max_points)
        self.velocity_data = deque(maxlen=max_points)
        self.is_moving_data = deque(maxlen=max_points)
        self.is_stalled_data = deque(maxlen=max_points)
        
        # Start time
        self.start_time = time.time()
        
        # Setup plot
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Encoder Real-Time Data', fontsize=14)
        
        # Position plot
        self.ax_pos = self.axes[0]
        self.line_pos, = self.ax_pos.plot([], [], 'b-', linewidth=1.5, label='Position')
        self.ax_pos.set_ylabel('Position (counts)')
        self.ax_pos.set_title('Encoder Position')
        self.ax_pos.grid(True, alpha=0.3)
        self.ax_pos.legend()
        
        # Velocity plot
        self.ax_vel = self.axes[1]
        self.line_vel, = self.ax_vel.plot([], [], 'g-', linewidth=1.5, label='Velocity')
        self.ax_vel.set_ylabel('Velocity (pulses/sec)')
        self.ax_vel.set_title('Encoder Velocity')
        self.ax_vel.grid(True, alpha=0.3)
        self.ax_vel.legend()
        
        # Status plot (moving/stalled)
        self.ax_status = self.axes[2]
        self.line_moving, = self.ax_status.plot([], [], 'g-', linewidth=2, label='Moving', alpha=0.7)
        self.line_stalled, = self.ax_status.plot([], [], 'r-', linewidth=2, label='Stalled', alpha=0.7)
        self.ax_status.set_ylabel('Status')
        self.ax_status.set_xlabel('Time (seconds)')
        self.ax_status.set_title('Motor Status')
        self.ax_status.set_ylim(-0.1, 1.1)
        self.ax_status.set_yticks([0, 1])
        self.ax_status.set_yticklabels(['No', 'Yes'])
        self.ax_status.grid(True, alpha=0.3)
        self.ax_status.legend()
        
        plt.tight_layout()
        
    def parse_line(self, line):
        """Parse a line of encoder data"""
        # Format: POS,<position>,<velocity>,<is_moving>,<is_stalled>,<pinA>,<pinB>
        match = re.match(r'POS,(-?\d+),(-?\d+\.?\d*),(\d),(\d),(\d),(\d)', line.strip())
        if match:
            position = int(match.group(1))
            velocity = float(match.group(2))
            is_moving = int(match.group(3))
            is_stalled = int(match.group(4))
            pin_a = int(match.group(5))
            pin_b = int(match.group(6))
            return position, velocity, is_moving, is_stalled, pin_a, pin_b
        return None
    
    def update_data(self, line):
        """Update data from a line of encoder output"""
        parsed = self.parse_line(line)
        if parsed:
            position, velocity, is_moving, is_stalled, pin_a, pin_b = parsed
            
            current_time = time.time() - self.start_time
            
            self.time_data.append(current_time)
            self.position_data.append(position)
            self.velocity_data.append(velocity)
            self.is_moving_data.append(is_moving)
            self.is_stalled_data.append(is_stalled)
            
            return True
        return False
    
    def update_plot(self, frame):
        """Update the plot animation"""
        if len(self.time_data) == 0:
            return []
        
        times = list(self.time_data)
        
        # Update position plot
        self.line_pos.set_data(times, list(self.position_data))
        self.ax_pos.relim()
        self.ax_pos.autoscale_view()
        
        # Update velocity plot
        self.line_vel.set_data(times, list(self.velocity_data))
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()
        
        # Update status plot
        self.line_moving.set_data(times, list(self.is_moving_data))
        self.line_stalled.set_data(times, list(self.is_stalled_data))
        self.ax_status.relim()
        self.ax_status.set_xlim(min(times) if times else 0, max(times) if times else 10)
        
        return [self.line_pos, self.line_vel, self.line_moving, self.line_stalled]
    
    def run(self):
        """Run the plotter, reading from stdin"""
        print("Encoder Plotter - Reading from stdin")
        print("Waiting for encoder data...")
        print("(Make sure encoder script is running and piping to this script)")
        print()
        
        # Setup animation
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)
        
        # Read from stdin
        try:
            for line in sys.stdin:
                line = line.strip()
                if line.startswith('POS,'):
                    self.update_data(line)
        except KeyboardInterrupt:
            print("\nStopping plotter...")
        
        plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time encoder data plotter')
    parser.add_argument('--max-points', type=int, default=500,
                       help='Maximum number of data points to display (default: 500)')
    
    args = parser.parse_args()
    
    plotter = EncoderPlotter(max_points=args.max_points)
    plotter.run()


if __name__ == "__main__":
    main()













