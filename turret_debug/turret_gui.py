#!/usr/bin/env python3
"""
Turret Control GUI - Graphical User Interface
Modern GUI for turret debugging and control

Usage:
    python turret_gui.py [port]
    
    port: Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
"""

import serial
import sys
import time
import threading
import argparse
from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math


class TurretController:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.top_pos = 90
        self.bottom_pos = 90
        self.top_min = 60
        self.top_max = 120
        self.bottom_min = 0
        self.bottom_max = 180
        self.motor1_speed = 0
        self.motor2_speed = 0
        
    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            time.sleep(2)
            return True
        except serial.SerialException as e:
            return False
    
    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def send_command(self, command: str, read_response: bool = True) -> Optional[str]:
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            self.ser.write((command + '\n').encode())
            self.ser.flush()
            
            if not read_response:
                return None
                
            response = ""
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response += line + "\n"
                        if line.startswith("OK:") or line.startswith("ERROR:"):
                            break
                time.sleep(0.01)
            return response.strip() if response else None
        except:
            return None
    
    def update_status(self):
        """Update internal status from Arduino"""
        resp = self.send_command("STATUS")
        if resp:
            for line in resp.split('\n'):
                if 'Top servo position:' in line:
                    try:
                        self.top_pos = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Bottom servo position:' in line:
                    try:
                        self.bottom_pos = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Top limits' in line:
                    try:
                        parts = line.split('MIN:')[1].split(',')
                        self.top_min = int(parts[0].strip())
                        self.top_max = int(parts[1].split('MAX:')[1].strip())
                    except:
                        pass
                elif 'Bottom limits' in line:
                    try:
                        parts = line.split('MIN:')[1].split(',')
                        self.bottom_min = int(parts[0].strip())
                        self.bottom_max = int(parts[1].split('MAX:')[1].strip())
                    except:
                        pass


class TurretGUI:
    def __init__(self, controller: TurretController):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Turret Control System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.step_size = 5
        self.fine_step = 1
        self.update_thread = None
        self.running = False
        
        self.setup_ui()
        self.setup_keyboard_bindings()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = tk.Label(main_frame, text="TURRET CONTROL SYSTEM", 
                               font=('Arial', 20, 'bold'), bg='#2b2b2b', fg='#00ff00')
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Left column - Turret visualization
        left_frame = ttk.LabelFrame(main_frame, text="Turret Status", padding="10")
        left_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Canvas for turret visualization
        self.canvas = tk.Canvas(left_frame, width=400, height=300, bg='#1e1e1e', highlightthickness=0)
        self.canvas.grid(row=0, column=0, pady=10)
        
        # Position displays
        pos_frame = ttk.Frame(left_frame)
        pos_frame.grid(row=1, column=0, pady=10)
        
        # Top servo
        top_frame = ttk.Frame(pos_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        tk.Label(top_frame, text="Top Servo:", font=('Arial', 12, 'bold'), bg='#2b2b2b', fg='#ffffff').grid(row=0, column=0, padx=5)
        self.top_pos_label = tk.Label(top_frame, text="90°", font=('Arial', 12), bg='#2b2b2b', fg='#00ff00', width=10)
        self.top_pos_label.grid(row=0, column=1, padx=5)
        self.top_pos_bar = ttk.Progressbar(top_frame, length=200, mode='determinate', maximum=180)
        self.top_pos_bar.grid(row=0, column=2, padx=5)
        
        # Bottom servo
        bottom_frame = ttk.Frame(pos_frame)
        bottom_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        tk.Label(bottom_frame, text="Bottom Servo:", font=('Arial', 12, 'bold'), bg='#2b2b2b', fg='#ffffff').grid(row=0, column=0, padx=5)
        self.bottom_pos_label = tk.Label(bottom_frame, text="90°", font=('Arial', 12), bg='#2b2b2b', fg='#00ff00', width=10)
        self.bottom_pos_label.grid(row=0, column=1, padx=5)
        self.bottom_pos_bar = ttk.Progressbar(bottom_frame, length=200, mode='determinate', maximum=180)
        self.bottom_pos_bar.grid(row=0, column=2, padx=5)
        
        # Limits display
        limits_frame = ttk.LabelFrame(left_frame, text="Limits", padding="5")
        limits_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        self.limits_label = tk.Label(limits_frame, 
                                     text="Top: 60°-120° | Bottom: 0°-180°",
                                     font=('Arial', 10), bg='#2b2b2b', fg='#ffff00')
        self.limits_label.pack()
        
        # Right column - Controls
        right_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Movement controls
        move_frame = ttk.LabelFrame(right_frame, text="Movement", padding="5")
        move_frame.pack(fill=tk.X, pady=5)
        
        # Top servo controls
        tk.Label(move_frame, text="Top Servo:", font=('Arial', 10, 'bold'), bg='#2b2b2b', fg='#ffffff').pack(anchor=tk.W)
        top_btn_frame = ttk.Frame(move_frame)
        top_btn_frame.pack(pady=5)
        ttk.Button(top_btn_frame, text="▲", command=lambda: self.move_top(self.step_size), width=3).grid(row=0, column=1, padx=2)
        ttk.Button(top_btn_frame, text="▼", command=lambda: self.move_top(-self.step_size), width=3).grid(row=2, column=1, padx=2)
        ttk.Button(top_btn_frame, text="◄", command=lambda: self.move_top(-self.fine_step), width=3).grid(row=1, column=0, padx=2)
        ttk.Button(top_btn_frame, text="►", command=lambda: self.move_top(self.fine_step), width=3).grid(row=1, column=2, padx=2)
        
        # Bottom servo controls
        tk.Label(move_frame, text="Bottom Servo:", font=('Arial', 10, 'bold'), bg='#2b2b2b', fg='#ffffff').pack(anchor=tk.W, pady=(10,0))
        bottom_btn_frame = ttk.Frame(move_frame)
        bottom_btn_frame.pack(pady=5)
        ttk.Button(bottom_btn_frame, text="▲", command=lambda: self.move_bottom(self.step_size), width=3).grid(row=0, column=1, padx=2)
        ttk.Button(bottom_btn_frame, text="▼", command=lambda: self.move_bottom(-self.step_size), width=3).grid(row=2, column=1, padx=2)
        ttk.Button(bottom_btn_frame, text="◄", command=lambda: self.move_bottom(-self.fine_step), width=3).grid(row=1, column=0, padx=2)
        ttk.Button(bottom_btn_frame, text="►", command=lambda: self.move_bottom(self.fine_step), width=3).grid(row=1, column=2, padx=2)
        
        # Quick actions
        action_frame = ttk.LabelFrame(right_frame, text="Quick Actions", padding="5")
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Home", command=self.home).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Reset Limits", command=self.reset_limits).pack(fill=tk.X, pady=2)
        
        # Motor controls
        motor_frame = ttk.LabelFrame(right_frame, text="Motors", padding="5")
        motor_frame.pack(fill=tk.X, pady=5)
        
        # Motor 1
        m1_frame = ttk.Frame(motor_frame)
        m1_frame.pack(fill=tk.X, pady=2)
        tk.Label(m1_frame, text="Motor 1:", bg='#2b2b2b', fg='#ffffff').pack(side=tk.LEFT)
        self.motor1_var = tk.IntVar(value=0)
        self.motor1_scale = ttk.Scale(m1_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                     variable=self.motor1_var, command=self.update_motor1)
        self.motor1_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.motor1_label = tk.Label(m1_frame, text="0", bg='#2b2b2b', fg='#ffffff', width=4)
        self.motor1_label.pack(side=tk.LEFT)
        
        # Motor 2
        m2_frame = ttk.Frame(motor_frame)
        m2_frame.pack(fill=tk.X, pady=2)
        tk.Label(m2_frame, text="Motor 2:", bg='#2b2b2b', fg='#ffffff').pack(side=tk.LEFT)
        self.motor2_var = tk.IntVar(value=0)
        self.motor2_scale = ttk.Scale(m2_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.motor2_var, command=self.update_motor2)
        self.motor2_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.motor2_label = tk.Label(m2_frame, text="0", bg='#2b2b2b', fg='#ffffff', width=4)
        self.motor2_label.pack(side=tk.LEFT)
        
        # Presets
        preset_frame = ttk.LabelFrame(right_frame, text="Presets", padding="5")
        preset_frame.pack(fill=tk.X, pady=5)
        
        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.pack()
        for i in range(1, 10):
            btn = ttk.Button(preset_btn_frame, text=str(i), command=lambda n=i: self.set_preset(n), width=3)
            btn.grid(row=(i-1)//3, column=(i-1)%3, padx=2, pady=2)
        
        # Status/Log
        log_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="5")
        log_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.rowconfigure(2, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, bg='#1e1e1e', fg='#00ff00', 
                                                   font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Connection status
        self.status_label = tk.Label(main_frame, text="● Disconnected", font=('Arial', 10), 
                                    bg='#2b2b2b', fg='#ff0000')
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)
        
    def setup_keyboard_bindings(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<KeyPress-w>', lambda e: self.move_top(self.step_size))
        self.root.bind('<KeyPress-W>', lambda e: self.move_top(self.step_size))
        self.root.bind('<KeyPress-s>', lambda e: self.move_top(-self.step_size))
        self.root.bind('<KeyPress-S>', lambda e: self.move_top(-self.step_size))
        self.root.bind('<KeyPress-a>', lambda e: self.move_bottom(-self.step_size))
        self.root.bind('<KeyPress-A>', lambda e: self.move_bottom(-self.step_size))
        self.root.bind('<KeyPress-d>', lambda e: self.move_bottom(self.step_size))
        self.root.bind('<KeyPress-D>', lambda e: self.move_bottom(self.step_size))
        self.root.bind('<KeyPress-q>', lambda e: self.move_top(self.fine_step))
        self.root.bind('<KeyPress-Q>', lambda e: self.move_top(self.fine_step))
        self.root.bind('<KeyPress-e>', lambda e: self.move_top(-self.fine_step))
        self.root.bind('<KeyPress-E>', lambda e: self.move_top(-self.fine_step))
        self.root.bind('<KeyPress-z>', lambda e: self.move_bottom(-self.fine_step))
        self.root.bind('<KeyPress-Z>', lambda e: self.move_bottom(-self.fine_step))
        self.root.bind('<KeyPress-x>', lambda e: self.move_bottom(self.fine_step))
        self.root.bind('<KeyPress-X>', lambda e: self.move_bottom(self.fine_step))
        self.root.bind('<KeyPress-space>', lambda e: self.home())
        self.root.bind('<KeyPress-h>', lambda e: self.home())
        self.root.bind('<KeyPress-H>', lambda e: self.home())
        self.root.bind('<Up>', lambda e: self.move_top(self.step_size))
        self.root.bind('<Down>', lambda e: self.move_top(-self.step_size))
        self.root.bind('<Left>', lambda e: self.move_bottom(-self.step_size))
        self.root.bind('<Right>', lambda e: self.move_bottom(self.step_size))
        
        # Number keys for presets
        for i in range(1, 10):
            self.root.bind(f'<KeyPress-{i}>', lambda e, n=i: self.set_preset(n))
        
    def log(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        
    def move_top(self, delta: int):
        new_pos = self.controller.top_pos + delta
        new_pos = max(self.controller.top_min, min(self.controller.top_max, new_pos))
        if new_pos != self.controller.top_pos:
            self.controller.send_command(f"TOP:{new_pos}", read_response=False)
            self.controller.top_pos = new_pos
            self.update_display()
            self.log(f"Top servo: {new_pos}°")
    
    def move_bottom(self, delta: int):
        new_pos = self.controller.bottom_pos + delta
        new_pos = max(self.controller.bottom_min, min(self.controller.bottom_max, new_pos))
        if new_pos != self.controller.bottom_pos:
            self.controller.send_command(f"BOTTOM:{new_pos}", read_response=False)
            self.controller.bottom_pos = new_pos
            self.update_display()
            self.log(f"Bottom servo: {new_pos}°")
    
    def home(self):
        self.controller.send_command("HOME", read_response=False)
        self.controller.top_pos = 90
        self.controller.bottom_pos = 90
        self.update_display()
        self.log("Moved to home position")
    
    def reset_limits(self):
        self.controller.send_command("SET_TOP_MIN:60", read_response=False)
        self.controller.send_command("SET_TOP_MAX:120", read_response=False)
        self.controller.send_command("SET_BOTTOM_MIN:0", read_response=False)
        self.controller.send_command("SET_BOTTOM_MAX:180", read_response=False)
        self.controller.top_min = 60
        self.controller.top_max = 120
        self.controller.bottom_min = 0
        self.controller.bottom_max = 180
        self.update_display()
        self.log("Limits reset")
    
    def set_preset(self, num: int):
        presets = {
            1: (45, 45), 2: (90, 90), 3: (135, 135),
            4: (45, 135), 5: (135, 45), 6: (0, 90),
            7: (180, 90), 8: (90, 0), 9: (90, 180),
        }
        if num in presets:
            top, bottom = presets[num]
            self.controller.send_command(f"TOP:{top}", read_response=False)
            self.controller.send_command(f"BOTTOM:{bottom}", read_response=False)
            self.controller.top_pos = top
            self.controller.bottom_pos = bottom
            self.update_display()
            self.log(f"Preset {num}: Top={top}°, Bottom={bottom}°")
    
    def update_motor1(self, value):
        speed = int(float(value))
        self.motor1_label.config(text=str(speed))
        self.controller.send_command(f"MOTOR1:{speed}", read_response=False)
        self.controller.motor1_speed = speed
    
    def update_motor2(self, value):
        speed = int(float(value))
        self.motor2_label.config(text=str(speed))
        self.controller.send_command(f"MOTOR2:{speed}", read_response=False)
        self.controller.motor2_speed = speed
    
    def draw_turret(self):
        """Draw turret visualization on canvas"""
        self.canvas.delete("all")
        width = 400
        height = 300
        center_x = width // 2
        center_y = height // 2
        
        # Draw base
        self.canvas.create_oval(center_x - 60, center_y + 80, center_x + 60, center_y + 140,
                               fill='#444444', outline='#666666', width=2)
        
        # Draw bottom servo arm (rotates around center)
        bottom_angle = math.radians(self.controller.bottom_pos - 90)
        bottom_arm_length = 50
        bottom_end_x = center_x + math.cos(bottom_angle) * bottom_arm_length
        bottom_end_y = center_y + 80 + math.sin(bottom_angle) * bottom_arm_length
        
        self.canvas.create_line(center_x, center_y + 110, bottom_end_x, bottom_end_y,
                               fill='#00ff00', width=4)
        self.canvas.create_circle(bottom_end_x, bottom_end_y, 8, fill='#00ff00', outline='#ffffff')
        
        # Draw top servo mount
        top_mount_x = bottom_end_x
        top_mount_y = bottom_end_y - 20
        
        self.canvas.create_rectangle(top_mount_x - 15, top_mount_y - 10,
                                    top_mount_x + 15, top_mount_y + 10,
                                    fill='#555555', outline='#777777', width=2)
        
        # Draw top servo arm
        top_angle = math.radians(self.controller.top_pos - 90)
        top_arm_length = 40
        top_end_x = top_mount_x + math.cos(top_angle) * top_arm_length
        top_end_y = top_mount_y + math.sin(top_angle) * top_arm_length
        
        self.canvas.create_line(top_mount_x, top_mount_y, top_end_x, top_end_y,
                               fill='#ff00ff', width=4)
        self.canvas.create_circle(top_end_x, top_end_y, 6, fill='#ff00ff', outline='#ffffff')
        
        # Draw angle indicators
        self.canvas.create_text(center_x, height - 20, text=f"Bottom: {self.controller.bottom_pos}°",
                               fill='#00ff00', font=('Arial', 10))
        self.canvas.create_text(top_mount_x, top_mount_y - 25, text=f"Top: {self.controller.top_pos}°",
                               fill='#ff00ff', font=('Arial', 10))
    
    def update_display(self):
        """Update all display elements"""
        # Update position labels
        self.top_pos_label.config(text=f"{self.controller.top_pos}°")
        self.bottom_pos_label.config(text=f"{self.controller.bottom_pos}°")
        
        # Update progress bars
        self.top_pos_bar['value'] = self.controller.top_pos
        self.bottom_pos_bar['value'] = self.controller.bottom_pos
        
        # Update limits
        self.limits_label.config(
            text=f"Top: {self.controller.top_min}°-{self.controller.top_max}° | "
                 f"Bottom: {self.controller.bottom_min}°-{self.controller.bottom_max}°"
        )
        
        # Update motor displays
        self.motor1_var.set(self.controller.motor1_speed)
        self.motor1_label.config(text=str(self.controller.motor1_speed))
        self.motor2_var.set(self.controller.motor2_speed)
        self.motor2_label.config(text=str(self.controller.motor2_speed))
        
        # Redraw turret
        self.draw_turret()
    
    def status_update_loop(self):
        """Background thread to update status"""
        while self.running:
            try:
                self.controller.update_status()
                self.root.after(0, self.update_display)
                time.sleep(1.0)  # Update every second
            except:
                pass
    
    def start(self):
        """Start the GUI"""
        # Try to connect
        self.log("Connecting to turret...")
        if self.controller.connect():
            self.status_label.config(text="● Connected", fg='#00ff00')
            self.log("Connected successfully!")
            self.controller.update_status()
            self.update_display()
            
            # Start status update thread
            self.running = True
            self.update_thread = threading.Thread(target=self.status_update_loop, daemon=True)
            self.update_thread.start()
        else:
            self.status_label.config(text="● Connection Failed", fg='#ff0000')
            self.log("Failed to connect! Check port and try again.")
            messagebox.showerror("Connection Error", 
                               f"Failed to connect to {self.controller.port}\n\n"
                               "Please check:\n"
                               "- Port is correct\n"
                               "- Arduino is connected\n"
                               "- No other program is using the port")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI loop
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window close"""
        self.running = False
        self.log("Shutting down...")
        try:
            self.controller.send_command("HOME", read_response=False)
            self.controller.send_command("MOTOR1:0", read_response=False)
            self.controller.send_command("MOTOR2:0", read_response=False)
        except:
            pass
        self.controller.disconnect()
        self.root.destroy()


# Add create_circle method to Canvas
def create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = create_circle


def list_serial_ports():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
    else:
        print("No serial ports found")


def main():
    parser = argparse.ArgumentParser(description='Turret Control GUI')
    parser.add_argument('port', nargs='?', help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--list-ports', '-l', action='store_true', help='List available serial ports')
    parser.add_argument('--baudrate', '-b', type=int, default=115200, help='Baud rate (default: 115200)')
    
    args = parser.parse_args()
    
    if args.list_ports:
        list_serial_ports()
        return
    
    if not args.port:
        print("Error: Serial port required")
        print("Use --list-ports to see available ports")
        parser.print_help()
        sys.exit(1)
    
    controller = TurretController(args.port, baudrate=args.baudrate)
    app = TurretGUI(controller)
    app.start()


if __name__ == '__main__':
    main()

