import serial
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

# ---------------- Serial Connection ---------------- #

PORT = "COM5"     # CHANGE THIS TO YOUR ESP32 PORT
BAUD = 115200

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except:
    ser = None
    print("Could not open serial port. GUI will still load.")


def send(cmd):
    """Send a command string over serial."""
    if ser:
        ser.write((cmd + "\n").encode())
    else:
        print(f"(No serial) Sent: {cmd}")


# ---------------- Current Servo Angles ---------------- #
# GUI tracks angles so it can store "points"
current_s1 = 41    # default home angle
current_s2 = 90

# step size for each WASD click
STEP = 2

# ---------------- Movement Functions ---------------- #

def up():
    global current_s1
    current_s1 += STEP
    send("W")

def down():
    global current_s1
    current_s1 -= STEP
    send("S")

def left():
    global current_s2
    current_s2 -= STEP
    send("A")

def right():
    global current_s2
    current_s2 += STEP
    send("D")

def home():
    global current_s1, current_s2
    current_s1 = 41
    current_s2 = 90
    send("H")


# ---------------- Points Storage ---------------- #

points = {}   # dictionary: {"pointName": (s1, s2)}

def create_point():
    global points, current_s1, current_s2

    name = simpledialog.askstring("Point Name", "Enter a name for this point:")
    if not name:
        return

    points[name] = (current_s1, current_s2)
    update_points_menu()

def clear_points():
    global points
    if messagebox.askyesno("Confirm", "Clear all saved points?"):
        points.clear()
        update_points_menu()

def send_to_point():
    """Send turret to selected saved point."""
    selection = point_var.get()
    if selection in points:
        s1, s2 = points[selection]
        cmd = f"S1:{s1},S2:{s2}"
        send(cmd)
    else:
        messagebox.showinfo("Error", "No point selected.")

def update_points_menu():
    """Refresh dropdown menu when points change."""
    menu = point_dropdown["menu"]
    menu.delete(0, "end")

    if points:
        for name in points:
            menu.add_command(label=name, command=lambda v=name: point_var.set(v))
        point_var.set(list(points.keys())[0])
    else:
        point_var.set("No Points")


# ---------------- GUI Setup ---------------- #

root = tk.Tk()
root.title("ESP32 Turret Controller")
root.geometry("")
root.resizable(False, False)

label = tk.Label(root, text="Turret Control", font=("Arial", 18, "bold"))
label.pack(pady=10)

# ---------------- Movement Buttons ---------------- #

frame = tk.Frame(root)
frame.pack(pady=10)

btn_up = tk.Button(frame, text="↑", font=("Arial", 24), width=4, height=2, command=up)
btn_left = tk.Button(frame, text="←", font=("Arial", 24), width=4, height=2, command=left)
btn_right = tk.Button(frame, text="→", font=("Arial", 24), width=4, height=2, command=right)
btn_down = tk.Button(frame, text="↓", font=("Arial", 24), width=4, height=2, command=down)

btn_up.grid(row=0, column=1)
btn_left.grid(row=1, column=0)
btn_right.grid(row=1, column=2)
btn_down.grid(row=2, column=1)

btn_home = tk.Button(root, text="HOME", font=("Arial", 16), width=12, height=2, command=home)
btn_home.pack(pady=10)

# ---------------- Point Management UI ---------------- #

section = tk.Label(root, text="Saved Points", font=("Arial", 16))
section.pack(pady=5)

point_var = tk.StringVar(value="No Points")

point_dropdown = ttk.OptionMenu(root, point_var, "No Points")
point_dropdown.pack(pady=5)

btn_create = tk.Button(root, text="Create Point", font=("Arial", 14), width=15, command=create_point)
btn_create.pack(pady=5)

btn_go = tk.Button(root, text="Go To Point", font=("Arial", 14), width=15, command=send_to_point)
btn_go.pack(pady=5)

btn_clear = tk.Button(root, text="Clear Points", font=("Arial", 14), width=15, command=clear_points)
btn_clear.pack(pady=5)


# ---------------- Keyboard Bindings ---------------- #

def key_pressed(event):
    key = event.char.lower()
    if key == 'w': up()
    elif key == 's': down()
    elif key == 'a': left()
    elif key == 'd': right()
    elif key == 'h': home()

root.bind("<Key>", key_pressed)

# ---------------- Run GUI ---------------- #

root.mainloop()
