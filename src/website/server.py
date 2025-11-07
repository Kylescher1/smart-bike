from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit, join_room
import cv2, threading, time, io, base64, numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ===== Video Capture =====
camera = cv2.VideoCapture(0)  # USB / IP camera

def generate_video():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ===== Shared Data =====
data_lock = threading.Lock()
latest_value = 0.0
latest_plot = ""

# ===== Generate Matplotlib Plot =====
def generate_plot():
    y = np.linspace(0, 2*np.pi, 100)
    f = np.sin(y + time.time())
    fig, ax = plt.subplots()
    ax.plot(y, f)
    ax.set_title("Live Sine Wave")
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ===== Background Data Thread =====
def data_updater():
    global latest_value, latest_plot
    while True:
        with data_lock:
            latest_value = np.sin(time.time()) + np.random.normal(0, 0.05)
            latest_plot = generate_plot()
        # Push all data in one WebSocket event
        socketio.emit("dashboard_update", {
            "value": latest_value,
            "plot": latest_plot
        }, namespace="/viewer", room="viewers")
        time.sleep(1)  # update every second

threading.Thread(target=data_updater, daemon=True).start()

# ===== Handle Bike Data =====
@socketio.on("bike_data", namespace="/bike")
def handle_bike_data(data):
    print("[Bike Stream]", data)
    # Broadcast to viewers
    socketio.emit("dashboard_update", {
        "bike": data
    }, namespace="/viewer", room="viewers")
    # Example response
    emit("bike_response", {"config": {"max_speed": 35, "units": "km/h"}})

# ===== Viewer Connections =====
@socketio.on("connect", namespace="/viewer")
def viewer_connect():
    sid = request.sid
    join_room("viewers")
    print(f"[Viewer Connected] SID: {sid}")

# ===== Dashboard Page =====
@app.route("/")
def index():
    return render_template("index.html")

# ===== Main =====
if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"Open http://{ip}:5000 on your LAN")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
