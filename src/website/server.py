from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import io, base64, threading, time, matplotlib.pyplot as plt

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ====== FAKE DATA GENERATOR (Replace with ISP feed) ======
data_lock = threading.Lock()
latest_value = 0.0

def data_updater():
    global latest_value
    while True:
        with data_lock:
            latest_value = np.sin(time.time()) + np.random.normal(0, 0.05)
        socketio.emit("data_update", {"value": latest_value})
        time.sleep(1)  # update every second

threading.Thread(target=data_updater, daemon=True).start()

# ====== MATPLOTLIB IMAGE GENERATOR ======
def generate_plot():
    with data_lock:
        y = np.linspace(0, 2*np.pi, 100)
        f = np.sin(y + time.time())
    fig, ax = plt.subplots()
    ax.plot(y, f)
    ax.set_title("Live Sine Wave")
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/plot")
def api_plot():
    return jsonify({"img": generate_plot()})

# ====== VIDEO STREAM (USB / IP camera) ======
camera = cv2.VideoCapture(0)  # change to IP stream or other index

def generate_video():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ====== ENTRY POINT ======
if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"Open http://{ip}:5000 on your LAN")
    socketio.run(app, host="0.0.0.0", port=5000)
