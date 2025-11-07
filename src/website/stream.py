import socketio
import time
import random
import threading

class BikeStreamer:
    def __init__(self, server_ip, server_port=5000):
        self.sio = socketio.Client(reconnection=True)
        self.server_url = f"http://{server_ip}:{server_port}"
        self.namespace = "/bike"
        self.connected = False

        @self.sio.event(namespace=self.namespace)
        def connect():
            self.connected = True
            print("✅ Connected to server")

        @self.sio.event(namespace=self.namespace)
        def disconnect():
            self.connected = False
            print("❌ Disconnected from server")

        @self.sio.on("bike_response", namespace=self.namespace)
        def bike_response(data):
            print("📡 Response from server:", data)

    def connect(self):
        self.sio.connect(self.server_url, namespaces=[self.namespace], transports=['websocket'])

    def send_data(self, data):
        if self.connected:
            self.sio.emit("bike_data", data, namespace=self.namespace)
        else:
            print("⚠️ Not connected")

    def start_fake_stream(self, interval=2):
        def loop():
            while True:
                if self.connected:
                    data = {
                        "speed": round(random.uniform(10, 30),2),
                        "battery": random.randint(70,100),
                        "imu": {
                            "ax": round(random.uniform(-1,1),3),
                            "ay": round(random.uniform(-1,1),3),
                            "az": round(random.uniform(-1,1),3),
                        },
                        "timestamp": time.time(),
                    }
                    self.send_data(data)
                    print("🚴 Sent:", data)
                time.sleep(interval)
        threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    SERVER_IP = "129.21.70.149"  # replace with your server LAN IP
    streamer = BikeStreamer(SERVER_IP)
    streamer.connect()
    streamer.start_fake_stream()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.sio.disconnect()
