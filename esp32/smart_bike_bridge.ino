/*
 * Smart Bike ESP32 Data Bridge
 * 
 * This ESP32 acts as a WiFi Access Point and web server to receive data
 * from the ROCK Pi (camera stream, depth map, LIDAR) and serve it to clients
 * for visualization.
 * 
 * Requirements:
 * - ESP32 board with WiFi support
 * - Arduino ESP32 board package installed
 * - WebSockets library (use Library Manager: WebSockets by Markus Sattler)
 * 
 * Pin Connections:
 * - None required for WiFi communication
 * - Optional: Status LED on GPIO 2
 */

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ========== Configuration ==========
const char* AP_SSID = "SmartBike_AP";
const char* AP_PASSWORD = "smartbike123";  // Change this!
const int AP_CHANNEL = 1;

// Server ports
const int WEB_PORT = 80;
const int WS_PORT = 81;

// Status LED (built-in LED on most ESP32 boards)
#define LED_PIN 2

// ========== Global Objects ==========
WebServer server(WEB_PORT);
WebSocketsServer webSocket = WebSocketsServer(WS_PORT);

// Data buffers
String cameraFrameBase64 = "";
String depthMapBase64 = "";
String lidarDataJSON = "[]";
unsigned long lastCameraUpdate = 0;
unsigned long lastDepthUpdate = 0;
unsigned long lastLidarUpdate = 0;

// ========== HTML Web Interface ==========
const char* htmlPage = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Smart Bike Visualization</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            overflow-x: hidden;
        }
        .header {
            background: #2d2d2d;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        .panel {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .panel h2 {
            margin-bottom: 15px;
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        canvas {
            width: 100%;
            height: auto;
            border: 2px solid #444;
            border-radius: 4px;
            background: #000;
            display: block;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }
        .status.connected { background: #4CAF50; }
        .status.disconnected { background: #f44336; }
        .lidar-container {
            position: relative;
            width: 100%;
            height: 400px;
        }
        .info {
            margin-top: 10px;
            font-size: 12px;
            color: #aaa;
        }
        #lidarCanvas {
            width: 100%;
            height: 400px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚴 Smart Bike Visualization Dashboard</h1>
        <p>Connection Status: <span id="connectionStatus" class="status disconnected">Disconnected</span></p>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="panel">
                <h2>📷 Camera Stream (Left)</h2>
                <canvas id="cameraCanvas"></canvas>
                <div class="info">FPS: <span id="cameraFPS">0</span> | Last Update: <span id="cameraTime">Never</span></div>
            </div>
            
            <div class="panel">
                <h2>🗺️ Depth Map</h2>
                <canvas id="depthCanvas"></canvas>
                <div class="info">FPS: <span id="depthFPS">0</span> | Last Update: <span id="depthTime">Never</span></div>
            </div>
        </div>
        
        <div class="panel">
            <h2>📡 LIDAR Scan</h2>
            <div class="lidar-container">
                <canvas id="lidarCanvas"></canvas>
            </div>
            <div class="info">Points: <span id="lidarPoints">0</span> | FPS: <span id="lidarFPS">0</span> | Last Update: <span id="lidarTime">Never</span></div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket('ws://' + window.location.hostname + ':81/');
        let cameraFPS = 0, depthFPS = 0, lidarFPS = 0;
        let cameraFrameCount = 0, depthFrameCount = 0, lidarFrameCount = 0;
        let lastFPSUpdate = Date.now();
        
        // Canvas setup
        const cameraCanvas = document.getElementById('cameraCanvas');
        const depthCanvas = document.getElementById('depthCanvas');
        const lidarCanvas = document.getElementById('lidarCanvas');
        const cameraCtx = cameraCanvas.getContext('2d');
        const depthCtx = depthCanvas.getContext('2d');
        const lidarCtx = lidarCanvas.getContext('2d');
        
        // Set canvas sizes
        cameraCanvas.width = 640;
        cameraCanvas.height = 480;
        depthCanvas.width = 640;
        depthCanvas.height = 480;
        lidarCanvas.width = 600;
        lidarCanvas.height = 400;
        
        // Connection status
        ws.onopen = () => {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'status connected';
        };
        
        ws.onclose = () => {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'status disconnected';
        };
        
        ws.onerror = () => {
            document.getElementById('connectionStatus').textContent = 'Error';
            document.getElementById('connectionStatus').className = 'status disconnected';
        };
        
        // Message handling
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'camera') {
                    drawImage(cameraCanvas, cameraCtx, data.frame);
                    cameraFrameCount++;
                    document.getElementById('cameraTime').textContent = new Date().toLocaleTimeString();
                } else if (data.type === 'depth') {
                    drawImage(depthCanvas, depthCtx, data.frame);
                    depthFrameCount++;
                    document.getElementById('depthTime').textContent = new Date().toLocaleTimeString();
                } else if (data.type === 'lidar') {
                    drawLidar(lidarCtx, data.points);
                    lidarFrameCount++;
                    document.getElementById('lidarPoints').textContent = data.points.length;
                    document.getElementById('lidarTime').textContent = new Date().toLocaleTimeString();
                }
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };
        
        // Draw image from base64
        function drawImage(canvas, ctx, base64Data) {
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = 'data:image/jpeg;base64,' + base64Data;
        }
        
        // Draw LIDAR scan
        function drawLidar(ctx, points) {
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            
            if (!points || points.length === 0) return;
            
            const centerX = ctx.canvas.width / 2;
            const centerY = ctx.canvas.height / 2;
            const scale = Math.min(centerX, centerY) / 8000; // Scale for 8m max distance
            
            // Draw grid
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            for (let r = 1000; r <= 8000; r += 1000) {
                ctx.beginPath();
                ctx.arc(centerX, centerY, r * scale, 0, Math.PI * 2);
                ctx.stroke();
            }
            
            // Draw points
            points.forEach(point => {
                if (point.d_mm && point.a_deg !== undefined) {
                    const angle = (90 - point.a_deg) * Math.PI / 180; // Convert to radians
                    const distance = point.d_mm * scale;
                    const x = centerX + distance * Math.cos(angle);
                    const y = centerY - distance * Math.sin(angle);
                    
                    // Color by distance
                    const intensity = Math.min(255, point.d_mm / 8000 * 255);
                    ctx.fillStyle = `rgb(${255-intensity}, ${intensity}, 0)`;
                    ctx.beginPath();
                    ctx.arc(x, y, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }
        
        // Update FPS counters
        setInterval(() => {
            const now = Date.now();
            const elapsed = (now - lastFPSUpdate) / 1000;
            
            cameraFPS = Math.round(cameraFrameCount / elapsed);
            depthFPS = Math.round(depthFrameCount / elapsed);
            lidarFPS = Math.round(lidarFrameCount / elapsed);
            
            document.getElementById('cameraFPS').textContent = cameraFPS;
            document.getElementById('depthFPS').textContent = depthFPS;
            document.getElementById('lidarFPS').textContent = lidarFPS;
            
            cameraFrameCount = 0;
            depthFrameCount = 0;
            lidarFrameCount = 0;
            lastFPSUpdate = now;
        }, 1000);
    </script>
</body>
</html>
)";

// ========== WebSocket Event Handler ==========
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected!\n", num);
            break;
        case WStype_CONNECTED:
            {
                IPAddress ip = webSocket.remoteIP(num);
                Serial.printf("[%u] Connected from %d.%d.%d.%d url: %s\n", num, ip[0], ip[1], ip[2], ip[3], payload);
            }
            break;
        case WStype_TEXT:
            // Handle incoming text (could be used for control commands)
            Serial.printf("[%u] Received: %s\n", num, payload);
            break;
        default:
            break;
    }
}

// ========== HTTP Server Handlers ==========
void handleRoot() {
    server.send(200, "text/html", htmlPage);
}

void handleCamera() {
    if (server.hasArg("frame")) {
        cameraFrameBase64 = server.arg("frame");
        lastCameraUpdate = millis();
        
        // Broadcast to all WebSocket clients
        DynamicJsonDocument doc(1024);
        doc["type"] = "camera";
        doc["frame"] = cameraFrameBase64;
        String jsonStr;
        serializeJson(doc, jsonStr);
        webSocket.broadcastTXT(jsonStr);
        
        server.send(200, "text/plain", "OK");
    } else {
        server.send(400, "text/plain", "Missing frame parameter");
    }
}

void handleDepth() {
    if (server.hasArg("frame")) {
        depthMapBase64 = server.arg("frame");
        lastDepthUpdate = millis();
        
        // Broadcast to all WebSocket clients
        DynamicJsonDocument doc(1024);
        doc["type"] = "depth";
        doc["frame"] = depthMapBase64;
        String jsonStr;
        serializeJson(doc, jsonStr);
        webSocket.broadcastTXT(jsonStr);
        
        server.send(200, "text/plain", "OK");
    } else {
        server.send(400, "text/plain", "Missing frame parameter");
    }
}

void handleLidar() {
    if (server.hasArg("data")) {
        lidarDataJSON = server.arg("data");
        lastLidarUpdate = millis();
        
        // Broadcast to all WebSocket clients
        DynamicJsonDocument doc(8192);
        doc["type"] = "lidar";
        JsonArray points = doc.createNestedArray("points");
        
        // Parse incoming JSON and add to points array
        DynamicJsonDocument incoming(8192);
        deserializeJson(incoming, lidarDataJSON);
        if (incoming.is<JsonArray>()) {
            for (JsonObject point : incoming.as<JsonArray>()) {
                JsonObject p = points.createNestedObject();
                if (point.containsKey("d_mm")) p["d_mm"] = point["d_mm"];
                if (point.containsKey("a_deg")) p["a_deg"] = point["a_deg"];
                if (point.containsKey("q")) p["q"] = point["q"];
            }
        }
        
        String jsonStr;
        serializeJson(doc, jsonStr);
        webSocket.broadcastTXT(jsonStr);
        
        server.send(200, "text/plain", "OK");
    } else {
        server.send(400, "text/plain", "Missing data parameter");
    }
}

void handleStatus() {
    DynamicJsonDocument doc(512);
    doc["camera_last"] = lastCameraUpdate;
    doc["depth_last"] = lastDepthUpdate;
    doc["lidar_last"] = lastLidarUpdate;
    doc["uptime"] = millis();
    
    String jsonStr;
    serializeJson(doc, jsonStr);
    server.send(200, "application/json", jsonStr);
}

// ========== Setup ==========
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    // Setup LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);
    
    Serial.println("\n=== Smart Bike ESP32 Bridge ===");
    
    // Start WiFi Access Point
    Serial.print("Starting Access Point...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL);
    
    IPAddress IP = WiFi.softAPIP();
    Serial.println(" Done!");
    Serial.print("AP IP address: ");
    Serial.println(IP);
    Serial.print("SSID: ");
    Serial.println(AP_SSID);
    Serial.print("Password: ");
    Serial.println(AP_PASSWORD);
    
    // Setup HTTP server routes
    server.on("/", handleRoot);
    server.on("/camera", HTTP_POST, handleCamera);
    server.on("/depth", HTTP_POST, handleDepth);
    server.on("/lidar", HTTP_POST, handleLidar);
    server.on("/status", HTTP_GET, handleStatus);
    server.begin();
    Serial.println("HTTP Server started on port 80");
    
    // Setup WebSocket server
    webSocket.begin();
    webSocket.onEvent(webSocketEvent);
    Serial.println("WebSocket Server started on port 81");
    
    Serial.println("\n=== Ready! Connect to the WiFi network ===");
    Serial.println("Open http://192.168.4.1 in your browser");
    
    // Blink LED to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, LOW);
        delay(200);
        digitalWrite(LED_PIN, HIGH);
        delay(200);
    }
}

// ========== Main Loop ==========
void loop() {
    server.handleClient();
    webSocket.loop();
    
    // Heartbeat LED
    static unsigned long lastBlink = 0;
    if (millis() - lastBlink > 2000) {
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        lastBlink = millis();
    }
}

