#!/bin/bash
# Helper script to start YOLO web stream with ngrok tunnel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODEL="yolo/models/yolo11n.rknn"
SOURCE="0"
PORT="8080"
PASSWORD=""
TRACK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --password)
            PASSWORD="$2"
            shift 2
            ;;
        --track)
            TRACK="--track"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--source SOURCE] [--port PORT] [--password PASSWORD] [--track]"
            exit 1
            ;;
    esac
done

# Kill any existing processes first
echo "🛑 Stopping any existing web servers..."
pkill -f "yolo_web_server" 2>/dev/null
sleep 2

# Kill anything on the port
lsof -ti :$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 1

echo "🚀 Starting YOLO Web Stream Server..."
echo ""

# Build command array to properly handle passwords with spaces/special chars
CMD_ARGS=("python3" "yolo_web_server.py" "--model" "$MODEL" "--source" "$SOURCE" "--port" "$PORT")
if [ -n "$PASSWORD" ]; then
    CMD_ARGS+=("--password" "$PASSWORD")
fi
if [ -n "$TRACK" ]; then
    CMD_ARGS+=("$TRACK")
fi

# Add optimized streaming settings
CMD_ARGS+=("--jpeg-quality" "55" "--max-width" "960" "--stream-fps" "30")

# Start web server in background
echo "📡 Starting web server on port $PORT..."
if [ -n "$PASSWORD" ]; then
    echo "🔑 Password protection: ENABLED"
    echo "   Username: (any value)"
    echo "   Password: $PASSWORD"
else
    echo "🔓 Password protection: DISABLED"
fi
echo ""

# Start the server
"${CMD_ARGS[@]}" &
WEB_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if ! kill -0 $WEB_PID 2>/dev/null; then
    echo "❌ Failed to start web server"
    exit 1
fi

echo "✅ Web server started (PID: $WEB_PID)"
echo ""
echo "🌐 Starting ngrok tunnel..."
echo ""

# Start ngrok
"$SCRIPT_DIR/ngrok" http $PORT

# Cleanup on exit
trap "kill $WEB_PID 2>/dev/null; exit" INT TERM


