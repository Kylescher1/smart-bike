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

# Wait for server to be ready and listening on the port
echo "⏳ Waiting for server to be ready..."
MAX_WAIT=30
WAITED=0
PORT_READY=0
while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if process is still running
    if ! kill -0 $WEB_PID 2>/dev/null; then
        echo ""
        echo "❌ Web server process died unexpectedly"
        exit 1
    fi
    
    # Check if port is listening (try IPv4 first, then IPv6)
    if (timeout 1 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null) || (nc -z 127.0.0.1 $PORT 2>/dev/null); then
        PORT_READY=1
        # Try to make an HTTP request to verify server is responding
        if command -v curl >/dev/null 2>&1; then
            if curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
                echo ""
                echo "✅ Web server is ready and responding on port $PORT"
                break
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -q -O /dev/null --timeout=2 "http://127.0.0.1:$PORT/" 2>/dev/null; then
                echo ""
                echo "✅ Web server is ready and responding on port $PORT"
                break
            fi
        else
            # If no curl/wget, just check port is listening and wait a bit more
            if [ $PORT_READY -eq 1 ]; then
                sleep 2
                echo ""
                echo "✅ Web server is listening on port $PORT"
                break
            fi
        fi
    fi
    
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo ""
    echo "❌ Timeout waiting for server to start on port $PORT"
    echo "   Check server logs for errors"
    kill $WEB_PID 2>/dev/null
    exit 1
fi

echo ""
echo "✅ Web server started successfully (PID: $WEB_PID)"
echo ""
echo "🌐 Starting ngrok tunnel..."
echo ""

# Start ngrok (it will connect to localhost:PORT)
# The readiness check above ensures the server is listening before ngrok starts
"$SCRIPT_DIR/ngrok" http $PORT

# Cleanup on exit
trap "kill $WEB_PID 2>/dev/null; exit" INT TERM


