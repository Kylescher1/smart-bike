#!/bin/bash
# Helper script to stop existing web server and start a new one

PORT=${1:-8080}
SOURCE=${2:-3}
PASSWORD=${3:-"your-password"}

echo "🛑 Stopping any existing web servers on port $PORT..."

# Find and kill existing server
PID=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Found process $PID using port $PORT, stopping it..."
    kill $PID
    sleep 2
    # Force kill if still running
    if kill -0 $PID 2>/dev/null; then
        kill -9 $PID
        sleep 1
    fi
    echo "✅ Stopped old server"
else
    echo "✅ No existing server found"
fi

# Verify port is free
if lsof -i :$PORT >/dev/null 2>&1; then
    echo "⚠️  Warning: Port $PORT may still be in use"
else
    echo "✅ Port $PORT is free"
fi

echo ""
echo "🚀 Starting new web server..."
echo ""

cd /home/radxa/smart-bike

python3 yolo_web_server.py \
  --model yolo/models/yolo11n.rknn \
  --source $SOURCE \
  --port $PORT \
  --password "$PASSWORD" \
  --jpeg-quality 55 \
  --max-width 960 \
  --stream-fps 30

