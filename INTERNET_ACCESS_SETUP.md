# Internet Access Setup Guide

This guide explains how to make your YOLO web stream accessible from the internet.

## Quick Start

1. **Start the web server:**
   ```bash
   python yolo_web_server.py --model yolo/models/yolo11n.rknn --source 0 --port 8080
   ```

2. **Access locally:**
   - Open `http://<your-local-ip>:8080` in your browser
   - Find your local IP with: `hostname -I` or `ip addr`

3. **Make it accessible from internet** (choose one method below)

---

## Method 1: Port Forwarding (Recommended for Permanent Access)

### Step 1: Find Your Public IP
```bash
curl ifconfig.me
# or visit: https://whatismyipaddress.com
```

### Step 2: Configure Router Port Forwarding

1. Access your router admin panel (usually `192.168.1.1` or `192.168.0.1`)
2. Navigate to "Port Forwarding" or "Virtual Server" settings
3. Add a new rule:
   - **External Port:** 8080 (or any port you prefer)
   - **Internal IP:** Your device's local IP (e.g., `192.168.1.100`)
   - **Internal Port:** 8080
   - **Protocol:** TCP
   - **Name:** YOLO Stream

4. Save and apply changes

### Step 3: Access from Internet
- Open `http://<your-public-ip>:8080` from any device
- Replace `<your-public-ip>` with the IP from Step 1

### Security Note:
⚠️ **Important:** Add password protection when exposing to internet:
```bash
python yolo_web_server.py --model yolo/models/yolo11n.rknn --source 0 --port 8080 --password "your-secure-password"
```

---

## Method 2: ngrok (Easiest, No Router Config)

### Step 1: Install ngrok
```bash
# Download from: https://ngrok.com/download
# Or on Linux:
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
```

### Step 2: Sign up and get auth token
1. Sign up at https://dashboard.ngrok.com/signup
2. Get your authtoken from the dashboard
3. Configure: `ngrok config add-authtoken <your-token>`

### Step 3: Start ngrok tunnel
```bash
# In a separate terminal:
ngrok http 8080
```

### Step 4: Access via ngrok URL
- ngrok will display a URL like: `https://abc123.ngrok.io`
- Open this URL from any device on the internet
- The URL changes each time you restart ngrok (unless you have a paid plan)

---

## Method 3: Cloudflare Tunnel (Free, Permanent URL)

### Step 1: Install cloudflared
```bash
# On Linux:
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
```

### Step 2: Create tunnel
```bash
cloudflared tunnel --url http://localhost:8080
```

### Step 3: Access via provided URL
- Cloudflare will provide a URL like: `https://random-words.trycloudflare.com`
- This URL works from anywhere on the internet

---

## Method 4: localtunnel (Simple Alternative)

### Step 1: Install
```bash
npm install -g localtunnel
```

### Step 2: Create tunnel
```bash
lt --port 8080
```

### Step 3: Access via provided URL
- You'll get a URL like: `https://random-name.loca.lt`
- Access from any device

---

## Security Best Practices

### 1. Use Password Protection
```bash
python yolo_web_server.py --password "strong-password-here" ...
```

### 2. Use HTTPS (with reverse proxy)
- Set up nginx or Apache as reverse proxy with SSL
- Or use Cloudflare Tunnel (includes HTTPS automatically)

### 3. Firewall Rules
- Only allow specific IPs if possible
- Use fail2ban to block brute force attempts

### 4. Change Default Port
- Use a non-standard port (not 80, 8080, etc.)
- Example: `--port 23456`

---

## Troubleshooting

### Can't access from internet?
1. **Check firewall:**
   ```bash
   sudo ufw allow 8080/tcp
   # or
   sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
   ```

2. **Verify port forwarding:**
   - Test with: `telnet <your-public-ip> 8080`
   - Or use online port checker: https://www.yougetsignal.com/tools/open-ports/

3. **Check if server is listening:**
   ```bash
   netstat -tuln | grep 8080
   ```

### Connection timeout?
- Your ISP might block incoming connections
- Use ngrok or Cloudflare Tunnel instead

### Slow performance?
- Reduce camera resolution in the script
- Lower FPS target
- Use `--no-fast-capture` if threading causes issues

---

## Example Commands

### Basic (local network only):
```bash
python yolo_web_server.py --model yolo/models/yolo11n.rknn --source 0
```

### With password (for internet access):
```bash
python yolo_web_server.py \
  --model yolo/models/yolo11n.rknn \
  --source 0 \
  --port 8080 \
  --password "my-secure-password" \
  --track
```

### Custom port and settings:
```bash
python yolo_web_server.py \
  --model yolo/models/yolo11n.rknn \
  --source 0 \
  --port 9000 \
  --conf 0.3 \
  --track \
  --password "secret123"
```

---

## Mobile Access

The web interface is mobile-friendly! Just open the URL on your phone's browser:
- Works on iOS Safari, Chrome, Firefox
- Responsive design adapts to screen size
- Touch-friendly controls

---

## Need Help?

- Check server logs for errors
- Verify camera is working: `python yolo/rknn_inference.py --source 0`
- Test local access first before setting up internet access
- Ensure all dependencies are installed: `pip install -r requirements.txt`

