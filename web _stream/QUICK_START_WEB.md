# Quick Start: Internet Web Stream

## 🚀 Fastest Way to Get Online (ngrok)

### 1. Install ngrok
```bash
# Download and install ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/

# Sign up at https://dashboard.ngrok.com (free)
# Get your authtoken and configure:
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### 2. Start the web server
```bash
python yolo_web_server.py --model yolo/models/yolo11n.rknn --source 0 --port 8080 --password "your-password"
```

### 3. In another terminal, start ngrok
```bash
ngrok http 8080
```

### 4. Copy the ngrok URL
- You'll see something like: `https://abc123.ngrok.io`
- Open this URL in any browser, anywhere in the world!
- Enter the password you set

**Done!** 🎉 Your stream is now accessible from the internet.

---

## 📱 Access from Your Phone

1. Get the URL (from ngrok, port forwarding, or Cloudflare Tunnel)
2. Open it in your phone's browser
3. Enter password if set
4. Enjoy live YOLO detection on your phone!

---

## 🔒 Security Tips

**Always use a password when exposing to internet:**
```bash
python yolo_web_server.py --password "strong-password" ...
```

**Change the default port:**
```bash
python yolo_web_server.py --port 23456 ...
```

---

## 🆘 Troubleshooting

**Server won't start?**
- Check if port is in use: `lsof -i :8080`
- Try a different port: `--port 9000`

**Can't see video?**
- Check camera: `python yolo/rknn_inference.py --source 0`
- Check server logs for errors

**Slow performance?**
- Lower camera resolution in code
- Disable tracking: remove `--track` flag

---

## 📖 Full Documentation

See `INTERNET_ACCESS_SETUP.md` for detailed setup instructions including:
- Port forwarding
- Cloudflare Tunnel
- localtunnel
- Security best practices

