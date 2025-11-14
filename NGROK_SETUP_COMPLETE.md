# ✅ ngrok Setup Complete!

Your ngrok is now configured and ready to use!

## 🎯 Quick Start

### Option 1: Use the helper script (easiest)
```bash
./start_web_stream.sh --model yolo/models/yolo11n.rknn --source 0 --password "your-password"
```

This will:
1. Start the web server
2. Automatically start ngrok tunnel
3. Show you the public URL

### Option 2: Manual start

**Terminal 1 - Start web server:**
```bash
python3 yolo_web_server.py --model yolo/models/yolo11n.rknn --source 0 --port 8080 --password "your-password"
```

**Terminal 2 - Start ngrok:**
```bash
./ngrok http 8080
```

## 📍 ngrok Location

- **Executable:** `/home/radxa/smart-bike/ngrok`
- **Config:** `/home/radxa/.config/ngrok/ngrok.yml`
- **Authtoken:** ✅ Configured

## 🌐 Getting Your Public URL

When you run `./ngrok http 8080`, you'll see output like:

```
Forwarding   https://abc123.ngrok.io -> http://localhost:8080
```

**Copy that URL** and open it in any browser from anywhere in the world!

## 🔒 Security

**Always use a password when exposing to internet:**
```bash
--password "your-secure-password"
```

## 📱 Mobile Access

1. Start the web server and ngrok
2. Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)
3. Open it on your phone's browser
4. Enter the password
5. Enjoy live YOLO detection!

## 🛠️ Troubleshooting

**ngrok not found?**
- Use: `./ngrok` (from the project directory)
- Or add to PATH: `export PATH=$PATH:/home/radxa/smart-bike`

**Connection refused?**
- Make sure web server is running first
- Check the port matches (default: 8080)

**Want a permanent URL?**
- Upgrade to ngrok paid plan
- Or use Cloudflare Tunnel (free, permanent)

## 📖 More Info

See `INTERNET_ACCESS_SETUP.md` for detailed setup options.


