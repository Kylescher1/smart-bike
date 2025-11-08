# 🚀 Quick Start - Web Camera Streaming

## Step 1: Install Dependencies

```bash
pip install -r requirements_web.txt
```

This installs:
- Flask (web server)
- OpenCV (camera/image processing)
- NumPy (array processing)
- Dill (config loading)

## Step 2: Verify Setup

```bash
python test_web_stream.py
```

You should see all ✅ checks pass.

## Step 3: Start the Server

```bash
python web_stream.py
```

You'll see something like:
```
🌐 WEB STREAMING SERVER READY
📱 Open on your phone: http://192.168.1.XXX:5000
💻 Or locally: http://localhost:5000
```

## Step 4: Open on Your Phone

1. Make sure phone and laptop are on the **same WiFi**
2. Open your phone's browser (Safari/Chrome)
3. Type the IP address shown above
4. You're streaming! 🎉

---

## 📱 Using the Interface

### Debug Mode (Green Button)
- Simple depth map viewing
- Real-time streaming
- Good for quick checks

### Calibrate Mode (Blue Button)
- Tap "⚙️ Calibration Parameters" to expand
- Adjust sliders to tune the depth map
- Changes apply in real-time
- Tap "💾 Save Parameters" when done

---

## ⚙️ Key Features

✅ **Switch Modes** - Toggle between debug and calibrate  
✅ **Real-time Tuning** - See parameter changes instantly  
✅ **Mobile-Friendly** - Designed for phone screens  
✅ **Save Config** - Saves directly to config.dill  
✅ **No App Needed** - Works in any browser  

---

## 🔧 Troubleshooting

**Can't connect from phone?**
- Check both devices are on same WiFi
- Try the IP shown in terminal output
- Check laptop firewall isn't blocking port 5000

**Dependencies won't install?**
```bash
# Try upgrading pip first
python -m pip install --upgrade pip
pip install -r requirements_web.txt
```

**Camera won't start?**
- Close debug.py or calibrate_disparity.py if running
- Make sure config.dill exists (run config_setup.py)
- Check camera connections

---

## 📚 Full Documentation

See `WEB_STREAM_GUIDE.md` for:
- Detailed parameter explanations
- Tuning tips and strategies
- Common workflows
- Performance optimization
- Advanced troubleshooting

---

Enjoy your remote depth camera streaming! 🎉

