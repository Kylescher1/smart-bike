# 📱 Smart Bike - Web Depth Camera Streaming Guide

Stream your depth camera feed to your phone for remote debugging and calibration!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Make Sure Your Camera is Calibrated

Ensure you have run `config_setup.py` and have a valid `config.dill` file with calibration data.

### 3. Start the Web Server

```bash
python web_stream.py
```

You'll see output like:
```
🌐 WEB STREAMING SERVER READY
📱 Open on your phone: http://192.168.1.100:5000
💻 Or locally: http://localhost:5000
```

### 4. Connect from Your Phone

1. Make sure your phone and laptop are on the **same WiFi network**
2. Open your phone's web browser (Safari, Chrome, etc.)
3. Type in the IP address shown in the terminal
4. You're streaming! 🎉

---

## 📋 Features

### 🔍 Debug Mode
- Simple depth map visualization
- Real-time streaming
- Minimal overlay
- Perfect for quick checks

**How to use:**
1. Click the **"Debug Mode"** button (green)
2. Watch the live depth map
3. That's it!

### 🎛️ Calibrate Mode
- Full parameter tuning interface
- Real-time parameter updates
- Mobile-friendly sliders
- Save parameters directly to config

**How to use:**
1. Click the **"Calibrate Mode"** button (blue)
2. Tap **"⚙️ Calibration Parameters"** to expand controls
3. Adjust sliders to tune depth map quality
4. Watch changes in real-time
5. When satisfied, tap **"💾 Save Parameters"**

---

## 🎚️ Key Parameters Explained

### Core Stereo Parameters

**Block Size** (5-51, odd numbers)
- Size of the matching window
- Smaller = more detail, more noise
- Larger = smoother, less detail
- Start: 11-15

**Num Disparities K** (1-16)
- Multiplier for disparity range (actual = K × 16)
- Higher = can see closer objects
- Lower = faster processing
- Start: 2-4

**Uniqueness Ratio** (0-100)
- Filters ambiguous matches
- Higher = stricter filtering, fewer false matches
- Lower = more data, but noisier
- Start: 1-10

**Pre Filter Cap** (1-100)
- Limits pre-filter response
- Helps with over-exposed areas
- Start: 30-50

### Filtering & Refinement

**Use WLS Filter** (ON/OFF)
- Weighted Least Squares post-filtering
- Dramatically improves depth map quality
- Slower but much better results
- **Recommended: ON**

**WLS Lambda** (100-10000)
- Smoothing strength
- Higher = smoother result
- Start: 2000-3000

**WLS Sigma** (0.1-5.0)
- Edge-aware filtering strength
- Higher = preserves edges better
- Start: 1.5-2.5

**Use Morphological** (ON/OFF)
- Cleans up noise with morphological operations
- Recommended: ON

**Morph Iterations** (1-20)
- Number of morphological passes
- More = cleaner but more aggressive
- Start: 5

### Pre-processing

**Down Sample %** (0-100)
- Reduce image size before processing
- Higher % = faster but less detail
- Start: 50-60

**Crop** (0-300 pixels)
- Crop edges of image
- Removes unreliable border areas
- Start: 100-150

**Near/Far Cutoff %** (0-100)
- Depth range filtering
- Cuts off too-near or too-far objects
- Start: Near 70, Far 5

---

## 💡 Tuning Tips

### For Best Quality:
1. Start in **Debug Mode** to see current output
2. Switch to **Calibrate Mode**
3. Turn **WLS Filter OFF** while tuning core parameters
4. Adjust **Block Size** and **Num Disparities K** first
5. Fine-tune **Uniqueness Ratio** to reduce noise
6. Turn **WLS Filter ON** to see final result
7. Adjust **WLS Lambda** and **Sigma** for smoothness
8. **Save Parameters** when happy!

### For Speed:
1. Increase **Down Sample %** (60-80%)
2. Decrease **Num Disparities K** (1-2)
3. Turn **WLS Filter OFF**
4. Reduce **Morph Iterations** (1-3)

### For Detail:
1. Decrease **Down Sample %** (0-40%)
2. Decrease **Block Size** (5-9)
3. Increase **Num Disparities K** (4-8)
4. Turn **WLS Filter ON**
5. Increase **WLS Lambda** (5000+)

---

## 🔧 Troubleshooting

### Can't Connect from Phone?

**Check WiFi:**
- Phone and laptop must be on same network
- Corporate/school WiFi may block connections
- Try a mobile hotspot if needed

**Check Firewall:**
```bash
# Windows: Allow Python through firewall
# Or temporarily disable firewall for testing
```

**Check IP Address:**
```bash
# Get your laptop's IP address
ipconfig  # Windows
ifconfig  # Linux/Mac
```

### Stream is Laggy?

1. **Increase Down Sample %** - reduces image size
2. **Turn off WLS** - faster processing
3. **Reduce Num Disparities K** - less computation
4. **Check WiFi signal** - move closer to router

### Parameters Not Saving?

- Make sure `config.dill` exists and is writable
- Check terminal for error messages
- Try running with administrator/sudo if permission denied

### Camera Not Starting?

- Make sure cameras are not in use by other programs
- Close `debug.py` or `calibrate_disparity.py` if running
- Check camera ports in `config.dill` match your hardware
- Run `config_setup.py` if cameras have changed

---

## 🆚 Comparison: Web Stream vs Desktop Tools

| Feature | web_stream.py | debug.py | calibrate_disparity.py |
|---------|---------------|----------|------------------------|
| **View from phone** | ✅ Yes | ❌ No | ❌ No |
| **Real-time tuning** | ✅ Yes | ❌ No | ✅ Yes |
| **Save parameters** | ✅ Yes | ❌ No | ✅ Yes |
| **Mobile-friendly** | ✅ Yes | ❌ No | ❌ No |
| **Recording video** | ❌ No | ✅ Yes | ❌ No |
| **All parameters** | ⚠️ Key ones | ❌ View only | ✅ All |

**Use `web_stream.py` when:**
- You want to check camera from across the room
- You're testing while riding (phone on handlebar)
- You want quick parameter adjustments on the go
- You don't have access to the laptop directly

**Use `calibrate_disparity.py` when:**
- You need ALL parameters (advanced tuning)
- You want desktop trackbar precision
- You're doing detailed calibration work

**Use `debug.py` when:**
- You just want to see if it works
- You're recording test videos
- You need simple monitoring

---

## 🔐 Security Note

This server runs without authentication. It's designed for **local network use only**.

**Safe:**
- Using on home WiFi
- Using on mobile hotspot
- Local network at your workshop

**Not Safe:**
- Exposing to internet without protection
- Using on public WiFi without VPN
- Port forwarding to the web

If you need remote access from outside your network, consider:
- Setting up a VPN
- Adding authentication to Flask app
- Using SSH tunnel

---

## 🎯 Common Workflows

### Quick Check While Riding
1. Mount phone on handlebar
2. Start `web_stream.py` on laptop
3. Connect phone browser to stream
4. See what the bike sees in real-time!

### Remote Calibration
1. Position laptop with camera near test area
2. Go to remote location with phone
3. Connect to web stream
4. Switch to Calibrate Mode
5. Adjust parameters while seeing results
6. Save when optimal

### Team Debugging
1. Start web stream on main laptop
2. Share IP with team members
3. Multiple people can view simultaneously
4. One person adjusts, all see changes

---

## 📊 Performance

**Typical Latency:**
- Local network: 100-300ms
- Same room WiFi: 50-150ms
- Farther WiFi: 200-500ms

**Bandwidth Usage:**
- ~2-5 Mbps depending on resolution
- Down Sample % greatly affects bandwidth

**Battery Impact:**
- Laptop: Minimal extra drain
- Phone: ~10-15% per hour of viewing

---

## 🐛 Known Issues

1. **First frame delay** - Takes 1-2 seconds to start streaming (normal)
2. **Slider precision** - Web sliders less precise than desktop trackbars
3. **No multi-touch** - Can't adjust multiple parameters simultaneously
4. **Refresh needed** - Sometimes need to refresh browser if stream freezes

---

## 🚀 Future Improvements

Potential additions (let me know if you want these!):
- [ ] Password protection
- [ ] HTTPS support
- [ ] Recording from web interface
- [ ] Side-by-side view (raw + depth)
- [ ] Parameter presets
- [ ] Historical parameter comparison
- [ ] Multi-camera support
- [ ] Stats/graphs overlay

---

## 📞 Need Help?

If you run into issues:
1. Check terminal output for error messages
2. Try refreshing the browser
3. Restart the web server
4. Make sure `config.dill` is up to date
5. Test with `debug.py` first to verify camera works

Happy streaming! 🎉

