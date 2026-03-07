# 🎮 BlurBerry + Twitch Streaming Setup Guide

## 📋 Prerequisites

### 1. Install FFmpeg
Download and install FFmpeg for video streaming:
- Windows: https://ffmpeg.org/download.html
- Add FFmpeg to system PATH
- Verify with: `ffmpeg -version`

### 2. Get Twitch Stream Key
1. Go to: https://www.twitch.tv/
2. Login to your account
3. Go to: Creator Dashboard → Settings → Stream
4. Copy your "Stream Key"
5. Keep it secure - don't share it!

## 🚀 Quick Start

### 1. Install Dependencies
```bash
install_twitch.bat
```

### 2. Run Face Recognition + Twitch Stream
```bash
venv\Scripts\python.exe twitch_face_blur_camera.py
```

### 3. Setup Process
1. **Enter your Twitch stream key** when prompted
2. **Press 'E'** to enroll your face
3. **Press 'S'** to start streaming
4. **Virtual Camera** starts for Zoom/Meet
5. **Twitch Stream** starts automatically

## 🎮 Controls

### **During Enrollment:**
- **'E'** - Start face enrollment
- **'S'** - Skip enrollment and start streaming
- **'Q'** - Quit

### **During Streaming:**
- **'T'** - Toggle Twitch streaming on/off
- **'Q'** - Quit application

## 🎥 Features

### **✅ What You Get:**
- **Face Recognition** - Your face stays clear, others get blurred
- **Virtual Camera** - Stream to Zoom/Meet as "OBS Virtual Camera"
- **Twitch Streaming** - Simultaneous stream to Twitch
- **Real-time Overlay** - Status, viewer count, timestamps
- **Privacy Protection** - Automatic face blurring for non-enrolled faces

### **🎮 Streaming Quality:**
- **Resolution**: 1280x720 (HD)
- **FPS**: 30 frames per second
- **Bitrate**: 2500 kbps
- **Codec**: H.264 (optimized for streaming)

## 🔧 Configuration

### **Adjust Stream Settings:**
```python
# In twitch_face_blur_camera.py, modify:
'-b:v', '2500k',  # Bitrate (increase for better quality)
'-maxrate', '2500k',  # Maximum bitrate
'-bufsize', '5000k',  # Buffer size
```

### **Adjust Face Blur:**
```python
# Change blur kernel size:
blurred = cv2.GaussianBlur(face_region, (51, 51), 0)  # Higher = more blur
```

## 🎯 Success Indicators

### **✅ Working When:**
1. Face enrollment completes successfully
2. Virtual camera shows "🎥 VIRTUAL CAMERA"
3. Twitch shows "🎮 TWITCH LIVE"
4. Your face stays clear in stream
5. Other faces get blurred automatically
6. Zoom/Meet receives virtual camera feed
7. Twitch receives live stream

## 🛠️ Troubleshooting

### **Virtual Camera Issues:**
- Install OBS Studio: https://obsproject.com/
- Start OBS before running the script
- Check if "OBS Virtual Camera" is available

### **Twitch Streaming Issues:**
- Verify FFmpeg is installed and in PATH
- Check your stream key is correct
- Ensure good internet connection
- Check Twitch stream health dashboard

### **Face Detection Issues:**
- Ensure good lighting
- Face should be clearly visible
- Try re-enrolling if detection is poor

## 🎮 Pro Tips

### **For Better Streaming:**
1. **Good lighting** - Improves face detection
2. **Stable internet** - Prevents stream drops
3. **Clear background** - Better face recognition
4. **Test locally** - Verify before going live

### **Privacy Best Practices:**
1. **Enroll your face** in good lighting
2. **Test with friends** - Verify blurring works
3. **Check stream** before important meetings
4. **Keep stream key** secure and private

## 🎥 Stream URLs

### **Your Streams Will Be Available At:**
- **Twitch**: https://www.twitch.tv/your_channel
- **Zoom/Meet**: Select "OBS Virtual Camera" as camera

---

## 🚀 Ready to Stream!

**"Install dependencies, run the script, and start your privacy-protected stream!"** 🎮✅
