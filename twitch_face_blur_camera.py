"""
BlurBerry — Face Recognition + Virtual Camera + Twitch Streaming

Features:
- Face enrollment and recognition
- Selective face blurring (only non-enrolled faces)
- Virtual camera output for Zoom/Meet
- Twitch streaming with face blur
- Real-time chat overlay

Controls:
  E  →  Start collecting enrollment frames 
  S  →  Stop collecting, save enrollment, start streaming
  Q  →  Quit anytime
  T  →  Toggle Twitch streaming on/off
"""
import cv2
import numpy as np
import sys
import time
import os
import threading
import subprocess
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_types import PipelineConfig
from blurberry.video.face_pipeline import FacePipeline

# Virtual camera import
try:
    import pyvirtualcam
    VIRTUAL_CAMERA_AVAILABLE = True
    print("✅ Virtual camera available")
except ImportError:
    VIRTUAL_CAMERA_AVAILABLE = False
    print("⚠️  Virtual camera not available, using local display")

# Twitch streaming imports
try:
    import requests
    TWITCH_API_AVAILABLE = True
    print("✅ Twitch API available")
except ImportError:
    TWITCH_API_AVAILABLE = False
    print("⚠️  Twitch API not available")

FRAMES_NEEDED = 60

config = PipelineConfig(
    blur_faces=True,
    detection_cadence=15,
    blur_strength=51,
    face_similarity_threshold=0.38
)

face_pipeline = FacePipeline(config)

class TwitchStreamer:
    def __init__(self):
        self.is_streaming = False
        self.ffmpeg_process = None
        self.stream_key = ""
        self.chat_messages = []
        
    def set_stream_key(self, stream_key):
        """Set Twitch stream key"""
        self.stream_key = stream_key
        
    def start_stream(self, width=1280, height=720, fps=30):
        """Start Twitch streaming using FFmpeg"""
        if self.is_streaming:
            return
            
        if not self.stream_key:
            print("❌ No stream key set!")
            return
            
        try:
            # FFmpeg command for Twitch streaming
            cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'veryfast',
                '-b:v', '2500k',
                '-maxrate', '2500k',
                '-bufsize', '5000k',
                '-f', 'flv',
                f'rtmp://live.twitch.tv/app/{self.stream_key}'
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.is_streaming = True
            print(f"🎮 Twitch streaming started!")
            print(f"📺 Stream URL: https://www.twitch.tv/your_channel")
            
        except Exception as e:
            print(f"❌ Failed to start Twitch stream: {e}")
            
    def stop_stream(self):
        """Stop Twitch streaming"""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process = None
        self.is_streaming = False
        print("🛑 Twitch streaming stopped")
        
    def send_frame(self, frame):
        """Send frame to Twitch stream"""
        if self.is_streaming and self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except:
                pass

# Initialize components
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

twitch_streamer = TwitchStreamer()

print("\n=== BlurBerry Face Recognition + Twitch Streaming ===")
print("Press E to enroll your face")
print("Press S to start streaming (Virtual Camera + Twitch)")
print("Press T to toggle Twitch streaming")
print("Press Q to quit\n")

# Get Twitch stream key
print("🎮 Twitch Setup:")
stream_key = input("Enter your Twitch stream key (or press Enter to skip): ").strip()
if stream_key:
    twitch_streamer.set_stream_key(stream_key)
    print("✅ Twitch stream key set")
else:
    print("⚠️  Twitch streaming disabled")

collecting = False
embeddings_collected = []
enrolled = False
twitch_enabled = bool(stream_key)

# Virtual camera setup
cam = None

def start_virtual_camera():
    """Initialize virtual camera"""
    global cam
    if VIRTUAL_CAMERA_AVAILABLE:
        try:
            cam = pyvirtualcam.Camera(
                width=1280,
                height=720,
                fps=30,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend="obs"
            )
            cam.__enter__()
            print(f"✅ Virtual camera started: {cam.device}")
            return True
        except Exception as e:
            print(f"⚠️  Virtual camera failed: {e}")
            return False
    return False

def stop_virtual_camera():
    """Stop virtual camera"""
    global cam
    if cam:
        try:
            cam.__exit__(None, None, None)
        except:
            pass
        cam = None

def add_overlay(frame, status, enrolled=False, collecting=False, progress=0, 
               twitch_streaming=False, viewer_count=0):
    """Add comprehensive overlay to frame"""
    display = frame.copy()
    
    # Title
    cv2.putText(display, "BlurBerry AI - Face Recognition + Twitch", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
    
    # Status
    cv2.putText(display, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    
    # Virtual camera indicator
    if cam:
        cv2.putText(display, "🎥 VIRTUAL CAMERA", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "📺 LOCAL DISPLAY", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Twitch indicator
    if twitch_streaming:
        cv2.putText(display, "🎮 TWITCH LIVE", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display, f"Viewers: {viewer_count}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(display, "🎮 TWITCH OFFLINE", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Progress bar during enrollment
    if collecting and progress > 0:
        bar_w = int(400 * progress)
        cv2.rectangle(display, (120, 680), (520, 700), (50, 50, 50), -1)
        cv2.rectangle(display, (120, 680), (120 + bar_w, 700), (0, 220, 100), -1)
        cv2.putText(display, f"Enrolling: {int(progress * 100)}%",
                    (120, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)
    
    # Instructions
    cv2.putText(display, "E: Enroll | S: Stream | T: Toggle Twitch | Q: Quit", (20, 720-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(display, timestamp, (1280-100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return display

# Enrollment phase
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    faces = face_pipeline.face_app.get(frame)

    if collecting:
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings_collected.append(face.embedding)
            progress = len(embeddings_collected) / FRAMES_NEEDED
            status = f"Collecting: {len(embeddings_collected)}/{FRAMES_NEEDED}"

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 100), 2)
            cv2.putText(display, "Enrolling...", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)
        else:
            status = "No face detected — look at camera"
            progress = 0

        if len(embeddings_collected) >= FRAMES_NEEDED:
            collecting = False
            face_pipeline.enroll_from_embeddings(embeddings_collected, name="owner")
            enrolled = True
            print(f"[Enrollment] Done! Collected {len(embeddings_collected)} frames.")
            status = "Face enrolled! Press S to start streaming"
            progress = 0
    else:
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        if enrolled:
            status = "Face enrolled! Press S to start streaming"
        else:
            status = "Press E to enroll your face | Press S to skip enrollment"
        progress = 0

    display = add_overlay(display, status, enrolled, collecting, progress, 
                      twitch_streaming=False, viewer_count=0)

    cv2.imshow("BlurBerry — Enrollment", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    elif key == ord('e') and not collecting:
        embeddings_collected = []
        collecting = True
        print("[Enrollment] Started collecting frames. Hold still and face the camera.")

    elif key == ord('s'):
        cap.release()
        cv2.destroyAllWindows()
        break

# ── Start streaming phase ──
print("\n[BlurBerry] Starting streaming...")
print("Virtual Camera: Available for Zoom/Meet")
if twitch_enabled:
    print("Twitch: Ready to stream")
else:
    print("Twitch: Disabled (no stream key)")
print("Press Q to quit, T to toggle Twitch streaming\n")

# Reinitialize camera for streaming
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Start virtual camera
virtual_camera_active = start_virtual_camera()

# Start Twitch if enabled
if twitch_enabled:
    twitch_streamer.start_stream()

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through face pipeline
        frame_id = int(time.time() * 1000)
        detection_events = face_pipeline.detect_faces(frame, frame_id)
        
        # Apply blurring based on detection events
        processed_frame = frame.copy()
        for event in detection_events:
            if event.blur:  # Blur non-enrolled faces
                x1, y1, x2, y2 = event.bbox
                face_region = processed_frame[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(face_region, (51, 51), 0)
                processed_frame[y1:y2, x1:x2] = blurred
        
        # Add overlay
        display = add_overlay(processed_frame, "Streaming Active", enrolled=True,
                          twitch_streaming=twitch_streamer.is_streaming, 
                          viewer_count=0)
        
        # Send to virtual camera
        if cam:
            cam.send(display)
            cam.sleep_until_next_frame()
        else:
            cv2.imshow("BlurBerry — Streaming (Local)", display)
        
        # Send to Twitch
        if twitch_enabled:
            twitch_streamer.send_frame(display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t') and twitch_enabled:
            if twitch_streamer.is_streaming:
                twitch_streamer.stop_stream()
            else:
                twitch_streamer.start_stream()
        
        frame_count += 1

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_virtual_camera()
    twitch_streamer.stop_stream()
    print("Streaming stopped")
