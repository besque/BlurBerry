"""
BlurBerry — Webcam enrollment with Virtual Camera output

Controls:
  E  →  Start collecting enrollment frames 
  S  →  Stop collecting, save enrollment, start privacy stream to virtual camera
  Q  →  Quit anytime

After enrollment, the face recognition and blurring will be streamed to OBS Virtual Camera
for use in Zoom, Google Meet, etc.
"""
import cv2
import numpy as np
import sys
import time
import os

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

FRAMES_NEEDED = 60   # collect 60 frames for enrollment — good balance of speed vs accuracy

config = PipelineConfig(
    blur_faces=True,
    detection_cadence=15,
    blur_strength=51,
    face_similarity_threshold=0.38
)

face_pipeline = FacePipeline(config)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n=== BlurBerry Enrollment with Virtual Camera ===")
print("Press E to start collecting your face")
print("Press S when done to save and start VIRTUAL CAMERA stream")
print("Press Q to quit\n")

collecting = False
embeddings_collected = []
enrolled = False

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
            print("🎥 Open Zoom/Meet and select 'OBS Virtual Camera'")
            return True
        except Exception as e:
            print(f"⚠️  Virtual camera failed: {e}")
            print("📺 Using local display instead")
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

def add_overlay(frame, status, enrolled=False, collecting=False, progress=0):
    """Add status overlay to frame"""
    display = frame.copy()
    
    # Status text
    cv2.putText(display, "BlurBerry AI - Face Recognition Virtual Camera", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
    
    cv2.putText(display, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    
    # Virtual camera indicator
    if cam:
        cv2.putText(display, "🎥 VIRTUAL CAMERA ACTIVE", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "📺 LOCAL DISPLAY", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Progress bar during enrollment
    if collecting and progress > 0:
        bar_w = int(400 * progress)
        cv2.rectangle(display, (120, 680), (520, 700), (50, 50, 50), -1)
        cv2.rectangle(display, (120, 680), (120 + bar_w, 700), (0, 220, 100), -1)
        cv2.putText(display, f"Enrolling: {int(progress * 100)}%",
                    (120, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)
    
    # Instructions
    cv2.putText(display, "E: Enroll | S: Start Stream | Q: Quit", (20, 720-20),
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
        # Try to grab embedding from largest face
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings_collected.append(face.embedding)

            progress = len(embeddings_collected) / FRAMES_NEEDED
            status = f"Collecting: {len(embeddings_collected)}/{FRAMES_NEEDED}"

            # Draw box around face being enrolled
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 100), 2)
            cv2.putText(display, "Enrolling...", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)
        else:
            status = "No face detected — look at camera"
            progress = 0

        # Auto-finish when enough frames collected
        if len(embeddings_collected) >= FRAMES_NEEDED:
            collecting = False
            face_pipeline.enroll_from_embeddings(embeddings_collected, name="owner")
            enrolled = True
            print(f"[Enrollment] Done! Collected {len(embeddings_collected)} frames.")
            status = "Face enrolled! Press S to start virtual camera stream"
            progress = 0
    else:
        # Draw face boxes while idle
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        if enrolled:
            status = "Face enrolled! Press S to start virtual camera stream"
        else:
            status = "Press E to enroll your face | Press S to skip enrollment"
        progress = 0

    # Add overlay
    display = add_overlay(display, status, enrolled, collecting, progress)

    # Show locally during enrollment
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
        break   # move to stream

# ── Start privacy stream to virtual camera ──
print("\n[BlurBerry] Starting privacy stream to virtual camera...")
print("Your enrolled face will NOT be blurred. All others will be.")
print("🎥 Stream is available in Zoom/Meet as 'OBS Virtual Camera'")
print("Press Q to quit.\n")

# Reinitialize camera for streaming
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Start virtual camera
virtual_camera_active = start_virtual_camera()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through face pipeline
        frame_id = int(time.time() * 1000)  # Use timestamp as frame ID
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
        display = add_overlay(processed_frame, "Face Recognition Active", enrolled=True)
        
        # Send to virtual camera or show locally
        if cam:
            cam.send(display)
            cam.sleep_until_next_frame()
        else:
            cv2.imshow("BlurBerry — Privacy Stream (Local)", display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_virtual_camera()
    print("Virtual camera stream stopped")
