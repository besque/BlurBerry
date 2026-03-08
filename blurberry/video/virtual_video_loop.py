"""
Virtual VideoLoop - Enhanced VideoLoop with Virtual Camera Output

Replaces local display with OBS Virtual Camera streaming for Zoom/Discord integration.
Optimized for low latency: latest-frame BGR buffer (no JPEG encode/decode in hot path).
"""
import cv2
import numpy as np
import threading
import time
from typing import List, Optional, Callable
from datetime import datetime

# Virtual camera import
try:
    import pyvirtualcam
    VIRTUAL_CAMERA_AVAILABLE = True
    print("✅ Virtual camera available")
except ImportError:
    VIRTUAL_CAMERA_AVAILABLE = False
    print("⚠️  Virtual camera not available, using local display")

from shared_types import DetectionEvent, PipelineConfig, AV_DELAY_SECONDS
from blurberry.video.face_pipeline import FacePipeline
from blurberry.video.blur_compositor import apply_blurs, draw_debug_overlay
from blurberry.video.tracker import MultiObjectTracker

OUTPUT_DELAY_SECONDS = AV_DELAY_SECONDS

# Target FPS for virtual camera (match OBS default; lower = less CPU if pipeline can't keep up)
VIRTUAL_CAM_FPS = 30

class VirtualVideoLoop:
    def __init__(self, config: PipelineConfig, source: int = 0, debug: bool = False):
        self.config = config
        self.source = source
        self.debug = debug
        self.running = False

        self.face_pipeline = FacePipeline(config)
        self.tracker = MultiObjectTracker(max_age=45, min_hits=1, iou_threshold=0.25)
        self.external_detectors: List[Callable] = []

        # Latest-frame buffer: BGR numpy array (no encode/decode = lower latency)
        self.current_output_frame: Optional[bytes] = None  # kept for any consumer expecting JPEG bytes
        self._latest_bgr: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        self.frame_id = 0
        self.last_object_detection_frame = -999
        self.fps = 0

        # Stores latest non-face detections between object detection runs
        self._last_object_events: List[DetectionEvent] = []
        
        # Virtual camera setup
        self.cam = None
        self.virtual_camera_active = False
        # Lightweight overlay for virtual camera (set False to skip overlay for max FPS)
        self._light_overlay = True

    def add_detector(self, detector_fn: Callable):
        self.external_detectors.append(detector_fn)

    def enroll_face(self, image_paths: List[str], name: str = "owner"):
        self.face_pipeline.enroll_face(image_paths, name)

    def enroll_from_embeddings(self, embeddings, name: str = "owner"):
        self.face_pipeline.enroll_from_embeddings(embeddings, name)

    def start_virtual_camera(self, width=1280, height=720, fps=30):
        """Initialize virtual camera for streaming"""
        if not VIRTUAL_CAMERA_AVAILABLE:
            print("⚠️  Virtual camera not available, using local display")
            return False
            
        try:
            self.cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend="obs"
            )
            self.cam.__enter__()
            self.virtual_camera_active = True
            print(f"✅ Virtual camera started: {self.cam.device}")
            print("🎥 Available in Zoom/Discord as 'OBS Virtual Camera'")
            return True
        except Exception as e:
            print(f"⚠️  Virtual camera failed: {e}")
            print("📺 Falling back to local display")
            return False

    def stop_virtual_camera(self):
        """Stop virtual camera"""
        if self.cam:
            try:
                self.cam.__exit__(None, None, None)
            except:
                pass
            self.cam = None
        self.virtual_camera_active = False

    def add_virtual_camera_overlay(self, frame, status="Virtual Camera Active"):
        """Add virtual camera specific overlays (full). Use _add_light_overlay for low-latency path."""
        display = frame.copy()
        
        # Title
        cv2.putText(display, "BlurBerry AI - Virtual Camera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
        
        # Status
        cv2.putText(display, status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        # Virtual camera indicator
        if self.virtual_camera_active:
            cv2.putText(display, "🎥 VIRTUAL CAMERA ACTIVE", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Available in Zoom/Discord as 'OBS Virtual Camera'", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "📺 LOCAL DISPLAY", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Instructions
        cv2.putText(display, "Press Q to quit", (20, 720-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display, timestamp, (1280-100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display

    def _add_light_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Minimal overlay for virtual camera (FPS + indicator only) to reduce latency."""
        if not self._light_overlay:
            return frame
        h, w = frame.shape[:2]
        # Single line: FPS + "BlurBerry"
        cv2.putText(frame, f"FPS: {self.fps:.1f} | BlurBerry", (10, min(28, h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
        return frame

    def _release_loop(self):
        """Release loop: grab latest BGR frame (no encode/decode), send to virtual cam or display.
        Runs at target FPS; always sends the most recent frame to minimize latency and avoid backlog.
        """
        frame_interval = 1.0 / VIRTUAL_CAM_FPS
        next_send_time = time.monotonic()

        while self.running:
            now = time.monotonic()
            # Grab latest frame only (drop stale; no queue buildup)
            with self.frame_lock:
                frame = self._latest_bgr
                self._latest_bgr = None

            if frame is not None:
                # Keep JPEG for any external consumer (e.g. web)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self.frame_lock:
                    self.current_output_frame = buf.tobytes()

                display_frame = self._add_light_overlay(frame.copy())

                if self.virtual_camera_active and self.cam:
                    try:
                        self.cam.send(display_frame)
                    except Exception as e:
                        print(f"⚠️  Virtual camera error: {e}")
                        cv2.imshow("BlurBerry AI - Local Display", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                            break
                else:
                    cv2.imshow("BlurBerry AI - Local Display", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

            # Throttle: use virtual camera's frame sync when active, else fixed interval
            if self.virtual_camera_active and self.cam:
                try:
                    self.cam.sleep_until_next_frame()
                except Exception:
                    pass
            else:
                next_send_time += frame_interval
                sleep_time = next_send_time - time.monotonic()
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                else:
                    next_send_time = time.monotonic()

        cv2.destroyAllWindows()

    def run(self, use_virtual_camera=True):
        """Run video loop with virtual camera option"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Set higher resolution for virtual camera
        if use_virtual_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True

        # Start virtual camera if requested
        if use_virtual_camera:
            self.start_virtual_camera()

        release_thread = threading.Thread(
            target=self._release_loop, daemon=True, name="FrameRelease"
        )
        release_thread.start()

        fps_counter = 0
        fps_start = time.monotonic()

        if use_virtual_camera:
            print(f"[VirtualVideoLoop] Running with virtual camera. Available after {OUTPUT_DELAY_SECONDS}s.")
        else:
            print(f"[VideoLoop] Running. Window appears after {OUTPUT_DELAY_SECONDS}s.")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_id += 1
            capture_time = time.monotonic()

            # ── Faces: run EVERY frame (no cadence) ──
            face_events = []
            try:
                face_events = self.face_pipeline.detect_faces(frame, self.frame_id)
            except Exception as e:
                print(f"[VideoLoop] Face error: {e}")

            # ── Other detectors: run every N frames ──
            should_detect_objects = (
                self.frame_id - self.last_object_detection_frame
            ) >= self.config.detection_cadence

            if should_detect_objects:
                self._last_object_events = []
                for detector in self.external_detectors:
                    try:
                        self._last_object_events.extend(detector(frame, self.frame_id))
                    except Exception as e:
                        print(f"[VideoLoop] Detector error: {e}")
                # Update tracker with object detections
                self.last_object_detection_frame = self.frame_id
                tracked_objects = self.tracker.update(self._last_object_events)
            else:
                # Let tracker predict forward for objects
                tracked_objects = self.tracker.update([])

            # Combine: face events (fresh every frame) + tracked object events
            all_events = face_events + tracked_objects

            # ── Blur ──
            output = apply_blurs(frame, all_events, self.config)
            if self.debug:
                output = draw_debug_overlay(output, all_events)

            # FPS
            fps_counter += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                self.fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.monotonic()

            # Add FPS to frame (so release thread can show it or we use it in light overlay)
            cv2.putText(output, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write latest BGR to buffer (release thread sends it; no encode in hot path)
            with self.frame_lock:
                self._latest_bgr = output

        cap.release()
        self.running = False

    def stop(self):
        """Stop video loop and virtual camera"""
        self.running = False
        self.stop_virtual_camera()
