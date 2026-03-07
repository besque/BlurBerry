"""
BlurBerry AI — FastAPI Web Server
Person 4 owns this file.

Endpoints:
  GET  /                  → serve index.html
  POST /enroll            → enroll a face (multipart image upload)
  POST /process_frame     → process a single JPEG frame, return blurred JPEG
  GET  /status            → pipeline health check
  POST /config            → update PipelineConfig at runtime
  GET  /audio/events      → drain audio PII event queue for UI display
  GET  /audio/transcript  → latest live transcript string
  WebSocket /ws/stream    → real-time frame streaming
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from shared_types import PipelineConfig

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[app] %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BlurBerry AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
# Automatically serves React build if frontend/dist exists, else falls back to static/
REACT_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(REACT_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(REACT_DIST, "assets")), name="assets")
    SERVE_REACT = True
    print("[app] Serving React build from frontend/dist")
else:
    os.makedirs(STATIC_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    SERVE_REACT = False
    print("[app] Serving plain HTML from static/")

# ── Pydantic models ───────────────────────────────────────────────────────────
class ConfigUpdate(BaseModel):
    blur_faces:        Optional[bool] = None
    blur_plates:       Optional[bool] = None
    blur_cards:        Optional[bool] = None
    blur_nsfw:         Optional[bool] = None
    blur_text_pii:     Optional[bool] = None
    detection_cadence: Optional[int]  = None
    blur_strength:     Optional[int]  = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes → OpenCV BGR frame."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return frame

def _encode_frame(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode OpenCV BGR frame → JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI — React build or plain HTML fallback."""
    if SERVE_REACT:
        html_path = os.path.join(REACT_DIST, "index.html")
    else:
        html_path = os.path.join(STATIC_DIR, "index.html")

    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return HTMLResponse("<h1>BlurBerry AI</h1><p>index.html not found.</p>")


@app.get("/status")
async def status():
    """Health check — returns pipeline readiness and current config."""
    config: PipelineConfig = getattr(app.state, "config", PipelineConfig())
    return {
        "status": "ok",
        "timestamp": time.time(),
        "pipeline_ready": hasattr(app.state, "video_loop"),
        "audio_ready":    hasattr(app.state, "audio"),
        "config": {
            "blur_faces":        config.blur_faces,
            "blur_plates":       config.blur_plates,
            "blur_cards":        config.blur_cards,
            "blur_nsfw":         config.blur_nsfw,
            "blur_text_pii":     config.blur_text_pii,
            "blur_strength":     config.blur_strength,
            "detection_cadence": config.detection_cadence,
        },
    }


@app.post("/config")
async def update_config(update: ConfigUpdate):
    """
    Dynamically toggle blur modules or change strength without restarting.
    Pushes updated config to VideoLoop immediately.
    """
    config: PipelineConfig = getattr(app.state, "config", PipelineConfig())

    for field, value in update.dict(exclude_none=True).items():
        setattr(config, field, value)
        log.info(f"Config updated: {field} = {value}")

    # Push to VideoLoop so it takes effect on the next frame
    if hasattr(app.state, "video_loop"):
        app.state.video_loop.config = config

    return {"status": "updated", "config": config.__dict__}


@app.post("/enroll")
async def enroll_face(file: UploadFile = File(...), name: str = "owner"):
    """
    Enroll a face so it won't be blurred.

    Person 1's API:
        loop.enroll_face(image_paths: List[str], name: str)
    We save the upload to a temp file, pass the path, then clean up.
    """
    if not hasattr(app.state, "video_loop"):
        raise HTTPException(status_code=503, detail="Video pipeline not initialised yet.")

    # Save upload to a temp file — Person 1 expects file paths, not numpy arrays
    import tempfile
    data = await file.read()
    suffix = os.path.splitext(file.filename or "face.jpg")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # Person 1: loop.enroll_face(image_paths: List[str], name: str)
        app.state.video_loop.enroll_face([tmp_path], name=name)
        log.info(f"Face enrolled: name='{name}', file={tmp_path}")
        return {"status": "enrolled", "name": name}
    except ValueError as e:
        # Person 1 raises ValueError if no face found in the image
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Enroll error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # always clean up temp file


@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Accept a JPEG/PNG frame, run the full blur pipeline, return blurred JPEG.

    Person 1's API:
        loop.process_frame(frame: np.ndarray) -> bytes  (JPEG bytes)
    """
    if not hasattr(app.state, "video_loop"):
        raise HTTPException(status_code=503, detail="Video pipeline not initialised yet.")

    data = await file.read()
    frame = _decode_upload(data)

    try:
        # Person 1: process_frame returns JPEG bytes directly
        output_bytes: bytes = app.state.video_loop.process_frame(frame)
        return Response(content=output_bytes, media_type="image/jpeg")
    except Exception as e:
        log.error(f"process_frame error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/events")
async def audio_events():
    """
    Drain Person 3's audio event queue and return pending PII events to the UI.

    Person 3's API:
        audio.get_events() -> List[dict]
        Each dict: {"type": "audio_pii", "text": str, "findings": [...], "timestamp": float}
    """
    if not hasattr(app.state, "audio"):
        return {"events": []}

    try:
        events = app.state.audio.get_events()
        return {"events": events}
    except Exception as e:
        log.warning(f"audio.get_events() error: {e}")
        return {"events": []}


@app.get("/audio/transcript")
async def audio_transcript():
    """
    Return the latest live transcript string from Person 3's pipeline.

    Person 3's API:
        audio.get_latest_transcript() -> str
    """
    if not hasattr(app.state, "audio"):
        return {"transcript": ""}

    try:
        transcript = app.state.audio.get_latest_transcript()
        return {"transcript": transcript}
    except Exception as e:
        log.warning(f"audio.get_latest_transcript() error: {e}")
        return {"transcript": ""}


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    WebSocket endpoint for real-time browser streaming.
    Client sends raw JPEG bytes → server returns blurred JPEG bytes.

    Uses Person 1's process_frame() under the hood.
    """
    await ws.accept()
    log.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_bytes()

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await ws.send_json({"error": "bad frame"})
                continue

            if hasattr(app.state, "video_loop"):
                try:
                    # Person 1: process_frame returns JPEG bytes
                    output_bytes = app.state.video_loop.process_frame(frame)
                except Exception as e:
                    log.warning(f"WS process_frame error: {e}")
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    output_bytes = buf.tobytes()
            else:
                # Pipeline not ready yet — mirror raw frame
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                output_bytes = buf.tobytes()

            await ws.send_bytes(output_bytes)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        await ws.close()