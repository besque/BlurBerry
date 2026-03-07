import sys
import os
import threading
import argparse
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

from shared_types import PipelineConfig


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="BlurBerry AI Privacy Filter")
    p.add_argument("--source",        default="0",
                   help="Video source: 0 for webcam, or path to video file")
    p.add_argument("--enroll",        nargs="+", default=None,
                   help="One or more image paths to enroll as the owner face "
                        "(e.g. --enroll photos/1.jpg photos/2.jpg photos/3.jpg)")
    p.add_argument("--port",          type=int, default=8000)
    p.add_argument("--host",          default="0.0.0.0")
    p.add_argument("--no-audio",      action="store_true", help="Disable audio pipeline")
    p.add_argument("--no-web",        action="store_true",
                   help="Run headless (OpenCV window only, no FastAPI)")
    p.add_argument("--blur-strength", type=int, default=55)
    p.add_argument("--cadence",       type=int, default=15,
                   help="Run detectors every N frames")
    p.add_argument("--debug",         action="store_true",
                   help="Show bounding box overlays on video")
    return p.parse_args()


# ── Build video pipeline ──────────────────────────────────────────────────────
def build_pipeline(config: PipelineConfig, source, debug: bool = False):
    """
    Instantiate VideoLoop (Person 1) and register all detectors via add_detector().
    Person 1's VideoLoop constructor: VideoLoop(config, source=0, debug=False)
    Person 1's registration API:      loop.add_detector(callable)
    """

    # ── Person 2: Plate + Card detector ──────────────────────────────────────
    # make_plate_card_detector() → returns detector.detect (a callable)
    # Signature: detect(frame: np.ndarray, frame_id: int) -> List[DetectionEvent]
    plate_card_fn = None
    try:
        from blurberry.video.object_detector import make_plate_card_detector
        plate_card_fn = make_plate_card_detector()
        log.info("✅ Plate + card detector loaded")
    except Exception as e:
        log.warning(f"⚠️  Plate/card detector unavailable: {e}")

    # ── Person 2: NSFW detector ───────────────────────────────────────────────
    # make_nsfw_detector() → returns detector.detect (a callable)
    nsfw_fn = None
    try:
        from blurberry.video.nsfw_detector import make_nsfw_detector
        nsfw_fn = make_nsfw_detector()
        log.info("✅ NSFW detector loaded")
    except Exception as e:
        log.warning(f"⚠️  NSFW detector unavailable: {e}")

    # ── Person 4 (self): Text PII detector ───────────────────────────────────
    # make_text_pii_detector() → returns detector.detect (a callable)
    text_pii_fn = None
    try:
        from blurberry.video.text_pii_detector import make_text_pii_detector
        text_pii_fn = make_text_pii_detector()
        log.info("✅ Text PII detector loaded")
    except Exception as e:
        log.warning(f"⚠️  Text PII detector unavailable: {e}")

    # ── Person 1: VideoLoop ───────────────────────────────────────────────────
    # Constructor: VideoLoop(config: PipelineConfig, source: int = 0, debug: bool = False)
    from blurberry.video.video_loop import VideoLoop
    loop = VideoLoop(config, source=source, debug=debug)

    # Register detectors via add_detector() — Person 1's explicit API
    if plate_card_fn:
        loop.add_detector(plate_card_fn)
    if nsfw_fn:
        loop.add_detector(nsfw_fn)
    if text_pii_fn:
        loop.add_detector(text_pii_fn)

    log.info(f"✅ VideoLoop ready (source={source}, debug={debug})")
    return loop


# ── Audio pipeline ────────────────────────────────────────────────────────────
def start_audio():
    """
    Person 3's AudioPipeline constructor:
        AudioPipeline(device_index=None, whisper_model_size="tiny", on_pii_detected=None)
    NOTE: does NOT accept a 'config' parameter — Person 3 manages its own config.
    audio.start() is BLOCKING — must run in a daemon thread.
    """
    try:
        from blurberry.audio.audio_pipeline import AudioPipeline

        def on_audio_pii(findings, text):
            """Callback invoked by AudioPipeline when PII is detected."""
            types = [pii_type for _, pii_type in findings]
            log.info(f"[AUDIO PII] {types} detected in: '{text}'")

        audio = AudioPipeline(
            whisper_model_size="tiny",      # ~40MB, fastest
            on_pii_detected=on_audio_pii,   # UI callback
            # device_index=None → uses default mic
        )

        # start() is blocking — run in daemon thread
        t = threading.Thread(target=audio.start, daemon=True, name="audio-pipeline")
        t.start()
        log.info("✅ Audio pipeline started (background thread)")
        return audio

    except Exception as e:
        log.warning(f"⚠️  Audio pipeline unavailable: {e}")
        return None


# ── Web server ────────────────────────────────────────────────────────────────
def start_web_server(loop, config, host: str, port: int):
    """
    Attach VideoLoop to app.state and start FastAPI in a daemon thread.
    loop.current_output_frame (bytes) + loop.frame_lock are used by streaming endpoints.
    """
    try:
        import uvicorn
        from blurberry.web.app import app

        # Attach pipeline objects so FastAPI routes can access them
        app.state.video_loop = loop
        app.state.config = config

        t = threading.Thread(
            target=lambda: uvicorn.run(app, host=host, port=port, log_level="warning"),
            daemon=True,
            name="web-server",
        )
        t.start()
        log.info(f"✅ Web server → http://localhost:{port}")
        return t

    except Exception as e:
        log.warning(f"⚠️  Web server failed to start: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    log.info("=" * 52)
    log.info("  🫐  BlurBerry AI — starting up")
    log.info("=" * 52)

    # Resolve source: int for webcam index, str for file path
    source = int(args.source) if args.source.isdigit() else args.source

    config = PipelineConfig(
        blur_faces=True,
        blur_plates=True,
        blur_cards=True,
        blur_nsfw=True,
        blur_text_pii=True,
        detection_cadence=args.cadence,    # default 15
        blur_strength=args.blur_strength,  # default 55
    )

    # Build video pipeline + register all detectors
    loop = build_pipeline(config, source, debug=args.debug)

    # ── Face enrollment ───────────────────────────────────────────────────────
    # Person 1's API: loop.enroll_face(image_paths: List[str], name: str = "owner")
    # Accepts a LIST of file paths — averages embeddings across all photos.
    # Best practice: enroll 3–5 photos for robustness.
    if args.enroll:
        try:
            loop.enroll_face(args.enroll, name="owner")
            log.info(f"✅ Face enrolled from {len(args.enroll)} image(s): {args.enroll}")
        except Exception as e:
            log.warning(f"⚠️  Face enrollment failed: {e}")
    else:
        log.info("ℹ️  No face enrolled — ALL faces will be blurred.")
        log.info("   Use --enroll photo1.jpg photo2.jpg to enroll your face.")

    # ── Audio pipeline ────────────────────────────────────────────────────────
    audio = None
    if not args.no_audio:
        audio = start_audio()
        # Poll audio events every 500ms as Person 3 recommends
        # (UI can call audio.get_events() and audio.get_latest_transcript())

    # ── Web server ────────────────────────────────────────────────────────────
    if not args.no_web:
        start_web_server(loop, config, args.host, args.port)
        log.info(f"   Open http://localhost:{args.port} in your browser")
    else:
        log.info("ℹ️  Web server disabled (--no-web). Using OpenCV window only.")

    # ── Video loop — BLOCKING ─────────────────────────────────────────────────
    # loop.run() opens webcam, processes frames, and shows OpenCV window.
    # Also updates loop.current_output_frame (JPEG bytes) for web streaming.
    # Press 'q' in the OpenCV window to quit.
    
    log.info("🚀 Running. Open http://localhost:{} in your browser.".format(args.port))
    try:
       loop.run_headless()    
    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down...")
    finally:
        if audio:
            try:
                audio.stop()
                log.info("Audio pipeline stopped")
            except Exception:
                pass
        log.info("👋 BlurBerry AI stopped")


if __name__ == "__main__":
    main()
