"""
Video pipeline checks that run the same way on macOS and Windows. Each one goes
through the backend this platform actually uses: camera enumeration (AVFoundation /
DirectShow), the detector model formats present (CoreML / ONNX / PyTorch), OCR
(Apple Vision / Windows OCR / EasyOCR) and window capture (Quartz / PrintWindow).
A test whose model or package is missing says so and returns instead of failing.
"""
import os
import sys
import tempfile

import cv2
import numpy as np

from blurberry.shared_types import DetectionEvent


def _skip(reason):
    print(f'  (skipped: {reason})')


def _clip(path, frames=10, size=(640, 360)):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    writer.release()


def test_text_pii_rules():
    from blurberry.video.text_pii_detector import _is_pii
    for text in ('ABCDE1234F', '9876543210', '+91 98765 43210', 'john.doe@gmail.com',
                 '1234 5678 9012', 'SBIN0001234', 'password: hunter2'):
        assert _is_pii(text), text
    for text in ('Meeting at 10', 'Hello world', 'Q3 roadmap'):
        assert not _is_pii(text), text


def test_tracker_coasts_then_drops():
    from blurberry.video.tracker import MultiObjectTracker
    tracker = MultiObjectTracker(max_age=5, min_hits=1)
    assert len(tracker.update([DetectionEvent('plate', (100, 100, 200, 150), 0.9, 0)])) == 1
    assert len([tracker.update([]) for _ in range(3)][-1]) == 1
    assert not [tracker.update([]) for _ in range(10)][-1]


def test_camera_list():
    from blurberry.video.camera import camera_names, default_camera
    names = camera_names()
    print(f'  cameras: {names}')
    assert isinstance(names, list)
    assert 0 <= default_camera() < max(1, len(names))


def test_video_file_source():
    from blurberry.video.camera import open_camera
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'clip.avi')
        _clip(path)
        cap = open_camera(path)
        ok, frame = cap.read()
        cap.release()
    assert ok and frame.shape == (360, 640, 3)


def test_detector_model_formats():
    """Every model format present loads and runs; the detector picks the platform's best."""
    from ultralytics import YOLO
    from blurberry.video.object_detector import MODELS_DIR, PlateCardDetector
    frame = np.full((720, 1280, 3), 128, np.uint8)
    found = []
    for name in ('plate_best', 'card_best'):
        for ext in ('.mlpackage', '_384x640.onnx', '.onnx', '.pt'):
            path = os.path.join(MODELS_DIR, name + ext)
            if not os.path.exists(path) or (ext == '.mlpackage' and sys.platform != 'darwin'):
                continue
            model = YOLO(path, task='detect')
            model(frame, imgsz=640, conf=0.2, verbose=False, device='cpu')
            found.append(name + ext)
    if not found:
        return _skip('no detector models; run python download_models.py')
    print(f'  ran: {", ".join(found)}')
    detector = PlateCardDetector()
    assert detector.plate_model and detector.card_model
    assert isinstance(detector.detect(frame, 0), list)


def test_ocr_finds_phone_number():
    from blurberry.video.text_pii_detector import TextPIIDetector, default_backend
    try:
        detector = TextPIIDetector(backend=default_backend())
    except ImportError as e:
        return _skip(f'no OCR engine ({e.name})')
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.putText(img, '98765 43210', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    events = detector.detect(img, 0)
    print(f'  backend: {detector.backend}')
    assert events, f'{detector.backend} OCR found no phone number'
    x1, y1, x2, y2 = events[0].bbox
    assert x1 < 120 and x2 > 400 and y1 < 200 < y2, events[0].bbox


def test_screen_and_window_capture():
    from blurberry.video.screen_capture import ScreenCapture, list_sources
    sources = list_sources()
    monitors = [s for s in sources if s['type'] == 'monitor']
    windows = [s for s in sources if s['type'] == 'window']
    assert monitors, sources
    for source in monitors[:1] + windows[:1]:
        capture = ScreenCapture()
        capture.set_source(source)
        ok, frame = capture.read()
        capture.release()
        assert ok and frame.shape == (720, 1280, 3), source['label']
    print(f'  captured: {monitors[0]["label"]}' + (f', window "{windows[0]["label"]}"' if windows else ''))


def test_face_model():
    try:
        import onnxruntime
        from insightface.app import FaceAnalysis
    except ImportError as e:
        return _skip(f'{e.name} not installed')
    providers = [p for p in ('CoreMLExecutionProvider', 'CPUExecutionProvider')
                 if p in onnxruntime.get_available_providers()]
    app = FaceAnalysis(name='buffalo_sc', providers=providers)
    app.prepare(ctx_id=0, det_size=(320, 320))
    assert app.get(np.zeros((360, 640, 3), np.uint8)) == []


def test_nsfw_model():
    try:
        from blurberry.video.nsfw_detector import make_nsfw_detector
    except ImportError as e:
        return _skip(f'{e.name} not installed')
    assert make_nsfw_detector()(np.zeros((360, 640, 3), np.uint8), 0) == []


def test_web_app_routes():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'clip.avi')
        _clip(path)
        os.environ['BLURBERRY_SOURCE'] = path
        try:
            from blurberry.web import server
        finally:
            del os.environ['BLURBERRY_SOURCE']
    client = server.app.test_client()
    assert client.get('/').status_code == 200
    status = client.get('/status').get_json()
    assert status['phase'] == 'idle' and set(status['settings']) >= {'faces', 'plates', 'voice'}
    assert client.post('/toggle/plates').get_json() == {'feature': 'plates', 'enabled': False}
    client.post('/toggle/plates')
