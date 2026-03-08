"""
BlurBerry Web Server
Phases: ready → collecting (E) → enrolled → streaming (S)
Run:    python blurberry/web/server.py
Open:   http://localhost:5000

Key fixes vs previous version:
  1. Object detection runs in a BACKGROUND THREAD — never blocks the main loop
  2. MultiObjectTracker wired in — blur follows moving objects between YOLO runs
  3. Face suppression — faces inside detected card boxes are not blurred separately
     (fixes Aadhaar card printing the face blur instead of the full card blur)
"""

import sys, os, time, threading, queue, pickle, cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from flask import Flask, Response, jsonify, send_from_directory

app    = Flask(__name__)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
ENROLLED_DIR  = os.path.join(ROOT, 'enrolled_faces')
ENROLLED_FILE = os.path.join(ENROLLED_DIR, 'owner.pkl')
os.makedirs(ENROLLED_DIR, exist_ok=True)

# ── Shared state ───────────────────────────────────────────────────────────────
_lock    = threading.Lock()
_jpeg    = None
_running = False

_state = {
    'phase':           'idle',
    'enroll_progress': 0,
    'enroll_msg':      '',
}

_settings = {
    'faces':  True,
    'plates': True,
    'cards':  True,
    'nsfw':   True,
}

_evt_start_collect = threading.Event()
_evt_start_stream  = threading.Event()

_cap          = None
_face_app     = None
_owner_embeds = []

ENROLL_FRAMES = 30

# ── Helpers ────────────────────────────────────────────────────────────────────

def _blur_region(frame, x1, y1, x2, y2, strength=55):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    k = strength if strength % 2 == 1 else strength + 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def _is_owner(embed):
    if not _owner_embeds:
        return False
    return max(_cosine(embed, e) for e in _owner_embeds) > 0.38

def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()

def _face_inside_card(face_bbox, card_boxes):
    """
    Returns True if the centre of face_bbox falls inside any card box.
    Used to suppress InsightFace blurring the printed photo on an Aadhaar card
    instead of blurring the whole card.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    fcx = (fx1 + fx2) / 2
    fcy = (fy1 + fy2) / 2
    for (cx1, cy1, cx2, cy2) in card_boxes:
        if cx1 <= fcx <= cx2 and cy1 <= fcy <= cy2:
            return True
    return False

# ── Init ───────────────────────────────────────────────────────────────────────

def _init_camera():
    global _cap
    _cap = cv2.VideoCapture(0)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _cap.set(cv2.CAP_PROP_FPS, 30)
    return _cap.isOpened()

def _init_face_app():
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        print('[BlurBerry] insightface (buffalo_sc) OK')
    except Exception as e:
        print(f'[BlurBerry] insightface unavailable: {e}')

# ── Phase 1: ready ─────────────────────────────────────────────────────────────

def _ready_phase():
    global _jpeg
    with _lock:
        _state['phase'] = 'ready'

    ok, frame = _cap.read()
    if ok:
        with _lock:
            _jpeg = _encode_jpeg(frame)

    print('[BlurBerry] Ready — press E in the UI to start enrollment')
    _evt_start_collect.clear()

    while not _evt_start_collect.wait(timeout=0.05):
        ok, frame = _cap.read()
        if not ok:
            continue
        with _lock:
            _jpeg = _encode_jpeg(frame)

    _collect_phase()

# ── Phase 2: collecting embeddings ────────────────────────────────────────────

def _collect_phase():
    global _owner_embeds, _jpeg

    with _lock:
        _state['phase']           = 'collecting'
        _state['enroll_progress'] = 0
        _state['enroll_msg']      = 'Look at the camera'

    if os.path.exists(ENROLLED_FILE):
        os.remove(ENROLLED_FILE)
    _owner_embeds = []

    collected   = []
    enroll_tick = 0
    print(f'[Enroll] Collecting {ENROLL_FRAMES} embeddings...')

    while len(collected) < ENROLL_FRAMES:
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        display = frame.copy()
        embed   = None

        if _face_app and enroll_tick % 2 == 0:
            try:
                small = cv2.resize(frame, (640, 360))
                faces = _face_app.get(small)
                if faces:
                    sx = frame.shape[1] / 640
                    sy = frame.shape[0] / 360
                    f  = max(faces,
                             key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    embed = f.normed_embedding
                    collected.append(embed.copy())
                    x1 = int(f.bbox[0]*sx); y1 = int(f.bbox[1]*sy)
                    x2 = int(f.bbox[2]*sx); y2 = int(f.bbox[3]*sy)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 229, 160), 2)
            except Exception:
                pass
        enroll_tick += 1

        pct = int(len(collected) / ENROLL_FRAMES * 100)
        msg = ('No face detected — move closer'
               if embed is None else f'{len(collected)}/{ENROLL_FRAMES}')

        with _lock:
            _state['enroll_progress'] = pct
            _state['enroll_msg']      = msg
            _jpeg = _encode_jpeg(display)

    with open(ENROLLED_FILE, 'wb') as f:
        pickle.dump(collected, f)
    _owner_embeds = collected
    print(f'[Enroll] Done — {len(collected)} embeddings saved')

    with _lock:
        _state['phase']           = 'enrolled'
        _state['enroll_progress'] = 100
        _state['enroll_msg']      = 'Enrolled!'

    print('[BlurBerry] Enrolled — press S in the UI to start stream')
    _evt_start_stream.clear()

    while not _evt_start_stream.wait(timeout=0.05):
        ok, frame = _cap.read()
        if not ok:
            continue
        with _lock:
            _jpeg = _encode_jpeg(frame)

    _streaming_thread()

# ── Phase 3: streaming ─────────────────────────────────────────────────────────

def _streaming_thread():
    global _jpeg, _running

    with _lock:
        _state['phase'] = 'streaming'

    # ── Virtual camera ──
    vcam = None
    try:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, backend='obs')
        print('[BlurBerry] OBS virtual camera started')
    except Exception as e:
        print(f'[BlurBerry] Virtual camera skipped: {e}')

    # ── Object detector ──
    obj_det = None
    try:
        from blurberry.video.object_detector import PlateCardDetector
        obj_det = PlateCardDetector()
        print('[BlurBerry] Object detector loaded')
    except Exception as e:
        print(f'[BlurBerry] Object detector skipped: {e}')

    # ── NSFW detector ──
    nsfw_detect = None
    try:
        from blurberry.video.nsfw_detector import make_nsfw_detector
        nsfw_detect = make_nsfw_detector()
        print('[BlurBerry] NSFW detector loaded')
    except Exception as e:
        print(f'[BlurBerry] NSFW detector skipped: {e}')

    # ── Tracker ──
    from blurberry.video.tracker import MultiObjectTracker
    tracker = MultiObjectTracker(max_age=8, min_hits=1, iou_threshold=0.15)

    # ── MediaPipe fallback if InsightFace unavailable ──
    mp_face = None
    if not _face_app:
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────────────
    # Background object detection thread
    # YOLO is slow (~200-400ms on CPU). Running it on the main loop thread
    # blocks everything and causes the 1-second delay before blur appears.
    # Fix: put it in its own thread. Main loop feeds frames in, reads results out.
    # ────────────────────────────────────────────────────────────────────────
    _det_queue      = queue.Queue(maxsize=1)   # only keep the freshest frame
    _det_results    = {'objs': [], 'nsfw': [], 'fire': 0}
    _det_results_lk = threading.Lock()

    def _detection_worker():
        frame_counter = 0
        while _running:
            try:
                frame_id, frame = _det_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            new_objs  = []
            new_nsfw  = []

            if obj_det:
                try:
                    new_objs = obj_det.detect(frame, frame_id)
                except Exception as e:
                    print(f'[ObjDet] {e}')

            # NSFW runs less often — every 3rd time the worker fires
            frame_counter += 1
            if nsfw_detect and frame_counter % 3 == 0:
                try:
                    new_nsfw = nsfw_detect(frame, frame_id)
                except Exception as e:
                    print(f'[NSFW] {e}')

            with _det_results_lk:
                _det_results['objs'] = new_objs
                _det_results['nsfw'] = new_nsfw
                _det_results['fire'] += 1

    det_thread = threading.Thread(
        target=_detection_worker, daemon=True, name='ObjDetWorker'
    )
    det_thread.start()
    print('[BlurBerry] Object detection worker started')

    # ── Main streaming loop ──
    frame_id        = 0
    last_fire       = 0
    cached_nsfw     = []
    FACE_W, FACE_H  = 640, 360      # resolution for face detection
    OBJ_CADENCE     = 3             # feed a frame to detection worker every N frames

    print('[BlurBerry] Streaming...')

    while _running:
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_id += 1

        with _lock:
            s = dict(_settings)

        output = frame.copy()
        ih, iw = frame.shape[:2]
        sx     = iw / FACE_W
        sy     = ih / FACE_H

        # ── Feed frame to background detection worker ──
        # put_nowait drops the frame if the worker is still busy — that's fine,
        # we'd rather drop a frame than block the main loop
        if frame_id % OBJ_CADENCE == 0 and (obj_det or nsfw_detect):
            try:
                _det_queue.put_nowait((frame_id, frame.copy()))
            except queue.Full:
                pass   # worker still running previous frame — skip

        # ── Collect latest detection results ──
        with _det_results_lk:
            current_fire = _det_results['fire']
            latest_objs  = list(_det_results['objs'])
            latest_nsfw  = list(_det_results['nsfw'])

        # ── Update tracker only when worker fired new results ──
        if current_fire != last_fire:
            tracker.update_from_detections(latest_objs)
            cached_nsfw = latest_nsfw
            last_fire   = current_fire

        # ── Advance Kalman filters every frame ──
        # This makes blur boxes follow movement smoothly between YOLO runs.
        # Without this, boxes stay frozen at their last detected position.
        tracker.predict_all()
        tracked_objs = tracker.get_active_boxes(frame_id)

        # ── Build card box list for face suppression ──
        # If a card is detected, we blur the full card. We must NOT also blur
        # the printed face on the card via InsightFace — that would be redundant
        # and means the card number stays visible. Suppress any face whose centre
        # falls inside a card bounding box.
        card_boxes = [
            ev.bbox for ev in tracked_objs
            if ev.type in ('card', 'plate')
        ]

        # ── Face detection and blur (runs every frame) ──
        if s['faces']:
            if _face_app:
                try:
                    small = cv2.resize(frame, (FACE_W, FACE_H))
                    for f in _face_app.get(small):
                        # Scale bbox back to full resolution
                        x1 = int(f.bbox[0] * sx) - 20
                        y1 = int(f.bbox[1] * sy) - 20
                        x2 = int(f.bbox[2] * sx) + 20
                        y2 = int(f.bbox[3] * sy) + 20
                        face_bbox = (x1, y1, x2, y2)

                        # Skip if this face is inside a detected card box
                        if _face_inside_card(face_bbox, card_boxes):
                            continue

                        if not _is_owner(f.normed_embedding):
                            _blur_region(output, x1, y1, x2, y2, 71)
                except Exception:
                    pass

            elif mp_face:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_face.process(rgb)
                    if res.detections:
                        for det in res.detections:
                            bb = det.location_data.relative_bounding_box
                            x1 = int(bb.xmin * iw) - 20
                            y1 = int(bb.ymin * ih) - 20
                            x2 = int((bb.xmin + bb.width)  * iw) + 20
                            y2 = int((bb.ymin + bb.height) * ih) + 20
                            face_bbox = (x1, y1, x2, y2)
                            if _face_inside_card(face_bbox, card_boxes):
                                continue
                            _blur_region(output, x1, y1, x2, y2, 71)
                except Exception:
                    pass

        # ── Object blur (plates and cards from tracker) ──
        for ev in tracked_objs:
            if ev.type == 'plate' and s['plates']:
                _blur_region(output, *ev.bbox, strength=61)
            elif ev.type == 'card' and s['cards']:
                _blur_region(output, *ev.bbox, strength=71)

        # ── NSFW blur ──
        if s['nsfw']:
            for ev in cached_nsfw:
                _blur_region(output, *ev.bbox, strength=71)

        # ── Publish frame ──
        with _lock:
            _jpeg = _encode_jpeg(output)

        # ── Virtual camera ──
        if vcam:
            try:
                out_send = (cv2.resize(output, (1280, 720))
                            if output.shape[1] != 1280 or output.shape[0] != 720
                            else output)
                vcam.send(cv2.cvtColor(out_send, cv2.COLOR_BGR2RGB))
                vcam.sleep_until_next_frame()
            except Exception:
                pass

    # ── Cleanup ──
    if vcam:
        vcam.close()
    with _lock:
        _state['phase'] = 'idle'
    print('[BlurBerry] Stopped.')

# ── Entry ──────────────────────────────────────────────────────────────────────

def _start_thread():
    global _running, _jpeg
    _running = True
    if not _init_camera():
        print('[BlurBerry] ERROR: cannot open webcam')
        _running = False
        return
    with _lock:
        _state['phase'] = 'starting'
    ok, frame = _cap.read()
    if ok:
        with _lock:
            _jpeg = _encode_jpeg(frame)
    _init_face_app()
    _ready_phase()

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    r = send_from_directory(STATIC, 'index.html')
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    r.headers['Pragma']        = 'no-cache'
    return r

@app.route('/start', methods=['POST'])
def start():
    global _running
    with _lock:
        already = _running
    if not already:
        threading.Thread(target=_start_thread, daemon=True).start()
    return jsonify({'ok': True})

@app.route('/keypress/<key>', methods=['POST'])
def keypress(key):
    k = key.lower()
    with _lock:
        phase = _state['phase']
    if k == 'e' and phase == 'ready':
        _evt_start_collect.set()
        return jsonify({'ok': True, 'action': 'start_collect'})
    if k == 's' and phase == 'enrolled':
        _evt_start_stream.set()
        return jsonify({'ok': True, 'action': 'start_stream'})
    return jsonify({'ok': False, 'reason': f'key {k!r} not valid in phase {phase!r}'})

@app.route('/toggle/<feature>', methods=['POST'])
def toggle(feature):
    if feature not in _settings:
        return jsonify({'error': 'unknown'}), 400
    with _lock:
        _settings[feature] = not _settings[feature]
        val = _settings[feature]
    return jsonify({'feature': feature, 'enabled': val})

@app.route('/status')
def get_status():
    with _lock:
        return jsonify({
            'running':         _running,
            'phase':           _state['phase'],
            'enroll_progress': _state['enroll_progress'],
            'enroll_msg':      _state['enroll_msg'],
            'settings':        dict(_settings),
        })

@app.route('/stop', methods=['POST'])
def stop():
    global _running
    _running = False
    return jsonify({'ok': True})

def _gen():
    while True:
        with _lock:
            frame = _jpeg
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    r = Response(_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    r.headers['Pragma']        = 'no-cache'
    return r

if __name__ == '__main__':
    print('BlurBerry → http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)