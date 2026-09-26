"""
BlurBerry Web Server
Phases: idle → starting → pre_stream (enroll faces) → streaming
Run:    python blurberry/web/server.py
Open:   http://localhost:5000   (override with PORT=xxxx)

Threads while streaming:
  frame readers  → camera (and screen, when shared) keep only their newest frame
  detector worker→ plates/cards (~15 Hz) and NSFW (~2 Hz), off the video thread
  text PII worker→ OCR on its own thread (camera ~1 Hz, screen ~5 Hz)
  stream thread  → faces every frame, tracks object boxes between detections, blurs, outputs

BLURBERRY_SOURCE=<camera index | video file> picks the camera input
(default: the first camera that isn't a virtual one such as OBS Virtual Camera).

Named face whitelist, screen sharing, text PII and the blur slider are ported
from CenServe (censerve/web/server.py).
"""

import sys, os, time, threading, pickle, cv2, numpy as np
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from flask import Flask, Response, jsonify, send_from_directory, request

app    = Flask(__name__)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
ENROLLED_DIR  = os.path.join(ROOT, 'enrolled_faces')
ENROLLED_FILE = os.path.join(ENROLLED_DIR, 'faces.pkl')
os.makedirs(ENROLLED_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_ENROLLED   = 5
ENROLL_FRAMES  = 30
OWNER_THRESHOLD = 0.38

VIRTUAL_CAMERA_WORDS = ('obs', 'virtual')

def _default_camera() -> int:
    """Index of the first real camera. Once the OBS Virtual Camera is installed it
    usually sorts first on macOS, and reading it would feed our own output back in.
    Mirrors OpenCV's AVFoundation order: video + muxed devices sorted by uniqueID."""
    if sys.platform != 'darwin':
        return 0
    try:
        import AVFoundation as AV
        devs = (list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
                + list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)))
        devs.sort(key=lambda d: str(d.uniqueID()))
        for i, d in enumerate(devs):
            name = str(d.localizedName())
            if not any(w in name.lower() for w in VIRTUAL_CAMERA_WORDS):
                print(f'[BlurBerry] Camera: {name} (index {i})')
                return i
    except Exception as e:
        print(f'[BlurBerry] Camera list unavailable ({e}); using camera 0')
    return 0

_src          = os.environ.get('BLURBERRY_SOURCE', '')
SOURCE        = (int(_src) if _src.isdigit() else _src) if _src else _default_camera()
CAM_W, CAM_H  = 1280, 720
SCREEN_FPS    = 15
JPEG_QUALITY  = 90
OBJ_INTERVAL  = 1 / 20   # seconds between plate/card detections
NSFW_INTERVAL = 0.5      # seconds between NSFW detections
TEXT_INTERVAL = {'camera': 1.0, 'screen': 0.2}   # seconds between OCR runs
TEXT_TTL_FRAMES = 30     # drop OCR results older than this

# Objects (fast branch): blurred on the first detection that reaches START_CONF;
# weaker detections keep an existing box. Boxes stay at the detector's size and
# are dropped after a few frames without a detection, so nothing lingers.
OBJ_TRACK_MAX_AGE = {'card': 6, 'plate': 10}
OBJ_MIN_HITS      = 1
OBJ_COAST_GROW    = 0.0

# ── Shared state ───────────────────────────────────────────────────────────────
_lock    = threading.Lock()
_jpeg    = None
_running = False

_blur_strength = 55  # 11-99, odd

_state = {
    'phase': 'idle',
    'fps':   0.0,
}

_settings = {
    'faces':    True,
    'plates':   True,
    'cards':    True,
    'nsfw':     True,
    'text_pii': True,
}

_cap            = None   # camera _FrameReader
_screen_reader  = None   # screen _FrameReader, while sharing
_source_mode    = 'camera'
_face_app       = None
_obj_det        = None
_nsfw_detect    = None
_text_worker    = None
_enrolled_faces = {}     # name → [embedding, ...]

_evt_begin_stream = threading.Event()

_enroll_state = {
    'active':    False,
    'name':      '',
    'collected': [],
    'progress':  0,
    'msg':       '',
    'done':      False,
}
_enroll_lock = threading.Lock()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _blur_region(frame, x1, y1, x2, y2, strength=55, oval=False):
    """Blur a region in place. oval=True limits it to an ellipse (for faces)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    # Shrink the region to a few cells across, then smooth it back up. The cell count
    # is relative to the region, so a face close to the camera is hidden as well as a
    # small one (a fixed kernel leaves big faces recognisable). Strength 11 → 16 cells,
    # 55 → ~9, 99 → 3.
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    cells = float(np.interp(strength, [11, 99], [16, 3]))
    scale = cells / max(rw, rh)
    small = cv2.resize(roi, (max(1, round(rw * scale)), max(1, round(rh * scale))),
                       interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    k = max(3, (min(rw, rh) // max(1, int(cells))) | 1)
    blurred = cv2.GaussianBlur(blurred, (k, k), 0)
    if oval:
        mask = np.zeros((rh, rw), np.uint8)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
        roi[mask > 0] = blurred[mask > 0]
    else:
        roi[:] = blurred

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def _is_owner(embed):
    for embeds in _enrolled_faces.values():
        for e in embeds:
            if _cosine(embed, e) > OWNER_THRESHOLD:
                return True
    return False

def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()

def _blur_faces(frame, output, strength):
    if not _face_app:
        return
    try:
        for f in _face_app.get(frame):
            if not _is_owner(f.normed_embedding):
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                _blur_region(output, x1 - 20, y1 - 20, x2 + 20, y2 + 20, strength, oval=True)
    except Exception:
        pass

# ── Enrolled faces persistence ─────────────────────────────────────────────────

def _save_enrolled_faces():
    with open(ENROLLED_FILE, 'wb') as f:
        pickle.dump(_enrolled_faces, f)

def _load_enrolled_faces():
    global _enrolled_faces
    if os.path.exists(ENROLLED_FILE):
        with open(ENROLLED_FILE, 'rb') as f:
            _enrolled_faces = pickle.load(f)
        return
    _enrolled_faces = {}
    # The single-owner version saved a plain list of embeddings in owner.pkl
    legacy = os.path.join(ENROLLED_DIR, 'owner.pkl')
    if os.path.exists(legacy):
        with open(legacy, 'rb') as f:
            embeds = pickle.load(f)
        if isinstance(embeds, list) and embeds:
            _enrolled_faces['You'] = embeds
            _save_enrolled_faces()
            print('[BlurBerry] Imported the previously enrolled face as "You"')

# ── Enrollment frame collector ─────────────────────────────────────────────────

def _collect_enrollment_frame(frame):
    with _enroll_lock:
        if not _enroll_state['active']:
            return
        name = _enroll_state['name']

    small = cv2.resize(frame, (640, 360))
    faces = None
    if _face_app:
        try:
            faces = _face_app.get(small)
        except Exception:
            pass

    with _enroll_lock:
        if not _enroll_state['active']:
            return
        if not faces:
            _enroll_state['msg'] = 'No face — move closer'
            return
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        _enroll_state['collected'].append(f.normed_embedding.copy())
        count = len(_enroll_state['collected'])
        _enroll_state['progress'] = int(count / ENROLL_FRAMES * 100)
        _enroll_state['msg'] = f'{count} / {ENROLL_FRAMES}'
        if count >= ENROLL_FRAMES:
            _enrolled_faces[name] = list(_enroll_state['collected'])
            _enroll_state['active'] = False
            _enroll_state['done'] = True
            _enroll_state['msg'] = f'{name} enrolled'
            _save_enrolled_faces()

# ── Frame readers and detector worker ──────────────────────────────────────────

class _FrameReader:
    """Pulls frames on its own thread and keeps only the newest one.

    read()   → the next frame the stream thread hasn't seen (blocks briefly)
    latest() → (id, frame) of the newest frame, for the detector workers
    `cap` is anything with read() → (ok, frame) and release(): a cv2.VideoCapture
    or a ScreenCapture. pace_fps throttles sources that don't pace themselves;
    loop rewinds a video file at its end (for testing).
    """
    def __init__(self, cap, pace_fps=0.0, loop=False):
        self.cap   = cap
        self.pace  = pace_fps
        self.loop  = loop
        self._cond  = threading.Condition()
        self._frame = None
        self._id    = 0
        self._seen  = 0
        self._stop  = False
        threading.Thread(target=self._run, daemon=True, name='frames').start()

    def _run(self):
        period = 1.0 / self.pace if self.pace else 0.0
        due = time.monotonic()
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._id += 1
                self._cond.notify_all()
            if period:
                due = max(due + period, time.monotonic() - period)
                time.sleep(max(0.0, due - time.monotonic()))
        self.cap.release()

    def read(self, timeout=1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._id > self._seen, timeout):
                return False, None
            self._seen = self._id
            return True, self._frame

    def latest(self):
        with self._cond:
            return self._id, self._frame

    def release(self):
        self._stop = True


def _active_reader():
    if _source_mode == 'screen' and _screen_reader is not None:
        return _screen_reader
    return _cap


class _DetectorWorker(threading.Thread):
    """Runs plate/card and NSFW detection off the stream thread, so a slow
    inference never stalls the video. Results are tagged with the source mode
    they came from; the stream picks them up via snapshot()."""
    def __init__(self, obj_det, nsfw_detect):
        super().__init__(daemon=True, name='detectors')
        self.obj_det     = obj_det
        self.nsfw_detect = nsfw_detect
        self._lock = threading.Lock()
        self._seq  = 0
        self._mode = None
        self._objs = []
        self._nsfw = []
        self._halt = threading.Event()

    def snapshot(self):
        with self._lock:
            return self._seq, self._mode, self._objs, self._nsfw

    def stop(self):
        self._halt.set()

    def run(self):
        last_nsfw = 0.0
        nsfw, nsfw_mode = [], None
        while not self._halt.is_set():
            t0 = time.monotonic()
            mode = _source_mode
            reader = _active_reader()
            fid, frame = reader.latest() if reader else (0, None)
            if frame is None:
                self._halt.wait(0.01)
                continue

            objs = []
            if self.obj_det:
                try:
                    objs = self.obj_det.detect(frame, fid)
                except Exception:
                    objs = []

            if mode != nsfw_mode:
                nsfw, nsfw_mode, last_nsfw = [], mode, 0.0
            if self.nsfw_detect and t0 - last_nsfw >= NSFW_INTERVAL:
                last_nsfw = t0
                try:
                    nsfw = self.nsfw_detect(frame, fid)
                except Exception:
                    nsfw = []

            with self._lock:
                self._objs, self._nsfw, self._mode = objs, nsfw, mode
                self._seq += 1
            self._halt.wait(max(0.0, OBJ_INTERVAL - (time.monotonic() - t0)))

# ── Init ───────────────────────────────────────────────────────────────────────

def _init_camera():
    global _cap
    cap = cv2.VideoCapture(SOURCE)
    is_file = isinstance(SOURCE, str)
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return False
    _cap = _FrameReader(cap, pace_fps=(cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0.0,
                        loop=is_file)
    return True

def _init_face_app():
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        # CoreML runs on the Neural Engine on Macs (~3x faster, identical results);
        # onnxruntime drops it silently where it isn't available.
        _face_app = FaceAnalysis(name='buffalo_sc',
                                 providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        print('[BlurBerry] insightface (buffalo_sc) OK')
    except Exception as e:
        print(f'[BlurBerry] insightface unavailable: {e}')

def _init_detectors():
    """Load the detectors and run one warm-up inference each. First inference
    compiles CoreML and builds the predictor (~3 s); doing it here, during
    'starting', keeps it from freezing the stream later."""
    global _obj_det, _nsfw_detect, _text_worker
    warm = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    try:
        from blurberry.video.object_detector import PlateCardDetector
        _obj_det = PlateCardDetector()
        _obj_det.detect(warm, -1)
        _obj_det._prev_gray, _obj_det._cached = None, []
        print('[BlurBerry] Object detector loaded')
    except Exception as e:
        print(f'[BlurBerry] Object detector skipped: {e}')
    try:
        from blurberry.video.nsfw_detector import make_nsfw_detector
        _nsfw_detect = make_nsfw_detector()
        _nsfw_detect(warm, -1)
        print('[BlurBerry] NSFW detector loaded')
    except Exception as e:
        print(f'[BlurBerry] NSFW detector skipped: {e}')
    try:
        from blurberry.video.text_pii_detector import TextPIIWorker
        if sys.platform == 'darwin':
            backend = 'vision'
        else:
            try:
                import paddleocr  # noqa: F401
                backend = 'paddle'
            except ImportError:
                backend = 'easy'
        _text_worker = TextPIIWorker(backend=backend, mode='camera')
        print(f'[BlurBerry] Text PII worker started (backend={backend})')
    except Exception as e:
        print(f'[BlurBerry] Text PII skipped: {e}')

def _open_vcam():
    try:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=CAM_W, height=CAM_H, fps=30,
                                   fmt=pyvirtualcam.PixelFormat.BGR, backend='obs')
        print('[BlurBerry] OBS virtual camera started')
        return vcam
    except Exception as e:
        print(f'[BlurBerry] Virtual camera skipped: {e}')
        return None

# ── Pre-stream phase: live preview with face blur, enroll faces ────────────────

def _pre_stream_thread():
    global _jpeg
    _evt_begin_stream.clear()
    print('[BlurBerry] Pre-stream — enroll faces, then click Start Stream')

    while _running and not _evt_begin_stream.is_set():
        ok, frame = _cap.read()
        if not ok:
            continue
        output = frame.copy()
        with _lock:
            do_faces, strength = _settings['faces'], _blur_strength
        if do_faces:
            _blur_faces(frame, output, strength)
        _collect_enrollment_frame(frame)
        with _lock:
            _jpeg = _encode_jpeg(output)

# ── Streaming ──────────────────────────────────────────────────────────────────

def _streaming_thread():
    global _jpeg, _running
    from blurberry.video.tracker import MultiObjectTracker
    from blurberry.video.object_detector import START_CONF

    with _lock:
        _state['phase'] = 'streaming'

    vcam   = _open_vcam()
    worker = _DetectorWorker(_obj_det, _nsfw_detect)
    worker.start()

    def new_tracker():
        return MultiObjectTracker(max_age=OBJ_TRACK_MAX_AGE, min_hits=OBJ_MIN_HITS,
                                  iou_threshold=0.2, start_conf=START_CONF,
                                  coast_grow=OBJ_COAST_GROW, exact_size=True)

    tracker      = new_tracker()
    frame_id     = 0
    seen_seq     = 0
    tracked      = []
    cached_nsfw  = []
    mode_seen    = None
    mode_fid     = 0
    text_sent_t  = 0.0
    text_hold    = deque(maxlen=2)   # last two OCR results, so one missed read doesn't unblur
    text_seen_fid = -1
    fps_count, fps_t0 = 0, time.monotonic()
    print('[BlurBerry] Streaming...')

    while _running:
        mode = _source_mode
        reader = _active_reader()
        ok, frame = reader.read()
        if not ok:
            continue
        is_screen = mode == 'screen'

        if mode != mode_seen:
            # New source: boxes from the old one mean nothing here
            tracker, tracked, cached_nsfw, mode_seen, mode_fid = new_tracker(), [], [], mode, frame_id
            text_hold.clear()
            if _text_worker:
                _text_worker.set_mode(mode)

        with _lock:
            s = dict(_settings)
            strength = _blur_strength

        output = frame.copy()

        # ── Faces (camera only), every frame ────────────────────────────────
        if not is_screen and s['faces']:
            _blur_faces(frame, output, strength)
        if not is_screen:
            _collect_enrollment_frame(frame)
        elif _cap is not None:
            _, cam = _cap.latest()
            if cam is not None:
                _collect_enrollment_frame(cam)

        # ── Plates / cards / NSFW from the worker ───────────────────────────
        # Fresh detections correct the tracker; in between it predicts where boxes moved
        seq, det_mode, objs, nsfw = worker.snapshot()
        if seq != seen_seq and det_mode == mode:
            seen_seq    = seq
            tracked     = tracker.update(objs)
            cached_nsfw = nsfw
        else:
            tracked = tracker.update([])

        for ev in tracked:
            if (ev.type == 'plate' and s['plates']) or (ev.type == 'card' and s['cards']):
                _blur_region(output, *ev.bbox, strength)

        if s['nsfw']:
            for ev in cached_nsfw:
                _blur_region(output, *ev.bbox, strength)

        # ── Text PII ────────────────────────────────────────────────────────
        if s['text_pii'] and _text_worker:
            now = time.monotonic()
            if now - text_sent_t >= TEXT_INTERVAL[mode]:
                text_sent_t = now
                _text_worker.submit_frame(frame, frame_id)
            fid = _text_worker.last_result_fid
            if fid != text_seen_fid and fid >= mode_fid:
                text_seen_fid = fid
                text_hold.append(_text_worker.latest_events)
            if text_seen_fid >= 0 and frame_id - text_seen_fid <= TEXT_TTL_FRAMES:
                for events in text_hold:
                    for ev in events:
                        _blur_region(output, *ev.bbox, strength)

        # ── Output ──────────────────────────────────────────────────────────
        with _lock:
            _jpeg = _encode_jpeg(output)

        if vcam:
            try:
                if output.shape[1] != CAM_W or output.shape[0] != CAM_H:
                    output = cv2.resize(output, (CAM_W, CAM_H))
                vcam.send(output)   # pacing comes from the source itself
            except Exception:
                pass

        frame_id += 1
        fps_count += 1
        if time.monotonic() - fps_t0 >= 1.0:
            with _lock:
                _state['fps'] = round(fps_count / (time.monotonic() - fps_t0), 1)
            fps_count, fps_t0 = 0, time.monotonic()

    # ── Cleanup ─────────────────────────────────────────────────────────────
    worker.stop()
    if _screen_reader:
        _screen_reader.release()
    _cap.release()
    if vcam:
        vcam.close()
    with _lock:
        _state['phase'] = 'idle'
        _state['fps'] = 0.0
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
    if _face_app is None:
        _init_face_app()
    if _obj_det is None and _nsfw_detect is None:
        _init_detectors()
    _load_enrolled_faces()
    with _lock:
        _state['phase'] = 'pre_stream'
    _pre_stream_thread()
    if _running:
        _streaming_thread()
    else:
        _cap.release()
        with _lock:
            _state['phase'] = 'idle'

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    r = send_from_directory(STATIC, 'index.html')
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    r.headers['Pragma'] = 'no-cache'
    return r

@app.route('/start', methods=['POST'])
def start():
    with _lock:
        already = _running
    if not already:
        threading.Thread(target=_start_thread, daemon=True).start()
    return jsonify({'ok': True})

@app.route('/blur_strength', methods=['POST'])
def set_blur_strength():
    global _blur_strength
    data = request.get_json(silent=True) or {}
    try:
        val = int(data.get('value', _blur_strength))
    except (TypeError, ValueError):
        return jsonify({'error': 'value must be a number'}), 400
    val = max(11, min(99, val))
    if val % 2 == 0:
        val += 1
    with _lock:
        _blur_strength = val
    return jsonify({'blur_strength': val})

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
        s = dict(_settings)
        s['blur_strength'] = _blur_strength
        return jsonify({
            'running':  _running,
            'phase':    _state['phase'],
            'fps':      _state['fps'],
            'source':   _source_mode,
            'settings': s,
        })

@app.route('/stop', methods=['POST'])
def stop():
    global _running
    _running = False
    return jsonify({'ok': True})

@app.route('/screens', methods=['GET'])
def get_screens():
    try:
        from blurberry.video.screen_capture import list_sources
        screens = list_sources()
    except Exception as e:
        print(f'[Screens] {e}')
        screens = []
    return jsonify({'screens': screens})

@app.route('/source', methods=['POST'])
def set_source():
    global _source_mode, _screen_reader
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'camera')
    if mode == 'screen':
        src = data.get('source') or {}
        if src.get('type') not in ('monitor', 'window'):
            return jsonify({'error': 'source required for screen mode'}), 400
        source = {'type': src['type'], 'label': str(src.get('label', ''))}
        for key in ('index', 'wid', 'hwnd'):
            if key in src:
                source[key] = int(src[key])
        from blurberry.video.screen_capture import ScreenCapture
        sc = ScreenCapture()
        sc.set_source(source)
        old, _screen_reader = _screen_reader, _FrameReader(sc, pace_fps=SCREEN_FPS)
        _source_mode = 'screen'
        if old:
            old.release()
        print(f'[BlurBerry] Source → screen: {source["label"]}')
    else:
        _source_mode = 'camera'
        old, _screen_reader = _screen_reader, None
        if old:
            old.release()
        print('[BlurBerry] Source → camera')
    return jsonify({'ok': True, 'mode': _source_mode})

# ── Face enrollment routes ─────────────────────────────────────────────────────

@app.route('/faces', methods=['GET'])
def get_faces():
    return jsonify({'faces': list(_enrolled_faces.keys())})

@app.route('/faces/<name>', methods=['DELETE'])
def delete_face(name):
    if name in _enrolled_faces:
        del _enrolled_faces[name]
        _save_enrolled_faces()
    return jsonify({'ok': True})

@app.route('/enroll/start', methods=['POST'])
def enroll_start():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()[:40]
    if not name:
        return jsonify({'error': 'name is required'}), 400
    if len(_enrolled_faces) >= MAX_ENROLLED and name not in _enrolled_faces:
        return jsonify({'error': f'Maximum {MAX_ENROLLED} faces allowed'}), 400
    with _enroll_lock:
        _enroll_state['active']    = True
        _enroll_state['name']      = name
        _enroll_state['collected'] = []
        _enroll_state['progress']  = 0
        _enroll_state['msg']       = 'Look at the camera'
        _enroll_state['done']      = False
    return jsonify({'ok': True})

@app.route('/enroll/status', methods=['GET'])
def enroll_status():
    with _enroll_lock:
        return jsonify({
            'active':   _enroll_state['active'],
            'name':     _enroll_state['name'],
            'progress': _enroll_state['progress'],
            'msg':      _enroll_state['msg'],
            'done':     _enroll_state['done'],
        })

@app.route('/enroll/cancel', methods=['POST'])
def enroll_cancel():
    with _enroll_lock:
        _enroll_state['active'] = False
        _enroll_state['msg']    = 'Cancelled'
    return jsonify({'ok': True})

@app.route('/stream/start', methods=['POST'])
def stream_start():
    with _lock:
        phase = _state['phase']
    if phase != 'pre_stream':
        return jsonify({'error': f'Cannot start stream in phase: {phase}'}), 400
    _evt_begin_stream.set()
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
    r.headers['Pragma'] = 'no-cache'
    return r

def _request_camera_access():
    """macOS only lets the main thread show the camera permission prompt,
    but the camera is opened later from a worker thread. Probe it here once."""
    cap = cv2.VideoCapture(SOURCE)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        print('[BlurBerry] WARNING: webcam not available. On macOS, allow camera access for '
              'your terminal app in System Settings → Privacy & Security → Camera, then restart.')

if __name__ == '__main__':
    if sys.platform == 'darwin' and isinstance(SOURCE, int):
        _request_camera_access()
    port = int(os.environ.get('PORT', 5000))
    print(f'BlurBerry → http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, threaded=True)
