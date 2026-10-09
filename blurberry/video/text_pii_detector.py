"""
Text PII detector — OCRs a frame and blurs text that looks like personal data
(PAN, Aadhaar, phone, card numbers, emails, IFSC, OTPs, visible passwords).

OCR backends:
  'vision'   Apple Vision (macOS, on-device, fast; the default on Macs)
  'windows'  Windows.Media.Ocr (Windows 10/11, on-device, fast; the default on Windows)
  'paddle'   PaddleOCR
  'easy'     EasyOCR (fallback everywhere; slow on CPU)

Ported from CenServe (censerve/video/text_pii_detector.py).
"""
import re
import cv2
import numpy as np
import os, sys
import threading
import time
from collections import deque

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

from typing import List, Tuple
from blurberry.shared_types import DetectionEvent


# ── PII regex patterns ────────────────────────────────────────────────────────

_RAW_PATTERNS = [
    re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),                            # PAN
    re.compile(r'[\w.+-]+@[\w-]+\.[\w]{2,}'),                          # Email (strict)
    re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),                          # IFSC
    re.compile(r'(?i)(password|passwd|pwd)\s*[:\-=]\s*\S+'),           # Visible password
]

_DIGIT_PATTERNS = [
    re.compile(r'\d{12}'),              # Aadhaar
    re.compile(r'[6-9]\d{9}'),          # Indian mobile
    re.compile(r'\d{13,19}'),           # Credit/debit card
    re.compile(r'\d{6}'),              # OTP / PIN
]


def _is_pii(text: str) -> bool:
    """Ultra-permissive PII check. Errs on the side of blurring."""
    raw = text.strip()
    if not raw:
        return False

    if any(p.search(raw) for p in _RAW_PATTERNS):
        return True

    compact = re.sub(r'\s+', '', raw)

    # Any token with @ that has chars on both sides is treated as email
    if '@' in compact:
        user, _, domain = compact.partition('@')
        if len(user) >= 1 and len(domain) >= 2:
            return True

    digits_only = re.sub(r'\D', '', raw)
    if len(digits_only) >= 6 and any(p.search(digits_only) for p in _DIGIT_PATTERNS):
        return True

    # Phone-like: +91, 0-prefixed 10-digit, etc.
    phone_compact = re.sub(r'[\s\-\(\)]', '', raw)
    if re.search(r'(\+\d{1,3})?\d{10,}', phone_compact):
        return True

    return False


def _merge_adjacent_tokens(
    ocr_results: list,
    x_gap_px: int = 30,
    y_overlap_ratio: float = 0.5,
) -> list:
    """
    Merge OCR tokens that sit on the same line and are close together.
    This recovers emails/phones that OCR splits across boxes, e.g.
    ["gaganrh@gmail", ".com"] or ["+91", "98765", "43210"].
    Returns a new list with the same (pts, text, conf) format.
    """
    if len(ocr_results) <= 1:
        return ocr_results

    def _bbox(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    items = []
    for (pts, text, conf) in ocr_results:
        x1, y1, x2, y2 = _bbox(pts)
        items.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                      'text': text, 'conf': conf, 'merged': False})

    items.sort(key=lambda t: (t['y1'], t['x1']))

    merged = []
    for i, a in enumerate(items):
        if a['merged']:
            continue
        group_text = a['text']
        gx1, gy1, gx2, gy2 = a['x1'], a['y1'], a['x2'], a['y2']
        gconf = a['conf']

        for j in range(i + 1, len(items)):
            b = items[j]
            if b['merged']:
                continue
            h_a = gy2 - gy1
            h_b = b['y2'] - b['y1']
            overlap_top = max(gy1, b['y1'])
            overlap_bot = min(gy2, b['y2'])
            overlap = max(0, overlap_bot - overlap_top)
            if overlap < y_overlap_ratio * min(h_a, h_b):
                continue
            gap = b['x1'] - gx2
            if gap > x_gap_px:
                continue
            if gap < -0.5 * min(gx2 - gx1, b['x2'] - b['x1']):
                continue

            group_text = group_text + b['text']
            gx2 = max(gx2, b['x2'])
            gy1 = min(gy1, b['y1'])
            gy2 = max(gy2, b['y2'])
            gconf = min(gconf, b['conf'])
            b['merged'] = True

        pts_merged = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]
        merged.append((pts_merged, group_text, gconf))

    return merged


def default_backend() -> str:
    if sys.platform == 'darwin':
        return 'vision'
    if sys.platform == 'win32':
        return 'windows'
    try:
        import paddleocr  # noqa: F401
        return 'paddle'
    except ImportError:
        return 'easy'


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class _WindowsOCR:
    """The OCR engine built into Windows 10/11, through pywinrt. Construction runs one
    recognition, so a missing language pack or API mismatch shows up where the caller
    can fall back to EasyOCR rather than on every frame."""

    def __init__(self):
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter
        self._bitmap, self._bgra8, self._writer = SoftwareBitmap, BitmapPixelFormat.BGRA8, DataWriter
        self._threads = threading.local()
        self._join_apartment()
        self.engine = (OcrEngine.try_create_from_user_profile_languages()
                       or OcrEngine.try_create_from_language(Language('en-US')))
        if self.engine is None:
            raise RuntimeError('no OCR language pack installed (Settings > Time & language > Language)')
        probe = np.full((80, 360, 3), 255, np.uint8)
        cv2.putText(probe, '98765 43210', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        self.read(probe)

    def _join_apartment(self):
        """Blocking on a WinRT call needs a multi-threaded COM apartment, once per thread."""
        if getattr(self._threads, 'ready', False):
            return
        from winrt.runtime import MTA, init_apartment
        try:
            init_apartment(MTA)
        except Exception:
            pass
        self._threads.ready = True

    def read(self, bgr: np.ndarray) -> list:
        self._join_apartment()
        h, w = bgr.shape[:2]
        writer = self._writer()
        writer.write_bytes(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA).tobytes())
        bitmap = self._bitmap.create_copy_from_buffer(writer.detach_buffer(), self._bgra8, w, h)
        out = []
        for line in self.engine.recognize_async(bitmap).get().lines:
            rects = [word.bounding_rect for word in line.words]
            if not rects:
                continue
            x1, y1 = min(r.x for r in rects), min(r.y for r in rects)
            x2, y2 = max(r.x + r.width for r in rects), max(r.y + r.height for r in rects)
            out.append(([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], line.text.strip(), 1.0))
        return out


# ── Detector class ────────────────────────────────────────────────────────────

class TextPIIDetector:
    def __init__(self, backend: str = 'easy', mode: str = 'camera'):
        """
        backend: 'vision', 'windows', 'paddle' or 'easy'; see default_backend().
        mode: 'camera' or 'screen'.
        """
        self.backend = backend
        self.mode = mode
        self.reader = None

        if backend == 'vision':
            try:
                import Vision  # noqa: F401
                print('[TextPII] Apple Vision OCR loaded')
                return
            except Exception as e:
                print(f'[TextPII] Apple Vision unavailable ({e}), falling back to EasyOCR')
                backend = 'easy'
                self.backend = 'easy'

        if backend == 'windows':
            try:
                self.reader = _WindowsOCR()
                print('[TextPII] Windows OCR loaded')
                return
            except Exception as e:
                print(f'[TextPII] Windows OCR unavailable ({e}), falling back to EasyOCR')
                backend = 'easy'
                self.backend = 'easy'

        if backend == 'paddle':
            try:
                from paddleocr import PaddleOCR
                self.reader = PaddleOCR(
                    use_angle_cls=False, lang='en',
                    show_log=False, use_gpu=False,
                )
                print('[TextPII] PaddleOCR loaded')
            except Exception as e:
                print(f'[TextPII] PaddleOCR failed ({e}), falling back to EasyOCR')
                backend = 'easy'
                self.backend = 'easy'

        if backend == 'easy':
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=_cuda_available(), verbose=False)
            print('[TextPII] EasyOCR loaded')

    def _run_ocr(self, img: np.ndarray) -> list:
        """Returns list of (pts, text, conf). pts = 4 corner points."""
        if self.backend == 'vision':
            return self._ocr_vision(img)
        if self.backend == 'windows':
            return self.reader.read(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.backend == 'paddle':
            return self._ocr_paddle(rgb)
        return self._ocr_easy(rgb)

    def _ocr_vision(self, bgr: np.ndarray) -> list:
        import Vision
        from Foundation import NSData
        h, w = bgr.shape[:2]
        _, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        data = NSData.dataWithBytes_length_(jpg.tobytes(), len(jpg))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)   # correction "fixes" emails and IDs
        handler.performRequests_error_([req], None)
        out = []
        for obs in req.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            bb = obs.boundingBox()                 # normalised, origin bottom-left
            x, bw = bb.origin.x * w, bb.size.width * w
            y, bh = (1 - bb.origin.y - bb.size.height) * h, bb.size.height * h
            pts = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            out.append((pts, str(cands[0].string()).strip(), float(cands[0].confidence())))
        return out

    def _ocr_easy(self, rgb: np.ndarray) -> list:
        results = []
        for (pts, text, conf) in self.reader.readtext(rgb):
            if isinstance(text, str):
                text = text.strip()
            results.append((pts, text, conf))
        return results

    def _ocr_paddle(self, rgb: np.ndarray) -> list:
        result = self.reader.ocr(rgb, cls=False)
        out = []
        if not result or not result[0]:
            return out
        for line in result[0]:
            # v2 format: [pts, (text, conf)]
            pts = line[0]
            text_conf = line[1]
            text = str(text_conf[0]).strip()
            conf = float(text_conf[1])
            out.append((pts, text, conf))
        return out

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        h, w = frame.shape[:2]

        crop_top = 0
        if self.mode == 'screen':
            crop_top = int(h * 0.10)
            crop_bot = int(h * 0.90)
            roi = frame[crop_top:crop_bot, :]
        else:
            roi = frame
            crop_bot = h

        rh, rw = roi.shape[:2]
        target = 1280 if self.mode == 'screen' else 640
        scale = target / max(rh, rw)
        if scale < 1.0:
            small = cv2.resize(roi, (int(rw * scale), int(rh * scale)))
        else:
            small = roi
            scale = 1.0
        inv = 1.0 / scale
        CONF_THRESHOLD = 0.30

        events = []
        try:
            t0 = time.time()
            ocr_results = self._run_ocr(small)
            elapsed = time.time() - t0

            # Merge adjacent tokens so split emails/phones get concatenated
            merged_results = _merge_adjacent_tokens(ocr_results, x_gap_px=35)
            all_results = ocr_results + merged_results

            seen_texts = set()
            for (pts, text, conf) in all_results:
                if conf < CONF_THRESHOLD:
                    continue

                if not _is_pii(text):
                    continue

                # Deduplicate: don't blur same text region twice
                text_key = text.strip().lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = int(min(xs) * inv) - 10
                y1 = int(min(ys) * inv) - 10
                x2 = int(max(xs) * inv) + 10
                y2 = int(max(ys) * inv) + 10

                y1 += crop_top
                y2 += crop_top

                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(w, x2);  y2 = min(h, y2)

                events.append(DetectionEvent(
                    type='text_pii',
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    frame_id=frame_id,
                    blur=True,
                ))


        except Exception as e:
            print(f'[TextPII] OCR error: {e}')

        return events


def make_text_pii_detector(backend: str = 'easy', mode: str = 'camera') -> callable:
    d = TextPIIDetector(backend=backend, mode=mode)
    return d.detect


# ── Async worker ──────────────────────────────────────────────────────────────

# Keep results for this many frame_ids (for display-buffer lookup)
_EVENTS_BY_FID_MAX = 90
# When using display delay, process frames in order. Queue holds ~delay*fps so
# the frame we display (oldest in buffer) has been processed by the time we show it.
_PENDING_QUEUE_MAX = 50


class TextPIIWorker:
    """
    Runs TextPIIDetector on a background thread.
    The main loop calls submit_frame() which never blocks.
    Results are available as latest_events (with TTL) or get_events_for_frame(fid)
    when using a display delay so OCR has time to finish for that frame.
    Uses a small in-order queue so that when the server buffers video by AV_DELAY_SECONDS,
    each displayed frame has its OCR result ready (processed in order, stored by frame_id).
    """

    def __init__(self, backend: str = 'easy', mode: str = 'camera', ordered: bool = False):
        """ordered=False keeps only the newest frame, so results never fall behind
        the live picture; ordered=True queues frames for a delayed display buffer."""
        self._detector = TextPIIDetector(backend=backend, mode=mode)
        self._ordered = ordered
        self._lock = threading.Lock()
        self._pending_queue: deque = deque(maxlen=_PENDING_QUEUE_MAX)  # (frame, fid)
        self._events: List[DetectionEvent] = []
        self._last_result_fid: int = -1
        self._events_by_fid: dict = {}  # frame_id -> list of events (for delayed display)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def backend(self) -> str:
        return self._detector.backend

    def submit_frame(self, frame: np.ndarray, frame_id: int):
        """Queue a frame for OCR. Non-blocking; if queue full, newest is dropped so we keep oldest (for display delay)."""
        with self._lock:
            if not self._ordered:
                self._pending_queue.clear()
            elif len(self._pending_queue) >= _PENDING_QUEUE_MAX:
                return
            self._pending_queue.append((frame.copy(), frame_id))

    def set_mode(self, mode: str):
        """'camera' or 'screen' — follows the video source when it's switched."""
        self._detector.mode = mode

    def get_events_for_frame(self, frame_id: int) -> List[DetectionEvent]:
        """Return OCR results for a specific frame_id (for use with display buffer)."""
        with self._lock:
            return list(self._events_by_fid.get(frame_id, []))

    @property
    def latest_events(self) -> List[DetectionEvent]:
        with self._lock:
            return list(self._events)

    @property
    def last_result_fid(self) -> int:
        with self._lock:
            return self._last_result_fid

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            with self._lock:
                if not self._pending_queue:
                    item = None
                else:
                    item = self._pending_queue.popleft()

            if item is None:
                time.sleep(0.05)
                continue

            frame, fid = item
            try:
                results = self._detector.detect(frame, fid)
                with self._lock:
                    self._events = results
                    self._last_result_fid = fid
                    self._events_by_fid[fid] = results
                    if len(self._events_by_fid) > _EVENTS_BY_FID_MAX:
                        for k in sorted(self._events_by_fid.keys())[:-_EVENTS_BY_FID_MAX]:
                            del self._events_by_fid[k]
            except Exception as e:
                print(f'[TextPII Worker] error: {e}')

            time.sleep(0.02)
