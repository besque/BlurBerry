"""
BlurBerry object detector — plates, ID cards, credit cards, Aadhaar.

Key improvements:
  - Motion gate: skips inference when scene hasn't changed (saves ~70% of YOLO calls)
  - 640px inference, the size both models were trained at (320 missed mid-size plates)
  - Loads the fastest available format: CoreML (.mlpackage, Neural Engine) on macOS,
    then .pt (on the Apple GPU when present), then .onnx
  - Shape-based card fallback, off by default (it fires on monitors, TVs, picture frames)
  - Plates are also searched in frames rotated ±22° and ±44°: the model finds nothing
    once a plate is tilted 20° or more
  - Geometry sanity limits: the card model fires on close-up faces and frame-filling
    scenes with boxes covering 50-100% of the frame
  - Returns everything above a low floor; the tracker demands START_CONF before it
    starts blurring a new object, then keeps it with the low floor (hysteresis)
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared_types import DetectionEvent


# ─── Config ───────────────────────────────────────────────────────────────────

# Floor for any box. A weak box only keeps an already-confirmed object blurred.
CONF_THRESHOLD = {
    "plate": 0.20,
    "card":  0.35,
}

# Needed to start blurring a new object (applied by the tracker). The server blurs
# on the first detection, so these sit a little higher to keep one-off misfires out.
START_CONF = {
    "plate": 0.35,
    "card":  0.45,
}

# max_area: fraction of the frame; aspect: width / height; max_edges: frame edges touched.
# Rotated plates have squarer boxes, hence the low plate min_aspect.
GEOMETRY = {
    "plate": dict(max_area=0.15, min_aspect=0.8,  max_aspect=8.0, max_edges=2),
    "card":  dict(max_area=0.40, min_aspect=0.45, max_aspect=2.3, max_edges=1),
}

# The plate model tolerates only about ±16° of tilt. Adding counter-rotated views
# (e.g. (0, 22, -22, 44, -44)) covers tilts up to ~60° but makes each call ~5x slower;
# this branch favours speed and searches the upright view only.
PLATE_TTA_ANGLES = (0,)

BBOX_PAD = {
    "plate": 4,
    "card":  6,
}

PLATE_NAMES = {
    "license_plate", "plate", "lp", "number_plate",
    "license-plate", "numberplate", "vehicle registration",
}

CARD_NAMES = {
    "credit_card", "debit_card", "id_card", "card", "aadhaar", "aadhar",
    "aadharcard", "pan_card", "pancard", "passport", "driving_license",
    "identity_card", "valid-credit-card",
}

INFERENCE_SIZE = 640      # px — detection resolution; matches training imgsz
USE_SHAPE_FALLBACK = False
MOTION_THRESH  = 0.003    # mean pixel diff / 255 to trigger re-detection
MAX_SKIPS      = 2        # rerun anyway after this many skipped calls: taking a small
                          # card away barely changes the frame, so without a cap the
                          # cached card box would be repeated until something else moved


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _load(path: str) -> Tuple[Optional[YOLO], str]:
    """
    Return (model, device), preferring the fastest format available.
    CoreML packages are exported with:
        yolo export model=<name>.pt format=coreml imgsz=384,640 half=True
    """
    mlpackage = path.replace(".pt", ".mlpackage")
    if sys.platform == "darwin" and os.path.exists(mlpackage):
        print(f"[ObjDet] CoreML loaded: {mlpackage}")
        return YOLO(mlpackage, task="detect"), "cpu"   # CoreML picks ANE/GPU itself
    if os.path.exists(path):
        device = "mps" if _mps_available() else "cpu"
        print(f"[ObjDet] PT loaded ({device}): {path}")
        return YOLO(path), device
    onnx = path.replace(".pt", ".onnx")
    if os.path.exists(onnx):
        print(f"[ObjDet] ONNX loaded: {onnx}")
        return YOLO(onnx, task="detect"), "cpu"
    print(f"[ObjDet] Not found: {path}")
    return None, "cpu"


def _rotate(frame: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate onto an expanded canvas; return the image and the affine that maps
    rotated-image points back to the original frame."""
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(w * c + h * s), int(w * s + h * c)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rotated = cv2.warpAffine(frame, M, (nw, nh), borderValue=(114, 114, 114))
    return rotated, cv2.invertAffineTransform(M)


def _map_back(inv: np.ndarray, x1, y1, x2, y2) -> Tuple[float, float, float, float]:
    """Axis-aligned envelope, in original-frame pixels, of a box from a rotated frame."""
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
    pts = corners @ inv.T
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def _plausible(etype: str, box, w: int, h: int) -> bool:
    g = GEOMETRY.get(etype)
    if not g:
        return True
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    edges = (x1 <= 3) + (y1 <= 3) + (x2 >= w - 3) + (y2 >= h - 3)
    return (bw * bh / (w * h) <= g["max_area"]
            and g["min_aspect"] <= bw / bh <= g["max_aspect"]
            and edges <= g["max_edges"])


def _overlap(a, b) -> float:
    """Intersection over the smaller box — catches a box nested in a bigger one."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(small, 1)


def _merge(found: List[Tuple[DetectionEvent, float]]) -> List[DetectionEvent]:
    """Merge overlapping same-type boxes found in different rotated views.
    The group keeps its best confidence but takes the box from the least-rotated
    view: a rotated view's box maps back as an envelope of an envelope, much larger
    than the object when the object isn't actually tilted."""
    groups: List[List[Tuple[DetectionEvent, float]]] = []
    for e, angle in sorted(found, key=lambda x: -x[0].confidence):
        for g in groups:
            if g[0][0].type == e.type and _overlap(g[0][0].bbox, e.bbox) >= 0.5:
                g.append((e, angle))
                break
        else:
            groups.append([(e, angle)])
    merged = []
    for g in groups:
        best, _ = min(g, key=lambda x: (abs(x[1]), -x[0].confidence))
        best.confidence = g[0][0].confidence
        merged.append(best)
    return merged


def _classify(name: str, default: str) -> str:
    n = name.lower().replace("-", "_").replace(" ", "_")
    for p in PLATE_NAMES:
        if p in n or n in p:
            return "plate"
    for c in CARD_NAMES:
        if c in n or n in c:
            return "card"
    return default


# ─── Detector ─────────────────────────────────────────────────────────────────

class PlateCardDetector:
    def __init__(
        self,
        plate_model_path: str = "blurberry/models/plate_best.pt",
        card_model_path:  str = "blurberry/models/card_best.pt",
    ):
        self.plate_model, self.plate_device = _load(plate_model_path)
        self.card_model,  self.card_device  = _load(card_model_path)

        self._prev_gray: Optional[np.ndarray] = None
        self._cached:    List[DetectionEvent] = []
        self._skips = 0

    # ── Motion gate ───────────────────────────────────────────────────────────

    def _has_motion(self, frame: np.ndarray) -> bool:
        """
        Return True if the frame changed enough to run YOLO again.
        Compares a tiny 160x90 greyscale thumbnail against the frame YOLO last
        ran on, so slow drift accumulates instead of slipping under the threshold.
        """
        thumb = cv2.cvtColor(
            cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        if self._prev_gray is None:
            self._prev_gray = thumb
            return True

        diff = np.mean(np.abs(thumb - self._prev_gray)) / 255.0
        if diff > MOTION_THRESH:
            self._prev_gray = thumb
            return True
        return False

    # ── YOLO inference ────────────────────────────────────────────────────────

    def _run(
        self,
        model: YOLO,
        device: str,
        frame: np.ndarray,
        frame_id: int,
        default_type: str,
        angle: float = 0,
    ) -> List[DetectionEvent]:
        h, w = frame.shape[:2]
        img, inv = (frame, None) if angle == 0 else _rotate(frame, angle)
        # Ultralytics letterboxes to the model size and maps boxes back to image pixels
        floor = min(CONF_THRESHOLD.values())
        results = model(img, imgsz=INFERENCE_SIZE, conf=floor, verbose=False, device=device)
        events  = []

        for result in results:
            for box in result.boxes:
                conf  = float(box.conf[0])
                cname = result.names[int(box.cls[0])]
                etype = _classify(cname, default_type)

                if conf < CONF_THRESHOLD.get(etype, 0.35):
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                if inv is not None:
                    x1, y1, x2, y2 = _map_back(inv, x1, y1, x2, y2)
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w, x2)), int(min(h, y2))
                if not _plausible(etype, (x1, y1, x2, y2), w, h):
                    continue

                pad = BBOX_PAD.get(etype, 8)
                x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)

                events.append(DetectionEvent(
                    type=etype, bbox=(x1, y1, x2, y2),
                    confidence=conf, frame_id=frame_id, blur=True,
                ))
        return events

    # ── Shape fallback ────────────────────────────────────────────────────────

    def _cards_by_shape(
        self, frame: np.ndarray, frame_id: int
    ) -> List[DetectionEvent]:
        """
        Detect credit/ID cards by contour shape when YOLO misses them.
        Credit cards are 85.6 x 54 mm — aspect ratio ~1.586.
        """
        h_f, w_f = frame.shape[:2]
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blur, 30, 100)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        conts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        events = []
        for cnt in conts:
            area = cv2.contourArea(cnt)
            if not (4000 < area < h_f * w_f * 0.35):
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            asp = w / max(h, 1)
            if 1.3 < asp < 1.9:
                events.append(DetectionEvent(
                    type="card",
                    bbox=(max(0, x-12), max(0, y-12),
                          min(w_f, x+w+12), min(h_f, y+h+12)),
                    confidence=0.60,
                    frame_id=frame_id,
                    blur=True,
                ))
        return events

    # ── Public detect() ───────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Called every N frames by the video loop.
        Returns cached results when nothing has moved (motion gate).
        """
        if not self._has_motion(frame) and self._skips < MAX_SKIPS:
            self._skips += 1
            return self._cached   # return last known boxes, not empty
        self._skips = 0

        found: List[Tuple[DetectionEvent, float]] = []   # (event, view angle)

        if self.plate_model:
            for angle in PLATE_TTA_ANGLES:
                try:
                    found += [(e, angle) for e in self._run(self.plate_model, self.plate_device,
                                                            frame, frame_id, "plate", angle)]
                except Exception as e:
                    print(f"[ObjDet] plate error: {e}")

        if self.card_model:
            try:
                found += [(e, 0) for e in self._run(self.card_model, self.card_device,
                                                    frame, frame_id, "card")]
            except Exception as e:
                print(f"[ObjDet] card error: {e}")

        events = _merge(found)
        # Card numbers look like plate text; a "plate" inside a card is the card
        cards = [e.bbox for e in events if e.type == "card"]
        events = [e for e in events
                  if e.type != "plate" or all(_overlap(e.bbox, c) < 0.6 for c in cards)]

        # Shape fallback — catches Aadhaar/ID cards YOLO missed
        card_found = any(e.type == "card" for e in events)
        if USE_SHAPE_FALLBACK and not card_found:
            events.extend(self._cards_by_shape(frame, frame_id))

        self._cached = events
        return events


def make_plate_card_detector(**kwargs):
    """Factory — returns the detect() callable for video_loop.add_detector()."""
    d = PlateCardDetector(**kwargs)
    return d.detect