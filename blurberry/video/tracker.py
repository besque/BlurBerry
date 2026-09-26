"""
Lightweight SORT tracker wrapper.
SORT uses Kalman filter + Hungarian algorithm to track bounding boxes across frames.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
from shared_types import DetectionEvent


def iou(bb_test, bb_gt):
    """Intersection over Union between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_gt = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / union if union > 0 else 0


def overlap_small(a, b):
    """Intersection over the smaller box: a box nested inside a bigger one scores high
    even when their IoU is low (e.g. a tight box vs a rotated-view envelope)."""
    iw = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))
    return iw * ih / small if small > 0 else 0


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox, event_type, blur=True):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.1
        self.kf.Q[4:, 4:] *= 0.1
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(1, y2 - y1)
        self.kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.event_type = event_type
        self.blur = blur
        self.last_box = tuple(bbox)
        self.last_center = (cx, cy)
        self.hits = 1   # the detection that spawned this track counts
        self.age = 0

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(1, y2 - y1)
        self.kf.update(np.array([[cx], [cy], [s], [r]]))
        self.time_since_update = 0
        self.hits += 1
        self.last_box = tuple(bbox)
        self.last_center = (cx, cy)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def get_bbox_exact(self, lead_frames=2.0):
        """The last detected box, moved by how far the filter thinks the object has
        travelled since. Keeps the detector's exact size instead of the smoothed one,
        stretched only on the side the object is moving towards (by lead_frames of
        motion) so a moving object doesn't slide out from under a tight box."""
        dx = self.kf.x[0, 0] - self.last_center[0]
        dy = self.kf.x[1, 0] - self.last_center[1]
        vx, vy = self.kf.x[4, 0] * lead_frames, self.kf.x[5, 0] * lead_frames
        x1, y1, x2, y2 = self.last_box
        x1, x2 = x1 + dx + min(vx, 0), x2 + dx + max(vx, 0)
        y1, y2 = y1 + dy + min(vy, 0), y2 + dy + max(vy, 0)
        return (int(x1), int(y1), int(x2), int(y2))

    def get_bbox(self):
        """Current state as (x1, y1, x2, y2). Does not advance the filter."""
        x = self.kf.x
        cx, cy, s, r = x[0,0], x[1,0], x[2,0], x[3,0]
        w = np.sqrt(abs(s * r))
        h = abs(s) / max(w, 1)
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))


class MultiObjectTracker:
    """
    max_age     frames a track survives without a detection; an int, or {type: frames}
    start_conf  {type: conf} a detection must reach to start a new track; weaker
                detections can only keep an existing track alive (hysteresis).
    min_hits    detections needed before a track is reported (1 = immediately).
    coast_grow  while a track goes undetected its box grows by this fraction per
                frame (capped at coast_grow_max), so a moving object stays covered.
    exact_size  report boxes at the detector's size (moved with the object) rather
                than the Kalman-smoothed size.
    """
    def __init__(self, max_age=60, min_hits=1, iou_threshold=0.15,
                 start_conf=None, coast_grow=0.0, coast_grow_max=0.4, exact_size=False):
        self.exact_size = exact_size
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.start_conf = start_conf or {}
        self.coast_grow = coast_grow
        self.coast_grow_max = coast_grow_max
        self.trackers: List[KalmanBoxTracker] = []

    def _max_age(self, event_type):
        if isinstance(self.max_age, dict):
            return self.max_age.get(event_type, max(self.max_age.values()))
        return self.max_age

    def update(self, detections: List[DetectionEvent]) -> List[DetectionEvent]:
        """
        Feed new detections. Returns active tracked boxes as DetectionEvents.
        Call this every frame — pass empty list on non-detection frames.
        """
        # Predict existing trackers
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())

        # Match detections to trackers via IoU
        matched_indices = set()
        if detections and predicted_boxes:
            iou_matrix = np.zeros((len(detections), len(predicted_boxes)))
            gate = np.zeros_like(iou_matrix, dtype=bool)
            for d_idx, det in enumerate(detections):
                for t_idx, pred in enumerate(predicted_boxes):
                    if det.type != self.trackers[t_idx].event_type:
                        continue
                    iou_matrix[d_idx, t_idx] = iou(det.bbox, pred)
                    gate[d_idx, t_idx] = (iou_matrix[d_idx, t_idx] >= self.iou_threshold
                                          or overlap_small(det.bbox, pred) >= 0.6)

            row_ind, col_ind = linear_sum_assignment(-(iou_matrix + gate))
            for r, c in zip(row_ind, col_ind):
                if gate[r, c]:
                    self.trackers[c].update(detections[r].bbox)
                    self.trackers[c].blur = detections[r].blur
                    matched_indices.add(r)

        # Spawn new trackers for unmatched detections confident enough to start one
        for i, det in enumerate(detections):
            if i not in matched_indices and det.confidence >= self.start_conf.get(det.type, 0.0):
                self.trackers.append(
                    KalmanBoxTracker(det.bbox, det.type, det.blur)
                )

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self._max_age(t.event_type)]

        # Return active tracked boxes
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits:
                x1, y1, x2, y2 = t.get_bbox_exact() if self.exact_size else t.get_bbox()
                grow = min(self.coast_grow * t.time_since_update, self.coast_grow_max)
                if grow > 0:
                    dx, dy = (x2 - x1) * grow / 2, (y2 - y1) * grow / 2
                    x1, y1, x2, y2 = int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy)
                results.append(DetectionEvent(
                    type=t.event_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=0.9,
                    frame_id=-1,
                    blur=t.blur
                ))
        return results