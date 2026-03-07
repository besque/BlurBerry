import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared_types import DetectionEvent


class PlateCardDetector:
    PLATE_CLASSES = ["license_plate", "plate", "LP", "number_plate"]
    CARD_CLASSES = ["credit_card", "id_card", "card", "aadhaar", "passport", "debit_card", "valid-credit-card"]

    def __init__(
        self,
        plate_model_path: str = "blurberry/models/plate_best.pt",
        card_model_path: Optional[str] = "blurberry/models/card_best.pt",
        confidence_threshold: float = 0.35,  # raised to avoid false positives
        device: str = "cpu"
    ):
        self.conf = confidence_threshold
        self.device = device

        # Load plate model
        if os.path.exists(plate_model_path):
            self.plate_model = YOLO(plate_model_path)
            print(f"[PlateCardDetector] Loaded plate model: {plate_model_path}")
            print(f"[PlateCardDetector] Plate model classes: {self.plate_model.names}")
        else:
            print(f"[PlateCardDetector] WARNING: Plate model not found at {plate_model_path}")
            self.plate_model = None

        # Load card model (optional)
        self.card_model = None
        if card_model_path and os.path.exists(card_model_path):
            self.card_model = YOLO(card_model_path)
            print(f"[PlateCardDetector] Loaded card model: {card_model_path}")
            print(f"[PlateCardDetector] Card model classes: {self.card_model.names}")
        else:
            print(f"[PlateCardDetector] No card model loaded (optional)")

    def _yolo_to_events(self, results, frame_id: int, default_type: str) -> List[DetectionEvent]:
        """Convert YOLO results to DetectionEvent list."""
        events = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls_name = result.names[int(box.cls[0])].lower()

                # Determine event type from class name
                event_type = default_type
                for c in self.CARD_CLASSES:
                    if c.lower() in cls_name:
                        event_type = "card"
                        break
                for c in self.PLATE_CLASSES:
                    if c.lower() in cls_name:
                        event_type = "plate"
                        break

                events.append(DetectionEvent(
                    type=event_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    frame_id=frame_id,
                    blur=True
                ))
        return events

    def _shape_based_card_detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Fallback: detect any card-shaped rectangle.
        Works for Aadhaar, ID cards, credit cards regardless of what's printed.
        Credit card standard aspect ratio = 85.6mm / 53.98mm = ~1.586
        """
        events = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_frame, w_frame = frame.shape[:2]
        min_area = (w_frame * h_frame) * 0.03   # at least 3% of frame
        max_area = (w_frame * h_frame) * 0.60   # at most 60% of frame

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue

            # Approximate to polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Must be roughly rectangular (4-6 corners)
            if len(approx) < 4 or len(approx) > 6:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)

            # Card aspect ratio ~1.586, generous tolerance for angled holding
            if 1.1 < aspect < 2.0:
                events.append(DetectionEvent(
                    type="card",
                    bbox=(x, y, x + w, y + h),
                    confidence=0.65,
                    frame_id=frame_id,
                    blur=True
                ))

        return events

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Main detection function. Returns list of DetectionEvents.
        Runs YOLO for plates and cards, falls back to shape detection for cards.
        """
        events = []

        # Downscale frame for speed
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1.0:
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small_frame = frame
            scale = 1.0

        # Plate detection (YOLO)
        if self.plate_model is not None:
            plate_results = self.plate_model(
                small_frame, conf=self.conf, verbose=False, device=self.device
            )
            plate_events = self._yolo_to_events(plate_results, frame_id, "plate")
            for e in plate_events:
                x1, y1, x2, y2 = e.bbox
                e.bbox = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
            events.extend(plate_events)

        # Card detection (YOLO first)
        if self.card_model is not None:
            card_results = self.card_model(
                small_frame, conf=self.conf, verbose=False, device=self.device
            )
            card_events = self._yolo_to_events(card_results, frame_id, "card")
            for e in card_events:
                x1, y1, x2, y2 = e.bbox
                e.bbox = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
            events.extend(card_events)

        # Shape-based fallback — only runs if YOLO didn't find a card
        card_already_found = any(e.type == "card" for e in events)
        if not card_already_found:
            shape_events = self._shape_based_card_detect(frame, frame_id)
            events.extend(shape_events)

        return events


# Convenience function for registering with pipeline
def make_plate_card_detector(**kwargs):
    """Returns a detect() function ready to plug into the main pipeline."""
    detector = PlateCardDetector(**kwargs)
    return detector.detect





# import cv2
# import numpy as np
# from ultralytics import YOLO
# from typing import List, Optional
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from shared_types import DetectionEvent


# class PlateCardDetector:
#     PLATE_CLASSES = ["license_plate", "plate", "LP", "number_plate"] 
#     CARD_CLASSES = ["credit_card", "id_card", "card", "aadhaar", "passport", "debit_card","valid-credit-card"]
    
#     def __init__(
#         self,
#         plate_model_path: str = "blurberry/models/plate_best.pt",
#         card_model_path: Optional[str] = "blurberry/models/card_best.pt",
#         confidence_threshold: float = 0.4,
#         device: str = "cpu"  # "cpu" or "0" for GPU
#     ):
#         self.conf = confidence_threshold
#         self.device = device
        
#         if os.path.exists(plate_model_path):
#             self.plate_model = YOLO(plate_model_path)
#             print(f"[PlateCardDetector] Loaded plate model: {plate_model_path}")
#             print(f"[PlateCardDetector] Plate model classes: {self.plate_model.names}")
#         else:
#             print(f"[PlateCardDetector] WARNING: Plate model not found at {plate_model_path}")
#             self.plate_model = None
        
#         # Load card model (optional)
#         self.card_model = None
#         if card_model_path and os.path.exists(card_model_path):
#             self.card_model = YOLO(card_model_path)
#             print(f"[PlateCardDetector] Loaded card model: {card_model_path}")
#             print(f"[PlateCardDetector] Card model classes: {self.card_model.names}")
#         else:
#             print(f"[PlateCardDetector] No card model loaded (optional)")

#     def _yolo_to_events(self, results, frame_id: int, default_type: str) -> List[DetectionEvent]:
#         """Convert YOLO results to DetectionEvent list."""
#         events = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
#                 conf = float(box.conf[0])
#                 cls_name = result.names[int(box.cls[0])].lower()
                
#                 # Determine event type from class name
#                 event_type = default_type
#                 for c in self.CARD_CLASSES:
#                     if c.lower() in cls_name:
#                         event_type = "card"
#                         break
#                 for c in self.PLATE_CLASSES:
#                     if c.lower() in cls_name:
#                         event_type = "plate"
#                         break
                
#                 events.append(DetectionEvent(
#                     type=event_type,
#                     bbox=(x1, y1, x2, y2),
#                     confidence=conf,
#                     frame_id=frame_id,
#                     blur=True
#                 ))
#         return events

#     def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
#         """
#         Main detection function. Returns list of DetectionEvents.
#         This is the function you register with the main video pipeline.
#         """
#         print(f"[DEBUG] detect() called on frame {frame_id}")
#         events = []
        
#         # Run on downscaled frame for speed
#         h, w = frame.shape[:2]
#         scale = 640 / max(h, w)
#         if scale < 1.0:
#             small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
#         else:
#             small_frame = frame
#             scale = 1.0
        
#         # Plate detection
#         if self.plate_model is not None:
#             plate_results = self.plate_model(
#                 small_frame,
#                 conf=self.conf,
#                 verbose=False,
#                 device=self.device
#             )
#             plate_events = self._yolo_to_events(plate_results, frame_id, "plate")
#             # Scale bounding boxes back to original resolution
#             for e in plate_events:
#                 x1, y1, x2, y2 = e.bbox
#                 e.bbox = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
#             events.extend(plate_events)
        
#         # Card detection
#         if self.card_model is not None:
#             card_results = self.card_model(
#                 small_frame,
#                 conf=self.conf,
#                 verbose=False,
#                 device=self.device
#             )
#             card_events = self._yolo_to_events(card_results, frame_id, "card")
#             for e in card_events:
#                 x1, y1, x2, y2 = e.bbox
#                 e.bbox = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
#             events.extend(card_events)
        
#         print(f"[DEBUG] events found: {events}") 
#         return events


# # Convenience function for registering with pipeline
# def make_plate_card_detector(**kwargs):
#     """Returns a detect() function ready to plug into the main pipeline."""
#     detector = PlateCardDetector(**kwargs)
#     return detector.detect