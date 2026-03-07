import cv2
import numpy as np
import sys
sys.path.insert(0, ".")

from shared_types import DetectionEvent, PipelineConfig
from blurberry.video.object_detector import PlateCardDetector
from blurberry.video.nsfw_detector import NSFWDetector
from blurberry.video.blur_compositor import apply_blurs, draw_debug_overlay

config = PipelineConfig()

plate_detector = PlateCardDetector()
nsfw_detector = NSFWDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_id = 0
last_events = []  # ← stores detections across frames

print("Press 'q' to quit. Hold a license plate or credit card in front of camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    
    # Only run detection every 15 frames (expensive)
    if frame_id % 15 == 0:
        plate_events = plate_detector.detect(frame, frame_id)
        nsfw_events = nsfw_detector.detect(frame, frame_id)
        last_events = plate_events + nsfw_events  # ← UPDATE last_events
        
        if last_events:
            print(f"Frame {frame_id}: {[(e.type, round(e.confidence,2)) for e in last_events]}")

    # ← Use last_events EVERY frame (not all_events)
    output = apply_blurs(frame, last_events, config)
    output = draw_debug_overlay(output, last_events)
    
    cv2.imshow("Object Detection Test", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()