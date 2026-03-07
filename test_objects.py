import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

from shared_types import PipelineConfig
from blurberry.video.face_pipeline import FacePipeline
from blurberry.video.object_detector import PlateCardDetector
from blurberry.video.nsfw_detector import NSFWDetector
from blurberry.video.blur_compositor import apply_blurs, draw_debug_overlay

config = PipelineConfig()

face_pipeline = FacePipeline(config)
plate_detector = PlateCardDetector(confidence_threshold=0.3)  # lower threshold
nsfw_detector = NSFWDetector(confidence_threshold=0.3)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_id = 0
last_object_events = []  # persists between frames like faces

print("✅ Running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # Faces — run every frame (same as Person 1)
    face_events = face_pipeline.detect_faces(frame, frame_id)

    # Objects — run every 15 frames, PERSIST results between detections
    if frame_id % 15 == 0:
        plate_events = plate_detector.detect(frame, frame_id)
        nsfw_events = nsfw_detector.detect(frame, frame_id)
        last_object_events = plate_events + nsfw_events
        if last_object_events:
            print(f"Frame {frame_id}: {[(e.type, round(e.confidence,2)) for e in last_object_events]}")

    # Combine — fresh face events + persisted object events
    all_events = face_events + last_object_events

    output = apply_blurs(frame, all_events, config)
    output = draw_debug_overlay(output, all_events)

    cv2.imshow("BlurBerry Test", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






