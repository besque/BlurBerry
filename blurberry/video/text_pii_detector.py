import cv2
import numpy as np
import re
import spacy
from paddleocr import PaddleOCR
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared_types import DetectionEvent

# PII Regex patterns — Indian context (mirrors audio pipeline patterns)
PII_PATTERNS = {
    "aadhaar":        re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan_card":       re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "indian_phone":   re.compile(r"\b[6-9]\d{9}\b"),
    "email":          re.compile(r"\b[\w.+-]+@[\w-]+\.\w{2,}\b"),
    "credit_card":    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "ip_address":     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ifsc_code":      re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    # Fixed: negative lookahead to avoid false-positives overlapping with email pattern
    "upi_id":         re.compile(r"\b[\w.+-]+@(?![\w-]+\.\w{2,})[\w]+\b"),
    "password_label": re.compile(r"(?i)\b(password|passwd|pin|secret|otp)\s*[:=\s]\s*\S+"),
    "bank_account":   re.compile(r"\b\d{9,18}\b"),
}


class TextPIIDetector:
    def __init__(self, lang: str = "en", confidence_threshold: float = 0.7):
        """
        lang: "en" for English only, or ["en", "hi"] for Hindi + English
        confidence_threshold: minimum OCR confidence to consider a text region
        """
        self.conf = confidence_threshold

        print("[TextPII] Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
        )
        self.nlp = spacy.load("en_core_web_sm")
        print("[TextPII] Ready")

    def _is_pii(self, text: str) -> bool:
        """Return True if text contains any PII pattern or sensitive NER entity."""
        # Regex pass
        for pattern in PII_PATTERNS.values():
            if pattern.search(text):
                return True

        # spaCy NER pass — flag names, locations if they appear in sensitive context
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "LOC"]:
                return True

        return False

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Run OCR on frame and return DetectionEvents for text regions containing PII.
        Registered with VideoLoop via loop.add_detector(make_text_pii_detector()).
        """
        try:
            # PaddleOCR returns: [[[bbox_points], (text, confidence)], ...]
            results = self.ocr.ocr(frame, cls=True)
        except Exception as e:
            print(f"[TextPII] OCR error: {e}")
            return []

        if not results or results[0] is None:
            return []

        events = []
        for line in results[0]:
            if line is None:
                continue

            try:
                bbox_points, (text, conf) = line
            except (ValueError, TypeError):
                continue

            # Skip low-confidence OCR reads
            if conf < self.conf:
                continue

            # Skip if text contains no PII
            if not self._is_pii(text):
                continue

            # bbox_points: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            xs = [int(p[0]) for p in bbox_points]
            ys = [int(p[1]) for p in bbox_points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            # Add padding so blur fully covers the text
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            events.append(DetectionEvent(
                type="text_pii",
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                frame_id=frame_id,
                blur=True,
            ))
            print(f"[TextPII] Detected '{text}' → blurring at {(x1, y1, x2, y2)}")

        return events


def make_text_pii_detector(**kwargs) -> callable:
    """
    Factory function — returns detector.detect ready to plug into VideoLoop.
    Usage in main.py:
        from blurberry.video.text_pii_detector import make_text_pii_detector
        loop.add_detector(make_text_pii_detector())
    """
    detector = TextPIIDetector(**kwargs)
    return detector.detect