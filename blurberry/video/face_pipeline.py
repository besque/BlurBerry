import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from typing import List
import pickle
import os
from blurberry.shared_types import DetectionEvent, PipelineConfig

ENROLLED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "enrolled_faces")


class FacePipeline:
    def __init__(self, config: PipelineConfig, enrolled_dir: str = ENROLLED_DIR):
        self.config = config
        self.enrolled_dir = enrolled_dir
        os.makedirs(enrolled_dir, exist_ok=True)

        self.face_app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection", "recognition"]
        )
        self.face_app.prepare(ctx_id=-1, det_size=(320, 320))

        self.enrolled_embeddings: List[np.ndarray] = []
        self._load_enrolled_faces()
        print(f"[FacePipeline] Loaded {len(self.enrolled_embeddings)} enrolled face(s)")

    def _load_enrolled_faces(self):
        """One mean embedding per person, from this pipeline's own files (one array each),
        the web app's faces.pkl ({name: [embeddings]}) and the old owner.pkl (a list)."""
        self.enrolled_embeddings = []
        for f in sorted(os.listdir(self.enrolled_dir)):
            if not f.endswith(".pkl"):
                continue
            with open(os.path.join(self.enrolled_dir, f), "rb") as fp:
                data = pickle.load(fp)
            for person in (data.values() if isinstance(data, dict) else [data]):
                emb = np.asarray(person, dtype=np.float32)
                if emb.ndim == 2:
                    emb = emb.mean(axis=0)
                if emb.ndim == 1 and emb.size:
                    self.enrolled_embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))

    def enroll_from_embeddings(self, embeddings: List[np.ndarray], name: str = "owner"):
        """Average a list of embeddings collected from webcam and save."""
        if not embeddings:
            raise ValueError("No embeddings to enroll")
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        save_path = os.path.join(self.enrolled_dir, f"{name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(mean_emb, f)
        self._load_enrolled_faces()
        print(f"[FacePipeline] Enrolled '{name}' from {len(embeddings)} frames")

    def enroll_face(self, image_paths: List[str], name: str = "owner"):
        embeddings = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            faces = self.face_app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings.append(face.embedding)
        if not embeddings:
            raise ValueError("No faces found in any image")
        self.enroll_from_embeddings(embeddings, name)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _is_whitelisted(self, embedding: np.ndarray) -> bool:
        for enrolled in self.enrolled_embeddings:
            if self._cosine_similarity(embedding, enrolled) >= self.config.face_similarity_threshold:
                return True
        return False

    def detect_faces(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Single method — runs InsightFace detection + recognition every call.
        """
        faces = self.face_app.get(frame)
        events = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            is_whitelisted = self._is_whitelisted(face.embedding) if self.enrolled_embeddings else False
            events.append(DetectionEvent(
                type="face",
                bbox=(x1, y1, x2, y2),
                confidence=float(face.det_score),
                frame_id=frame_id,
                blur=not is_whitelisted
            ))
        return events