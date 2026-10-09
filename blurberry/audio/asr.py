"""
Streaming speech recognition (Vosk) that reports words in stream time.

Vosk gives three views of the audio it has heard:
  final results     exact word times, once an utterance ends
  partial text      updated on nearly every chunk, but without times
  partial words     word times, but they lag behind the text
The redactor needs the newest words fast, so it uses the partial text and times the
words not yet covered by partial words itself: they are spread over the voiced audio
that follows the last word with a known time.
"""
import json
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from blurberry.audio.spoken_pii import Word

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DEFAULT_MODEL = 'vosk-model-small-en-us-0.15'
FRAME_SECONDS = 0.01
ENERGY_HISTORY = 1000       # frames (10 s) kept for the voiced-audio estimate


def find_model(path: Optional[str] = None) -> Optional[str]:
    path = path or os.environ.get('BLURBERRY_VOSK_MODEL') or os.path.join(MODELS_DIR, DEFAULT_MODEL)
    return path if os.path.isdir(path) else None


def load_model(path: Optional[str] = None):
    from vosk import Model, SetLogLevel
    SetLogLevel(-2)     # partial-word alignment warns on every update
    found = find_model(path)
    if not found:
        raise FileNotFoundError(f'Vosk model not found ({path or DEFAULT_MODEL}); run python download_models.py')
    return Model(_low_latency_view(found))


LOW_LATENCY_CONF = '--determinize-max-delay=20\n--determinize-min-chunk-size=5\n'


def _low_latency_view(path: str) -> str:
    """Kaldi determinizes the lattice in chunks up to 60 frames behind the decoder, which
    is why word times lag the partial text by a second or more. A mirror of the model
    with a shorter determinize delay in its config brings them within a word or two.
    The mirror sits next to the model (same drive, so files can be hard links rather
    than copies); the downloaded model itself is left as it is."""
    import tempfile
    path = os.path.abspath(path)
    try:
        with open(os.path.join(path, 'conf', 'model.conf')) as f:
            conf = f.read()
    except OSError:
        return path
    if 'determinize-max-delay' in conf:
        return path
    for parent in (os.path.dirname(path), tempfile.gettempdir()):
        view = os.path.join(parent, os.path.basename(path) + '-lowlatency')
        try:
            _mirror(path, view, skip=os.path.join('conf', 'model.conf'))
            with open(os.path.join(view, 'conf', 'model.conf'), 'w') as f:
                f.write(conf.rstrip('\n') + '\n' + LOW_LATENCY_CONF)
            return view
        except OSError as e:
            error = e
    print(f'[ASR] Low-latency config unavailable ({error}); word times will lag')
    return path


def _mirror(src: str, dst: str, skip: str):
    """Recreate src under dst with hard links, or copies where links aren't possible
    (another drive, FAT). `skip` is left out so it can be written without touching src."""
    import shutil
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for name in files:
            target = os.path.join(dst, rel, name)
            if os.path.normpath(os.path.join(rel, name)) == skip or os.path.exists(target):
                continue
            try:
                os.link(os.path.join(root, name), target)
            except OSError:
                shutil.copy2(os.path.join(root, name), target)


class StreamingASR:
    def __init__(self, model, sample_rate: int, origin: int = 0):
        from vosk import KaldiRecognizer
        self.sr = sample_rate
        self.rec = KaldiRecognizer(model, sample_rate)
        self.rec.SetWords(True)
        self.origin = origin            # stream sample where this recognizer started
        self.fed = 0
        self.anchor = origin / sample_rate
        self.frame = max(1, int(sample_rate * FRAME_SECONDS))
        self.energy: deque = deque(maxlen=ENERGY_HISTORY)   # (time, rms)

    @property
    def now(self) -> float:
        return (self.origin + self.fed) / self.sr

    def accept(self, pcm: np.ndarray) -> Tuple[Optional[List[Word]], List[Word]]:
        """Feed float32 mono samples. Returns (final words or None, current partial words)."""
        t0 = self.now
        n = len(pcm) // self.frame * self.frame
        if n:
            rms = np.sqrt(np.mean(pcm[:n].reshape(-1, self.frame) ** 2, axis=1))
            self.energy.extend((t0 + i * FRAME_SECONDS, float(r)) for i, r in enumerate(rms))
        self.fed += len(pcm)

        data = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()
        if self.rec.AcceptWaveform(data):
            words = self._timed(json.loads(self.rec.Result()).get('result', []))
            if words:
                self.anchor = words[-1].end
            return words, []

        self.rec.SetPartialWords(False)
        text = json.loads(self.rec.PartialResult()).get('partial', '').split()
        self.rec.SetPartialWords(True)
        timed = self._timed(json.loads(self.rec.PartialResult()).get('partial_result', []))

        j = 0
        while j < min(len(text), len(timed)) and text[j] == timed[j].text:
            j += 1
        return None, timed[:j] + self._estimate(text[j:], timed[j - 1].end if j else self.anchor)

    def _timed(self, items) -> List[Word]:
        base = self.origin / self.sr
        return [Word(x['word'], base + x['start'], base + x['end']) for x in items]

    def _estimate(self, texts: List[str], t0: float) -> List[Word]:
        """Spread untimed words over the voiced frames after t0, in proportion to length."""
        if not texts:
            return []
        frames = [(t, r) for t, r in self.energy if t >= t0]
        times = []
        if frames:
            levels = np.array([r for _, r in self.energy])
            threshold = max(0.004, 3.0 * float(np.percentile(levels, 20)))
            times = [t for t, r in frames if r > threshold]
        if not times:
            times = list(np.arange(t0, max(self.now, t0 + FRAME_SECONDS), FRAME_SECONDS))
        weights = np.cumsum([0] + [len(w) + 2 for w in texts], dtype=float)
        weights /= weights[-1]
        n = len(times)
        out = []
        for k, w in enumerate(texts):
            a = times[min(n - 1, int(weights[k] * n))]
            b = times[min(n - 1, max(0, int(np.ceil(weights[k + 1] * n)) - 1))] + FRAME_SECONDS
            out.append(Word(w, a, max(b, a + FRAME_SECONDS), timed=False))
        return out
