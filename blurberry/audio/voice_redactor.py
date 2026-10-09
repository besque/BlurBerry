"""
Voice redaction: a broadcast-style delay line for the microphone.

  mic ─► ring buffer ──────────── held for DELAY seconds ────────────► output
            └─► ASR (100 ms chunks) ─► spoken PII detector ─► bleep spans ─┘

Every sample waits in the ring before it is played. Meanwhile the ASR transcribes it
and the detector marks the samples of sensitive words; as marked samples leave the
ring they are replaced by a 1 kHz bleep (or silence). The web server holds the video
back by the same delay, so lips stay in sync with the redacted audio.

Run on its own:
  python -m blurberry.audio.voice_redactor                  mic -> headphones, live
  python -m blurberry.audio.voice_redactor in.wav out.wav   a recording, simulated in real time

BLURBERRY_VOICE_DELAY  seconds of delay (default 1.0)
BLURBERRY_VOICE_MODE   bleep | mute
BLURBERRY_MIC          input device number or part of its name (default: system mic)
BLURBERRY_AUDIO_OUT    output device for the call (default: BlackHole / VB-Cable if
                       installed); 'none' to disable
BLURBERRY_VOSK_MODEL   path to a Vosk model folder
"""
import os
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from blurberry.audio.asr import StreamingASR, load_model
from blurberry.audio.spoken_pii import Word, detect

DELAY = float(os.environ.get('BLURBERRY_VOICE_DELAY', 1.0))
MODE = os.environ.get('BLURBERRY_VOICE_MODE', 'bleep')
CHUNK_SECONDS = 0.1
RING_SECONDS = 30
PASSTHROUGH_DELAY = 0.08        # redaction off: just enough to absorb device jitter
CONTEXT_SECONDS = 8.0           # finished words kept so triggers can arm across utterances
PAD_TIMED = (0.05, 0.05)        # margin around a word, seconds before / after
PAD_ESTIMATED = (0.15, 0.12)
COMMIT_AHEAD = 0.1              # provisional bleeps this close to playing become permanent
BLEEP_HZ = 1000
BLEEP_GAIN = 0.25
FADE_SECONDS = 0.005
RESYNC_SECONDS = 0.08           # an output this far off its playhead jumps back onto it
VIRTUAL_OUTPUTS = ('blackhole', 'vb-audio', 'cable input', 'loopback', 'soundflower')
VIRTUAL_INPUTS = VIRTUAL_OUTPUTS + ('cable output', 'obs', 'zoomaudio', 'teams')
VIRTUAL_MIC_APP = 'VB-Cable' if sys.platform == 'win32' else 'BlackHole'
MIC_SETTINGS = {'darwin': 'System Settings > Privacy & Security > Microphone',
                'win32': 'Settings > Privacy & security > Microphone'}


class SpanSet:
    """Sorted, non-overlapping [start, end) sample ranges to redact."""

    def __init__(self):
        self.spans: List[List[int]] = []

    def add(self, s: int, e: int) -> List[Tuple[int, int]]:
        """Merge [s, e) in; return the parts of it that weren't covered before."""
        if e <= s:
            return []
        new, cur = [], s
        for a, b in self.spans:
            if b <= cur:
                continue
            if a >= e:
                break
            if a > cur:
                new.append((cur, a))
            cur = max(cur, b)
        if cur < e:
            new.append((cur, e))
        if new:
            merged = []
            for a, b in sorted(self.spans + [[s, e]]):
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            self.spans = merged
        return new

    def covered(self, s: int, e: int) -> int:
        return sum(max(0, min(b, e) - max(a, s)) for a, b in self.spans)

    def mask(self, start: int, n: int, fade: int) -> Optional[np.ndarray]:
        """0..1 per sample of [start, start+n); ramps over `fade` samples outside each span."""
        m = None
        for a, b in self.spans:
            if b + fade <= start or a - fade >= start + n:
                continue
            if m is None:
                idx = np.arange(start, start + n)
                m = np.zeros(n, np.float32)
            d = np.maximum(np.maximum(a - idx, idx - (b - 1)), 0)
            np.maximum(m, np.clip(1 - d / fade, 0, 1), out=m)
        return m

    def prune(self, before: int):
        self.spans = [s for s in self.spans if s[1] > before]


class VoiceRedactor:
    """The delay line, ASR and detector, without any audio devices: feed() samples in,
    process() runs recognition, render() reads them back out delayed and redacted."""

    def __init__(self, sample_rate: int, model=None, delay: float = DELAY, mode: str = MODE):
        self.sr = sample_rate
        self.model = model or load_model()
        self.delay = delay
        self.mode = mode
        self.enabled = True
        self.ring = np.zeros(sample_rate * RING_SECONDS, np.float32)
        self.written = 0
        self.spans = SpanSet()          # committed: exact word times, or already playing
        self.provisional = SpanSet()    # estimated word times, replaced on every update
        self.lock = threading.Lock()
        self.chunk = int(sample_rate * CHUNK_SECONDS)
        self.asr: Optional[StreamingASR] = None
        self.asr_pos = 0
        self.context: deque = deque()
        self.history: deque = deque(maxlen=200)       # (text, label) of finished words
        self.partial: List[Tuple[str, Optional[str]]] = []
        self.redacted = 0
        self.finished = 0
        self.leaked = 0                                # samples of PII speech already played when found
        self.level = 0.0
        self.heard = False                             # any non-zero sample: a blocked mic gives pure zeros

    @property
    def active_delay(self) -> float:
        return self.delay if self.enabled else PASSTHROUGH_DELAY

    def playhead(self) -> int:
        return self.written - int(self.active_delay * self.sr)

    def feed(self, block: np.ndarray):
        self.heard = self.heard or bool(block.any())
        n, cap = len(block), len(self.ring)
        with self.lock:
            i = self.written % cap
            k = min(n, cap - i)
            self.ring[i:i + k] = block[:k]
            self.ring[:n - k] = block[k:]
            self.written += n

    def _read(self, pos: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        cap = len(self.ring)
        lo, hi = max(pos, self.written - cap, 0), min(pos + n, self.written)
        if hi > lo:
            i, k = lo % cap, hi - lo
            first = min(k, cap - i)
            out[lo - pos:lo - pos + first] = self.ring[i:i + first]
            out[lo - pos + first:hi - pos] = self.ring[:k - first]
        return out

    def render(self, pos: int, n: int) -> np.ndarray:
        with self.lock:
            out = self._read(pos, n)
            mask = None
            if self.enabled:
                fade = int(FADE_SECONDS * self.sr)
                masks = [m for m in (self.spans.mask(pos, n, fade), self.provisional.mask(pos, n, fade))
                         if m is not None]
                mask = np.maximum.reduce(masks) if masks else None
        if mask is None:
            return out
        tone = 0.0
        if self.mode == 'bleep':
            tone = BLEEP_GAIN * np.sin(2 * np.pi * BLEEP_HZ * np.arange(pos, pos + n) / self.sr).astype(np.float32)
        return out * (1 - mask) + tone * mask

    def process(self) -> bool:
        """Recognise the next chunk if it has arrived. False when there was nothing to do."""
        if self.written - self.asr_pos < self.chunk:
            return False
        if not self.enabled:
            self.asr, self.asr_pos = None, self.written
            with self.lock:
                self.partial = []
                self.provisional = SpanSet()
            return False
        if self.asr is None or self.written - self.asr_pos > len(self.ring) // 2:
            self.asr_pos = self.written - self.chunk
            self.asr = StreamingASR(self.model, self.sr, origin=self.asr_pos)
            self.context.clear()

        with self.lock:
            pcm = self._read(self.asr_pos, self.chunk)
        self.asr_pos += self.chunk
        self.level = float(np.sqrt(np.mean(pcm ** 2)))
        finals, partial = self.asr.accept(pcm)

        while self.context and self.context[0].end < self.asr.now - CONTEXT_SECONDS:
            self.context.popleft()
        ctx = list(self.context)
        words = ctx + (finals if finals is not None else partial)
        flags = detect(words)
        self._redact(words, flags)

        with self.lock:
            if finals is not None:
                for i, w in enumerate(finals, len(ctx)):
                    label = flags.get(i)
                    s, e = int(w.start * self.sr), int(w.end * self.sr)
                    if not label and e > s and self.spans.covered(s, e) > 0.5 * (e - s):
                        label = 'bleeped'
                    self.redacted += bool(label)
                    self.finished += 1
                    self.history.append((w.text, label))
                self.context.extend(finals)
                self.partial = []
            else:
                self.partial = [(w.text, flags.get(i)) for i, w in enumerate(partial, len(ctx))]
        return True

    def _redact(self, words: List[Word], flags: Dict[int, str]):
        """Words with exact times are committed. Estimated ones only replace the previous
        estimates, so a guess that improves on the next update doesn't widen the bleep;
        whatever part of an estimate is already playing is committed first."""
        played = self.playhead()
        sr = self.sr
        provisional = SpanSet()
        with self.lock:
            horizon = played + int(COMMIT_AHEAD * sr)
            for a, b in self.provisional.spans:
                if a < horizon:
                    self.spans.add(a, min(b, horizon))
            for i in flags:
                w = words[i]
                s, e = int(w.start * sr), int(w.end * sr)
                if w.timed:
                    for a, b in self.spans.add(s - int(PAD_TIMED[0] * sr), e + int(PAD_TIMED[1] * sr)):
                        self.leaked += max(0, min(b, e, played) - max(a, s))
                else:
                    provisional.add(s - int(PAD_ESTIMATED[0] * sr), e + int(PAD_ESTIMATED[1] * sr))
            self.provisional = provisional
            self.spans.prune(played - int((CONTEXT_SECONDS + 2) * sr))

    def transcript(self, n: int = 60) -> List[dict]:
        """Recent words for display. Redacted words carry only their label, never the text."""
        with self.lock:
            done = list(self.history)[-n:]
            partial = list(self.partial)
        out = [{'w': None if label else text, 'pii': label} for text, label in done]
        out += [{'w': None if label else text, 'pii': label, 'partial': True} for text, label in partial]
        return out


# ── Devices ───────────────────────────────────────────────────────────────────

def _is_virtual(name: str, words) -> bool:
    return any(w in name.lower() for w in words)


def _devices(sd, kind: str) -> List[Tuple[int, dict]]:
    """(index, info) of every device that can `kind`, those of the default host API first.
    Windows lists each device once per API (MME, DirectSound, WASAPI, WDM-KS); the default
    one (MME) resamples for us and opens reliably."""
    devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d[f'max_{kind}_channels'] > 0]
    return sorted(devices, key=lambda item: item[1]['hostapi'] != sd.default.hostapi)


def _device_from_hint(sd, hint, kind: str) -> Optional[int]:
    if hint is None or str(hint).strip() == '':
        return None
    if str(hint).isdigit():
        return int(hint)
    for i, d in _devices(sd, kind):
        if str(hint).lower() in d['name'].lower():
            return i
    raise ValueError(f'No {kind} audio device matching {hint!r}')


def find_input(sd, hint=None) -> int:
    idx = _device_from_hint(sd, hint, 'input')
    if idx is not None:
        return idx
    default = sd.default.device[0]
    if default is not None and default >= 0 and not _is_virtual(sd.query_devices(default)['name'], VIRTUAL_INPUTS):
        return default
    for i, d in _devices(sd, 'input'):
        if not _is_virtual(d['name'], VIRTUAL_INPUTS):
            return i
    raise RuntimeError('No microphone found')


def find_output(sd, hint=None) -> Optional[int]:
    if hint is not None and str(hint).strip().lower() == 'none':
        return None
    idx = _device_from_hint(sd, hint, 'output')
    if idx is not None:
        return idx
    for i, d in _devices(sd, 'output'):
        if _is_virtual(d['name'], VIRTUAL_OUTPUTS):
            return i
    return None


class _Sink:
    """An output device playing the redacted stream. Each device runs on its own clock,
    so each sink keeps its own read position and jumps back onto the playhead if it drifts."""

    def __init__(self, sd, core: VoiceRedactor, device: int):
        info = sd.query_devices(device, 'output')
        self.name = info['name']
        self.core = core
        self.pos = None
        self.stream = sd.OutputStream(device=device, samplerate=core.sr,
                                      channels=max(1, min(2, info['max_output_channels'])),
                                      dtype='float32', latency='low', callback=self._callback)
        self.stream.start()

    def _callback(self, out, frames, _time, _status):
        target = self.core.playhead()
        if self.pos is None or abs(self.pos - target) > RESYNC_SECONDS * self.core.sr:
            self.pos = target
        out[:] = self.core.render(self.pos, frames)[:, None]
        self.pos += frames

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class LiveVoice:
    """A VoiceRedactor wired to the microphone and output devices, on its own threads.
    'output' is the virtual mic the call app listens to; 'monitor' plays the same
    redacted audio on the default speakers/headphones so you can hear the bleeps."""

    def __init__(self, delay: float = DELAY, mode: str = MODE, mic=None, output=None):
        import sounddevice as sd
        self.sd = sd
        self.mic = find_input(sd, mic if mic is not None else os.environ.get('BLURBERRY_MIC'))
        info = sd.query_devices(self.mic, 'input')
        self.mic_name = info['name']
        self.core = VoiceRedactor(int(info['default_samplerate']), delay=delay, mode=mode)
        self.output = find_output(sd, output if output is not None else os.environ.get('BLURBERRY_AUDIO_OUT'))
        self.sinks: Dict[str, _Sink] = {}
        self.stream = None
        self.running = False
        self.started = 0.0
        self.error: Optional[str] = None

    def start(self):
        if self.running:
            return
        self.stream = self.sd.InputStream(device=self.mic, samplerate=self.core.sr, channels=1,
                                          dtype='float32', latency='low', callback=self._on_audio)
        self.stream.start()
        self.running, self.started = True, time.monotonic()
        if self.output is not None:
            self._open('output', self.output)
        threading.Thread(target=self._asr_loop, daemon=True, name='voice-asr').start()
        out = self.sinks['output'].name if 'output' in self.sinks else f'none (install {VIRTUAL_MIC_APP} for a virtual mic)'
        print(f'[Voice] {self.mic_name} -> {out}, {self.core.delay:.1f} s delay, {self.core.mode}')

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        for sink in self.sinks.values():
            sink.close()
        self.sinks.clear()

    def _open(self, key: str, device: int):
        try:
            self.sinks[key] = _Sink(self.sd, self.core, device)
        except Exception as e:
            self.error = f'{key} device failed: {e}'
            print(f'[Voice] {self.error}')

    def _on_audio(self, indata, _frames, _time, _status):
        self.core.feed(indata[:, 0])

    def _asr_loop(self):
        while self.running:
            try:
                if not self.core.process():
                    time.sleep(0.01)
            except Exception as e:
                self.error = f'ASR error: {e}'
                print(f'[Voice] {self.error}')
                time.sleep(0.5)

    def set_enabled(self, on: bool):
        self.core.enabled = bool(on)

    def set_monitor(self, on: bool):
        if on and 'monitor' not in self.sinks and self.running:
            self._open('monitor', self.sd.default.device[1])
        elif not on and 'monitor' in self.sinks:
            self.sinks.pop('monitor').close()

    def video_delay(self) -> float:
        """How long to hold each video frame. The audio also spends the output device's
        latency in its buffers (~10 ms on CoreAudio, 100 ms or more on Windows MME), so that
        is added to keep lips in sync."""
        if not (self.running and self.core.enabled):
            return 0.0
        sink = self.sinks.get('output')
        return self.core.delay + (sink.stream.latency if sink else 0.0)

    def status(self) -> dict:
        c = self.core
        error = self.error
        if not error and self.running and not c.heard and time.monotonic() - self.started > 3:
            where = MIC_SETTINGS.get(sys.platform, 'your system settings')
            error = f'Microphone is silent: allow microphone access for Python / your terminal in {where}'
        return {
            'running': self.running,
            'enabled': c.enabled,
            'delay': c.delay,
            'mode': c.mode,
            'input': self.mic_name,
            'output': self.sinks['output'].name if 'output' in self.sinks else None,
            'monitor': 'monitor' in self.sinks,
            'level': round(c.level, 4),
            'redacted': c.redacted,
            'leaked_ms': int(c.leaked * 1000 / c.sr),
            'error': error,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def simulate(samples: np.ndarray, sr: int, delay: float = DELAY, mode: str = MODE,
             model=None) -> Tuple[np.ndarray, VoiceRedactor]:
    """Run a recording through the redactor as if it were live: 10 ms in, 10 ms out.
    Returns the redacted audio (aligned with the input) and the redactor, for its stats."""
    core = VoiceRedactor(sr, model=model, delay=delay, mode=mode)
    block = sr // 100
    tail = np.zeros(int(sr * (delay + 1.5)), np.float32)
    data = np.concatenate([samples.astype(np.float32), tail])
    out = []
    for i in range(0, len(data) - block + 1, block):
        core.feed(data[i:i + block])
        while core.process():
            pass
        out.append(core.render(core.playhead() - block, block))
    lag = int(delay * sr)
    return np.concatenate(out)[lag:lag + len(samples)], core


def _print_transcript(core: VoiceRedactor):
    print(' '.join(f"[{e['pii'].upper()}]" if e['pii'] else e['w'] for e in core.transcript(10_000)))


def _run_file(src: str, dst: str):
    import wave
    with wave.open(src) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit('Expected 16-bit PCM WAV')
    pcm = np.frombuffer(raw, np.int16).reshape(-1, ch)[:, 0].astype(np.float32) / 32768
    out, core = simulate(pcm, sr)
    with wave.open(dst, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
    _print_transcript(core)
    print(f'\n{core.redacted} words redacted, {int(core.leaked * 1000 / sr)} ms of PII speech leaked '
          f'(delay {core.delay:.1f} s) -> {dst}')


def _run_live():
    import sounddevice as sd
    voice = LiveVoice(output=os.environ.get('BLURBERRY_AUDIO_OUT', str(sd.default.device[1])))
    voice.start()
    print('Speak; you hear yourself delayed, with PII bleeped. Use headphones. Ctrl+C to stop.\n')
    shown = 0
    try:
        while True:
            time.sleep(0.25)
            fresh = voice.core.finished - shown
            if fresh:
                shown = voice.core.finished
                words = list(voice.core.history)[-fresh:]
                print(' '.join(f'[{label.upper()}]' if label else text for text, label in words))
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        s = voice.status()
        print(f"\n{s['redacted']} words redacted, {s['leaked_ms']} ms leaked")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        _run_file(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        _run_live()
    else:
        sys.exit(__doc__)
