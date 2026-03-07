import threading
import time
import queue
import collections
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy import signal
from .vad import VAD
from .pii_detector import PIIDetector


class AudioPipeline:
    def __init__(self, input_device=None, output_device=None, whisper_model_size="tiny", on_pii_detected=None):
        # Constants
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = 2.5
        self.CHUNK_SECONDS = 2.0
        self.BLEEP_HZ = 1000
        
        # Calculate derived constants
        self.CHUNK_SAMPLES = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
        
        # Device configuration
        self.input_device = input_device
        self.output_device = output_device
        
        # Initialize components
        self.vad = VAD()
        self.pii_detector = PIIDetector()
        self.whisper_model = WhisperModel(
            whisper_model_size,
            device="cpu",
            compute_type="int8"
        )
        
        # Callback for PII detection
        self.on_pii_detected = on_pii_detected
        
        # Audio buffers and state
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES + self.CHUNK_SAMPLES)
        self._muted_ranges = []  # List of (start_sample, end_sample) tuples
        self._latest_transcript = ""
        self.event_queue = queue.Queue()
        
        # Threading control
        self._running = False
        self._capture_thread = None
        self._detection_thread = None
        self._playback_thread = None
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        
        # Start threads
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        
        self._capture_thread.start()
        self._detection_thread.start()
        self._playback_thread.start()
        
        # Start audio streams
        self._start_audio_streams()
    
    def stop(self):
        if not self._running:
            return
        
        self._running = False
        
        # Stop audio streams
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        
        # Wait for threads to finish
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._detection_thread:
            self._detection_thread.join(timeout=1.0)
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
    
    def _start_audio_streams(self):
        # Input stream configuration
        input_kwargs = {
            'samplerate': self.SAMPLE_RATE,
            'channels': 1,
            'dtype': 'float32',
            'blocksize': 1024,
            'callback': self._input_callback
        }
        if self.input_device is not None:
            input_kwargs['device'] = self.input_device
        
        self.input_stream = sd.InputStream(**input_kwargs)
        self.input_stream.start()
        
        # Output stream configuration
        output_kwargs = {
            'samplerate': self.SAMPLE_RATE,
            'channels': 1,
            'dtype': 'float32',
            'blocksize': self.PLAYBACK_CHUNK_SAMPLES
        }
        if self.output_device is not None:
            output_kwargs['device'] = self.output_device
        
        self.output_stream = sd.OutputStream(**output_kwargs)
        self.output_stream.start()
    
    def _input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Input callback status: {status}")
        
        # Add audio data to buffer
        audio_chunk = indata.flatten()
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
    
    def _capture_loop(self):
        # This loop is handled by the input callback
        pass
    
    def _detection_loop(self):
        while self._running:
            time.sleep(0.5)  # Every 500ms
            
            if len(self.audio_buffer) < self.CHUNK_SAMPLES:
                continue
            
            # Get latest chunk
            chunk = np.array(list(self.audio_buffer)[-self.CHUNK_SAMPLES:])
            
            # VAD check
            if not self.vad.is_speech(chunk):
                continue
            
            # Transcribe with Whisper
            try:
                segments, info = self.whisper_model.transcribe(chunk, language="en", word_timestamps=True)
                transcript = " ".join([segment.text for segment in segments])
                self._latest_transcript = transcript
                
                # Detect PII
                if self.pii_detector.contains_pii(transcript):
                    findings = self.pii_detector.detect_pii(transcript)
                    
                    # Calculate muted ranges based on word timestamps
                    for segment in segments:
                        for word in segment.words:
                            word_text = word.word.lower().strip()
                            # Check if this word contains PII
                            if self.pii_detector.contains_pii(word_text):
                                start_sample = int(word.start * self.SAMPLE_RATE)
                                end_sample = int(word.end * self.SAMPLE_RATE)
                                self._muted_ranges.append((start_sample, end_sample))
                    
                    # Emit event
                    event = {
                        "type": "audio_pii",
                        "text": transcript,
                        "findings": findings,
                        "timestamp": time.time()
                    }
                    self.event_queue.put(event)
                    
                    # Call callback if provided
                    if self.on_pii_detected:
                        self.on_pii_detected(event)
                
            except Exception as e:
                print(f"Detection error: {e}")
    
    def _playback_loop(self):
        time.sleep(self.DELAY_SECONDS)  # Wait 2.5s on startup
        
        while self._running:
            if len(self.audio_buffer) < self.PLAYBACK_CHUNK_SAMPLES:
                time.sleep(0.01)
                continue
            
            # Get playback chunk
            playback_chunk = np.array(list(self.audio_buffer)[:self.PLAYBACK_CHUNK_SAMPLES])
            
            # Apply muting for PII ranges
            current_sample = 0
            for start_sample, end_sample in self._muted_ranges[:]:
                if start_sample >= len(playback_chunk):
                    break
                
                # Calculate overlap with current chunk
                chunk_start = max(0, start_sample)
                chunk_end = min(len(playback_chunk), end_sample)
                
                if chunk_start < chunk_end:
                    # Replace with bleep tone
                    t = np.arange(chunk_start, chunk_end) / self.SAMPLE_RATE
                    bleep = 0.3 * np.sin(2 * np.pi * self.BLEEP_HZ * t)
                    playback_chunk[chunk_start:chunk_end] = bleep
            
            # Write to output
            try:
                self.output_stream.write(playback_chunk.reshape(-1, 1))
            except Exception as e:
                print(f"Output error: {e}")
            
            # Remove processed samples from buffer
            for _ in range(self.PLAYBACK_CHUNK_SAMPLES):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
            
            # Clean up old muted ranges
            current_time = time.time()
            self._muted_ranges = [(s, e) for s, e in self._muted_ranges 
                                 if s > len(self.audio_buffer)]
    
    def get_latest_transcript(self) -> str:
        return self._latest_transcript
    
    def get_events(self) -> list:
        events = []
        try:
            while True:
                event = self.event_queue.get_nowait()
                events.append(event)
        except queue.Empty:
            pass
        return events