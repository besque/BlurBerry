import threading
import time
import queue
import collections
import numpy as np
import sounddevice as sd
import re
import os


class SimplePIIDetector:
    """Simplified PII detector without heavy dependencies"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+91|0)?[6-9]\d{9}\b',
            'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        }
        
        # Load custom slurs
        slurs_path = os.path.join("blurberry", "data", "wordlists", "custom_slurs.txt")
        self.custom_slurs = []
        try:
            with open(slurs_path, 'r') as f:
                self.custom_slurs = [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            pass
    
    def detect_pii(self, text):
        findings = []
        text_lower = text.lower()
        
        # Regex patterns
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append((pii_type, match.group()))
        
        # Custom slurs
        for slur in self.custom_slurs:
            if slur in text_lower:
                findings.append(('custom_slur', slur))
        
        return findings
    
    def contains_pii(self, text):
        return len(self.detect_pii(text)) > 0


class SimpleAudioPipeline:
    """Simplified audio pipeline for testing without PyTorch"""
    
    def __init__(self, on_pii_detected=None):
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = 2.5
        self.CHUNK_SECONDS = 2.0
        self.BLEEP_HZ = 1000
        
        self.CHUNK_SAMPLES = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)
        
        self.pii_detector = SimplePIIDetector()
        self.on_pii_detected = on_pii_detected
        
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES + self.CHUNK_SAMPLES)
        self._muted_ranges = []
        self._latest_transcript = ""
        self.event_queue = queue.Queue()
        
        self._running = False
        self._capture_thread = None
        self._playback_thread = None
        self._detection_thread = None
        
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
        
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._detection_thread:
            self._detection_thread.join(timeout=1.0)
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
    
    def _start_audio_streams(self):
        # Input stream
        self.input_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=1024,
            callback=self._input_callback
        )
        self.input_stream.start()
        
        # Output stream
        self.output_stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=self.PLAYBACK_CHUNK_SAMPLES
        )
        self.output_stream.start()
    
    def _input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Input callback status: {status}")
        
        audio_chunk = indata.flatten()
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
    
    def _capture_loop(self):
        pass  # Handled by callback
    
    def _detection_loop(self):
        while self._running:
            time.sleep(0.5)
            
            if len(self.audio_buffer) < self.CHUNK_SAMPLES:
                continue
            
            # Simulate transcription (in real pipeline, this would be Whisper)
            chunk = np.array(list(self.audio_buffer)[-self.CHUNK_SAMPLES:])
            
            # Simple speech detection (energy threshold)
            energy = np.mean(chunk ** 2)
            if energy < 0.001:  # Silent or very quiet
                continue
            
            # Simulate transcript with placeholder text
            # In real pipeline, this would come from Whisper
            simulated_transcript = "User is speaking about contact information"
            
            # Add some variation to make it interesting
            import random
            if random.random() < 0.3:
                simulated_transcript = "My email is john.doe@example.com"
            elif random.random() < 0.3:
                simulated_transcript = "Call me at 9876543210"
            elif random.random() < 0.3:
                simulated_transcript = "My PAN is ABCDE1234F"
            
            self._latest_transcript = simulated_transcript
            
            # Detect PII
            if self.pii_detector.contains_pii(simulated_transcript):
                findings = self.pii_detector.detect_pii(simulated_transcript)
                
                # Mark random segment for muting (simulation)
                start_sample = len(self.audio_buffer) - self.CHUNK_SAMPLES + random.randint(0, self.CHUNK_SAMPLES - 1000)
                end_sample = start_sample + 1000
                self._muted_ranges.append((start_sample, end_sample))
                
                # Emit event
                event = {
                    "type": "audio_pii",
                    "text": simulated_transcript,
                    "findings": findings,
                    "timestamp": time.time()
                }
                self.event_queue.put(event)
                
                if self.on_pii_detected:
                    self.on_pii_detected(event)
    
    def _playback_loop(self):
        time.sleep(self.DELAY_SECONDS)
        
        while self._running:
            if len(self.audio_buffer) < self.PLAYBACK_CHUNK_SAMPLES:
                time.sleep(0.01)
                continue
            
            playback_chunk = np.array(list(self.audio_buffer)[:self.PLAYBACK_CHUNK_SAMPLES])
            
            # Apply muting
            for start_sample, end_sample in self._muted_ranges[:]:
                if start_sample >= len(playback_chunk):
                    break
                
                chunk_start = max(0, start_sample)
                chunk_end = min(len(playback_chunk), end_sample)
                
                if chunk_start < chunk_end:
                    t = np.arange(chunk_start, chunk_end) / self.SAMPLE_RATE
                    bleep = 0.3 * np.sin(2 * np.pi * self.BLEEP_HZ * t)
                    playback_chunk[chunk_start:chunk_end] = bleep
            
            # Write to output
            try:
                self.output_stream.write(playback_chunk.reshape(-1, 1))
            except Exception as e:
                print(f"Output error: {e}")
            
            # Remove processed samples
            for _ in range(self.PLAYBACK_CHUNK_SAMPLES):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
            
            # Clean up old muted ranges
            self._muted_ranges = [(s, e) for s, e in self._muted_ranges 
                                 if s > len(self.audio_buffer)]
    
    def get_latest_transcript(self):
        return self._latest_transcript
    
    def get_events(self):
        events = []
        try:
            while True:
                event = self.event_queue.get_nowait()
                events.append(event)
        except queue.Empty:
            pass
        return events


def pii_callback(event):
    print(f"\n🚨 PII DETECTED!")
    print(f"Text: {event['text']}")
    print(f"Findings: {event['findings']}")
    print(f"Timestamp: {event['timestamp']}")
    print()


def signal_handler(sig, frame):
    print('\nStopping audio pipeline...')
    pipeline.stop()
    exit(0)


if __name__ == "__main__":
    import signal
    import sys
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("BlurBerry Simple Audio Pipeline Test")
    print("=" * 50)
    print("This is a simplified version for testing without PyTorch")
    print("It simulates transcription and demonstrates PII detection")
    print("=" * 50)
    
    # Print available devices
    devices = sd.query_devices()
    print(f"\nFound {len(devices)} audio devices")
    print(f"Default input: {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
    
    print("\nStarting simple audio pipeline...")
    print("Speak into your microphone - the pipeline will simulate transcription")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    pipeline = SimpleAudioPipeline(on_pii_detected=pii_callback)
    
    try:
        pipeline.start()
        
        while True:
            time.sleep(1)
            transcript = pipeline.get_latest_transcript()
            if transcript:
                print(f"Transcript: {transcript}")
            
            events = pipeline.get_events()
            for event in events:
                if event['type'] != 'audio_pii':
                    print(f"Event: {event}")
    
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        pipeline.stop()
    except Exception as e:
        print(f"Error: {e}")
        pipeline.stop()
