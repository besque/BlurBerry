import time
import threading
import queue
import numpy as np
import re
import os
from typing import List, Tuple, Dict


class TranscriptionModule:
    """Transcribes audio and detects slurs/PII with timestamps"""
    
    def __init__(self):
        # PII/Slur detection patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+91|0)?[6-9]\d{9}\b',
            'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'bank_account': r'\b\d{10,18}\b',
            'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            'upi_id': r'\b[A-Za-z0-9._-]+@[a-z]{2,}\b',
            'password': r'\b(?:my password|your password|the password|password is|pass is|secret is|key is)\s+[^.!?]*\b',
            'address': r'\b(?:my address|your address|the address|address is|i live|live at|from|located at|street|road|apartment|flat|house)\b'
        }
        
        # Load custom slurs
        slurs_path = os.path.join("blurberry", "data", "wordlists", "custom_slurs.txt")
        self.custom_slurs = []
        try:
            with open(slurs_path, 'r') as f:
                self.custom_slurs = [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            print("Warning: Custom slurs file not found")
        
        # Transcription state
        self.transcription_queue = queue.Queue()
        self.latest_transcript = ""
        self.detected_slurs = []  # List of (start_time, end_time, slur_type, text)
        self.transcription_events = queue.Queue()
        
        # Simulated transcription phrases for testing
        self.demo_phrases = [
            "Hello, this is a normal conversation",
            "My email is john.doe@example.com",
            "Please call me at 9876543210",
            "My PAN number is ABCDE1234F",
            "The password is secret123",
            "I live at 123 Main Street",
            "Just talking about regular stuff",
            "My Aadhaar number is 1234 5678 9012"
        ]
        
        # Threading
        self._running = False
        self._transcription_thread = None
    
    def start(self):
        """Start the transcription module"""
        if self._running:
            return
        
        self._running = True
        self._transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._transcription_thread.start()
        print("🎯 Transcription module started")
    
    def stop(self):
        """Stop the transcription module"""
        self._running = False
        if self._transcription_thread:
            self._transcription_thread.join(timeout=1.0)
        print("🛑 Transcription module stopped")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, chunk_start_time: float):
        """Add an audio chunk for transcription"""
        self.transcription_queue.put((audio_chunk, chunk_start_time))
    
    def detect_slurs(self, text: str) -> List[Tuple[str, str]]:
        """Detect slurs and PII in text"""
        findings = []
        text_lower = text.lower()
        
        # Regex patterns
        for slur_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append((slur_type, match.group()))
        
        # Custom slurs
        for slur in self.custom_slurs:
            if slur in text_lower:
                findings.append(('custom_slur', slur))
        
        return findings
    
    def _simulate_transcription(self, audio_chunk: np.ndarray) -> str:
        """Simulate transcription (replace with real Whisper later)"""
        # For now, return random demo phrases
        # In real implementation, this would use Whisper
        import random
        return random.choice(self.demo_phrases)
    
    def _simulate_word_timestamps(self, transcript: str, start_time: float) -> List[Dict]:
        """Simulate word-level timestamps"""
        words = transcript.split()
        word_timestamps = []
        
        # Simulate words taking ~0.3 seconds each
        for i, word in enumerate(words):
            word_start = start_time + (i * 0.3)
            word_end = word_start + 0.25
            word_timestamps.append({
                'word': word,
                'start': word_start,
                'end': word_end,
                'confidence': 0.9
            })
        
        return word_timestamps
    
    def _transcription_loop(self):
        """Main transcription loop"""
        print("🔄 Transcription loop started")
        
        while self._running:
            try:
                # Get audio chunk from queue (timeout to allow checking _running)
                audio_chunk, chunk_start_time = self.transcription_queue.get(timeout=0.1)
                
                # Skip empty chunks
                if len(audio_chunk) == 0:
                    continue
                
                # Check if chunk has speech (simple energy detection)
                energy = np.mean(audio_chunk ** 2)
                if energy < 0.00001:  # Very quiet, probably silence
                    continue
                
                print(f"🎤 Processing audio chunk (energy: {energy:.6f})")
                
                # Transcribe the audio
                transcript = self._simulate_transcription(audio_chunk)
                self.latest_transcript = transcript
                
                print(f"📝 Transcribed: {transcript}")
                
                # Get word timestamps
                word_timestamps = self._simulate_word_timestamps(transcript, chunk_start_time)
                
                # Detect slurs in transcript
                slur_findings = self.detect_slurs(transcript)
                
                if slur_findings:
                    print(f"🚨 Slurs detected: {slur_findings}")
                    
                    # Mark timestamps for detected slurs
                    for slur_type, slur_text in slur_findings:
                        # Find which word(s) contain the slur
                        for word_info in word_timestamps:
                            if slur_text.lower() in word_info['word'].lower():
                                # Convert time to sample indices for audio processing
                                start_sample = int((word_info['start'] - chunk_start_time) * 16000)
                                end_sample = int((word_info['end'] - chunk_start_time) * 16000)
                                
                                # Store slur detection info
                                slur_info = {
                                    'type': slur_type,
                                    'text': slur_text,
                                    'word': word_info['word'],
                                    'start_time': word_info['start'],
                                    'end_time': word_info['end'],
                                    'start_sample': start_sample,
                                    'end_sample': end_sample,
                                    'chunk_start_time': chunk_start_time
                                }
                                
                                self.detected_slurs.append(slur_info)
                                
                                # Emit event
                                event = {
                                    'type': 'slur_detected',
                                    'transcript': transcript,
                                    'slur_info': slur_info,
                                    'timestamp': time.time()
                                }
                                self.transcription_events.put(event)
                                
                                print(f"🔇 Marked for muting: {slur_text} at {word_info['start']:.2f}-{word_info['end']:.2f}s")
                
                # Emit transcription event
                event = {
                    'type': 'transcription',
                    'text': transcript,
                    'word_timestamps': word_timestamps,
                    'slurs_detected': len(slur_findings) > 0,
                    'timestamp': time.time()
                }
                self.transcription_events.put(event)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Transcription error: {e}")
    
    def get_latest_transcript(self) -> str:
        """Get the latest transcription"""
        return self.latest_transcript
    
    def get_detected_slurs(self) -> List[Dict]:
        """Get all detected slurs with timestamps"""
        return self.detected_slurs.copy()
    
    def get_events(self) -> List[Dict]:
        """Get all transcription events"""
        events = []
        try:
            while True:
                event = self.transcription_events.get_nowait()
                events.append(event)
        except queue.Empty:
            pass
        return events
    
    def clear_old_slurs(self, older_than_seconds: float = 10.0):
        """Clear slur detections older than specified time"""
        current_time = time.time()
        self.detected_slurs = [
            slur for slur in self.detected_slurs 
            if current_time - slur['chunk_start_time'] < older_than_seconds
        ]


# Test function
def test_transcription_module():
    """Test the transcription module"""
    print("Testing Transcription Module")
    print("=" * 40)
    
    # Create transcription module
    transcriber = TranscriptionModule()
    
    # Test slur detection
    test_texts = [
        "My email is john.doe@example.com",
        "Call me at 9876543210",
        "My password is secret123",
        "This is normal text"
    ]
    
    print("\nTesting slur detection:")
    for text in test_texts:
        slurs = transcriber.detect_slurs(text)
        print(f"'{text}' -> {slurs}")
    
    # Test with simulated audio
    print("\nTesting with simulated audio:")
    transcriber.start()
    
    # Simulate audio chunks
    for i in range(5):
        audio_chunk = np.random.randn(int(16000 * 2)) * 0.1  # 2 seconds of noise
        chunk_time = time.time()
        transcriber.add_audio_chunk(audio_chunk, chunk_time)
        time.sleep(0.5)
    
    # Get results
    time.sleep(2)  # Wait for processing
    print(f"\nLatest transcript: {transcriber.get_latest_transcript()}")
    print(f"Detected slurs: {len(transcriber.get_detected_slurs())}")
    
    events = transcriber.get_events()
    for event in events:
        print(f"Event: {event['type']} - {event}")
    
    transcriber.stop()


if __name__ == "__main__":
    test_transcription_module()
