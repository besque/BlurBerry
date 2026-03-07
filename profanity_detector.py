import sounddevice as sd
import speech_recognition as sr
import numpy as np
import threading
import time
import collections
import re
import sys


class ProfanityDetector:
    """Real-time profanity detection with delayed audio playback and bleep"""
    
    def __init__(self, delay_seconds=2.5):
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = delay_seconds
        self.BLEEP_DURATION = 1.0  # 1 second bleep
        self.BLEEP_HZ = 1000  # 1kHz tone
        
        # Audio buffer for delay
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES + self.PLAYBACK_CHUNK_SAMPLES)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Profanity word list - comprehensive
        self.PROFANITY_WORDS = [
            # Common swear words
            'fuck', 'fucking', 'fucked', 'fucker', 'motherfucker',
            'shit', 'shitting', 'shitted', 'shitter', 'bullshit',
            'damn', 'damned', 'goddamn', 'goddamned',
            'hell', 'hellish', 'hells',
            'bitch', 'bitching', 'bitched', 'son of a bitch',
            'ass', 'asshole', 'assholes', 'dumbass', 'smartass',
            'crap', 'crappy', 'crapped',
            'piss', 'pissed', 'pissing', 'pissed off',
            'bastard', 'bastards',
            'dick', 'dickhead', 'dicks',
            'cock', 'cocksucker', 'cocksucking',
            'pussy', 'pussies',
            'cunt', 'cunting', 'cunts',
            'whore', 'whores',
            'slut', 'sluts', 'slutty',
            'twat', 'twats',
            # Mild profanity
            'idiot', 'idiots', 'stupid', 'stupidity', 'dumb', 'dumber', 'dumbest',
            'moron', 'morons', 'loser', 'losers', 'jerk', 'jerks',
            'fool', 'fools', 'foolish', 'suck', 'sucking', 'sucks',
            'crap', 'crappy'
        ]
        
        # Create regex pattern for profanity detection
        self.profanity_pattern = re.compile(
            r'\b(' + '|'.join(self.PROFANITY_WORDS) + r')\b',
            re.IGNORECASE
        )
        
        # Threading control
        self._running = False
        self._capture_thread = None
        self._detection_thread = None
        self._playback_thread = None
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Profanity detection state
        self.profanity_detected = False
        self.bleep_active = False
        self.detected_words = []
        
        print("🚨 Profanity Detector Initialized")
        print(f"⏱️  Audio Delay: {self.DELAY_SECONDS}s")
        print(f"🔇 Bleep Duration: {self.BLEEP_DURATION}s")
        print(f"📝 Profanity Words: {len(self.PROFANITY_WORDS)} detected")
    
    def generate_bleep(self, duration=1.0):
        """Generate 1kHz bleep tone"""
        sample_rate = self.SAMPLE_RATE
        samples = int(sample_rate * duration)
        
        # Create 1kHz sine wave
        t = np.linspace(0, duration, samples, False)
        bleep = 0.3 * np.sin(2 * np.pi * self.BLEEP_HZ * t)
        
        return bleep.astype(np.float32)
    
    def detect_profanity(self, text):
        """Detect profanity in transcribed text"""
        matches = self.profanity_pattern.findall(text)
        if matches:
            return True, matches
        return False, []
    
    def _input_callback(self, indata, frames, time_info, status):
        """Audio input callback - captures audio continuously"""
        if status:
            print(f"Input status: {status}")
        
        # Add audio data to buffer
        audio_chunk = indata.flatten()
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
    
    def _capture_loop(self):
        """Audio capture loop - handled by callback"""
        pass
    
    def _detection_loop(self):
        """Profanity detection loop"""
        print("🔍 Profanity detection loop started")
        
        while self._running:
            time.sleep(1.0)  # Check every 1 second
            
            # Get recent audio for transcription
            if len(self.audio_buffer) < self.SAMPLE_RATE * 2:  # Need 2 seconds
                continue
            
            # Get audio chunk for transcription
            chunk = np.array(list(self.audio_buffer)[-int(self.SAMPLE_RATE * 2):])
            
            # Check if chunk has speech
            energy = np.mean(chunk ** 2)
            if energy < 0.00001:  # Too quiet
                continue
            
            print(f"🎤 Processing audio (energy: {energy:.6f})")
            
            # Transcribe the chunk
            try:
                # Convert to int16 for speech_recognition
                audio_data = (chunk * 32767).astype(np.int16)
                audio_data = sr.AudioData(audio_data.tobytes(), self.SAMPLE_RATE, 2)
                
                # Transcribe with Google Speech
                text = self.recognizer.recognize_google(audio_data)
                text = text.lower()
                
                print(f"📝 Transcribed: '{text}'")
                
                # Detect profanity
                has_profanity, profanity_words = self.detect_profanity(text)
                
                if has_profanity:
                    print(f"🚨 PROFANITY DETECTED: {profanity_words}")
                    self.profanity_detected = True
                    self.detected_words.extend(profanity_words)
                    
                    # Calculate when to bleep (in the future during playback)
                    current_time = time.time()
                    bleep_time = current_time + self.DELAY_SECONDS
                    self.bleep_time = bleep_time
                    
                    print(f"🔇 Will bleep at: {time.strftime('%H:%M:%S', time.localtime(bleep_time))}")
                else:
                    print("✅ No profanity detected")
                
            except sr.UnknownValueError:
                print("❌ Could not understand audio")
            except sr.RequestError as e:
                print(f"❌ Google Speech API Error: {e}")
            except Exception as e:
                print(f"❌ Detection error: {e}")
    
    def _playback_loop(self):
        """Audio playback loop with profanity muting"""
        time.sleep(self.DELAY_SECONDS)  # Wait for buffer to fill
        
        print("🔊 Playback loop started")
        print(f"⏱️  Audio delayed by {self.DELAY_SECONDS}s")
        
        while self._running:
            if len(self.audio_buffer) < self.PLAYBACK_CHUNK_SAMPLES:
                time.sleep(0.01)
                continue
            
            # Get playback chunk from buffer
            playback_chunk = np.array(list(self.audio_buffer)[:self.PLAYBACK_CHUNK_SAMPLES])
            
            # Check if we need to bleep right now
            current_time = time.time()
            if hasattr(self, 'bleep_time') and current_time >= self.bleep_time - 0.5:
                if current_time <= self.bleep_time + self.BLEEP_DURATION:
                    # We're in the bleep window
                    print("🔇 APPLYING BLEEP TONE!")
                    bleep = self.generate_bleep()
                    
                    # Replace entire chunk with bleep (for 1 second)
                    playback_chunk = bleep[:len(playback_chunk)]
                    self.bleep_active = True
                else:
                    # Bleep period is over
                    self.bleep_active = False
                    if hasattr(self, 'bleep_time'):
                        delattr(self, 'bleep_time')
            
            # Write to output
            try:
                self.output_stream.write(playback_chunk.reshape(-1, 1))
            except Exception as e:
                print(f"Output error: {e}")
            
            # Remove processed samples from buffer
            for _ in range(self.PLAYBACK_CHUNK_SAMPLES):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
    
    def _start_audio_streams(self):
        """Start audio input and output streams"""
        print(f"🎤 Starting input stream")
        print(f"🔊 Starting output stream")
        
        # Input stream configuration
        input_kwargs = {
            'samplerate': self.SAMPLE_RATE,
            'channels': 1,
            'dtype': 'float32',
            'blocksize': 1024,
            'callback': self._input_callback
        }
        
        try:
            self.input_stream = sd.InputStream(**input_kwargs)
            self.input_stream.start()
            print("✅ Input stream started")
        except Exception as e:
            print(f"❌ Input stream error: {e}")
            raise
        
        # Output stream configuration
        output_kwargs = {
            'samplerate': self.SAMPLE_RATE,
            'channels': 1,
            'dtype': 'float32',
            'blocksize': self.PLAYBACK_CHUNK_SAMPLES
        }
        
        try:
            self.output_stream = sd.OutputStream(**output_kwargs)
            self.output_stream.start()
            print("✅ Output stream started")
        except Exception as e:
            print(f"❌ Output stream error: {e}")
            raise
    
    def start(self):
        """Start profanity detection with delayed playback"""
        if self._running:
            return
        
        self._running = True
        
        print("🚨 Starting Profanity Detection System")
        print("🎤 Will detect profanity and bleep after delay")
        print(f"⏱️  Audio delay: {self.DELAY_SECONDS}s")
        print(f"🔇 Bleep duration: {self.BLEEP_DURATION}s")
        print("🎤 Speak into your microphone")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start audio streams
        self._start_audio_streams()
        
        # Start all threads
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        
        self._capture_thread.start()
        self._detection_thread.start()
        self._playback_thread.start()
        
        print("✅ All threads started")
    
    def stop(self):
        """Stop profanity detection"""
        if not self._running:
            return
        
        self._running = False
        print("🛑 Stopping Profanity Detection System")
        
        # Stop audio streams
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        
        # Wait for threads to finish
        for thread in [self._capture_thread, self._detection_thread, self._playback_thread]:
            if thread:
                thread.join(timeout=1.0)
        
        print(f"📊 Total profanity detected: {len(self.detected_words)}")
        print("✅ System stopped")
    
    def get_detected_words(self):
        """Get list of detected profanity words"""
        return self.detected_words.copy()


def signal_handler(sig, frame):
    print('\n🛑 Stopping profanity detection...')
    profanity_detector.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("🚨 PROFANITY DETECTION WITH REAL-TIME BLEEP")
    print("=" * 60)
    print("Features:")
    print("• 🎤 Real-time audio capture")
    print("• 🔍 Profanity detection (50+ words)")
    print("• ⏱️  Delayed audio playback (2.5s)")
    print("• 🔇 1-second bleep tone for profanity")
    print("• 📊 Detection statistics")
    print("=" * 60)
    print("🎤 Speak into your microphone")
    print("🚨 Any profanity will be bleeped after delay")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    # Set up signal handler
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start
    profanity_detector = ProfanityDetector(delay_seconds=2.5)
    
    try:
        profanity_detector.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Show status every 10 seconds
            if int(time.time()) % 10 == 0:
                detected = profanity_detector.get_detected_words()
                if detected:
                    print(f"📊 Total profanity detected: {len(detected)} - {detected[-5:]}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        profanity_detector.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        profanity_detector.stop()
