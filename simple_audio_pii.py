import sounddevice as sd
import speech_recognition as sr
import numpy as np
import threading
import time
import re
import sys


class SimpleAudioPII:
    """Simple audio PII detection with immediate bleep"""
    
    def __init__(self):
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION = 2.0  # 2-second chunks
        self.bleep_duration = 1.0  # 1 second bleep
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # PII patterns - simplified but comprehensive
        self.SIMPLIFIED_PII_PATTERNS = {
            # 🚨 Explicit Content
            'profanity': [
                r'\b(bad|word|swear|curse|damn|hell|shit|fuck|bitch|asshole|idiot|stupid|dumb|moron|loser)\b',
                r'\b(crap|suck|loser|jerk|fool)\b'
            ],
            
            # 🔐 Sensitive Information  
            'password': [
                r'\bmy password is\s+\w+',
                r'\bthe password is\s+\w+',
                r'\bpassword\s+is\s+\w+',
                r'\bmy pass is\s+\w+',
                r'\bmy pin is\s+\w+',
                r'\bmy secret is\s+\w+'
            ],
            
            # 🏠 Location Information
            'address': [
                r'\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)\b',
                r'\b\d+\s+[A-Za-z]+,\s*[A-Za-z]{2}\s+\d{5}\b',
                r'\b(?:House|Flat|Apartment|Apt)\s+\d+\b',
                r'\b\d+\s+[A-Za-z]+\s+(?:Street|St|Ave)\b'
            ],
            
            # 📞 Contact Information
            'phone': [
                r'\b\d{10}\b',
                r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
                r'\b(?:\+91|0)?[6-9]\d{9}\b'  # Indian format
            ],
            
            # 📧 Email Information
            'email': [
                r'\b[\w.+-]+@[\w-]+\.\w{2,}\b'
            ],
            
            # 🆔 Personal Identifiers
            'personal_info': [
                r'\bmy email is\s+[\w.+-]+@[\w-]+\.\w{2,}\b',
                r'\bmy number is\s+\d{10}\b',
                r'\bmy phone is\s+\d{10}\b',
                r'\bmy address is\s+.+\b'
            ]
        }
        
        # State
        self._running = False
        
        print("🎵 Simple Audio PII Detector Initialized")
        print("🔇 Will play 1-second bleep for any PII detected")
    
    def generate_bleep(self, duration=1.0):
        """Generate 1kHz bleep tone"""
        sample_rate = self.SAMPLE_RATE
        samples = int(sample_rate * duration)
        
        # Create 1kHz sine wave
        t = np.linspace(0, duration, samples, False)
        bleep = 0.3 * np.sin(2 * np.pi * 1000 * t)
        
        return bleep.astype(np.float32)
    
    def detect_pii_simple(self, text):
        """Simple PII detection with immediate bleep trigger"""
        
        for pii_type, patterns in self.SIMPLIFIED_PII_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True, pii_type
        
        return False, None
    
    def play_bleep(self):
        """Play bleep tone through speakers"""
        try:
            bleep = self.generate_bleep()
            sd.play(bleep, self.SAMPLE_RATE)
            sd.wait()  # Wait for bleep to finish
        except Exception as e:
            print(f"❌ Bleep playback error: {e}")
    
    def process_audio_loop(self):
        """Main processing loop"""
        print("🚀 Starting simple PII detection...")
        print("🎤 Listening for speech...")
        
        with self.microphone as source:
            print("🔧 Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self._running:
                try:
                    # Listen for audio
                    print("🎤 Listening...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    
                    # Transcribe
                    print("📝 Transcribing...")
                    text = self.recognizer.recognize_google(audio)
                    text = text.lower()
                    
                    print(f"📝 Heard: '{text}'")
                    
                    # Check for PII
                    has_pii, pii_type = self.detect_pii_simple(text)
                    
                    if has_pii:
                        print(f"🚨 {pii_type.upper()} DETECTED!")
                        print("🔇 PLAYING 1-SECOND BLEEP!")
                        self.play_bleep()
                        print("✅ Bleep complete - continuing to listen...")
                    else:
                        print("✅ No PII detected - safe content")
                    
                    print("-" * 50)
                    
                except sr.WaitTimeoutError:
                    continue  # No speech detected
                except sr.UnknownValueError:
                    print("❌ Could not understand audio")
                    continue
                except sr.RequestError as e:
                    print(f"❌ Google Speech API Error: {e}")
                    time.sleep(2)  # Wait before retrying
                    continue
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    continue
    
    def start(self):
        """Start PII detection"""
        if self._running:
            return
        
        self._running = True
        print("🎵 Simple Audio PII Detection Started")
        print("🔇 Will play 1-second bleep for any PII detected")
        print("🎤 Speak clearly into your microphone")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start processing in background thread
        processing_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        processing_thread.start()
        
        return processing_thread
    
    def stop(self):
        """Stop PII detection"""
        self._running = False
        print("🛑 Simple Audio PII Detection Stopped")


def signal_handler(sig, frame):
    print('\n🛑 Stopping audio PII detection...')
    simple_pii.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("🎵 SIMPLE AUDIO PII DETECTION")
    print("=" * 60)
    print("Will transcribe audio and play 1-second bleep for:")
    print("• 🚨 Profanity/Swear words")
    print("• 🔐 Passwords and secrets")  
    print("• 🏠 Addresses and locations")
    print("• 📞 Phone numbers")
    print("• 📧 Email addresses")
    print("• 🆔 Personal information")
    print("=" * 60)
    print("🎤 Speak clearly into your microphone")
    print("🔇 Any PII detected will trigger a 1-second bleep")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    # Set up signal handler
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start
    simple_pii = SimpleAudioPII()
    
    try:
        processing_thread = simple_pii.start()
        
        # Keep main thread alive
        if processing_thread:
            processing_thread.join()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        simple_pii.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        simple_pii.stop()
