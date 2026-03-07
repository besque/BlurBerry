import threading
import time
import collections
import numpy as np
import sounddevice as sd
import signal
import sys
from real_transcription import RealTranscriptionModule


class TranscriptionDisplay:
    """Audio pipeline with clear transcription display"""
    
    def __init__(self, input_device=None, output_device=None, delay_seconds=2.5):
        # Constants
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = delay_seconds
        self.BLEEP_HZ = 1000
        
        # Derived constants
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)
        self.CHUNK_SECONDS = 3.0
        self.CHUNK_SAMPLES = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        
        # Device configuration
        self.input_device = input_device
        self.output_device = output_device
        
        # Audio buffer
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES + self.CHUNK_SAMPLES)
        
        # Real transcription module
        self.transcriber = RealTranscriptionModule()
        
        # Display state
        self.latest_transcript = ""
        self.transcription_history = []
        self.max_history = 10
        
        # Threading control
        self._running = False
        self._capture_thread = None
        self._transcription_thread = None
        self._display_thread = None
        self._playback_thread = None
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Slur detection
        self.detected_slurs = []
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        
        # Start transcription module
        self.transcriber.start()
        
        # Start threads
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        
        self._capture_thread.start()
        self._transcription_thread.start()
        self._display_thread.start()
        self._playback_thread.start()
        
        # Start audio streams
        self._start_audio_streams()
    
    def stop(self):
        if not self._running:
            return
        
        self._running = False
        
        # Stop transcription module
        self.transcriber.stop()
        
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
        if self._transcription_thread:
            self._transcription_thread.join(timeout=1.0)
        if self._display_thread:
            self._display_thread.join(timeout=1.0)
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
    
    def _start_audio_streams(self):
        print(f"🎤 Starting input stream with device {self.input_device}")
        print(f"🔊 Starting output stream with device {self.output_device}")
        
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
        
        try:
            self.input_stream = sd.InputStream(**input_kwargs)
            self.input_stream.start()
            print("✅ Input stream started successfully")
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
        if self.output_device is not None:
            output_kwargs['device'] = self.output_device
        
        try:
            self.output_stream = sd.OutputStream(**output_kwargs)
            self.output_stream.start()
            print("✅ Output stream started successfully")
        except Exception as e:
            print(f"❌ Output stream error: {e}")
            raise
    
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
    
    def _transcription_loop(self):
        """Process audio chunks for real transcription"""
        print("🔄 Real transcription loop started")
        
        while self._running:
            time.sleep(0.5)  # Check every 500ms
            
            if len(self.audio_buffer) < self.CHUNK_SAMPLES:
                continue
            
            # Get latest chunk for transcription
            chunk = np.array(list(self.audio_buffer)[-self.CHUNK_SAMPLES:])
            
            # Check if chunk has speech
            energy = np.mean(chunk ** 2)
            if energy < 0.00001:  # Very quiet
                continue
            
            print(f"🎤 Processing audio for transcription (energy: {energy:.6f})")
            
            # Get current time for this chunk
            chunk_time = time.time() - self.DELAY_SECONDS
            
            # Send to transcription module
            self.transcriber.add_audio_chunk(chunk, chunk_time)
            
            # Process transcription events
            self._process_transcription_events()
    
    def _process_transcription_events(self):
        """Process events from transcription module"""
        events = self.transcriber.get_events()
        
        for event in events:
            if event['type'] == 'slur_detected':
                slur_info = event['slur_info']
                
                print(f"\n🚨 SLUR DETECTED!")
                print(f"   Type: {slur_info['type']}")
                print(f"   Text: {slur_info['text']}")
                print(f"   Word: {slur_info['word']}")
                print(f"   Time: {slur_info['start_time']:.2f}-{slur_info['end_time']:.2f}s")
                print()
                
                self.detected_slurs.append(slur_info)
            
            elif event['type'] == 'transcription':
                transcript = event['text']
                self.latest_transcript = transcript
                
                # Add to history
                timestamp = time.strftime("%H:%M:%S")
                self.transcription_history.append({
                    'time': timestamp,
                    'text': transcript,
                    'slurs': event['slurs_detected']
                })
                
                # Keep only recent history
                if len(self.transcription_history) > self.max_history:
                    self.transcription_history.pop(0)
                
                # Show immediate transcription
                if event['slurs_detected']:
                    print(f"\n📝 [SLURS] {timestamp}: {transcript}")
                else:
                    print(f"\n📝 {timestamp}: {transcript}")
    
    def _display_loop(self):
        """Display transcription information continuously"""
        time.sleep(3)  # Wait for system to start
        
        while self._running:
            time.sleep(2)  # Update every 2 seconds
            
            # Clear screen and show current status
            self._show_transcription_display()
    
    def _show_transcription_display(self):
        """Show current transcription status"""
        # Clear screen (works on most terminals)
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("🎤 REAL-TIME TRANSCRIPTION DISPLAY")
        print("=" * 80)
        print(f"🔊 Audio Delay: {self.DELAY_SECONDS}s | 🎯 Status: {'Recording' if self._running else 'Stopped'}")
        print(f"📝 Latest: {self.latest_transcript if self.latest_transcript else '(Waiting for speech...)'}")
        print(f"🚨 Slurs Detected: {len(self.detected_slurs)}")
        print()
        
        print("📜 RECENT TRANSCRIPTIONS:")
        print("-" * 80)
        
        if not self.transcription_history:
            print("   (No transcriptions yet - start speaking!)")
        else:
            for entry in reversed(self.transcription_history[-5:]):  # Show last 5
                status = "🚨" if entry['slurs'] else "✅"
                print(f"   {status} [{entry['time']}] {entry['text']}")
        
        print()
        print("🎤 SPEAK INTO YOUR MICROPHONE TO SEE TRANSCRIPTION")
        print("🚨 SLURS/PII WILL BE DETECTED AND MARKED")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 80)
    
    def _playback_loop(self):
        """Play back audio (simple version without muting for now)"""
        time.sleep(self.DELAY_SECONDS)  # Wait for buffer to fill
        
        print("🔊 Playback loop started")
        
        while self._running:
            if len(self.audio_buffer) < self.PLAYBACK_CHUNK_SAMPLES:
                time.sleep(0.01)
                continue
            
            # Get playback chunk from buffer
            playback_chunk = np.array(list(self.audio_buffer)[:self.PLAYBACK_CHUNK_SAMPLES])
            
            # Write to output
            try:
                self.output_stream.write(playback_chunk.reshape(-1, 1))
            except Exception as e:
                print(f"Output error: {e}")
            
            # Remove processed samples from buffer
            for _ in range(self.PLAYBACK_CHUNK_SAMPLES):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
    
    def get_latest_transcript(self):
        """Get the latest transcription"""
        return self.latest_transcript
    
    def get_transcription_history(self):
        """Get transcription history"""
        return self.transcription_history.copy()


def signal_handler(sig, frame):
    print('\nStopping transcription display...')
    trans_display.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("🎤 Real-Time Transcription Display")
    print("=" * 50)
    print("See your speech transcribed in real-time!")
    print("Slurs and PII will be detected and highlighted")
    print("=" * 50)
    
    # Show available devices
    devices = sd.query_devices()
    print(f"\nAvailable Audio Devices:")
    print("-" * 25)
    
    for i, device in enumerate(devices):
        device_type = ""
        if device['max_input_channels'] > 0:
            device_type += "Input"
        if device['max_output_channels'] > 0:
            device_type += "Output"
        
        if device_type:
            print(f"  [{i}] {device['name']} ({device_type})")
    
    print(f"\nDefault input: {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
    
    # Use working devices
    input_device = 1  # Internal Microphone (AMD Audio)
    output_device = 4  # Microsoft Sound Mapper - Output
    
    print(f"\n🎤 Using Input Device [{input_device}]: {devices[input_device]['name']}")
    print(f"🔊 Using Output Device [{output_device}]: {devices[output_device]['name']}")
    
    print(f"\n🚀 Starting transcription display...")
    print("Speak into your microphone and watch the transcription appear!")
    print("The display will update every 2 seconds")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start transcription display
    trans_display = TranscriptionDisplay(
        input_device=input_device,
        output_device=output_device,
        delay_seconds=2.5
    )
    
    try:
        trans_display.start()
        
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping transcription display...")
        trans_display.stop()
    except Exception as e:
        print(f"Error: {e}")
        trans_display.stop()
