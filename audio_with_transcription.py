import threading
import time
import collections
import numpy as np
import sounddevice as sd
import signal
import sys
from transcription_module import TranscriptionModule


class AudioWithTranscription:
    """Audio pipeline with transcription and slur muting"""
    
    def __init__(self, input_device=None, output_device=None, delay_seconds=2.5):
        # Constants
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = delay_seconds
        self.BLEEP_HZ = 1000
        
        # Derived constants
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
        self.CHUNK_SECONDS = 2.0  # 2-second chunks for transcription
        self.CHUNK_SAMPLES = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        
        # Device configuration
        self.input_device = input_device
        self.output_device = output_device
        
        # Audio buffer - rolling buffer for delay
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES + self.CHUNK_SAMPLES)
        
        # Transcription module
        self.transcriber = TranscriptionModule()
        
        # Muting ranges (start_sample, end_sample) relative to audio_buffer
        self.muted_ranges = []
        
        # Threading control
        self._running = False
        self._capture_thread = None
        self._transcription_thread = None
        self._playback_thread = None
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Callback for slur detection
        self.slur_callback = None
    
    def set_slur_callback(self, callback):
        """Set callback function for when slurs are detected"""
        self.slur_callback = callback
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        
        # Start transcription module
        self.transcriber.start()
        
        # Start threads
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        
        self._capture_thread.start()
        self._transcription_thread.start()
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
        """Process audio chunks for transcription"""
        print("🔄 Transcription loop started")
        
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
                
                print(f"🚨 Slur detected: {slur_info['type']} - '{slur_info['text']}'")
                print(f"   Time: {slur_info['start_time']:.2f}-{slur_info['end_time']:.2f}s")
                print(f"   Samples: {slur_info['start_sample']}-{slur_info['end_sample']}")
                
                # Calculate where this slur will be in the audio buffer
                # The slur was detected in a chunk that started at chunk_start_time
                # We need to map this to the current buffer position
                current_time = time.time()
                time_since_chunk = current_time - slur_info['chunk_start_time']
                samples_since_chunk = int(time_since_chunk * self.SAMPLE_RATE)
                
                # Calculate position in current buffer
                buffer_position = len(self.audio_buffer) - samples_since_chunk
                
                # Add muted range (relative to current buffer end)
                start_pos = buffer_position + slur_info['start_sample']
                end_pos = buffer_position + slur_info['end_sample']
                
                # Only add if it's in the future (will be in buffer)
                if start_pos > 0 and start_pos < len(self.audio_buffer):
                    self.muted_ranges.append((start_pos, end_pos))
                    print(f"🔇 Will mute at buffer position {start_pos}-{end_pos}")
                
                # Call user callback if provided
                if self.slur_callback:
                    self.slur_callback(event)
            
            elif event['type'] == 'transcription':
                transcript = event['text']
                if event['slurs_detected']:
                    print(f"📝 Transcription (slurs detected): {transcript}")
                else:
                    print(f"📝 Transcription: {transcript}")
    
    def _playback_loop(self):
        """Play back audio with slur muting"""
        time.sleep(self.DELAY_SECONDS)  # Wait for buffer to fill
        
        print("🔊 Playback loop started - you should hear delayed audio with slur muting")
        
        while self._running:
            if len(self.audio_buffer) < self.PLAYBACK_CHUNK_SAMPLES:
                time.sleep(0.01)
                continue
            
            # Get playback chunk from buffer
            playback_chunk = np.array(list(self.audio_buffer)[:self.PLAYBACK_CHUNK_SAMPLES])
            
            # Apply muting for detected slurs
            chunk_start = 0
            chunk_end = self.PLAYBACK_CHUNK_SAMPLES
            
            for mute_start, mute_end in self.muted_ranges[:]:
                # Check if muted range overlaps with current chunk
                overlap_start = max(chunk_start, mute_start)
                overlap_end = min(chunk_end, mute_end)
                
                if overlap_start < overlap_end:
                    # Replace with bleep tone
                    print(f"🔇 Applying bleep tone at samples {overlap_start}-{overlap_end}")
                    
                    t = np.arange(overlap_start, overlap_end) / self.SAMPLE_RATE
                    bleep = 0.3 * np.sin(2 * np.pi * self.BLEEP_HZ * t)
                    playback_chunk[int(overlap_start):int(overlap_end)] = bleep
            
            # Write to output
            try:
                self.output_stream.write(playback_chunk.reshape(-1, 1))
            except Exception as e:
                print(f"Output error: {e}")
            
            # Remove processed samples from buffer
            for _ in range(self.PLAYBACK_CHUNK_SAMPLES):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
            
            # Update muted ranges (remove old ones)
            self.muted_ranges = [(s - self.PLAYBACK_CHUNK_SAMPLES, e - self.PLAYBACK_CHUNK_SAMPLES) 
                                for s, e in self.muted_ranges if s > self.PLAYBACK_CHUNK_SAMPLES]
    
    def get_latest_transcript(self):
        """Get the latest transcription"""
        return self.transcriber.get_latest_transcript()
    
    def get_detected_slurs(self):
        """Get all detected slurs"""
        return self.transcriber.get_detected_slurs()


def slur_detected_callback(event):
    """Callback function for when slurs are detected"""
    print(f"\n🚨 CALLBACK: Slur detected!")
    print(f"   Type: {event['slur_info']['type']}")
    print(f"   Text: {event['slur_info']['text']}")
    print(f"   Word: {event['slur_info']['word']}")
    print(f"   Time: {event['slur_info']['start_time']:.2f}-{event['slur_info']['end_time']:.2f}s")
    print()


def signal_handler(sig, frame):
    print('\nStopping audio with transcription...')
    audio_trans.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("Audio Pipeline with Transcription and Slur Muting")
    print("=" * 60)
    print("Records audio, transcribes it, detects slurs, and mutes them")
    print("=" * 60)
    
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
    input_device = 1  # Microphone (AB13X USB Audio)
    output_device = 4  # Microsoft Sound Mapper - Output
    
    print(f"\n🎤 Using Input Device [{input_device}]: {devices[input_device]['name']}")
    print(f"🔊 Using Output Device [{output_device}]: {devices[output_device]['name']}")
    
    print(f"\nStarting 2.5s audio delay with transcription...")
    print("Speak into your microphone - slurs will be replaced with bleep tones")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start audio with transcription
    audio_trans = AudioWithTranscription(
        input_device=input_device,
        output_device=output_device,
        delay_seconds=2.5
    )
    
    # Set callback for slur detection
    audio_trans.set_slur_callback(slur_detected_callback)
    
    try:
        audio_trans.start()
        
        # Keep running and show status
        while True:
            time.sleep(2)
            transcript = audio_trans.get_latest_transcript()
            if transcript:
                print(f"Latest: {transcript}")
            
            slurs = audio_trans.get_detected_slurs()
            if slurs:
                print(f"Total slurs detected: {len(slurs)}")
    
    except KeyboardInterrupt:
        print("\nStopping audio with transcription...")
        audio_trans.stop()
    except Exception as e:
        print(f"Error: {e}")
        audio_trans.stop()
