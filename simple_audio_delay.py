import threading
import time
import collections
import numpy as np
import sounddevice as sd
import signal
import sys


class SimpleAudioDelay:
    """Simple audio pipeline that records and plays back with delay"""
    
    def __init__(self, input_device=None, output_device=None, delay_seconds=2.5):
        # Constants
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = delay_seconds
        
        # Derived constants
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
        
        # Device configuration
        self.input_device = input_device
        self.output_device = output_device
        
        # Audio buffer - rolling buffer for delay
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES)
        
        # Threading control
        self._running = False
        self._capture_thread = None
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
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        
        self._capture_thread.start()
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
    
    def _playback_loop(self):
        time.sleep(self.DELAY_SECONDS)  # Wait for buffer to fill
        
        print("🔊 Playback loop started - you should hear delayed audio now")
        
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


def signal_handler(sig, frame):
    print('\nStopping audio delay...')
    audio_delay.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("Simple Audio Delay Test")
    print("=" * 40)
    print("Records audio and plays it back after delay")
    print("=" * 40)
    
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
    
    # Use working devices from our tests
    input_device = 1  # Internal Microphone (AMD Audio)
    output_device = 4  # Speakers (Realtek(R) Audio)
    
    print(f"\n🎤 Using Input Device [{input_device}]: {devices[input_device]['name']}")
    print(f"🔊 Using Output Device [{output_device}]: {devices[output_device]['name']}")
    
    print(f"\nStarting 2.5s audio delay...")
    print("Speak into your microphone - you'll hear it back after the delay")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start audio delay
    audio_delay = SimpleAudioDelay(
        input_device=input_device,
        output_device=output_device,
        delay_seconds=2.5
    )
    
    try:
        audio_delay.start()
        
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping audio delay...")
        audio_delay.stop()
    except Exception as e:
        print(f"Error: {e}")
        audio_delay.stop()
