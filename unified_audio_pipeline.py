import threading
import time
import collections
import numpy as np
import sounddevice as sd
import signal
import sys
import os
from real_transcription import RealTranscriptionModule


class UnifiedAudioPipeline:
    """Complete audio pipeline with transcription and playback running simultaneously"""
    
    def __init__(self, input_device=None, output_device=None, delay_seconds=2.5):
        # Constants
        self.SAMPLE_RATE = 16000
        self.DELAY_SECONDS = delay_seconds
        self.BLEEP_HZ = 1000
        
        # Derived constants
        self.DELAY_SAMPLES = int(self.SAMPLE_RATE * self.DELAY_SECONDS)
        self.PLAYBACK_CHUNK_SAMPLES = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
        self.CHUNK_SECONDS = 3.0  # 3-second chunks for transcription
        self.CHUNK_SAMPLES = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        
        # Device configuration
        self.input_device = input_device
        self.output_device = output_device
        
        # Audio buffer - rolling buffer for delay
        self.audio_buffer = collections.deque(maxlen=self.DELAY_SAMPLES + self.CHUNK_SAMPLES)
        
        # Real transcription module
        self.transcriber = RealTranscriptionModule()
        
        # Muting ranges (start_sample, end_sample) relative to audio_buffer
        self.muted_ranges = []
        
        # Display state
        self.latest_transcript = ""
        self.transcription_history = []
        self.max_history = 10
        self.detected_slurs = []
        
        # Threading control
        self._running = False
        self._capture_thread = None
        self._transcription_thread = None
        self._playback_thread = None
        self._display_thread = None
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Status tracking
        self.start_time = time.time()
        self.audio_chunks_processed = 0
        self.transcriptions_made = 0
        self.slurs_detected_count = 0
    
    def start(self):
        """Start all components simultaneously"""
        if self._running:
            return
        
        self._running = True
        self.start_time = time.time()
        
        print("🚀 Starting all pipeline components simultaneously...")
        
        # Start transcription module first
        self.transcriber.start()
        
        # Start all threads at once
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        
        # Start all threads
        self._capture_thread.start()
        self._transcription_thread.start()
        self._playback_thread.start()
        self._display_thread.start()
        
        # Start audio streams
        self._start_audio_streams()
        
        print("✅ All components started successfully!")
    
    def stop(self):
        """Stop all components"""
        if not self._running:
            return
        
        self._running = False
        
        print("🛑 Stopping all pipeline components...")
        
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
        if self._display_thread:
            self._display_thread.join(timeout=1.0)
        
        print("✅ All components stopped")
    
    def _start_audio_streams(self):
        """Start audio input and output streams"""
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
        """Audio input callback - runs continuously"""
        if status:
            print(f"Input callback status: {status}")
        
        # Add audio data to buffer
        audio_chunk = indata.flatten()
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
        
        self.audio_chunks_processed += 1
    
    def _capture_loop(self):
        """Audio capture loop - handled by callback"""
        pass
    
    def _transcription_loop(self):
        """Process audio chunks for real transcription"""
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
            
            print(f"🎤 Processing audio chunk #{self.audio_chunks_processed} (energy: {energy:.6f})")
            
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
                self.slurs_detected_count += 1
                
                # Calculate where this slur will be in the audio buffer
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
            
            elif event['type'] == 'transcription':
                transcript = event['text']
                self.latest_transcript = transcript
                self.transcriptions_made += 1
                
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
                    print(f"📝 [SLURS] {timestamp}: {transcript}")
                else:
                    print(f"📝 {timestamp}: {transcript}")
    
    def _playback_loop(self):
        """Play back audio with slur muting"""
        time.sleep(self.DELAY_SECONDS)  # Wait for buffer to fill
        
        print("🔊 Playback loop started - you should hear your voice with delay")
        
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
    
    def _display_loop(self):
        """Display real-time status and transcription"""
        time.sleep(3)  # Wait for system to start
        
        while self._running:
            time.sleep(2)  # Update every 2 seconds
            self._show_status_display()
    
    def _show_status_display(self):
        """Show current pipeline status"""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        runtime = time.time() - self.start_time
        
        print("=" * 80)
        print("🎤 UNIFIED AUDIO PIPELINE - TRANSCRIPTION + PLAYBACK")
        print("=" * 80)
        print(f"⏱️  Runtime: {runtime:.1f}s | 🎯 Status: {'🔴 Recording' if self._running else '⏹️ Stopped'}")
        print(f"🎤 Audio Chunks: {self.audio_chunks_processed} | 📝 Transcriptions: {self.transcriptions_made}")
        print(f"🚨 Slurs Detected: {self.slurs_detected_count} | 🔇 Muted Ranges: {len(self.muted_ranges)}")
        print()
        
        print(f"📝 LATEST TRANSCRIPT: {self.latest_transcript if self.latest_transcript else '(Waiting for speech...)'}")
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
        print("🎤 SPEAK INTO MICROPHONE:")
        print("   • Your speech appears as transcription above")
        print("   • You'll hear your voice back after 2.5s delay")
        print("   • Slurs/PII will be detected and bleeped")
        print("   • All processing happens in real-time")
        print()
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 80)
    
    def get_latest_transcript(self):
        """Get the latest transcription"""
        return self.latest_transcript
    
    def get_transcription_history(self):
        """Get transcription history"""
        return self.transcription_history.copy()
    
    def get_stats(self):
        """Get pipeline statistics"""
        return {
            'runtime': time.time() - self.start_time,
            'audio_chunks': self.audio_chunks_processed,
            'transcriptions': self.transcriptions_made,
            'slurs_detected': self.slurs_detected_count,
            'muted_ranges': len(self.muted_ranges)
        }


def signal_handler(sig, frame):
    print('\n🛑 Stopping unified audio pipeline...')
    unified.stop()
    print(f"\n📊 Final Stats: {unified.get_stats()}")
    sys.exit(0)


if __name__ == "__main__":
    print("🎤 Unified Audio Pipeline - Transcription + Playback")
    print("=" * 60)
    print("Real-time transcription AND audio playback with slur muting")
    print("Both start simultaneously and run together")
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
    input_device = 1  # Internal Microphone (AMD Audio)
    output_device = 4  # Microsoft Sound Mapper - Output
    
    print(f"\n🎤 Using Input Device [{input_device}]: {devices[input_device]['name']}")
    print(f"🔊 Using Output Device [{output_device}]: {devices[output_device]['name']}")
    
    print(f"\n🚀 Starting unified pipeline...")
    print("✅ Transcription will start immediately")
    print("✅ Audio playback will start after 2.5s delay")
    print("✅ Both run simultaneously")
    print("✅ Slurs will be detected and bleeped in real-time")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start unified pipeline
    unified = UnifiedAudioPipeline(
        input_device=input_device,
        output_device=output_device,
        delay_seconds=2.5
    )
    
    try:
        unified.start()
        
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping unified audio pipeline...")
        unified.stop()
        print(f"\n📊 Final Stats: {unified.get_stats()}")
    except Exception as e:
        print(f"Error: {e}")
        unified.stop()
