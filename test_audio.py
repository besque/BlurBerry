import sounddevice as sd
import time
import signal
import sys
from blurberry.audio.audio_pipeline import AudioPipeline


def pii_callback(event):
    print(f"\n🚨 PII DETECTED!")
    print(f"Text: {event['text']}")
    print(f"Findings: {event['findings']}")
    print(f"Timestamp: {event['timestamp']}")
    print()


def signal_handler(sig, frame):
    print('\nStopping audio pipeline...')
    pipeline.stop()
    sys.exit(0)


if __name__ == "__main__":
    print("BlurBerry Audio Pipeline Test")
    print("=" * 40)
    
    # Print available audio devices
    print("\nAvailable Audio Devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = "Input" if device['max_input_channels'] > 0 else ""
        device_type += "Output" if device['max_output_channels'] > 0 else ""
        if device_type:
            print(f"  [{i}] {device['name']} ({device_type})")
    
    print("\nStarting audio pipeline...")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start pipeline
    pipeline = AudioPipeline(
        whisper_model_size="tiny",
        on_pii_detected=pii_callback
    )
    
    try:
        pipeline.start()
        
        # Main loop - print transcript every second
        while True:
            time.sleep(1)
            transcript = pipeline.get_latest_transcript()
            if transcript:
                print(f"Transcript: {transcript}")
            
            # Print any events
            events = pipeline.get_events()
            for event in events:
                if event['type'] != 'audio_pii':  # PII events handled by callback
                    print(f"Event: {event}")
    
    except KeyboardInterrupt:
        print("\nStopping audio pipeline...")
        pipeline.stop()
    except Exception as e:
        print(f"Error: {e}")
        pipeline.stop()
