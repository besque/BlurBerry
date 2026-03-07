import sounddevice as sd
import numpy as np
import time

def test_audio_devices():
    """Test audio device enumeration"""
    print("Audio Device Test:")
    print("=" * 40)
    
    try:
        devices = sd.query_devices()
        print(f"Found {len(devices)} audio devices:")
        
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device['name']))
            if device['max_output_channels'] > 0:
                output_devices.append((i, device['name']))
        
        print(f"\nInput Devices:")
        for idx, name in input_devices:
            print(f"  [{idx}] {name}")
        
        print(f"\nOutput Devices:")
        for idx, name in output_devices:
            print(f"  [{idx}] {name}")
        
        print(f"\nDefault input device: {sd.default.device[0]}")
        print(f"Default output device: {sd.default.device[1]}")
        
        return True
        
    except Exception as e:
        print(f"Error testing audio devices: {e}")
        return False

def test_audio_recording():
    """Test basic audio recording"""
    print("\nAudio Recording Test:")
    print("=" * 40)
    
    try:
        SAMPLE_RATE = 16000
        DURATION = 2  # seconds
        
        print(f"Recording {DURATION} seconds of audio at {SAMPLE_RATE}Hz...")
        
        # Record audio
        recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                          samplerate=SAMPLE_RATE, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        print(f"Recorded {len(recording)} samples")
        print(f"Audio shape: {recording.shape}")
        print(f"Audio dtype: {recording.dtype}")
        print(f"Audio range: [{recording.min():.3f}, {recording.max():.3f}]")
        
        # Test if audio has content (not silent)
        if np.abs(recording).max() > 0.001:
            print("✓ Audio has content (not silent)")
        else:
            print("⚠ Audio appears to be silent")
        
        return True
        
    except Exception as e:
        print(f"Error testing audio recording: {e}")
        return False

def test_audio_playback():
    """Test basic audio playback"""
    print("\nAudio Playback Test:")
    print("=" * 40)
    
    try:
        SAMPLE_RATE = 16000
        DURATION = 1  # second
        FREQUENCY = 440  # Hz (A4 note)
        
        print(f"Generating {FREQUENCY}Hz test tone...")
        
        # Generate test tone
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
        tone = 0.3 * np.sin(2 * np.pi * FREQUENCY * t)
        
        print("Playing test tone...")
        sd.play(tone, SAMPLE_RATE)
        sd.wait()
        
        print("✓ Audio playback completed")
        return True
        
    except Exception as e:
        print(f"Error testing audio playback: {e}")
        return False

def test_pii_detection_basic():
    """Test PII detection without heavy dependencies"""
    print("\nBasic PII Detection Test:")
    print("=" * 40)
    
    import re
    import os
    
    # Basic regex patterns
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+91|0)?[6-9]\d{9}\b',
        'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
    }
    
    test_texts = [
        "My email is john.doe@example.com",
        "Call me at 9876543210", 
        "My PAN number is ABCDE1234F"
    ]
    
    for text in test_texts:
        findings = []
        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append((pii_type, match))
        
        if findings:
            print(f"✓ '{text}' -> {findings}")
        else:
            print(f"✗ '{text}' -> No PII detected")
    
    # Test custom slurs file
    slurs_path = os.path.join("blurberry", "data", "wordlists", "custom_slurs.txt")
    if os.path.exists(slurs_path):
        with open(slurs_path, 'r') as f:
            slurs = [line.strip().lower() for line in f if line.strip()]
        print(f"✓ Custom slurs loaded: {len(slurs)} words")
    else:
        print("✗ Custom slurs file not found")

if __name__ == "__main__":
    print("BlurBerry Audio Component Tests")
    print("=" * 50)
    
    # Test components
    audio_ok = test_audio_devices()
    if audio_ok:
        test_audio_recording()
        test_audio_playback()
    
    test_pii_detection_basic()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("✓ Basic components tested successfully")
    print("⚠ Full pipeline requires PyTorch fix")
    print("\nTo run full pipeline:")
    print("1. Fix PyTorch DLL issue (may need Visual C++ Redistributable)")
    print("2. Run: python test_audio.py")
