import sys
import os

# Test imports step by step
print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")
    sys.exit(1)

try:
    import sounddevice as sd
    print("✓ sounddevice")
except Exception as e:
    print(f"✗ sounddevice: {e}")
    sys.exit(1)

try:
    import scipy
    print("✓ scipy")
except Exception as e:
    print(f"✗ scipy: {e}")
    sys.exit(1)

try:
    import better_profanity
    print("✓ better_profanity")
except Exception as e:
    print(f"✗ better_profanity: {e}")
    sys.exit(1)

print("\nTesting PII detector (without spaCy)...")

# Test PII detection without spaCy
import re

def test_pii_basic():
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+91|0)?[6-9]\d{9}\b',
        'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
    }
    
    test_text = "My email is john.doe@example.com and phone is 9876543210"
    findings = []
    
    for pii_type, pattern in patterns.items():
        matches = re.finditer(pattern, test_text, re.IGNORECASE)
        for match in matches:
            findings.append((pii_type, match.group()))
    
    print(f"PII Detection: {findings}")
    return len(findings) > 0

if test_pii_basic():
    print("✓ PII detection works")
else:
    print("✗ PII detection failed")

print("\nTesting audio devices...")
try:
    devices = sd.query_devices()
    print(f"✓ Found {len(devices)} audio devices")
    
    # Test basic audio recording
    SAMPLE_RATE = 16000
    recording = sd.rec(int(1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print(f"✓ Audio recording works: {len(recording)} samples")
    
except Exception as e:
    print(f"✗ Audio test failed: {e}")

print("\n" + "="*50)
print("Basic components are working!")
print("The full pipeline needs PyTorch/Whisper fix.")
print("You can test the simplified version:")
print("python test_pipeline_simple.py")
