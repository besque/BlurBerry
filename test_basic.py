import re
import os

def test_regex_patterns():
    """Test basic regex patterns without heavy dependencies"""
    
    patterns = {
        'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
        'indian_phone': r'\b(?:\+91|0)?[6-9]\d{9}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'bank_account': r'\b\d{10,18}\b',
        'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
        'upi_id': r'\b[A-Za-z0-9._-]+@[a-z]{2,}\b',
    }
    
    test_cases = [
        ("My email is test@example.com", "email"),
        ("Call me at 9876543210", "indian_phone"),
        ("My PAN is ABCDE1234F", "pan"),
        ("Aadhaar number 1234 5678 9012", "aadhaar"),
        ("Credit card 4111 1111 1111 1111", "credit_card"),
        ("IP address 192.168.1.1", "ip_address"),
        ("Bank account 1234567890123456", "bank_account"),
        ("IFSC code HDFC0001234", "ifsc"),
        ("UPI id user@okbank", "upi_id"),
    ]
    
    print("Testing PII Detection Patterns:")
    print("=" * 40)
    
    for text, expected_type in test_cases:
        pattern = patterns[expected_type]
        match = re.search(pattern, text, re.IGNORECASE)
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"{status}: {text} -> {expected_type}")
        if match:
            print(f"   Matched: {match.group()}")
    
    print("\nTesting custom slurs file:")
    print("=" * 40)
    
    slurs_path = os.path.join("blurberry", "data", "wordlists", "custom_slurs.txt")
    try:
        with open(slurs_path, 'r') as f:
            slurs = [line.strip().lower() for line in f if line.strip()]
        print(f"✓ Found {len(slurs)} custom slurs: {slurs}")
    except FileNotFoundError:
        print("✗ Custom slurs file not found")
    
    print("\nTesting folder structure:")
    print("=" * 40)
    
    required_files = [
        "blurberry/audio/__init__.py",
        "blurberry/audio/pii_detector.py", 
        "blurberry/audio/vad.py",
        "blurberry/audio/audio_pipeline.py",
        "blurberry/data/wordlists/custom_slurs.txt",
        "test_audio.py"
    ]
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"{status}: {file_path}")

if __name__ == "__main__":
    test_regex_patterns()
