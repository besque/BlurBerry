"""
Spoken PII detector tests. Transcripts are lowercase ASR output, several taken from
Vosk on synthetic speech, so they include its mistakes ("pen" for PAN, "to" for two).

Run: python -m tests voice_pii
"""
import os, sys

from blurberry.audio.spoken_pii import detect, redact_text, words_from_text, Word

CASES = [
    # (transcript, words that must be redacted, words that must stay)
    ("my phone number is nine eight seven six five four three two one zero",
     "nine eight seven six five four three two one zero", "my phone number is"),
    ("call me on ninety eight forty five sixty seven twenty one thirty",
     "ninety eight forty five sixty seven twenty one thirty", "call me on"),
    ("my number is double nine eight four five triple seven to one",
     "double nine eight four five triple seven to one", "my number is"),
    ("my on her number is four three two one eight seven six five two one zero nine",
     "four three two one eight seven six five two one zero nine", "my on her number is"),
    ("the card number is four thousand one hundred eleven one thousand and one hundred eleven",
     "four thousand one hundred eleven one thousand and one hundred eleven", "the card number is"),
    ("the o t v is five nine to eight one four", "five nine to eight one four", "the is"),
    ("my pen is a b c d one two three four f", "a b c d one two three four f", "my pen is"),
    ("my vehicle number is kay zero one eighty one two three four",
     "kay zero one eighty one two three four", "my vehicle number is"),
    ("my password is sunshine forty two", "sunshine forty two", "my password is"),
    ("my name is ammo and i live at forty two mg road bangalore",
     "ammo forty two mg road bangalore", "my name is and i live at"),
    ("my email is john dot doe at gmail dot com and that is it",
     "john dot doe at gmail dot com", "my email is and that is it"),
    ("you can mail me at amogh at the rate gmail dot com", "amogh at the rate gmail dot com", "you can mail me"),
    ("send it to amogh at gmail", "amogh at gmail", "send it to"),
    ("my email is john da da je male dot com", "john da da je male dot com", "my email is"),
    ("pay me at amogh at okaxis", "amogh at okaxis", "pay me"),
    ("the otp is five nine two eight", "five nine two eight", "the otp is"),
    ("my date of birth is the twenty fifth of march", "twenty fifth of march", "my date birth is the"),
    ("it is nine eight seven six five today", "nine eight seven six five", "it is today"),
    ("my pin is one nine", "one nine", "my pin is"),
    ("the otp is five nine to eight one for my password", "five nine to eight one for", "the otp is my password"),
]

CLEAN = [
    "the meeting is a ten thirty on the twenty fifth",
    "i have two kids and three dogs",
    "the number of people is five",
    "we sold twenty twenty four units",
    "look at this it is great",
    "my phone number is not something i share",
    "my name is not important",
    "i need a pen to write",
    "one of them is here",
    "there are four hundred people",
    "the code review is tomorrow",
    "i gave him two for one",
]


def _check(text, must, keep):
    words = words_from_text(text)
    flags = detect(words)
    toks = text.split()
    redacted = [toks[i] for i in sorted(flags)]
    assert redacted == must.split(), f"{text!r}\n  redacted {redacted}\n  expected {must.split()}"
    for w in keep.split():
        assert all(toks[i] != w or i not in flags for i in range(len(toks))), f"{text!r}: {w!r} redacted"


def test_cases():
    for text, must, keep in CASES:
        _check(text, must, keep)


def test_clean():
    for text in CLEAN:
        flags = detect(words_from_text(text))
        assert not flags, f"{text!r} -> {redact_text(text)}"


def test_streaming_prefixes():
    """Triggers arm redaction of the payload's first word, before the number looks like PII."""
    text = "my phone number is nine"
    assert redact_text(text).endswith('[PHONE]'), redact_text(text)
    text = "my email is john"
    assert redact_text(text).endswith('[EMAIL]'), redact_text(text)


def test_trigger_expires():
    words = [Word('my', 0, .2), Word('number', .2, .5), Word('is', .5, .6), Word('nine', 6.0, 6.3)]
    assert not detect(words)


def _fake_sd(default_device, devices):
    class SD:
        default = type('default', (), {'device': default_device, 'hostapi': 0})

        @staticmethod
        def query_devices(i=None, kind=None):
            return devices if i is None else devices[i]
    return SD


def _dev(name, api, ins, outs):
    return dict(name=name, hostapi=api, max_input_channels=ins, max_output_channels=outs)


def test_device_selection():
    """Windows lists each device once per host API (0 = MME, 2 = WASAPI here) with MME
    names cut at 31 characters; the default API's copy is picked and virtual cables
    are never used as the mic."""
    from blurberry.audio.voice_redactor import find_input, find_output
    win = [_dev('Microsoft Sound Mapper - Input', 0, 2, 0), _dev('Microphone (Realtek(R) Audio)', 0, 2, 0),
           _dev('CABLE Output (VB-Audio Virtual ', 0, 8, 0), _dev('Microsoft Sound Mapper - Output', 0, 0, 2),
           _dev('CABLE Input (VB-Audio Virtual Cable)', 2, 0, 8), _dev('CABLE Input (VB-Audio Virtual C', 0, 0, 8),
           _dev('Microphone (Realtek(R) Audio)', 2, 2, 0)]
    assert find_output(_fake_sd([0, 3], win)) == 5
    assert find_input(_fake_sd([0, 3], win), 'realtek') == 1
    assert find_input(_fake_sd([2, 3], win)) == 0
    assert find_output(_fake_sd([0, 3], win), 'none') is None
    mac = [_dev('MacBook Air Microphone', 0, 1, 0), _dev('MacBook Air Speakers', 0, 0, 2), _dev('BlackHole 2ch', 0, 2, 2)]
    assert find_output(_fake_sd([0, 1], mac)) == 2
    assert find_input(_fake_sd([2, 1], mac)) == 0
    assert find_output(_fake_sd([0, 1], mac[:2])) is None


def _synthesize(text: str, wav: str) -> bool:
    """Speak `text` into a 16-bit mono WAV with the OS voice: `say` on macOS,
    System.Speech (Windows PowerShell) on Windows. False when neither is available."""
    import shutil, subprocess
    if shutil.which('say') and shutil.which('afconvert'):
        aiff = wav + '.aiff'
        subprocess.run(['say', '-o', aiff, text], check=True)
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@48000', '-c', '1', aiff, wav], check=True)
        return True
    if sys.platform == 'win32' and shutil.which('powershell'):
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
            f"$s.SetOutputToWaveFile('{wav.replace(chr(39), chr(39) * 2)}', $f); $s.Speak('{text}'); $s.Dispose()")
        subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True)
        return True
    return False


def test_end_to_end():
    """Synthetic speech through the ASR and the delay line at the default delay: every
    digit after the trigger is bleeped before it plays, and the words around stay clear."""
    import tempfile, wave
    import numpy as np
    from blurberry.audio.asr import find_model
    from blurberry.audio.voice_redactor import simulate
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, 'speech.wav')
        text = 'Hi, my phone number is nine eight seven six five, four three two one zero. Thanks.'
        if not find_model() or not _synthesize(text, wav):
            print('  (end-to-end skipped: needs the Vosk model and a system voice)')
            return
        with wave.open(wav) as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    _, core = simulate(pcm, sr)
    labels = dict(core.history)
    digits = 'nine eight seven six five four three two one zero'.split()
    assert all(labels.get(d) == 'phone' for d in digits), core.history
    assert not any(labels.get(w) for w in ('hi', 'my', 'phone', 'number', 'thanks')), core.history
    assert core.leaked == 0, f'{core.leaked * 1000 // sr} ms of the number played before it was caught'

