# BlurBerry - AI-Powered Privacy Filter
[![YouTube Demo](https://img.shields.io/badge/Watch-Demo-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Bj_x8OvU-Kc)

## Overview

BlurBerry is a real-time privacy protection system that automatically blurs sensitive information in video streams. It uses AI models to detect and protect faces (whitelisting), license plates, credit cards, NSFW content, and text PII, making it ideal for video calls, streaming, and content creation.

## Features

- **Face Recognition**: Enroll your face to keep it clear while blurring others
- **Object Detection**: Automatically blurs license plates and credit cards
- **NSFW Filtering**: Blocks sensitive content in real-time
- **Text PII Protection**: Detects and blurs personally identifiable text
- **Voice PII Redaction**: Holds your mic (and video) ~1 s, transcribes on-device and bleeps spoken phone numbers, IDs, emails and passwords before they reach the call
- **Virtual Camera Output**: Streams processed video to OBS Virtual Camera for Zoom, Teams, Meet
- **Web Interface**: Browser-based control panel with live preview
- **Real-time Performance**: Optimized for smooth 30 FPS streaming

## Screenshots
![demo1](https://github.com/user-attachments/assets/9cea181a-c136-4cf7-88bd-2af40a54c175)
![demo3](https://github.com/user-attachments/assets/eec8e563-a871-41dc-b80c-ffec1d6cd05f)



## Quick Start

You need Python 3.10+, a webcam and [OBS Studio](https://obsproject.com/) (for the virtual camera). On Windows also install the [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first (see [Prerequisites](#prerequisites)).

```bash
git clone https://github.com/besque/BlurBerry.git
cd BlurBerry
```

| | Windows | macOS |
|---|---|---|
| **1. Install** (once) | `setup.bat` | `./setup.sh` |
| **2. Run** | `run.bat` | `./run.sh` |
| **3. Check it works** (optional) | `venv\Scripts\python -m tests` | `venv/bin/python -m tests` |

Step 2 opens http://localhost:5000 in your browser. Then:

1. Click **Start Virtual Camera**, enroll the faces that should stay visible, click **Start Stream**.
2. In Zoom, Teams or Meet, choose **OBS Virtual Camera** as the camera.
3. For voice redaction, install a virtual audio device ([VB-Cable](https://vb-audio.com/Cable/) on Windows, `brew install blackhole-2ch` on macOS) and choose **CABLE Output** / **BlackHole 2ch** as the microphone. See [Voice PII Redaction](#voice-pii-redaction).

On macOS, allow camera and microphone access for your terminal when asked. Stop BlurBerry with `Ctrl+C` in its window.

## Directory Structure

```
BlurBerry/
├── blurberry/
│   ├── audio/                    # Voice PII redaction
│   │   ├── voice_redactor.py     #   mic delay line, bleeps, audio devices
│   │   ├── asr.py                #   streaming speech recognition (Vosk)
│   │   └── spoken_pii.py         #   spoken PII rules (numbers, IDs, emails, triggers)
│   ├── video/
│   │   ├── camera.py             # Webcam choice and opening per platform
│   │   ├── face_pipeline.py      # Face detection and recognition
│   │   ├── object_detector.py    # License plate and card detection
│   │   ├── nsfw_detector.py      # NSFW content detection
│   │   ├── text_pii_detector.py  # OCR + text PII rules
│   │   ├── screen_capture.py     # Monitor / window capture for screen share
│   │   ├── tracker.py            # Object tracking between detections
│   │   └── video_loop.py, virtual_video_loop.py, blur_compositor.py   # used by scripts/
│   ├── web/
│   │   ├── server.py             # Flask server: the main app
│   │   └── static/index.html     # Web interface
│   ├── shared_types.py           # DetectionEvent, PipelineConfig
│   └── models/                   # Model files (downloaded, not in git)
├── scripts/                      # Standalone demos and debugging tools (OpenCV windows)
├── tests/                        # python -m tests
├── download_models.py            # Model downloader
├── setup.bat / setup.sh          # One-step setup (Windows / macOS)
├── run.bat / run.sh              # Start BlurBerry and open the browser
└── enrolled_faces/               # Saved face embeddings (not in git)
```

## Installation

### Prerequisites

- Windows 10/11 or macOS 12+
- Python 3.10+ (3.12 tested)
- Webcam
- [OBS Studio](https://obsproject.com/) 30+, for its virtual camera (Zoom/Teams/Meet integration)
- Windows only: the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (workload *Desktop development with C++*); `insightface` has no Windows wheels and compiles during install

### Setup

**Windows:** double-click `setup.bat`, then `run.bat`.

**macOS:**
```bash
./setup.sh
./run.sh
```

The setup scripts create `venv/`, install `requirements.txt` and run `download_models.py`. To do it by hand:

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate          # macOS
pip install -r requirements.txt
python download_models.py
```

`download_models.py` also builds the fastest model format for the platform: a CoreML package on macOS (Neural Engine), a 640x384 ONNX model elsewhere. If the download fails, get the files from the [v0.1-models release](https://github.com/besque/BlurBerry/releases/tag/v0.1-models) and put them in `blurberry/models/`.

## Usage

### Web Interface (Recommended)

1. Start BlurBerry with `run.bat` / `./run.sh`, or `python blurberry/web/server.py` (then open http://localhost:5000).
2. Click **Start Virtual Camera**, enroll the faces that should stay visible, then **Start Stream**.
3. In Zoom, Teams or Meet pick **OBS Virtual Camera** as the camera. Toggle protections and share a screen or window from the sidebar.

`BLURBERRY_SOURCE=<camera index | video file>` picks a different camera, or plays a video file instead (handy for testing).

### Platform Notes

The same code runs on both platforms; each uses its fastest local backend:

| | macOS | Windows |
|---|---|---|
| Camera | AVFoundation | DirectShow, MJPG (720p at 30 fps) |
| Plates / cards (YOLO) | CoreML on the Neural Engine | ONNX Runtime on CPU (PyTorch on CUDA if an NVIDIA GPU is set up) |
| Faces (InsightFace) | CoreML | ONNX Runtime on CPU |
| Text PII OCR | Apple Vision | Windows OCR (`Windows.Media.Ocr`), EasyOCR as fallback |
| Window capture | Quartz (covered windows still captured) | PrintWindow (covered windows still captured) |
| Voice output | BlackHole | VB-Cable |

On both, the default camera skips OBS Virtual Camera so BlurBerry never reads its own output.

### Voice PII Redaction

Works like a broadcast profanity delay: the mic (and the video, to keep lip sync) is held for 1 second while an on-device recognizer (Vosk) transcribes it, and spoken PII is replaced with a 1 kHz bleep before it leaves the buffer. Phrases like "my phone number is" arm redaction of what follows, so the first digit is already caught; digit-by-digit dictation, emails, UPI IDs and spelled IDs are caught without one.

1. Install a virtual audio device so call apps can hear the redacted audio:
   - macOS: [BlackHole](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`)
   - Windows: [VB-Cable](https://vb-audio.com/Cable/) (run the installer as administrator, then reboot)
2. Start BlurBerry; the redacted audio is played into BlackHole / VB-Cable automatically. In Zoom, Meet or Teams pick **BlackHole 2ch** (macOS) or **CABLE Output** (Windows) as the microphone.
3. The **Live Transcript** panel shows what was redacted; **🎧 Monitor** plays the call's audio on your headphones.

If the panel says the microphone is silent, allow microphone access: macOS → System Settings → Privacy & Security → Microphone (for your terminal); Windows → Settings → Privacy & security → Microphone (turn on *Let desktop apps access your microphone*).

Without the web UI:
```bash
python -m blurberry.audio.voice_redactor                  # mic → headphones, live
python -m blurberry.audio.voice_redactor in.wav out.wav   # redact a recording
python -m tests voice_pii                                 # detector + end-to-end tests
```

Environment variables: `BLURBERRY_VOICE_DELAY` (default `1.0`; `1.5` also catches the first digits of a number said with no lead-in), `BLURBERRY_VOICE_MODE` (`bleep` or `mute`), `BLURBERRY_MIC`, `BLURBERRY_AUDIO_OUT` (`none` to disable), `BLURBERRY_VOSK_MODEL` (e.g. `vosk-model-small-en-in-0.4` for Indian English).

### Scripts

Standalone demos with an OpenCV preview window; they predate the web app and use the simpler pipeline in `blurberry/video/video_loop.py`.

```bash
python scripts/virtual_camera.py             # all protections -> OBS Virtual Camera
python scripts/virtual_camera_face_only.py   # face blur only -> OBS Virtual Camera
python scripts/enroll_face.py                # enroll a face, then preview the stream
python scripts/detect_objects.py             # faces, plates, cards, NSFW with debug boxes
python scripts/debug_cards.py                # card/plate detections every frame
python scripts/track_objects.py              # tracker smoothing between detections
```

### Tests

```bash
python -m tests              # everything
python -m tests video        # tests/test_video.py: camera, models, OCR, screen capture, web app
python -m tests voice_pii    # tests/test_voice_pii.py
```

The video tests use whatever this machine runs on (CoreML or ONNX, Vision or Windows OCR, Quartz or PrintWindow), so running them on each platform checks that platform's path. Tests whose model or package is missing are skipped with a note.

## Configuration

### Privacy Settings

Toggle privacy features in the web interface or modify `PipelineConfig`:

```python
config = PipelineConfig(
    blur_faces=True,           # Blur non-enrolled faces
    blur_plates=True,          # Blur license plates
    blur_cards=True,           # Blur credit cards
    blur_nsfw=True,            # Blur NSFW content
    blur_text_pii=True,        # Blur text PII
    detection_cadence=10,      # Detection frequency (frames)
    blur_strength=51,          # Blur intensity
    face_similarity_threshold=0.38  # Face recognition threshold
)
```

### Virtual Camera Settings

- **Resolution**: 1280x720 at 30 FPS
- **Backend**: OBS Virtual Camera
- **Compatibility**: Zoom, Microsoft Teams, Google Meet, Discord

## Model Information

- **Face Detection**: InsightFace buffalo_sc (downloaded on first run)
- **Object Detection**: YOLO models for plates and cards (`download_models.py`)
- **NSFW Detection**: NudeNet
- **Text Detection**: Apple Vision (macOS), Windows OCR (Windows), EasyOCR (fallback)
- **Speech Recognition**: Vosk small English model (~40 MB, runs on CPU)

## Performance Optimization

- **Fastest backend per platform**: see Platform Notes
- **Motion gate**: plate/card detection is skipped while the scene is still
- **Threading**: detectors, OCR and speech recognition run off the video thread
- **Model warmup**: models are loaded and run once before streaming starts

## Troubleshooting

1. **Camera not found / wrong camera**: close other apps using it, or set `BLURBERRY_SOURCE` to its index. On macOS allow camera access for your terminal (System Settings → Privacy & Security → Camera).
2. **Low FPS on Windows**: some webcams ignore the MJPG request; try another USB port or a lower resolution in the camera's own software.
3. **Virtual camera not working**: install OBS Studio 30+ and start its virtual camera once from OBS.
4. **`insightface` fails to install on Windows**: install the Microsoft C++ Build Tools (see Prerequisites) and run `setup.bat` again.
5. **Text PII uses EasyOCR on Windows** (startup log says so): the Windows OCR language pack for your display language is missing; add it under Settings → Time & language → Language.
6. **Screen share is black on macOS**: allow Screen Recording for your terminal (System Settings → Privacy & Security → Screen Recording).
7. **Models not loading**: run `python download_models.py`.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review model download instructions
- Ensure all dependencies are properly installed

---

**BlurBerry provides enterprise-grade privacy protection for video communications, ensuring sensitive information remains confidential while maintaining professional video quality.**
