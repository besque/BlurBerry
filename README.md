# BlurBerry - AI-Powered Privacy Filter
[![YouTube Demo](https://img.shields.io/badge/Watch-Demo-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Bj_x8OvU-Kc)

## Overview

BlurBerry is a real-time privacy protection system that automatically blurs sensitive information in video streams. It uses AI models to detect and protect faces (whitelisting), license plates, credit cards, NSFW content, and text PII, making it ideal for video calls, streaming, and content creation.

## Features

- **Face Recognition**: Enroll your face to keep it clear while blurring others
- **Object Detection**: Automatically blurs license plates and credit cards
- **NSFW Filtering**: Blocks sensitive content in real-time
- **Text PII Protection**: Detects and blurs personally identifiable text
- **Virtual Camera Output**: Streams processed video to OBS Virtual Camera for Zoom, Teams, Meet
- **Web Interface**: Browser-based control panel with live preview
- **Real-time Performance**: Optimized for smooth 30 FPS streaming

## Screenshots
![demo1](https://github.com/user-attachments/assets/9cea181a-c136-4cf7-88bd-2af40a54c175)
![demo3](https://github.com/user-attachments/assets/eec8e563-a871-41dc-b80c-ffec1d6cd05f)



## Directory Structure

```
BlurBerry/
├── blurberry/
│   ├── video/
│   │   ├── face_pipeline.py      # Face detection and recognition
│   │   ├── object_detector.py    # License plate and card detection
│   │   ├── nsfw_detector.py      # NSFW content detection
│   │   ├── virtual_video_loop.py # Video processing with virtual camera
│   │   └── tracker.py            # Object tracking
│   ├── web/
│   │   ├── server.py             # Flask web server
│   │   └── static/
│   │       └── index.html        # Web interface
│   └── models/                   # AI model files (downloaded)
├── enrolled_faces/               # Saved face embeddings
├── virtual_camera_complete.py   # Complete virtual camera script
├── test_face.py                 # Face enrollment test
├── test_objects.py              # Object detection test
├── download_models.py           # Model downloader
└── venv/                        # Python virtual environment
```

## Installation

### Prerequisites

- Python 3.8+
- Windows 10/11 (recommended)
- Webcam
- OBS Virtual Camera (for Zoom/Teams integration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd BlurBerry
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI models:**
   ```bash
   python download_models.py
   ```

   If the download fails, manually download the models from:
   - https://github.com/besque/BlurBerry/releases/download/v0.1-models/plate_best.pt
   - https://github.com/besque/BlurBerry/releases/download/v0.1-models/card_best.pt
   - https://github.com/besque/BlurBerry/releases/download/v0.1-models/plate_best.onnx
   - https://github.com/besque/BlurBerry/releases/download/v0.1-models/card_best.onnx
   
   Save them to [blurberry/models/](cci:9://file:///d:/Allen%20Code%20Repository/BlurBerry/blurberry/models:0:0-0:0)

## Usage

### Web Interface (Recommended)

1. **Start the web server:**
   ```bash
   python blurberry\web\server.py
   ```

2. **Open browser:**
   ```
   http://localhost:5000
   ```

3. **Follow the on-screen instructions:**
   - Press 'E' to enroll your face
   - Press 'S' to start streaming
   - Use controls to toggle privacy features

### Command Line

**Complete Virtual Camera:**
```bash
python virtual_camera_complete.py
```

**Face-only Virtual Camera:**
```bash
python virtual_camera_face_only.py
```

**Test Individual Components:**
```bash
python test_face.py              # Face enrollment
python test_objects.py           # Object detection
python debug_card_detection.py   # Debug card detection
```

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

BlurBerry uses pre-trained AI models for detection:

- **Face Detection**: InsightFace buffalo_sc model
- **Object Detection**: YOLO models for plates and cards
- **NSFW Detection**: NudeNet model
- **Text Detection**: EasyOCR for PII extraction

Models are automatically downloaded to [blurberry/models/](cci:9://file:///d:/Allen%20Code%20Repository/BlurBerry/blurberry/models:0:0-0:0) directory.

## Performance Optimization

The system includes several optimizations:

- **Frame Skipping**: Processes detection every N frames
- **Resolution Scaling**: Uses smaller frames for inference
- **Model Warmup**: Pre-loads models to avoid first-frame lag
- **Threading**: Separate threads for video processing and streaming

## Troubleshooting

### Common Issues

1. **Camera not found**: Ensure webcam is connected and not in use by other applications
2. **Virtual camera not working**: Install OBS Virtual Camera and ensure it's running
3. **Face detection slow**: Try reducing `detection_cadence` or using smaller `det_size`
4. **Models not loading**: Run [download_models.py](cci:7://file:///d:/Allen%20Code%20Repository/BlurBerry/download_models.py:0:0-0:0) or manually download model files

### Dependencies

If you encounter import errors, install missing packages:

```bash
pip install insightface ultralytics nudenet pyvirtualcam opencv-python flask numpy
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review model download instructions
- Ensure all dependencies are properly installed

---

**BlurBerry provides enterprise-grade privacy protection for video communications, ensuring sensitive information remains confidential while maintaining professional video quality.**
