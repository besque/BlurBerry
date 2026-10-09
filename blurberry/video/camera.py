"""
Picking and opening the webcam on each platform.

macOS    AVFoundation (OpenCV's default).
Windows  DirectShow, asking for MJPG. OpenCV's default there, Media Foundation, can
         take many seconds to open, and most webcams only manage 720p at 30 fps as
         MJPG (uncompressed YUY2 tops out around 10 fps at that size).

On both, OBS Virtual Camera is skipped when choosing the default camera: once it is
installed it can be listed first, and reading it would loop BlurBerry's own output back in.
BLURBERRY_SOURCE=<camera index | video file> overrides the choice.
"""
import os
import sys
from typing import List, Union

import cv2

VIRTUAL_CAMERA_WORDS = ('obs', 'virtual')


def camera_names() -> List[str]:
    """Camera names in the order OpenCV numbers them on this platform."""
    if sys.platform == 'darwin':
        import AVFoundation as AV
        devices = (list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
                   + list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)))
        devices.sort(key=lambda d: str(d.uniqueID()))
        return [str(d.localizedName()) for d in devices]
    if sys.platform == 'win32':
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    return []


def default_camera() -> int:
    try:
        names = camera_names()
    except Exception as e:
        print(f'[Camera] Camera list unavailable ({e}); using camera 0')
        return 0
    for i, name in enumerate(names):
        if not any(w in name.lower() for w in VIRTUAL_CAMERA_WORDS):
            print(f'[Camera] {name} (index {i})')
            return i
    return 0


def configured_source() -> Union[int, str]:
    src = os.environ.get('BLURBERRY_SOURCE', '')
    if not src:
        return default_camera()
    return int(src) if src.isdigit() else src


def open_camera(source: Union[int, str], width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    """A capture for a camera index or a video file; check isOpened() on the result."""
    if isinstance(source, str):
        return cv2.VideoCapture(source)
    cap = None
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
