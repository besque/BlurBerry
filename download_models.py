import urllib.request
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blurberry", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "plate_best.pt": "https://github.com/besque/BlurBerry/releases/download/v0.1-models/plate_best.pt",
    "card_best.pt":  "https://github.com/besque/BlurBerry/releases/download/v0.1-models/card_best.pt",
    "plate_best.onnx": "https://github.com/besque/BlurBerry/releases/download/v0.1-models/plate_best.onnx",
    "card_best.onnx": "https://github.com/besque/BlurBerry/releases/download/v0.1-models/card_best.onnx"
}

def download(filename, url):
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  saved {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  manually download from: {url}")
        print(f"  and save to: {dest}")

VOSK_MODEL = "vosk-model-small-en-us-0.15"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

def download_vosk():
    """Speech model for voice redaction (~40 MB zip). Any Vosk model works: point
    BLURBERRY_VOSK_MODEL at another folder, e.g. vosk-model-small-en-in-0.4 for Indian English."""
    import tempfile, zipfile
    dest = os.path.join(MODELS_DIR, VOSK_MODEL)
    if os.path.isdir(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {VOSK_MODEL}...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "model.zip")
            urllib.request.urlretrieve(VOSK_URL, archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(MODELS_DIR)
        print(f"  saved {dest}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  manually download {VOSK_URL} and unzip it into {MODELS_DIR}")

def export_coreml(pt_name):
    """On macOS, build a CoreML copy so detection runs on the Neural Engine."""
    pt = os.path.join(MODELS_DIR, pt_name)
    dest = pt.replace(".pt", ".mlpackage")
    if os.path.exists(dest) or not os.path.exists(pt):
        return
    print(f"  exporting {pt_name} to CoreML...")
    try:
        from ultralytics import YOLO
        YOLO(pt).export(format="coreml", imgsz=(384, 640), half=True)
        print(f"  saved {dest}")
    except Exception as e:
        print(f"  CoreML export skipped ({e}); the .pt model will be used instead")

def export_onnx_wide(pt_name):
    """Off macOS, build a 640x384 ONNX copy: the trained 640 resolution, shaped for
    16:9 video, and faster on CPU than the .pt. Exported in a temp folder so the
    release .onnx next to the .pt isn't overwritten."""
    import shutil, tempfile
    pt = os.path.join(MODELS_DIR, pt_name)
    dest = pt.replace(".pt", "_384x640.onnx")
    if os.path.exists(dest) or not os.path.exists(pt):
        return
    print(f"  exporting {pt_name} to ONNX (640x384)...")
    try:
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pt = shutil.copy(pt, tmp)
            out = YOLO(tmp_pt).export(format="onnx", imgsz=(384, 640), simplify=False)
            shutil.move(out, dest)
        print(f"  saved {dest}")
    except Exception as e:
        print(f"  ONNX export skipped ({e}); the .pt model will be used instead")

print("BlurBerry — downloading model weights\n")
for name, url in MODELS.items():
    download(name, url)
download_vosk()
for name in ("plate_best.pt", "card_best.pt"):
    if sys.platform == "darwin":
        export_coreml(name)
    else:
        export_onnx_wide(name)
print("\nDone. Run: python blurberry/web/server.py")