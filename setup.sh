#!/bin/sh
# BlurBerry setup for macOS: virtual environment, dependencies, model files.
set -e
cd "$(dirname "$0")"
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
venv/bin/python download_models.py
echo
echo "Setup complete. Start BlurBerry with ./run.sh"
echo "For the virtual camera install OBS Studio; for voice redaction: brew install blackhole-2ch"
