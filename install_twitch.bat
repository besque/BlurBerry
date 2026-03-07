@echo off
echo Installing Twitch streaming dependencies...
cd /d "d:\Allen Code Repository\BlurBerry\venv\Scripts"

echo Installing pyvirtualcam...
pip.exe install pyvirtualcam

echo Installing python-ffmpeg-video...
pip.exe install ffmpeg-python

echo Installing requests for Twitch API...
pip.exe install requests

echo Installing obs-websocket...
pip.exe install obs-websocket-py

echo Installation complete!
pause
