@echo off
echo Removing Twitch-related files...
cd /d "d:\Allen Code Repository\BlurBerry"

echo Deleting install_twitch.bat...
if exist install_twitch.bat del install_twitch.bat

echo Deleting twitch_face_blur_camera.py...
if exist twitch_face_blur_camera.py del twitch_face_blur_camera.py

echo Deleting twitch_setup_guide.md...
if exist twitch_setup_guide.md del twitch_setup_guide.md

echo Cleanup complete!
echo Remaining files:
dir *.py *.bat *.md
pause
