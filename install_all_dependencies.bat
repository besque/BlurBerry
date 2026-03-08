@echo off
echo Installing All BlurBerry Dependencies
echo =====================================
cd /d "d:\Allen Code Repository\BlurBerry\venv\Scripts"

echo Installing ultralytics (YOLO)...
pip.exe install ultralytics

echo Installing nudenet (NSFW detection)...
pip.exe install nudenet

echo Installing pyvirtualcam...
pip.exe install pyvirtualcam

echo Installing other dependencies...
pip.exe install opencv-python numpy insightface

echo.
echo Installation complete!
echo Available scripts:
echo   - virtual_camera_complete.py
echo   - test_objects_virtual_camera.py
echo   - test_face_virtual_camera.py
echo   - test_objects.py
echo.
pause
