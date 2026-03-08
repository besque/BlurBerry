@echo off
echo Starting BlurBerry Objects Detection + Virtual Camera...
echo ========================================================
cd /d "d:\Allen Code Repository\BlurBerry"

echo Activating virtual environment...
call venv\Scripts\activate

echo Starting object detection with virtual camera...
echo This will detect objects and stream to Zoom/Discord
echo.
echo Instructions:
echo   1. Wait for virtual camera to start
echo   2. Open Zoom/Discord
echo   3. Select "OBS Virtual Camera" as your camera
echo   4. You should see object detection with privacy blurring!
echo   5. Press Q to quit
echo.

python test_objects_virtual_camera.py

echo.
echo Virtual camera stream stopped.
pause
