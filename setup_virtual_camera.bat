@echo off
echo BlurBerry Virtual Camera Setup
echo =============================
cd /d "d:\Allen Code Repository\BlurBerry"

echo.
echo Step 1: Installing virtual camera dependencies...
call venv\Scripts\activate
pip install pyvirtualcam

echo.
echo Step 2: Testing virtual camera availability...
python -c "import pyvirtualcam; print('✅ Virtual camera available')"

echo.
echo Setup complete!
echo.
echo Available tests:
echo   - test_objects_virtual_camera.py  (Object detection + virtual camera)
echo   - test_face_virtual_camera.py     (Face recognition + virtual camera)
echo.
echo Usage:
echo   1. Run: test_objects_virtual_camera.py
echo   2. Open Zoom/Discord and select "OBS Virtual Camera"
echo   3. You should see the BlurBerry processed video feed!
echo.
pause
