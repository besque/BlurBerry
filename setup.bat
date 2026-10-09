@echo off
rem BlurBerry setup for Windows: virtual environment, dependencies, model files.
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    py -3 -m venv venv 2>nul || python -m venv venv || goto :nopython
)

venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt || goto :failed
rem EasyOCR pulls in opencv-python-headless, which can replace the build with preview windows
venv\Scripts\python -m pip install --force-reinstall --no-deps opencv-python
venv\Scripts\python download_models.py

echo.
echo Setup complete. Start BlurBerry with run.bat
echo For the virtual camera install OBS Studio; for voice redaction install VB-Cable.
pause
exit /b 0

:nopython
echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
pause
exit /b 1

:failed
echo.
echo Installing the dependencies failed. If the error mentions insightface or "Microsoft Visual C++",
echo install the C++ Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/,
echo workload "Desktop development with C++"), then run setup.bat again.
pause
exit /b 1
