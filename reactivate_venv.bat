@echo off
echo Reactivating virtual environment...
cd /d "d:\Allen Code Repository\BlurBerry"
call venv\Scripts\activate
echo Virtual environment activated!
echo Current directory: %CD%
echo Python path: %PYTHONPATH%
cmd /k
