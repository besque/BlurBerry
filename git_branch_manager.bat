@echo off
echo Git Branch Manager for BlurBerry
echo ================================
cd /d "d:\Allen Code Repository\BlurBerry"

echo Current branch:
git branch --show-current

echo.
echo All available branches:
git branch -a

echo.
echo Remote branches:
git branch -r

pause
