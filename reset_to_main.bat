@echo off
echo Reset BlurBerry to Main Branch
echo ==============================
cd /d "d:\Allen Code Repository\BlurBerry"

echo Current status:
git status

echo.
echo Current branch:
git branch --show-current

echo.
echo Stashing any local changes...
git stash push -m "Stashing current work before reset"

echo.
echo Switching to main branch...
git checkout main

echo.
echo Pulling latest from main...
git pull origin main

echo.
echo Reset complete! Current status:
git status

echo.
echo Current branch:
git branch --show-current

echo.
echo If you want to restore your work later, run:
echo git stash pop

pause
