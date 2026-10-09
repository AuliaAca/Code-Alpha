@echo off
rem Live object detection + tracking from the default webcam. Press q in the window to quit.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" Task4\cli.py --source 0 --show %*
if errorlevel 1 pause
