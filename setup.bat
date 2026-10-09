@echo off
rem One-time setup on Windows: double-click this file or run  .\setup.bat
setlocal
cd /d "%~dp0"

rem Actually run Python: on Windows 10/11 a "python" shortcut can exist that only
rem opens the Microsoft Store, so checking that the command exists is not enough.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer was not found.
    echo  - Install it from https://www.python.org and tick "Add python.exe to PATH".
    echo  - If the Microsoft Store opens instead, turn off the python.exe and python3.exe
    echo    "App execution aliases" in Windows Settings, then open a new terminal.
    python --version 2>nul
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo [2/3] Installing packages - the first time this downloads about 1 GB, please wait ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed. Check your internet connection and run setup.bat again.
    pause
    exit /b 1
)

echo [3/3] Running the tests ...
".venv\Scripts\python.exe" -m pytest -q

echo.
echo Setup complete.
echo   run.bat          opens the web app with all four tasks
echo   run_webcam.bat   live webcam tracking with YOLO
pause
