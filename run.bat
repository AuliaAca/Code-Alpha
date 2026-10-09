@echo off
rem Opens the CodeLab web app (all four tasks) in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
