@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10, 3.11, or 3.12 and add it to PATH.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your internet connection and Python version.
  pause
  exit /b 1
)
echo Starting FuelScope dashboard...
python -m streamlit run app.py
pause
