@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (echo Python Launcher is missing. Install Python 3.12 first.& exit /b 1)
py -3.12 --version >nul 2>nul || (echo Python 3.12 is required. Install it from python.org.& exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo Setup complete. Use prepare_dataset.bat if the ICBHI dataset is in data\ICBHI_final_database, then train_model.bat, then run_app.bat.
pause
