@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup_windows.bat first.& pause& exit /b 1)
if not exist "data\ICBHI_final_database" (echo Dataset folder not found: data\ICBHI_final_database& echo Obtain the ICBHI 2017 dataset and extract its WAV and TXT files there.& pause& exit /b 1)
".venv\Scripts\python.exe" prepare_icbhi.py --dataset-dir "data\ICBHI_final_database"
pause
