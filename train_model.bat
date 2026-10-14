@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup_windows.bat first.& pause& exit /b 1)
if not exist "data\labels.csv" (echo Labels not found. Run prepare_dataset.bat first.& pause& exit /b 1)
set PYTHONPATH=src
echo Training may take several minutes on a CPU. This is a research prototype, not a medical diagnostic tool.
".venv\Scripts\python.exe" -u -m rural_stetho.train --manifest data\labels.csv --output models\best.pt --epochs 20 --batch-size 16
pause
