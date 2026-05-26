@echo off
REM ============================================================
REM  MTACD — Multi-Temporal Aerial Change Detection
REM  Helsinki Urban Change Detection (AMD / DirectML build)
REM  Double-click this file to launch the GUI.
REM ============================================================
REM
REM  Requires Python 3.12. torch-directml (the AMD GPU library)
REM  only has wheels for Python 3.10 – 3.12; newer versions are
REM  not yet supported. We use the `py` launcher so 3.12 is used
REM  even if a different Python is your system default.
REM
REM  Run Install.bat FIRST if you haven't already.
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo   MTACD — Multi-Temporal Aerial Change Detection
echo ============================================================
echo.

REM --- 1. Check for py launcher --------------------------------
where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: the 'py' launcher was not found.
    echo Install Python 3.12 from:
    echo     https://www.python.org/downloads/release/python-31210/
    echo Tick "Add python.exe to PATH" and "py launcher" during setup.
    echo.
    pause
    exit /b 1
)

REM --- 2. Check for Python 3.12 --------------------------------
py -3.12 --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.12 is not installed.
    echo torch-directml requires Python 3.10, 3.11, or 3.12.
    echo Install Python 3.12 from:
    echo     https://www.python.org/downloads/release/python-31210/
    echo Then run Install.bat to install all required libraries.
    echo.
    pause
    exit /b 1
)

REM --- 3. Launch -----------------------------------------------
echo Starting MTACD app...
echo.
py -3.12 app.py
if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------
    echo  The app exited with an error.  Common fixes:
    echo    "No module named ..."  ^>  run Install.bat first.
    echo    Other errors           ^>  check the console output above.
    echo ------------------------------------------------------------
    pause
)
