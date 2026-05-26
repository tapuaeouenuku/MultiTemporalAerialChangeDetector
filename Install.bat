@echo off
REM ============================================================
REM  MTACD — Multi-Temporal Aerial Change Detection
REM  Dependency installer (AMD / DirectML build)
REM ============================================================
REM
REM  What this script does:
REM    1. Verifies the Python 3.12 "py" launcher is available.
REM    2. Installs all Python packages from requirements.txt.
REM
REM  Key packages installed:
REM    torch / torchvision    — deep-learning framework
REM    torch-directml         — AMD GPU support on Windows
REM    numpy / Pillow         — array and image handling
REM    opencv-python          — image processing (CLAHE, morphology)
REM    scikit-image           — histogram matching for pair normalisation
REM    pyproj                 — coordinate projection (WGS84 <-> EPSG:3879)
REM    shapely / requests     — geometry and HTTP requests
REM    matplotlib             — result plots
REM
REM  Why Python 3.12:
REM    torch-directml currently only has wheels for Python 3.10–3.12.
REM
REM  Why --pre:
REM    torch-directml is published as a pre-release on PyPI.
REM    Without --pre, pip cannot find it.
REM
REM  First-time install: ~2 GB download, takes 5–15 minutes.
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo   MTACD — Installing dependencies
echo ============================================================
echo.

REM --- 1. Check py launcher ------------------------------------
where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: The 'py' launcher was not found.
    echo Install Python 3.12 from:
    echo     https://www.python.org/downloads/release/python-31210/
    echo Tick "Add python.exe to PATH" and "py launcher" during setup.
    echo.
    pause
    exit /b 1
)

REM --- 2. Check Python 3.12 ------------------------------------
py -3.12 --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.12 is not installed.
    echo torch-directml requires Python 3.10, 3.11, or 3.12.
    echo Install Python 3.12 from:
    echo     https://www.python.org/downloads/release/python-31210/
    echo You can keep your existing Python; the py launcher picks
    echo the right version using the -3.12 flag.
    echo.
    pause
    exit /b 1
)

echo Found:
py -3.12 --version
echo.

REM --- 3. Upgrade pip first ------------------------------------
echo Upgrading pip...
py -3.12 -m pip install --upgrade pip --quiet
echo.

REM --- 4. Install requirements ---------------------------------
echo Installing packages (torch-directml is ~2 GB, please wait)...
echo.
py -3.12 -m pip install --pre -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed. See the pip output above.
    echo Common causes:
    echo   - No internet / proxy      -- configure pip proxy and retry.
    echo   - Disk full                -- needs ~2 GB free space.
    echo   - Network interrupted      -- re-run this script.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Install complete!  Double-click Run.bat to start the app.
echo ============================================================
echo.
pause
