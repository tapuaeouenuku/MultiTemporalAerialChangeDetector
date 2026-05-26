@echo off
REM ============================================================
REM  MTACD — Unattended batch training
REM  Double-click this before you leave.
REM  It will generate 6000 tiles from 2025 imagery and train 1 model.
REM  Progress is logged to batch_train.log in this folder.
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo   MTACD Batch Training
echo   6000 tiles from 2025 imagery -- one robust model
echo   This will run for several hours.
echo   You can close this window; log is saved to batch_train.log
echo ============================================================
echo.

py -3.12 batch_train.py

echo.
echo ============================================================
echo   Done.  Check batch_train.log for results.
echo   Models are in the models\ folder.
echo ============================================================
echo.
pause
