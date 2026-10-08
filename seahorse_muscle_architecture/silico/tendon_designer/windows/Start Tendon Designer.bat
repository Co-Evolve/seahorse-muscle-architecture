@echo off
setlocal
cd /d "%~dp0"
title Tendon Designer

if not exist "app\index.html" (
  echo.
  echo   The app files were not found next to this file.
  echo   Please right-click the zip file first and choose "Extract All...",
  echo   then open the extracted folder and double-click this file again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting the Tendon Designer...
echo   Your browser opens in a moment. Keep this window open while you work.
echo   Close this window to stop the app.
echo.

rem 1) Python launcher "py" (installed with Python from python.org)
py -3 -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_py

rem 2) "python" on the PATH (skips the Microsoft Store placeholder, which fails this test)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if %errorlevel%==0 goto use_python

rem 3) No Python: use the web server built into Windows PowerShell
goto use_powershell

:use_py
py -3 "tools\serve.py" --root "app"
goto end

:use_python
python "tools\serve.py" --root "app"
goto end

:use_powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\serve.ps1"
goto end

:end
echo.
pause
