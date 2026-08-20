@echo off
rem Launch the Guardians Report local app and open it in the browser.
rem
rem Kept as a .cmd rather than a PowerShell script so the desktop shortcut runs
rem without touching the execution policy. The console window stays open while
rem the server runs -- closing it stops the server, which is the behaviour
rem people expect from a launcher.

title Guardians Report
cd /d "%~dp0.."

set PORT=8765
set PYTHON=%CD%\.venv\Scripts\python.exe

if not exist "%PYTHON%" (
  echo.
  echo   Could not find the virtual environment at:
  echo     %PYTHON%
  echo.
  echo   Create it with:  py -m venv .venv
  echo.
  pause
  exit /b 1
)

rem If the app is already running, just open it rather than failing on a port
rem clash -- double-clicking the shortcut twice should be harmless.
netstat -ano | findstr /r /c:"LISTENING.*:%PORT% " >nul 2>&1
if %errorlevel%==0 (
  echo   Already running - opening http://127.0.0.1:%PORT%
  start "" "http://127.0.0.1:%PORT%"
  timeout /t 2 >nul
  exit /b 0
)

echo.
echo   Guardians Report
echo   ----------------
echo   Starting on http://127.0.0.1:%PORT%
echo   Close this window to stop the server.
echo.

rem Open the browser slightly after the server starts. `start` returns
rem immediately, so the wait happens in a throwaway shell rather than blocking
rem the server itself.
start "" cmd /c "timeout /t 3 >nul & start """" ""http://127.0.0.1:%PORT%"""

"%PYTHON%" -m guards_report.cli serve --port %PORT%

rem Only reached if the server exits. Hold the window so the error is readable
rem instead of vanishing.
echo.
echo   Server stopped.
pause
