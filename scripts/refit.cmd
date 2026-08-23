@echo off
REM Weekly refit, for Windows Task Scheduler.
REM
REM Register it with (run as your own user, from the project root):
REM
REM   schtasks /create /tn "Guards report refit" /tr "\"%CD%\scripts\refit.cmd\"" ^
REM            /sc weekly /d MON /st 04:00 /f
REM
REM Add /ru "%USERNAME%" /rp to have it run when you are not logged in, and set
REM "Wake the computer to run this task" in Task Scheduler if the machine sleeps
REM -- a task that never fires looks exactly like a task that had nothing to do.
REM
REM Exit codes are meaningful and Task Scheduler records them:
REM   0  refit ran and was kept
REM   1  something failed
REM   2  refit ran and was rejected; the previous model is still in place

setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Cannot find .venv\Scripts\python.exe -- is this the project root?
  exit /b 1
)

".venv\Scripts\python.exe" "scripts\refit.py" %*
set CODE=%ERRORLEVEL%

REM Surfaced here as well as in data\refit.log, because a scheduled task that
REM only writes to a file it never tells you about is a task you will not read.
if %CODE% NEQ 0 echo Refit finished with exit code %CODE% -- see data\refit.log

exit /b %CODE%
