@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "runtime\python\python.exe" exit /b 1
"runtime\python\python.exe" -X utf8 "portable_launcher.py" stop
if errorlevel 1 pause
exit /b %ERRORLEVEL%
