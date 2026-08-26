@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "runtime\python\python.exe" (
  echo The portable Python runtime is missing. Please extract the complete release archive first.
  pause
  exit /b 1
)
"runtime\python\python.exe" -X utf8 "portable_launcher.py" doctor
set "XP_GACHA_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %XP_GACHA_EXIT_CODE%
