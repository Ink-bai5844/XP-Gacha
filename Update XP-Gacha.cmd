@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul 2>&1
for %%I in ("%~dp0.") do set "UPDATE_ROOT=%%~fI"
cd /d "%UPDATE_ROOT%"

set "UPDATE_SOURCE=%UPDATE_ROOT%\tools\update_xp_gacha.ps1"
set "UPDATE_WORKER=%TEMP%\XP-Gacha-Updater-%RANDOM%-%RANDOM%.ps1"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "PSModulePath=%USERPROFILE%\Documents\WindowsPowerShell\Modules;%ProgramFiles%\WindowsPowerShell\Modules;%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"

if not exist "%UPDATE_SOURCE%" (
  echo [XP-Gacha Update] Missing updater: "%UPDATE_SOURCE%"
  set "UPDATE_EXIT=2"
  goto :finish
)

if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"
copy /y "%UPDATE_SOURCE%" "%UPDATE_WORKER%" >nul
if errorlevel 1 (
  echo [XP-Gacha Update] Could not create the temporary update worker.
  set "UPDATE_EXIT=2"
  goto :finish
)

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%UPDATE_WORKER%" -PackageRoot "%UPDATE_ROOT%" %*
set "UPDATE_EXIT=%ERRORLEVEL%"
del /q "%UPDATE_WORKER%" >nul 2>&1

:finish
if not defined UPDATE_EXIT set "UPDATE_EXIT=1"
if not defined XP_GACHA_NO_PAUSE pause
exit /b %UPDATE_EXIT%
