@echo off
setlocal
cd /d "%~dp0"
if not exist "userdata" mkdir "userdata"
start "" explorer.exe "%~dp0userdata"
