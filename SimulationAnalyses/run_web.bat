@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Virtual env not found: %PY%
  echo Run: python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

set "CHROME="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
  set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
) else if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
  set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
)

if defined CHROME (
  start "" "%CHROME%" "http://127.0.0.1:8001"
) else (
  start "" chrome "http://127.0.0.1:8001"
)

echo Starting web server on http://127.0.0.1:8001
echo Press Ctrl+C to stop.
"%PY%" "%ROOT%webapp.py" --port 8001
