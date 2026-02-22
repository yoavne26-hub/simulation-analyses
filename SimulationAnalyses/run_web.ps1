$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  Write-Host "[ERROR] Virtual env not found: $python"
  Write-Host "Run: python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt"
  exit 1
}

$chrome = $null
$chrome64 = Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"
$chrome86 = Join-Path $env:ProgramFiles(x86) "Google\Chrome\Application\chrome.exe"
if (Test-Path $chrome64) {
  $chrome = $chrome64
} elseif (Test-Path $chrome86) {
  $chrome = $chrome86
}

if ($chrome) {
  Start-Process -FilePath $chrome -ArgumentList "http://127.0.0.1:8001"
} else {
  Start-Process -FilePath "chrome" -ArgumentList "http://127.0.0.1:8001"
}

Write-Host "Starting web server on http://127.0.0.1:8001"
Write-Host "Press Ctrl+C to stop."
& $python "$root\webapp.py" --port 8001
