$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop first."
}
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Update paths and LLM settings if needed."
}
docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

docker compose ps
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "XP-Gacha started: http://127.0.0.1:8000"
Write-Host "On first use, open Appendix to import project data, ZIP, or CSV files."
