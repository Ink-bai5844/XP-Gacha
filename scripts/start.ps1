$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop first."
}
if (Test-Path -LiteralPath ".env" -PathType Container) {
    throw ".env is a directory. Rename or remove that directory, then run this script again."
}
if (-not (Test-Path -LiteralPath ".env" -PathType Leaf)) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Update paths and LLM settings if needed."
}
docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$AppAddress = (docker compose port app 8000).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($AppAddress)) {
    throw "Could not determine the published XP-Gacha port."
}

$AppUrl = "http://$AppAddress"
$HealthUrl = "$AppUrl/"
$HealthDeadline = [DateTime]::UtcNow.AddSeconds(180)
$AppReady = $false
Write-Host "Waiting for XP-Gacha web service..."

while ([DateTime]::UtcNow -lt $HealthDeadline) {
    try {
        $HealthResponse = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 3
        if ($HealthResponse.StatusCode -eq 200) {
            $AppReady = $true
            break
        }
    }
    catch {
        # The application may still be starting; retry until the deadline.
    }
    Start-Sleep -Seconds 1
}

if (-not $AppReady) {
    Write-Host "XP-Gacha did not become healthy within 180 seconds."
    docker compose ps
    docker compose logs --tail 120 app
    throw "XP-Gacha application health check failed. Review the logs above."
}

docker compose ps
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "XP-Gacha started: $AppUrl"
Write-Host "On first use, open Appendix to import project data, ZIP, or CSV files."
