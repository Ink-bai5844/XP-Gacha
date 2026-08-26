$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $projectRoot "scripts\start.ps1"
$quotedStartScript = $startScript.Replace("'", "''")
$testCommand = "function global:docker { `$global:LASTEXITCODE = 37 }; & '$quotedStartScript'; exit `$LASTEXITCODE"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($testCommand))
$output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedCommand 2>&1
$exitCode = $LASTEXITCODE
$outputText = $output -join [Environment]::NewLine

if ($exitCode -ne 37) {
    throw "Expected start.ps1 to preserve Docker exit code 37, got $exitCode.`n$outputText"
}
if ($outputText -match "XP-Gacha started") {
    throw "start.ps1 printed a success message after Docker failed."
}

Write-Host "start.ps1 preserves Docker failures."
