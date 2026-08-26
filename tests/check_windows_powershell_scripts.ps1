$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$scriptPaths = Get-ChildItem -Path (Join-Path $projectRoot "scripts") -Filter "*.ps1"
$hasErrors = $false

foreach ($scriptPath in $scriptPaths) {
    $tokens = $null
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $scriptPath.FullName,
        [ref]$tokens,
        [ref]$parseErrors
    ) | Out-Null

    if ($parseErrors.Count -gt 0) {
        $hasErrors = $true
        foreach ($parseError in $parseErrors) {
            [Console]::Error.WriteLine(
                "$($scriptPath.Name):$($parseError.Extent.StartLineNumber): $($parseError.Message)"
            )
        }
    }
}

if ($hasErrors) {
    exit 1
}

Write-Host "Windows PowerShell scripts parsed successfully."
