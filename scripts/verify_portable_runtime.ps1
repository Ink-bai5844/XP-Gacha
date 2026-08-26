[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseRoot
)

$ErrorActionPreference = "Stop"
$resolvedRoot = [System.IO.Path]::GetFullPath($ReleaseRoot)
$python = Join-Path $resolvedRoot "runtime\python\python.exe"
$jobModule = Join-Path $resolvedRoot "server\job_tasks.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Portable Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $jobModule -PathType Leaf)) {
    throw "Portable job module is missing: $jobModule"
}

$oldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldNoUserSite = $env:PYTHONNOUSERSITE
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

try {
    Push-Location $resolvedRoot
    try {
        $probe = @'
import importlib.util
import pathlib
import sys

release_root = pathlib.Path(sys.argv[1]).resolve()
search_roots = {pathlib.Path(item).resolve() for item in sys.path if item}
if release_root not in search_roots:
    raise SystemExit(f"release root missing from sys.path: {release_root}")
if importlib.util.find_spec("server.job_tasks") is None:
    raise SystemExit("server.job_tasks is not importable")
print("portable module resolution OK")
'@
        $probePath = Join-Path ([System.IO.Path]::GetTempPath()) ("xp-gacha-portable-probe-" + [Guid]::NewGuid().ToString("N") + ".py")
        [System.IO.File]::WriteAllText($probePath, $probe, [System.Text.UTF8Encoding]::new($false))
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & $python -X utf8 $probePath $resolvedRoot 2>&1
            $probeExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $oldErrorActionPreference
            Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
        }
        if ($probeExitCode -ne 0) {
            throw "Portable job-module verification failed.`n$($output -join [Environment]::NewLine)"
        }
        $output
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:PYTHONDONTWRITEBYTECODE = $oldNoBytecode
    $env:PYTHONNOUSERSITE = $oldNoUserSite
    $env:PYTHONUTF8 = $oldUtf8
}
