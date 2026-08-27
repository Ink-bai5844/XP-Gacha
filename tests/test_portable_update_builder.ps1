$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$builder = Join-Path $projectRoot "scripts\build_portable_update.ps1"

& $builder -SelfTest
