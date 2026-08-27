[CmdletBinding()]
param(
    [string]$ReleaseRoot,
    [string]$OutputRoot,
    [string]$Version,
    [switch]$Force,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version 2.0

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))

$ProtectedPaths = @(
    "runtime",
    ".env",
    ".env.local",
    ".git",
    ".pytest_cache",
    ".streamlit",
    "__pycache__",
    "userdata",
    "data",
    "datacache",
    "b64_cache",
    "b64_tmp",
    "localimgtmp",
    "onlineimgtmp",
    "library",
    "manga_vectors",
    "models",
    "mysql",
    "config",
    "run",
    "updates",
    "logs",
    "tmp",
    "dictionaries",
    "portable-settings.env",
    "web/node_modules",
    "tools/error.json"
)
$ReplaceDirectories = @("web/dist")
$RequiredUpdateFiles = @(
    "BUILD-INFO.json",
    "SHA256SUMS.txt",
    "portable_launcher.py",
    "Update XP-Gacha.cmd",
    "tools/update_xp_gacha.ps1",
    "web/dist/index.html"
)
$AllowedProgramDirectories = @("Integration", "data_get", "data_processing", "server", "tools", "UI-imgs")
$AllowedRootFiles = @(
    "app.py", "config_docker.py", "config_empty.py", "config.py", "data_pipeline.py",
    "launcher.py", "ui_data_processing.py", "utils_charts.py", "utils_chat.py", "utils_core.py",
    "utils_cv.py", "utils_history.py", "utils_nlp.py", "utils_online_cover.py", "requirements.txt",
    "requirements-lock.txt", "README.md", "LICENSE", "BUILD-INFO.json", "SHA256SUMS.txt",
    "portable_launcher.py", "Start XP-Gacha.cmd", "Stop XP-Gacha.cmd", "Check XP-Gacha.cmd",
    "Open XP-Gacha Folder.cmd", "Update XP-Gacha.cmd",
    ("README_" + [char]0x4FBF + [char]0x643A + [char]0x7248 + ".md"),
    "THIRD_PARTY_NOTICES.md"
)

function Write-UpdateStep {
    param([string]$Message)
    Write-Host "[portable-update-build] $Message"
}

function Get-LowerSha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StreamSha256 {
    param([System.IO.Stream]$Stream)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $algorithm.ComputeHash($Stream)
        return ([System.BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-OptionalProperty {
    param(
        [object]$Object,
        [string]$Name,
        [object]$DefaultValue = $null
    )
    if ($null -eq $Object) {
        return $DefaultValue
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return $DefaultValue
    }
    return $property.Value
}

function Get-PortableRelativePath {
    param(
        [string]$FullPath,
        [string]$RootPath
    )
    $resolvedRoot = [System.IO.Path]::GetFullPath($RootPath).TrimEnd('\', '/')
    $resolvedPath = [System.IO.Path]::GetFullPath($FullPath)
    $prefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the release root: $resolvedPath"
    }
    $relativePath = $resolvedPath.Substring($prefix.Length).Replace('\', '/')
    Assert-SafePortablePath -Path $relativePath
    return $relativePath
}

function Assert-SafePortablePath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Update paths cannot be empty."
    }
    if ($Path.Contains("\") -or $Path.StartsWith("/") -or $Path.Contains(":")) {
        throw "Update path is not a relative POSIX path: $Path"
    }
    foreach ($segment in $Path.Split('/')) {
        if ([string]::IsNullOrWhiteSpace($segment) -or $segment -eq "." -or $segment -eq "..") {
            throw "Update path contains an unsafe segment: $Path"
        }
    }
}

function Test-ProtectedPortablePath {
    param([string]$Path)
    $normalized = $Path.Replace('\', '/').Trim('/')
    foreach ($segment in $normalized.Split('/')) {
        if ($segment.Equals("__pycache__", [System.StringComparison]::OrdinalIgnoreCase) -or
            $segment.Equals(".pytest_cache", [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    if ($normalized.StartsWith("data_processing/", [System.StringComparison]::OrdinalIgnoreCase) -and
        [System.IO.Path]::GetExtension($normalized) -in @('.jsonl', '.csv', '.zip', '.pkl', '.pickle', '.parquet')) {
        return $true
    }
    foreach ($protectedPath in $ProtectedPaths) {
        if ($normalized.Equals($protectedPath, [System.StringComparison]::OrdinalIgnoreCase) -or
            $normalized.StartsWith($protectedPath + "/", [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Test-ManagedPortablePath {
    param([string]$Path)
    $normalized = $Path.Replace('\', '/').Trim('/')
    if (Test-ProtectedPortablePath -Path $normalized) {
        return $false
    }
    if ($normalized.StartsWith("web/dist/", [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    if (-not $normalized.Contains('/')) {
        return $normalized -in $AllowedRootFiles
    }
    return $normalized.Split('/')[0] -in $AllowedProgramDirectories
}

function Assert-OutputOutsideReleaseRoot {
    param(
        [string]$ResolvedReleaseRoot,
        [string]$ResolvedOutputRoot
    )
    $releasePrefix = $ResolvedReleaseRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if ($ResolvedOutputRoot.Equals($ResolvedReleaseRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $ResolvedOutputRoot.StartsWith($releasePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "OutputRoot must not be the release root or one of its descendants."
    }
}

function Get-UpdateSourceFiles {
    param([string]$RootPath)
    $directoryStack = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
    $directoryStack.Push((Get-Item -LiteralPath $RootPath))
    while ($directoryStack.Count -gt 0) {
        $directory = $directoryStack.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $directory.FullName -Force) {
            $relativePath = Get-PortableRelativePath -FullPath $item.FullName -RootPath $RootPath
            if (Test-ProtectedPortablePath -Path $relativePath) {
                continue
            }
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Update snapshots do not accept reparse points: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $directoryStack.Push($item)
            }
            else {
                if (-not (Test-ManagedPortablePath -Path $relativePath)) {
                    throw "The complete release contains an unknown application-layer file: $relativePath"
                }
                Write-Output $item
            }
        }
    }
}

function New-PortableUpdatePackage {
    param(
        [string]$SourceRoot,
        [string]$DestinationRoot,
        [string]$TargetVersion,
        [switch]$ReplaceExisting,
        [switch]$SkipTagVerification
    )

    $resolvedReleaseRoot = [System.IO.Path]::GetFullPath($SourceRoot).TrimEnd('\', '/')
    if (-not (Test-Path -LiteralPath $resolvedReleaseRoot -PathType Container)) {
        throw "ReleaseRoot does not exist: $resolvedReleaseRoot"
    }
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\', '/')
    Assert-OutputOutsideReleaseRoot -ResolvedReleaseRoot $resolvedReleaseRoot -ResolvedOutputRoot $resolvedOutputRoot
    New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null

    $buildInfoPath = Join-Path $resolvedReleaseRoot "BUILD-INFO.json"
    $requirementsLockPath = Join-Path $resolvedReleaseRoot "requirements-lock.txt"
    if (-not (Test-Path -LiteralPath $buildInfoPath -PathType Leaf)) {
        throw "The complete portable release is missing BUILD-INFO.json."
    }
    if (-not (Test-Path -LiteralPath $requirementsLockPath -PathType Leaf)) {
        throw "The complete portable release is missing requirements-lock.txt."
    }

    $buildInfo = [System.IO.File]::ReadAllText(
        $buildInfoPath,
        [System.Text.Encoding]::UTF8
    ) | ConvertFrom-Json
    $sourceCommit = [string](Get-OptionalProperty -Object $buildInfo -Name "sourceCommit" -DefaultValue "")
    $sourceDirty = [bool](Get-OptionalProperty -Object $buildInfo -Name "sourceDirty" -DefaultValue $true)
    if ($sourceDirty) {
        throw "Refusing to publish automatic update assets from a dirty source tree. Commit the release first."
    }
    if ($sourceCommit -notmatch '^[0-9a-fA-F]{40}$') {
        throw "BUILD-INFO.json must contain the clean release source commit."
    }
    if (-not [bool](Get-OptionalProperty -Object $buildInfo -Name "blankFirstStartVerified" -DefaultValue $false)) {
        throw "Automatic update assets require a portable release that passed the blank first-start verification."
    }
    $metadataVersion = [string](Get-OptionalProperty -Object $buildInfo -Name "version" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($TargetVersion)) {
        $TargetVersion = $metadataVersion
    }
    elseif (-not [string]::IsNullOrWhiteSpace($metadataVersion) -and $TargetVersion -ne $metadataVersion) {
        throw "Requested version $TargetVersion does not match BUILD-INFO.json version $metadataVersion."
    }
    if ($TargetVersion -notmatch '^\d+\.\d+\.\d+$') {
        throw "Portable update versions must use numeric semantic versioning (for example, 0.2.7): $TargetVersion"
    }
    if (-not $SkipTagVerification) {
        $tagRef = "refs/tags/v$TargetVersion^{}"
        $tagCommit = (& git -C $ProjectRoot rev-parse $tagRef 2>$null)
        if ($LASTEXITCODE -ne 0 -or ([string]$tagCommit).Trim() -ne $sourceCommit) {
            throw "Tag v$TargetVersion must exist and resolve to BUILD-INFO sourceCommit $sourceCommit before update assets are built."
        }
    }

    $product = [string](Get-OptionalProperty -Object $buildInfo -Name "product" -DefaultValue "XP-Gacha")
    if ($product -ne "XP-Gacha") {
        throw "Unexpected release product: $product"
    }
    $runtime = Get-OptionalProperty -Object $buildInfo -Name "runtime"
    $python = Get-OptionalProperty -Object $runtime -Name "python"
    $mysql = Get-OptionalProperty -Object $runtime -Name "mysql"
    $pythonVersion = [string](Get-OptionalProperty -Object $python -Name "version" -DefaultValue "")
    $mysqlVersion = [string](Get-OptionalProperty -Object $mysql -Name "version" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($pythonVersion) -or [string]::IsNullOrWhiteSpace($mysqlVersion)) {
        throw "BUILD-INFO.json is missing the Python or MySQL runtime version."
    }

    $releaseName = "XP-Gacha-v$TargetVersion-portable-win64"
    $packageAsset = "$releaseName-update.zip"
    $manifestAsset = "$releaseName-update.json"
    $packagePath = Join-Path $resolvedOutputRoot $packageAsset
    $manifestPath = Join-Path $resolvedOutputRoot $manifestAsset
    $sidecarPath = "$packagePath.sha256"
    $finalArtifacts = @($packagePath, $manifestPath, $sidecarPath)
    $collisions = @($finalArtifacts | Where-Object { Test-Path -LiteralPath $_ })
    if ($collisions.Count -gt 0 -and -not $ReplaceExisting) {
        throw "Update assets already exist for version $TargetVersion. Use -Force to replace the complete asset set."
    }
    if ($ReplaceExisting) {
        foreach ($artifact in $finalArtifacts) {
            if (Test-Path -LiteralPath $artifact) {
                Remove-Item -LiteralPath $artifact -Force
            }
        }
    }

    $operationId = [Guid]::NewGuid().ToString("N")
    $snapshotRoot = Join-Path $resolvedOutputRoot ".update-snapshot-$operationId"
    $temporaryPackage = Join-Path $resolvedOutputRoot ".update-package-$operationId.zip"
    $temporaryManifest = Join-Path $resolvedOutputRoot ".update-manifest-$operationId.json"
    $temporarySidecar = Join-Path $resolvedOutputRoot ".update-package-$operationId.zip.sha256"
    $published = [System.Collections.Generic.List[string]]::new()

    try {
        New-Item -ItemType Directory -Path $snapshotRoot -Force | Out-Null
        $caseInsensitivePaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        $sourceFiles = @(Get-UpdateSourceFiles -RootPath $resolvedReleaseRoot)
        foreach ($sourceFile in $sourceFiles) {
            $relativePath = Get-PortableRelativePath -FullPath $sourceFile.FullName -RootPath $resolvedReleaseRoot
            if (-not $caseInsensitivePaths.Add($relativePath)) {
                throw "The release contains duplicate update paths that differ only by case: $relativePath"
            }
            $snapshotPath = Join-Path $snapshotRoot $relativePath.Replace('/', '\')
            $snapshotParent = Split-Path -Parent $snapshotPath
            New-Item -ItemType Directory -Path $snapshotParent -Force | Out-Null
            [System.IO.File]::Copy($sourceFile.FullName, $snapshotPath, $false)
        }

        $snapshotFiles = @(
            Get-ChildItem -LiteralPath $snapshotRoot -File -Recurse -Force |
                ForEach-Object {
                    [pscustomobject]@{
                        File = $_
                        Path = Get-PortableRelativePath -FullPath $_.FullName -RootPath $snapshotRoot
                    }
                } |
                Sort-Object -Property Path
        )
        if ($snapshotFiles.Count -eq 0) {
            throw "The release did not contain any application files for the update snapshot."
        }
        $snapshotPaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($snapshotFile in $snapshotFiles) {
            [void]$snapshotPaths.Add($snapshotFile.Path)
        }
        foreach ($requiredUpdateFile in $RequiredUpdateFiles) {
            if (-not $snapshotPaths.Contains($requiredUpdateFile)) {
                throw "The update snapshot is missing required application file: $requiredUpdateFile"
            }
        }

        $fileEntries = [System.Collections.Generic.List[object]]::new()
        foreach ($snapshotFile in $snapshotFiles) {
            if (Test-ProtectedPortablePath -Path $snapshotFile.Path) {
                throw "Protected path reached the update snapshot: $($snapshotFile.Path)"
            }
            $fileEntries.Add([ordered]@{
                path = $snapshotFile.Path
                sha256 = Get-LowerSha256 -Path $snapshotFile.File.FullName
                size = [Int64]$snapshotFile.File.Length
            })
        }

        Write-UpdateStep "Creating application-layer update ZIP for v$TargetVersion."
        $archive = [System.IO.Compression.ZipFile]::Open($temporaryPackage, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($snapshotFile in $snapshotFiles) {
                $entry = $archive.CreateEntry($snapshotFile.Path, [System.IO.Compression.CompressionLevel]::Optimal)
                $inputStream = [System.IO.File]::OpenRead($snapshotFile.File.FullName)
                $entryStream = $entry.Open()
                try {
                    $inputStream.CopyTo($entryStream)
                }
                finally {
                    $entryStream.Dispose()
                    $inputStream.Dispose()
                }
            }
        }
        finally {
            $archive.Dispose()
        }

        $packageSha256 = Get-LowerSha256 -Path $temporaryPackage
        $packageSize = [Int64](Get-Item -LiteralPath $temporaryPackage).Length
        $pythonArchiveSha256 = [string](Get-OptionalProperty -Object $python -Name "archiveSha256" -DefaultValue "")
        $mysqlArchiveSha256 = [string](Get-OptionalProperty -Object $mysql -Name "archiveSha256" -DefaultValue "")
        $visualCpp = Get-OptionalProperty -Object $runtime -Name "visualCppRuntime"
        $visualCppInstallerSha256 = [string](Get-OptionalProperty -Object $visualCpp -Name "installerSha256" -DefaultValue "")

        $manifest = [ordered]@{
            schema = 1
            product = "XP-Gacha"
            version = $TargetVersion
            platform = "windows-x64"
            updateKind = "portable-app-layer"
            packageAsset = $packageAsset
            packageSha256 = $packageSha256
            packageSize = $packageSize
            runtimeCompatibility = [ordered]@{
                requirementsLockSha256 = Get-LowerSha256 -Path $requirementsLockPath
                pythonVersion = $pythonVersion
                pythonArchiveSha256 = $pythonArchiveSha256
                mysqlVersion = $mysqlVersion
                mysqlArchiveSha256 = $mysqlArchiveSha256
                visualCppRuntimeInstallerSha256 = $visualCppInstallerSha256
            }
            files = @($fileEntries)
            replaceDirectories = @($ReplaceDirectories)
            protectedPaths = @($ProtectedPaths)
            sourceCommit = $sourceCommit
            sourceDirty = $sourceDirty
            createdAtUtc = [DateTime]::UtcNow.ToString("o")
        }
        $manifestJson = $manifest | ConvertTo-Json -Depth 8
        [System.IO.File]::WriteAllText($temporaryManifest, $manifestJson + "`n", [System.Text.UTF8Encoding]::new($false))
        [System.IO.File]::WriteAllText(
            $temporarySidecar,
            "$packageSha256  $packageAsset`n",
            [System.Text.UTF8Encoding]::new($false)
        )

        Move-Item -LiteralPath $temporaryPackage -Destination $packagePath
        $published.Add($packagePath)
        Move-Item -LiteralPath $temporaryManifest -Destination $manifestPath
        $published.Add($manifestPath)
        Move-Item -LiteralPath $temporarySidecar -Destination $sidecarPath
        $published.Add($sidecarPath)

        Write-UpdateStep "Update assets complete."
        return [pscustomobject]@{
            PackagePath = $packagePath
            ManifestPath = $manifestPath
            SidecarPath = $sidecarPath
            PackageSha256 = $packageSha256
            PackageSize = $packageSize
            FileCount = $fileEntries.Count
        }
    }
    catch {
        foreach ($publishedPath in $published) {
            Remove-Item -LiteralPath $publishedPath -Force -ErrorAction SilentlyContinue
        }
        throw
    }
    finally {
        Remove-Item -LiteralPath $snapshotRoot -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $temporaryPackage -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $temporaryManifest -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $temporarySidecar -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-PortableUpdateSelfTest {
    $testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("xp-gacha-update-selftest-" + [Guid]::NewGuid().ToString("N"))
    $fixtureRoot = Join-Path $testRoot "XP-Gacha-v9.8.7-portable-win64"
    $outputRoot = Join-Path $testRoot "out"
    try {
        New-Item -ItemType Directory -Path $fixtureRoot, $outputRoot -Force | Out-Null
        $fixtureFiles = [ordered]@{
            "app.py" = "print('fixture')`n"
            "server/main.py" = "fixture = True`n"
            "web/dist/index.html" = "<!doctype html><title>fixture</title>`n"
            "web/dist/assets/app.js" = "console.log('fixture');`n"
            "requirements-lock.txt" = "fixture-package==1.0`n"
            "SHA256SUMS.txt" = "fixture manifest`n"
            "portable_launcher.py" = "print('portable fixture')`n"
            "Update XP-Gacha.cmd" = "@echo off`r`n"
            "tools/update_xp_gacha.ps1" = "Write-Host 'update fixture'`n"
            "runtime/python/python.exe" = "protected runtime"
            "data/private.csv" = "protected data"
            "datacache/imports/private.json" = "protected cache"
            "dictionaries/custom.txt" = "protected dictionary"
            "portable-settings.env" = "LLM_API_KEY=secret"
            "tools/error.json" = "{`"private`":true}"
            "updates/backups/old-app.py" = "protected update backup"
            "logs/private.log" = "protected log"
        }
        $fixtureFiles[("README_" + [char]0x4FBF + [char]0x643A + [char]0x7248 + ".md")] = "portable readme`n"
        foreach ($fixtureEntry in $fixtureFiles.GetEnumerator()) {
            $fixturePath = Join-Path $fixtureRoot $fixtureEntry.Key.Replace('/', '\')
            New-Item -ItemType Directory -Path (Split-Path -Parent $fixturePath) -Force | Out-Null
            [System.IO.File]::WriteAllText($fixturePath, $fixtureEntry.Value, [System.Text.UTF8Encoding]::new($false))
        }
        $buildInfo = [ordered]@{
            product = "XP-Gacha"
            version = "9.8.7"
            sourceCommit = "0123456789abcdef0123456789abcdef01234567"
            sourceDirty = $false
            blankFirstStartVerified = $true
            runtime = [ordered]@{
                python = [ordered]@{
                    version = "3.13.15"
                    archiveSha256 = ("a" * 64)
                }
                mysql = [ordered]@{
                    version = "8.4.11"
                    archiveSha256 = ("b" * 64)
                }
                visualCppRuntime = [ordered]@{
                    installerSha256 = ("c" * 64)
                }
            }
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $fixtureRoot "BUILD-INFO.json"),
            ($buildInfo | ConvertTo-Json -Depth 6) + "`n",
            [System.Text.UTF8Encoding]::new($false)
        )

        $result = New-PortableUpdatePackage -SourceRoot $fixtureRoot -DestinationRoot $outputRoot `
            -TargetVersion "9.8.7" -SkipTagVerification
        foreach ($requiredOutput in @($result.PackagePath, $result.ManifestPath, $result.SidecarPath)) {
            if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf)) {
                throw "Self-test output is missing: $requiredOutput"
            }
        }

        $manifest = [System.IO.File]::ReadAllText(
            $result.ManifestPath,
            [System.Text.Encoding]::UTF8
        ) | ConvertFrom-Json
        if ($manifest.schema -ne 1 -or $manifest.product -ne "XP-Gacha" -or $manifest.version -ne "9.8.7") {
            throw "Self-test manifest identity fields are invalid."
        }
        if ($manifest.packageAsset -ne (Split-Path -Leaf $result.PackagePath)) {
            throw "Self-test package asset name is invalid."
        }
        if ($manifest.packageSha256 -ne (Get-LowerSha256 -Path $result.PackagePath)) {
            throw "Self-test package hash does not match the ZIP."
        }
        if ([Int64]$manifest.packageSize -ne [Int64](Get-Item -LiteralPath $result.PackagePath).Length) {
            throw "Self-test package size does not match the ZIP."
        }
        if ($manifest.runtimeCompatibility.requirementsLockSha256 -ne (Get-LowerSha256 -Path (Join-Path $fixtureRoot "requirements-lock.txt"))) {
            throw "Self-test requirements lock hash is invalid."
        }
        if ($manifest.runtimeCompatibility.pythonVersion -ne "3.13.15" -or $manifest.runtimeCompatibility.mysqlVersion -ne "8.4.11") {
            throw "Self-test runtime version compatibility fields are invalid."
        }
        foreach ($manifestFile in @($manifest.files)) {
            if ($manifestFile.path.Contains("\")) {
                throw "Self-test manifest contains a non-POSIX path: $($manifestFile.path)"
            }
            if (Test-ProtectedPortablePath -Path $manifestFile.path) {
                throw "Self-test manifest includes protected content: $($manifestFile.path)"
            }
        }
        if (@($manifest.replaceDirectories).Count -ne 1 -or $manifest.replaceDirectories[0] -ne "web/dist") {
            throw "Self-test replaceDirectories is invalid."
        }
        foreach ($requiredProtectedPath in $ProtectedPaths) {
            if ($requiredProtectedPath -notin @($manifest.protectedPaths)) {
                throw "Self-test protectedPaths is missing $requiredProtectedPath."
            }
        }

        $manifestFilesByPath = @{}
        foreach ($manifestFile in @($manifest.files)) {
            $manifestFilesByPath[[string]$manifestFile.path] = $manifestFile
        }
        $zip = [System.IO.Compression.ZipFile]::OpenRead($result.PackagePath)
        try {
            $zipFiles = @($zip.Entries | Where-Object { -not [string]::IsNullOrEmpty($_.Name) })
            if ($zipFiles.Count -ne $manifestFilesByPath.Count) {
                throw "Self-test ZIP and manifest file counts differ."
            }
            foreach ($entry in $zipFiles) {
                if ($entry.FullName.Contains("\")) {
                    throw "Self-test ZIP contains a non-POSIX path: $($entry.FullName)"
                }
                if (Test-ProtectedPortablePath -Path $entry.FullName) {
                    throw "Self-test ZIP includes protected content: $($entry.FullName)"
                }
                if (-not $manifestFilesByPath.ContainsKey($entry.FullName)) {
                    throw "Self-test ZIP entry is missing from the manifest: $($entry.FullName)"
                }
                $entryStream = $entry.Open()
                try {
                    $entryHash = Get-StreamSha256 -Stream $entryStream
                }
                finally {
                    $entryStream.Dispose()
                }
                $manifestFile = $manifestFilesByPath[$entry.FullName]
                if ($entryHash -ne $manifestFile.sha256 -or [Int64]$entry.Length -ne [Int64]$manifestFile.size) {
                    throw "Self-test ZIP entry hash or size mismatch: $($entry.FullName)"
                }
            }
        }
        finally {
            $zip.Dispose()
        }

        $expectedSidecar = "$($manifest.packageSha256)  $($manifest.packageAsset)"
        if ((Get-Content -LiteralPath $result.SidecarPath -Raw).Trim() -ne $expectedSidecar) {
            throw "Self-test SHA256 sidecar is invalid."
        }

        $collisionRejected = $false
        try {
            New-PortableUpdatePackage -SourceRoot $fixtureRoot -DestinationRoot $outputRoot `
                -TargetVersion "9.8.7" -SkipTagVerification | Out-Null
        }
        catch {
            $collisionRejected = $_.Exception.Message -like "*already exist*"
        }
        if (-not $collisionRejected) {
            throw "Self-test did not reject a same-version asset collision."
        }
        New-PortableUpdatePackage -SourceRoot $fixtureRoot -DestinationRoot $outputRoot `
            -TargetVersion "9.8.7" -ReplaceExisting -SkipTagVerification | Out-Null
        Write-Host "[portable-update-build] Self-test passed."
    }
    finally {
        Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($SelfTest) {
    Invoke-PortableUpdateSelfTest
    return
}

if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    throw "ReleaseRoot is required unless -SelfTest is used."
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Split-Path -Parent ([System.IO.Path]::GetFullPath($ReleaseRoot))
}

New-PortableUpdatePackage `
    -SourceRoot $ReleaseRoot `
    -DestinationRoot $OutputRoot `
    -TargetVersion $Version `
    -ReplaceExisting:$Force
