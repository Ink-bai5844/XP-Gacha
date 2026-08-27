[CmdletBinding()]
param(
    [string]$PackageRoot = "",
    [switch]$CheckOnly,
    [switch]$SkipRestart,
    [switch]$SelfTest,
    [string]$ReleaseMetadataPath = "",
    [string]$AssetDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$script:Repository = "Ink-bai5844/XP-Gacha"
$script:RepositoryUrl = "https://github.com/Ink-bai5844/XP-Gacha.git"
$script:ApiUrl = "https://api.github.com/repos/Ink-bai5844/XP-Gacha/releases/latest"
$script:UpdaterSchemaVersion = 1
$script:Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$script:ProtectedTopLevels = @(
    "data", "datacache", "b64_cache", "b64_tmp", "localimgtmp", "onlineimgtmp",
    "library", "manga_vectors", "models", "mysql", "config", "run", "logs",
    "tmp", "updates", "dictionaries", "runtime", "userdata", ".git", ".streamlit"
)
$script:ProtectedFiles = @(
    ".env", ".env.local", "portable-settings.env", "tools/error.json",
    "data_processing/title_translation_results.jsonl"
)
$script:AllowedProgramDirectories = @(
    "Integration", "data_get", "data_processing", "server", "tools", "UI-imgs"
)
$script:AllowedRootFiles = @(
    "app.py", "config_docker.py", "config_empty.py", "config.py", "data_pipeline.py",
    "launcher.py", "ui_data_processing.py", "utils_charts.py", "utils_chat.py",
    "utils_core.py", "utils_cv.py", "utils_history.py", "utils_nlp.py",
    "utils_online_cover.py", "requirements.txt", "requirements-lock.txt", "README.md",
    "LICENSE", "BUILD-INFO.json", "SHA256SUMS.txt", "portable_launcher.py",
    "Start XP-Gacha.cmd", "Stop XP-Gacha.cmd", "Check XP-Gacha.cmd",
    "Open XP-Gacha Folder.cmd", "Update XP-Gacha.cmd",
    ("README_" + [char]0x4FBF + [char]0x643A + [char]0x7248 + ".md"),
    "THIRD_PARTY_NOTICES.md"
)

function Write-UpdateMessage {
    param([string]$Message)
    Write-Host "[XP-Gacha Update] $Message"
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth 12
    $temporary = Join-Path $parent ("." + [System.IO.Path]::GetFileName($Path) + "." + [Guid]::NewGuid().ToString("N") + ".tmp")
    $replaceBackup = Join-Path $parent ("." + [System.IO.Path]::GetFileName($Path) + "." + [Guid]::NewGuid().ToString("N") + ".bak")
    $bytes = $script:Utf8NoBom.GetBytes($json + "`n")
    try {
        $stream = [System.IO.File]::Open($temporary, [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try {
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush($true)
        }
        finally {
            $stream.Dispose()
        }
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            [System.IO.File]::Replace($temporary, $Path, $replaceBackup, $true)
            Remove-Item -LiteralPath $replaceBackup -Force -ErrorAction SilentlyContinue
        }
        else {
            [System.IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $replaceBackup -Force -ErrorAction SilentlyContinue
    }
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "JSON file not found: $Path"
    }
    try {
        return ([System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8) | ConvertFrom-Json)
    }
    catch {
        throw "Invalid JSON file '$Path': $($_.Exception.Message)"
    }
}

function Get-ObjectProperty {
    param([object]$Object, [string]$Name, [object]$Default = $null)
    if ($null -eq $Object) {
        return $Default
    }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) {
            return $Object[$Name]
        }
        return $Default
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $Default
    }
    return $property.Value
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function ConvertTo-ReleaseVersion {
    param([string]$Value)
    $plain = ([string]$Value).Trim()
    if ($plain.StartsWith("v", [System.StringComparison]::OrdinalIgnoreCase)) {
        $plain = $plain.Substring(1)
    }
    if ($plain -notmatch '^\d+\.\d+\.\d+$') {
        throw "Unsupported release version: $Value"
    }
    try {
        return [version]$plain
    }
    catch {
        throw "Unsupported release version: $Value"
    }
}

function Normalize-RelativePath {
    param([string]$Path)
    return ([string]$Path).Replace('\', '/').Trim()
}

function Assert-SafeRelativePath {
    param([string]$Path)
    $normalized = Normalize-RelativePath $Path
    if (-not $normalized -or $normalized.Length -gt 240) {
        throw "Unsafe update path: '$Path'"
    }
    if ($normalized.StartsWith('/') -or $normalized.StartsWith('//') -or $normalized.Contains(':')) {
        throw "Absolute paths and alternate streams are forbidden: '$Path'"
    }
    $segments = @($normalized.Split('/'))
    foreach ($segment in $segments) {
        if (-not $segment -or $segment -eq '.' -or $segment -eq '..') {
            throw "Path traversal is forbidden: '$Path'"
        }
        if ($segment.EndsWith('.') -or $segment.EndsWith(' ')) {
            throw "Trailing dots or spaces are forbidden: '$Path'"
        }
        $deviceName = ($segment.Split('.')[0]).ToUpperInvariant()
        if ($deviceName -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$') {
            throw "Reserved Windows device name in update path: '$Path'"
        }
    }
    return $normalized
}

function Test-ProtectedRelativePath {
    param([string]$Path)
    $normalized = (Normalize-RelativePath $Path).TrimStart('/')
    $segments = @($normalized.Split('/'))
    foreach ($segment in $segments) {
        if ($segment.Equals("__pycache__", [System.StringComparison]::OrdinalIgnoreCase) -or
            $segment.Equals(".pytest_cache", [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    foreach ($file in $script:ProtectedFiles) {
        if ($normalized.Equals($file, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    $top = $normalized.Split('/')[0]
    foreach ($protected in $script:ProtectedTopLevels) {
        if ($top.Equals($protected, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    if ($normalized.StartsWith("data_processing/", [System.StringComparison]::OrdinalIgnoreCase)) {
        $extension = [System.IO.Path]::GetExtension($normalized)
        if ($extension -in @('.jsonl', '.csv', '.zip', '.pkl', '.pickle', '.parquet')) {
            return $true
        }
    }
    return $false
}

function Test-ManagedRelativePath {
    param([string]$Path)
    try {
        $normalized = Assert-SafeRelativePath $Path
    }
    catch {
        return $false
    }
    if (Test-ProtectedRelativePath $normalized) {
        return $false
    }
    if ($normalized.StartsWith("web/dist/", [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    if (-not $normalized.Contains('/')) {
        foreach ($allowed in $script:AllowedRootFiles) {
            if ($normalized.Equals($allowed, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
        return $false
    }
    $top = $normalized.Split('/')[0]
    foreach ($allowedDirectory in $script:AllowedProgramDirectories) {
        if ($top.Equals($allowedDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Assert-ManagedRelativePath {
    param([string]$Path)
    $normalized = Assert-SafeRelativePath $Path
    if (-not (Test-ManagedRelativePath $normalized)) {
        throw "The update package attempted to write a protected or unknown path: '$normalized'"
    }
    return $normalized
}

function Get-ChildFullPath {
    param([string]$Root, [string]$RelativePath)
    $normalized = Assert-SafeRelativePath $RelativePath
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $fullRoot ($normalized.Replace('/', '\'))))
    $prefix = $fullRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escaped its expected root: '$RelativePath'"
    }
    return $candidate
}

function Assert-NoReparseParents {
    param([string]$Root, [string]$RelativePath)
    $normalized = Assert-SafeRelativePath $RelativePath
    $current = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $segments = @($normalized.Split('/'))
    for ($index = 0; $index -lt ($segments.Count - 1); $index += 1) {
        $current = Join-Path $current $segments[$index]
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "A managed path crosses a reparse point: '$RelativePath'"
            }
        }
    }
}

function Assert-NotReparseTarget {
    param([string]$Path, [string]$Description = "managed target")
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "The $Description is a reparse point and cannot be updated safely: $Path"
        }
    }
}

function Remove-VerifiedTree {
    param([string]$Path, [string]$Parent)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $fullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $prefix = $fullParent + [System.IO.Path]::DirectorySeparatorChar
    if ($fullPath -eq $fullParent -or -not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside its verified parent: $fullPath"
    }
    Assert-NotReparseTarget -Path $fullPath -Description "directory selected for removal"
    Remove-Item -LiteralPath $fullPath -Recurse -Force
}

function Assert-AbsoluteChildPath {
    param([string]$Path, [string]$Parent, [string]$Description = "path")
    $fullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $prefix = $fullParent + [System.IO.Path]::DirectorySeparatorChar
    if ($fullPath -eq $fullParent -or
        -not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe $Description outside '$fullParent': $fullPath"
    }
    return $fullPath
}

function Get-ReleaseMetadata {
    if ($ReleaseMetadataPath) {
        return Read-JsonFile ([System.IO.Path]::GetFullPath($ReleaseMetadataPath))
    }
    $headers = @{
        Accept = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "XP-Gacha-Updater/1"
    }
    if ($env:GITHUB_TOKEN) {
        $headers.Authorization = "Bearer $($env:GITHUB_TOKEN)"
    }
    $lastError = $null
    for ($attempt = 1; $attempt -le 3; $attempt += 1) {
        try {
            return Invoke-RestMethod -Uri $script:ApiUrl -Headers $headers -Method Get -TimeoutSec 30
        }
        catch {
            $lastError = $_
            if ($attempt -lt 3) {
                Start-Sleep -Seconds $attempt
            }
        }
    }
    throw "Could not query the latest GitHub Release: $($lastError.Exception.Message)"
}

function Assert-StableRelease {
    param([object]$Release)
    if ([bool](Get-ObjectProperty $Release "draft" $false)) {
        throw "The latest GitHub Release is still a draft."
    }
    if ([bool](Get-ObjectProperty $Release "prerelease" $false)) {
        throw "The latest GitHub Release is a prerelease; the stable updater will not install it."
    }
    $tag = [string](Get-ObjectProperty $Release "tag_name" "")
    if ($tag -notmatch '^v\d+\.\d+\.\d+$') {
        throw "The latest GitHub Release has an invalid tag: '$tag'"
    }
    return $tag
}

function Get-ReleaseAsset {
    param([object]$Release, [string]$Name)
    $matches = @((Get-ObjectProperty $Release "assets" @()) | Where-Object { ([string]$_.name) -ceq $Name })
    if ($matches.Count -ne 1) {
        throw "Release asset '$Name' was not found exactly once. Use the full portable package for this version."
    }
    $asset = $matches[0]
    $state = [string](Get-ObjectProperty $asset "state" "uploaded")
    if ($state -ne "uploaded") {
        throw "Release asset '$Name' is not ready for download."
    }
    if (-not $AssetDirectory) {
        $urlText = [string](Get-ObjectProperty $asset "browser_download_url" "")
        $uri = $null
        if (-not [Uri]::TryCreate($urlText, [UriKind]::Absolute, [ref]$uri)) {
            throw "Release asset '$Name' has an invalid URL."
        }
        $expectedPrefix = "/$($script:Repository)/releases/download/"
        if ($uri.Scheme -ne "https" -or $uri.Host -ne "github.com" -or
            -not $uri.AbsolutePath.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Release asset '$Name' does not use the official GitHub download path."
        }
    }
    return $asset
}

function Receive-ReleaseAsset {
    param([object]$Asset, [string]$Destination)
    $name = [string]$Asset.name
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $partial = $Destination + ".partial"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    try {
        if ($AssetDirectory) {
            $source = Join-Path ([System.IO.Path]::GetFullPath($AssetDirectory)) $name
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
                throw "Offline release asset not found: $source"
            }
            Copy-Item -LiteralPath $source -Destination $partial -Force
        }
        else {
            $headers = @{ "User-Agent" = "XP-Gacha-Updater/1" }
            if ($env:GITHUB_TOKEN) {
                $headers.Authorization = "Bearer $($env:GITHUB_TOKEN)"
            }
            $lastError = $null
            for ($attempt = 1; $attempt -le 3; $attempt += 1) {
                try {
                    Invoke-WebRequest -Uri ([string]$Asset.browser_download_url) -Headers $headers `
                        -UseBasicParsing -TimeoutSec 120 -OutFile $partial
                    $lastError = $null
                    break
                }
                catch {
                    $lastError = $_
                    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
                    if ($attempt -lt 3) {
                        Start-Sleep -Seconds $attempt
                    }
                }
            }
            if ($null -ne $lastError) {
                throw $lastError
            }
        }
        $expectedSize = [Int64](Get-ObjectProperty $Asset "size" 0)
        $actualSize = (Get-Item -LiteralPath $partial).Length
        if ($expectedSize -gt 0 -and $actualSize -ne $expectedSize) {
            throw "Release asset '$name' has the wrong size ($actualSize instead of $expectedSize)."
        }
        $digest = [string](Get-ObjectProperty $Asset "digest" "")
        if ($digest) {
            if ($digest -notmatch '^sha256:([0-9a-fA-F]{64})$') {
                throw "Release asset '$name' has an unsupported digest."
            }
            $actualHash = Get-Sha256 $partial
            if ($actualHash -ne $Matches[1].ToLowerInvariant()) {
                throw "Release asset '$name' failed its GitHub SHA-256 digest check."
            }
        }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
    }
    finally {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    }
    return $Destination
}

function Get-ValidatedManifest {
    param([string]$Path, [string]$ExpectedTag)
    $manifest = Read-JsonFile $Path
    $schema = [int](Get-ObjectProperty $manifest "schemaVersion" (Get-ObjectProperty $manifest "schema" 0))
    if ($schema -ne $script:UpdaterSchemaVersion) {
        throw "Unsupported portable update manifest schema."
    }
    if ([string](Get-ObjectProperty $manifest "product" "") -ne "XP-Gacha") {
        throw "The update manifest is for a different product."
    }
    if ([string](Get-ObjectProperty $manifest "updateKind" "") -ne "portable-app-layer") {
        throw "The release asset is not an XP-Gacha portable app-layer update."
    }
    if ([string](Get-ObjectProperty $manifest "platform" "") -ne "windows-x64") {
        throw "The portable update manifest targets a different platform."
    }
    if ([bool](Get-ObjectProperty $manifest "sourceDirty" $false)) {
        throw "The portable update was built from uncommitted source changes and is not eligible for automatic installation."
    }
    $sourceCommit = [string](Get-ObjectProperty $manifest "sourceCommit" "")
    if ($sourceCommit -notmatch '^[0-9a-fA-F]{40}$') {
        throw "The portable update manifest has no valid source commit."
    }
    $manifestVersion = ConvertTo-ReleaseVersion ([string](Get-ObjectProperty $manifest "version" ""))
    $tagVersion = ConvertTo-ReleaseVersion $ExpectedTag
    if ($manifestVersion -ne $tagVersion) {
        throw "The update manifest version does not match Release tag $ExpectedTag."
    }
    $manifestTag = [string](Get-ObjectProperty $manifest "tag" $ExpectedTag)
    if ($manifestTag -ne $ExpectedTag) {
        throw "The update manifest tag does not match Release tag $ExpectedTag."
    }

    $payload = Get-ObjectProperty $manifest "payload" $null
    $assetName = [string](Get-ObjectProperty $manifest "packageAsset" "")
    $payloadHash = ([string](Get-ObjectProperty $manifest "packageSha256" "")).ToLowerInvariant()
    $payloadSize = [Int64](Get-ObjectProperty $manifest "packageSize" 0)
    if ($null -ne $payload) {
        $assetName = [string](Get-ObjectProperty $payload "assetName" (Get-ObjectProperty $payload "name" $assetName))
        $payloadHash = ([string](Get-ObjectProperty $payload "sha256" $payloadHash)).ToLowerInvariant()
        $payloadSize = [Int64](Get-ObjectProperty $payload "size" $payloadSize)
    }
    if ($assetName -notmatch '^XP-Gacha-v\d+\.\d+\.\d+-portable-win64-update\.zip$' -or
        $payloadHash -notmatch '^[0-9a-f]{64}$' -or $payloadSize -le 0) {
        throw "The update manifest payload metadata is invalid."
    }

    $files = @(Get-ObjectProperty $manifest "files" @())
    if ($files.Count -eq 0 -or $files.Count -gt 20000) {
        throw "The update manifest has an invalid file count."
    }
    $fileMap = [System.Collections.Generic.Dictionary[string, object]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    [Int64]$totalSize = 0
    foreach ($file in $files) {
        $relative = Assert-ManagedRelativePath ([string](Get-ObjectProperty $file "path" ""))
        $hash = ([string](Get-ObjectProperty $file "sha256" "")).ToLowerInvariant()
        $size = [Int64](Get-ObjectProperty $file "size" -1)
        if ($hash -notmatch '^[0-9a-f]{64}$' -or $size -lt 0 -or $size -gt 268435456) {
            throw "Invalid hash or size for update file '$relative'."
        }
        if ($fileMap.ContainsKey($relative)) {
            throw "Duplicate update path (case-insensitive): '$relative'"
        }
        $entry = [pscustomobject]@{ path = $relative; sha256 = $hash; size = $size }
        $fileMap.Add($relative, $entry)
        $totalSize += $size
        if ($totalSize -gt 1073741824) {
            throw "The portable app-layer update is unexpectedly large."
        }
    }
    foreach ($required in @(
        "BUILD-INFO.json", "SHA256SUMS.txt", "portable_launcher.py",
        "Update XP-Gacha.cmd", "tools/update_xp_gacha.ps1", "web/dist/index.html"
    )) {
        if (-not $fileMap.ContainsKey($required)) {
            throw "The portable update is missing required program file '$required'."
        }
    }
    $replaceDirectories = @(Get-ObjectProperty $manifest "replaceDirectories" @("web/dist"))
    if ($replaceDirectories.Count -ne 1 -or
        -not ([string]$replaceDirectories[0]).Equals("web/dist", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Only web/dist may be replaced as a complete directory."
    }
    return [pscustomobject]@{
        Raw = $manifest
        Version = $manifestVersion
        AssetName = $assetName
        PayloadSha256 = $payloadHash
        PayloadSize = $payloadSize
        FileMap = $fileMap
        Files = @($fileMap.Values | Sort-Object path)
        TotalSize = $totalSize
        ReplaceDirectories = @("web/dist")
        RuntimeCompatibility = Get-ObjectProperty $manifest "runtimeCompatibility" $null
    }
}

function Assert-RuntimeCompatibility {
    param([string]$Root, [object]$BuildInfo, [object]$Compatibility)
    if ($null -eq $Compatibility) {
        throw "The update manifest has no runtime compatibility information."
    }
    $runtime = Get-ObjectProperty $BuildInfo "runtime" $null
    $localPython = [string](Get-ObjectProperty (Get-ObjectProperty $runtime "python" $null) "version" "")
    $localMySql = [string](Get-ObjectProperty (Get-ObjectProperty $runtime "mysql" $null) "version" "")
    $requiredPython = [string](Get-ObjectProperty $Compatibility "pythonVersion" "")
    $requiredMySql = [string](Get-ObjectProperty $Compatibility "mysqlVersion" "")
    $requiredLockHash = ([string](Get-ObjectProperty $Compatibility "requirementsLockSha256" "")).ToLowerInvariant()
    $lockPath = Join-Path $Root "requirements-lock.txt"
    if (-not $localPython -or -not $localMySql -or -not $requiredPython -or -not $requiredMySql -or
        $requiredLockHash -notmatch '^[0-9a-f]{64}$' -or -not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        throw "This package does not expose a complete runtime compatibility fingerprint."
    }
    $localLockHash = Get-Sha256 $lockPath
    if ($localPython -ne $requiredPython -or $localMySql -ne $requiredMySql -or
        $localLockHash -ne $requiredLockHash) {
        throw ("The bundled runtime is not compatible with this incremental update. " +
            "Current Python/MySQL/lock: $localPython / $localMySql / $localLockHash; " +
            "required: $requiredPython / $requiredMySql / $requiredLockHash. " +
            "Download the full portable ZIP instead.")
    }
    $checks = @(
        @(
            "Python archive",
            [string](Get-ObjectProperty (Get-ObjectProperty $runtime "python" $null) "archiveSha256" ""),
            [string](Get-ObjectProperty $Compatibility "pythonArchiveSha256" "")
        ),
        @(
            "MySQL archive",
            [string](Get-ObjectProperty (Get-ObjectProperty $runtime "mysql" $null) "archiveSha256" ""),
            [string](Get-ObjectProperty $Compatibility "mysqlArchiveSha256" "")
        ),
        @(
            "Visual C++ runtime",
            [string](Get-ObjectProperty (Get-ObjectProperty $runtime "visualCppRuntime" $null) "installerSha256" ""),
            [string](Get-ObjectProperty $Compatibility "visualCppRuntimeInstallerSha256" "")
        )
    )
    foreach ($check in $checks) {
        $localHash = ([string]$check[1]).ToLowerInvariant()
        $requiredHash = ([string]$check[2]).ToLowerInvariant()
        if ($localHash -notmatch '^[0-9a-f]{64}$' -or $requiredHash -notmatch '^[0-9a-f]{64}$' -or
            $localHash -ne $requiredHash) {
            throw "$($check[0]) fingerprint is incompatible with this incremental update. Download the full portable ZIP instead."
        }
    }
}

function Assert-StagedReleaseMetadata {
    param([string]$StageRoot, [object]$ValidatedManifest)
    $buildInfo = Read-JsonFile (Join-Path $StageRoot "BUILD-INFO.json")
    if ([string](Get-ObjectProperty $buildInfo "product" "") -ne "XP-Gacha" -or
        (ConvertTo-ReleaseVersion ([string](Get-ObjectProperty $buildInfo "version" ""))) -ne $ValidatedManifest.Version) {
        throw "The staged BUILD-INFO.json does not match the target product/version."
    }
    if ([bool](Get-ObjectProperty $buildInfo "sourceDirty" $true) -or
        -not [bool](Get-ObjectProperty $buildInfo "blankFirstStartVerified" $false)) {
        throw "The staged portable release was not built cleanly and verified from a blank first start."
    }
    $manifestCommit = [string](Get-ObjectProperty $ValidatedManifest.Raw "sourceCommit" "")
    if ([string](Get-ObjectProperty $buildInfo "sourceCommit" "") -ne $manifestCommit) {
        throw "The staged BUILD-INFO source commit does not match the update manifest."
    }
    Assert-RuntimeCompatibility -Root $StageRoot -BuildInfo $buildInfo `
        -Compatibility $ValidatedManifest.RuntimeCompatibility
}

function Expand-VerifiedUpdateArchive {
    param([string]$ZipPath, [string]$Destination, [object]$ValidatedManifest)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $seenFiles = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $seenDirectories = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        if ($archive.Entries.Count -gt 25000) {
            throw "The update archive contains too many entries."
        }
        [Int64]$totalExpanded = 0
        foreach ($entry in $archive.Entries) {
            $rawName = ([string]$entry.FullName).Replace('\', '/').TrimEnd('/')
            if (-not $rawName) {
                continue
            }
            $relative = Assert-SafeRelativePath $rawName
            $isDirectory = -not [string]$entry.Name
            $unixType = (([int64]$entry.ExternalAttributes -shr 16) -band 0xF000)
            if ($unixType -eq 0xA000) {
                throw "Symbolic links are forbidden in update archives: '$relative'"
            }
            if ($isDirectory) {
                if ($seenFiles.Contains($relative)) {
                    throw "Archive file/directory collision: '$relative'"
                }
                [void]$seenDirectories.Add($relative)
                continue
            }
            $relative = Assert-ManagedRelativePath $relative
            if (-not $seenFiles.Add($relative)) {
                throw "Duplicate archive path (case-insensitive): '$relative'"
            }
            if (-not $ValidatedManifest.FileMap.ContainsKey($relative)) {
                throw "Archive contains a file not declared by the manifest: '$relative'"
            }
            $parts = @($relative.Split('/'))
            $parentText = ""
            for ($index = 0; $index -lt ($parts.Count - 1); $index += 1) {
                $parentText = if ($parentText) { "$parentText/$($parts[$index])" } else { $parts[$index] }
                if ($seenFiles.Contains($parentText)) {
                    throw "Archive file/directory collision: '$relative'"
                }
                [void]$seenDirectories.Add($parentText)
            }
            $expected = $ValidatedManifest.FileMap[$relative]
            if ([Int64]$entry.Length -ne [Int64]$expected.size) {
                throw "Archive size does not match the manifest for '$relative'."
            }
            if ($entry.Length -gt 0 -and $entry.CompressedLength -gt 0 -and
                ($entry.Length / [double]$entry.CompressedLength) -gt 1000) {
                throw "Suspicious compression ratio for '$relative'."
            }
            $totalExpanded += [Int64]$entry.Length
            if ($totalExpanded -gt 1073741824) {
                throw "The expanded update archive is unexpectedly large."
            }
            $target = Get-ChildFullPath -Root $Destination -RelativePath $relative
            $targetParent = Split-Path -Parent $target
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
            $inputStream = $entry.Open()
            $outputStream = [System.IO.File]::Open($target, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try {
                $inputStream.CopyTo($outputStream)
            }
            finally {
                $outputStream.Dispose()
                $inputStream.Dispose()
            }
            if ((Get-Sha256 $target) -ne [string]$expected.sha256) {
                throw "Archive file failed SHA-256 verification: '$relative'"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
    if ($seenFiles.Count -ne $ValidatedManifest.FileMap.Count) {
        $missing = @($ValidatedManifest.Files | Where-Object { -not $seenFiles.Contains([string]$_.path) } | Select-Object -ExpandProperty path)
        throw "Archive is missing manifest files: $($missing -join ', ')"
    }
}

function Read-InstalledFileManifest {
    param([string]$Path)
    $result = [System.Collections.Generic.Dictionary[string, string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "The installed package has no SHA256SUMS.txt baseline; use the full portable ZIP."
    }
    foreach ($line in [System.IO.File]::ReadAllLines($Path, [System.Text.Encoding]::UTF8)) {
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            continue
        }
        $relative = Normalize-RelativePath $Matches[2]
        if ((Test-ManagedRelativePath $relative) -and -not $result.ContainsKey($relative)) {
            $result.Add($relative, $Matches[1].ToLowerInvariant())
        }
    }
    return $result
}

function Test-IsInsideReplacedDirectory {
    param([string]$RelativePath)
    return $RelativePath.StartsWith("web/dist/", [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-PortableChangePlan {
    param([string]$Root, [string]$StageRoot, [object]$ValidatedManifest)
    $oldMap = Read-InstalledFileManifest (Join-Path $Root "SHA256SUMS.txt")
    $targetMap = $ValidatedManifest.FileMap
    $conflicts = [System.Collections.Generic.List[string]]::new()

    foreach ($old in $oldMap.GetEnumerator()) {
        $relative = [string]$old.Key
        $target = Get-ChildFullPath -Root $Root -RelativePath $relative
        Assert-NotReparseTarget -Path $target
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $currentHash = Get-Sha256 $target
            if ($currentHash -ne [string]$old.Value) {
                $incoming = if ($targetMap.ContainsKey($relative)) { [string]$targetMap[$relative].sha256 } else { "" }
                if ($currentHash -ne $incoming) {
                    $conflicts.Add($relative)
                }
            }
        }
    }
    foreach ($incoming in $ValidatedManifest.Files) {
        $relative = [string]$incoming.path
        $target = Get-ChildFullPath -Root $Root -RelativePath $relative
        Assert-NoReparseParents -Root $Root -RelativePath $relative
        Assert-NotReparseTarget -Path $target
        if ((Test-Path -LiteralPath $target -PathType Leaf) -and -not $oldMap.ContainsKey($relative)) {
            $currentHash = Get-Sha256 $target
            if ($currentHash -ne [string]$incoming.sha256 -and $relative -ne "SHA256SUMS.txt") {
                $conflicts.Add($relative)
            }
        }
    }
    if ($conflicts.Count -gt 0) {
        $unique = @($conflicts | Sort-Object -Unique)
        throw ("Locally modified managed files would be overwritten. Restore or back them up first: " +
            ($unique -join ', '))
    }

    $operations = [System.Collections.Generic.List[object]]::new()
    foreach ($incoming in $ValidatedManifest.Files) {
        $relative = [string]$incoming.path
        if (Test-IsInsideReplacedDirectory $relative) {
            continue
        }
        $target = Get-ChildFullPath -Root $Root -RelativePath $relative
        $same = (Test-Path -LiteralPath $target -PathType Leaf) -and
            ((Get-Sha256 $target) -eq [string]$incoming.sha256)
        if (-not $same) {
            $priority = if ($relative -in @("BUILD-INFO.json", "SHA256SUMS.txt")) { 100 } else { 10 }
            $operations.Add([pscustomobject]@{
                path = $relative
                action = "replace"
                source = Get-ChildFullPath -Root $StageRoot -RelativePath $relative
                priority = $priority
            })
        }
    }
    foreach ($old in $oldMap.GetEnumerator()) {
        $relative = [string]$old.Key
        if ((Test-IsInsideReplacedDirectory $relative) -or $targetMap.ContainsKey($relative)) {
            continue
        }
        $target = Get-ChildFullPath -Root $Root -RelativePath $relative
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $operations.Add([pscustomobject]@{
                path = $relative
                action = "delete"
                source = ""
                priority = 20
            })
        }
    }
    return [pscustomobject]@{
        Operations = @($operations | Sort-Object priority, path)
        ReplaceDirectories = @("web/dist")
    }
}

function Write-TransactionJournal {
    param([string]$JournalPath, [object]$State)
    Write-JsonFile -Path $JournalPath -Value $State
}

function Copy-FileAtomically {
    param([string]$Source, [string]$Destination)
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = Join-Path $parent (".xp-gacha-update-" + [Guid]::NewGuid().ToString("N") + ".tmp")
    $replaceBackup = Join-Path $parent (".xp-gacha-replaced-" + [Guid]::NewGuid().ToString("N") + ".bak")
    try {
        Copy-Item -LiteralPath $Source -Destination $temporary -Force
        if (Test-Path -LiteralPath $Destination -PathType Leaf) {
            [System.IO.File]::Replace($temporary, $Destination, $replaceBackup, $true)
            Remove-Item -LiteralPath $replaceBackup -Force -ErrorAction SilentlyContinue
        }
        else {
            [System.IO.File]::Move($temporary, $Destination)
        }
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $replaceBackup -Force -ErrorAction SilentlyContinue
    }
}

function Install-PortablePayload {
    param(
        [string]$Root,
        [string]$StageRoot,
        [object]$Plan,
        [string]$OldVersion,
        [string]$NewVersion,
        [bool]$WasRunning
    )
    $updatesRoot = Join-Path $Root "updates"
    $backupRoot = Join-Path (Join-Path $updatesRoot "backups") (
        ("v{0}-to-v{1}-{2}" -f $OldVersion, $NewVersion, [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss"))
    )
    New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
    $journalPath = Join-Path $updatesRoot "state.json"
    $state = [ordered]@{
        schemaVersion = 1
        phase = "applying"
        packageRoot = $Root
        oldVersion = $OldVersion
        newVersion = $NewVersion
        wasRunning = $WasRunning
        backupRoot = $backupRoot
        directories = [System.Collections.Generic.List[object]]::new()
        files = [System.Collections.Generic.List[object]]::new()
        startedAtUtc = [DateTime]::UtcNow.ToString("o")
    }
    Write-TransactionJournal $journalPath $state

    foreach ($relativeDirectory in $Plan.ReplaceDirectories) {
        $relativeDirectory = Normalize-RelativePath $relativeDirectory
        if ($relativeDirectory -ne "web/dist") {
            throw "Unexpected replace-directory operation: '$relativeDirectory'"
        }
        $targetDirectory = Get-ChildFullPath -Root $Root -RelativePath $relativeDirectory
        $stagedDirectory = Get-ChildFullPath -Root $StageRoot -RelativePath $relativeDirectory
        Assert-NoReparseParents -Root $Root -RelativePath ($relativeDirectory + "/.update-probe")
        Assert-NotReparseTarget -Path $targetDirectory -Description "managed web directory"
        $backupDirectory = Get-ChildFullPath -Root $backupRoot -RelativePath ("directories/" + $relativeDirectory)
        $existed = Test-Path -LiteralPath $targetDirectory -PathType Container
        if ($existed) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $backupDirectory) -Force | Out-Null
            Copy-Item -LiteralPath $targetDirectory -Destination $backupDirectory -Recurse -Force
        }
        $state.directories.Add([ordered]@{
            path = $relativeDirectory
            existed = $existed
            backupPath = $backupDirectory
        })
        Write-TransactionJournal $journalPath $state
        Remove-VerifiedTree -Path $targetDirectory -Parent $Root
        New-Item -ItemType Directory -Path (Split-Path -Parent $targetDirectory) -Force | Out-Null
        Copy-Item -LiteralPath $stagedDirectory -Destination $targetDirectory -Recurse -Force
    }

    foreach ($operation in $Plan.Operations) {
        $relative = [string]$operation.path
        $target = Get-ChildFullPath -Root $Root -RelativePath $relative
        Assert-NoReparseParents -Root $Root -RelativePath $relative
        Assert-NotReparseTarget -Path $target
        $existed = Test-Path -LiteralPath $target -PathType Leaf
        $backupPath = Get-ChildFullPath -Root $backupRoot -RelativePath ("files/" + $relative)
        if ($existed) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $backupPath) -Force | Out-Null
            Copy-Item -LiteralPath $target -Destination $backupPath -Force
        }
        $state.files.Add([ordered]@{
            path = $relative
            action = [string]$operation.action
            existed = $existed
            backupPath = $backupPath
        })
        Write-TransactionJournal $journalPath $state
        if ([string]$operation.action -eq "delete") {
            Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        }
        else {
            Copy-FileAtomically -Source ([string]$operation.source) -Destination $target
        }
    }
    $state.phase = "verifying"
    Write-TransactionJournal $journalPath $state
    return [pscustomobject]@{ State = $state; JournalPath = $journalPath }
}

function Undo-PortableTransaction {
    param([object]$State, [string]$JournalPath, [string]$ExpectedRoot)
    $root = [System.IO.Path]::GetFullPath([string](Get-ObjectProperty $State "packageRoot" ""))
    $expected = [System.IO.Path]::GetFullPath($ExpectedRoot).TrimEnd('\', '/')
    if (-not $root.TrimEnd('\', '/').Equals($expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The pending update journal belongs to a different package location and will not be applied."
    }
    $updatesRoot = Join-Path $expected "updates"
    $expectedJournal = Join-Path $updatesRoot "state.json"
    if (-not ([System.IO.Path]::GetFullPath($JournalPath)).Equals(
        [System.IO.Path]::GetFullPath($expectedJournal),
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "The rollback journal path is not the package update journal."
    }
    $backupContainer = Join-Path $updatesRoot "backups"
    $backupRoot = Assert-AbsoluteChildPath `
        -Path ([string](Get-ObjectProperty $State "backupRoot" "")) `
        -Parent $backupContainer -Description "rollback backup root"
    $files = @(Get-ObjectProperty $State "files" @())
    for ($index = $files.Count - 1; $index -ge 0; $index -= 1) {
        $record = $files[$index]
        $relative = Assert-ManagedRelativePath ([string]$record.path)
        $target = Get-ChildFullPath -Root $root -RelativePath $relative
        Assert-NoReparseParents -Root $root -RelativePath $relative
        Assert-NotReparseTarget -Path $target -Description "rollback target"
        if ([bool]$record.existed) {
            $backup = Assert-AbsoluteChildPath -Path ([string]$record.backupPath) `
                -Parent $backupRoot -Description "rollback file backup"
            if (-not (Test-Path -LiteralPath $backup -PathType Leaf)) {
                throw "Rollback backup is missing: $backup"
            }
            Copy-FileAtomically -Source $backup -Destination $target
        }
        else {
            Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        }
    }
    $directories = @(Get-ObjectProperty $State "directories" @())
    for ($index = $directories.Count - 1; $index -ge 0; $index -= 1) {
        $record = $directories[$index]
        $relative = [string]$record.path
        if ($relative -ne "web/dist") {
            throw "Unsafe rollback directory in journal: '$relative'"
        }
        $target = Get-ChildFullPath -Root $root -RelativePath $relative
        Assert-NoReparseParents -Root $root -RelativePath ($relative + "/.rollback-probe")
        Assert-NotReparseTarget -Path $target -Description "rollback web directory"
        Remove-VerifiedTree -Path $target -Parent $root
        if ([bool]$record.existed) {
            $backup = Assert-AbsoluteChildPath -Path ([string]$record.backupPath) `
                -Parent $backupRoot -Description "rollback directory backup"
            if (-not (Test-Path -LiteralPath $backup -PathType Container)) {
                throw "Rollback directory backup is missing: $backup"
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Copy-Item -LiteralPath $backup -Destination $target -Recurse -Force
        }
    }
    Remove-Item -LiteralPath $JournalPath -Force -ErrorAction SilentlyContinue
}

function Complete-PortableTransaction {
    param([object]$Transaction)
    $state = $Transaction.State
    $state.phase = "committed"
    $state.completedAtUtc = [DateTime]::UtcNow.ToString("o")
    Write-TransactionJournal -JournalPath ([string]$Transaction.JournalPath) -State $state
    $lastUpdatePath = Join-Path ([string]$state.packageRoot) "updates/last-update.json"
    try {
        Write-JsonFile -Path $lastUpdatePath -Value $state
        Remove-Item -LiteralPath ([string]$Transaction.JournalPath) -Force -ErrorAction Stop
    }
    catch {
        Write-UpdateMessage "The update committed successfully; its cleanup journal will be finalized on the next check."
    }
}

function Invoke-PortableLauncher {
    param([string]$Root, [string[]]$Arguments, [switch]$AllowFailure, [switch]$Quiet)
    $python = Join-Path $Root "runtime/python/python.exe"
    $launcher = Join-Path $Root "portable_launcher.py"
    if ($Quiet) {
        & $python $launcher @Arguments *> $null
    }
    else {
        & $python $launcher @Arguments
    }
    $code = $LASTEXITCODE
    if (-not $AllowFailure -and $code -ne 0) {
        throw "Portable launcher '$($Arguments -join ' ')' failed with exit code $code."
    }
    return $code
}

function Get-PortableRunningState {
    param([string]$Root)
    $code = Invoke-PortableLauncher -Root $Root -Arguments @("status") -AllowFailure -Quiet
    if ($code -eq 0) { return $true }
    if ($code -eq 1) { return $false }
    throw "The portable instance is starting or unhealthy. Stop it before updating."
}

function Stop-PortableInstance {
    param([string]$Root)
    [void](Invoke-PortableLauncher -Root $Root -Arguments @("stop"))
}

function Start-PortableInstance {
    param([string]$Root)
    $python = Join-Path $Root "runtime/python/python.exe"
    $hadRestartFlag = Test-Path Env:XP_GACHA_UPDATE_RESTART
    $previousRestartFlag = $env:XP_GACHA_UPDATE_RESTART
    try {
        $env:XP_GACHA_UPDATE_RESTART = "1"
        $process = Start-Process -FilePath $python -ArgumentList @("portable_launcher.py", "start", "--no-browser") `
            -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    }
    finally {
        if ($hadRestartFlag) {
            $env:XP_GACHA_UPDATE_RESTART = $previousRestartFlag
        }
        else {
            Remove-Item Env:XP_GACHA_UPDATE_RESTART -ErrorAction SilentlyContinue
        }
    }
    try {
        $deadline = [DateTime]::UtcNow.AddSeconds(330)
        while ([DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Milliseconds 800
            $code = Invoke-PortableLauncher -Root $Root -Arguments @("status") -AllowFailure -Quiet
            if ($code -eq 0) {
                return
            }
            if ($process.HasExited) {
                throw "The updated portable instance exited during startup. Check logs/app.log."
            }
        }
        throw "The updated portable instance did not become healthy within 330 seconds."
    }
    catch {
        try { Stop-PortableInstance $Root } catch { }
        if (-not $process.HasExited) {
            try {
                $process.Kill()
                $process.WaitForExit(10000) | Out-Null
            }
            catch { }
        }
        throw
    }
}

function Acquire-UpdateLock {
    param([string]$Path)
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    try {
        return [System.IO.File]::Open($Path, [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    }
    catch {
        throw "Another XP-Gacha update is already running for this installation."
    }
}

function Initialize-PortableUpdateWorkspace {
    param([string]$Root)
    $updatesRoot = Join-Path $Root "updates"
    if (Test-Path -LiteralPath $updatesRoot) {
        $updatesItem = Get-Item -LiteralPath $updatesRoot -Force
        if (-not $updatesItem.PSIsContainer -or
            ($updatesItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "The portable updates path must be a normal directory, not a file or reparse point."
        }
    }
    else {
        New-Item -ItemType Directory -Path $updatesRoot -Force | Out-Null
    }
    foreach ($name in @("backups", "downloads", "staging")) {
        $path = Join-Path $updatesRoot $name
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path -Force
            if (-not $item.PSIsContainer -or
                ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "The portable update workspace '$name' must be a normal directory."
            }
        }
        else {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
        }
    }
    return $updatesRoot
}

function Recover-PendingPortableTransaction {
    param([string]$Root)
    $journalPath = Join-Path $Root "updates/state.json"
    if (-not (Test-Path -LiteralPath $journalPath -PathType Leaf)) {
        return
    }
    $state = Read-JsonFile $journalPath
    if ([string](Get-ObjectProperty $state "phase" "") -eq "committed") {
        Write-JsonFile -Path (Join-Path $Root "updates/last-update.json") -Value $state
        Remove-Item -LiteralPath $journalPath -Force
        Write-UpdateMessage "A completed update journal was finalized; no rollback was needed."
        return
    }
    Write-UpdateMessage "An incomplete update transaction was found; restoring its program files."
    Stop-PortableInstance $Root
    Undo-PortableTransaction -State $state -JournalPath $journalPath -ExpectedRoot $Root
    if ([bool](Get-ObjectProperty $state "wasRunning" $false) -and -not $SkipRestart) {
        Start-PortableInstance $Root
    }
    Write-UpdateMessage "The interrupted update was rolled back; continuing this update check from the restored version."
}

function Assert-FreeSpace {
    param([string]$Root, [Int64]$RequiredBytes)
    try {
        $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($Root))
        if ($drive.AvailableFreeSpace -lt $RequiredBytes) {
            throw "Not enough free disk space for staging and rollback backups."
        }
    }
    catch [System.ArgumentException] {
        # UNC/test paths may not expose DriveInfo. Copy operations will still fail safely if full.
    }
}

function Invoke-PortableUpdate {
    param([string]$Root, [object]$Release, [string]$Tag)
    $updatesRoot = Initialize-PortableUpdateWorkspace $Root
    $lockPath = Join-Path $updatesRoot "update.lock"
    $lock = Acquire-UpdateLock $lockPath
    try {
        Recover-PendingPortableTransaction $Root
        $buildInfoPath = Join-Path $Root "BUILD-INFO.json"
        $buildInfo = Read-JsonFile $buildInfoPath
        $currentVersionText = [string](Get-ObjectProperty $buildInfo "version" "")
        $currentVersion = ConvertTo-ReleaseVersion $currentVersionText
        $targetVersion = ConvertTo-ReleaseVersion $Tag
        Write-UpdateMessage "Portable package: v$currentVersion -> $Tag"
        if ($targetVersion -lt $currentVersion) {
            Write-UpdateMessage "The local package is newer than the latest stable Release; no downgrade was performed."
            return
        }
        if ($targetVersion -eq $currentVersion) {
            Write-UpdateMessage "Already up to date."
            return
        }

        $assetBase = "XP-Gacha-$Tag-portable-win64-update"
        $manifestName = "$assetBase.json"
        $zipName = "$assetBase.zip"
        $checksumName = "$zipName.sha256"
        $manifestAsset = Get-ReleaseAsset -Release $Release -Name $manifestName
        $zipAsset = Get-ReleaseAsset -Release $Release -Name $zipName
        $checksumAsset = Get-ReleaseAsset -Release $Release -Name $checksumName
        $manifestAssetSize = [Int64](Get-ObjectProperty $manifestAsset "size" 0)
        $zipAssetSize = [Int64](Get-ObjectProperty $zipAsset "size" 0)
        $checksumAssetSize = [Int64](Get-ObjectProperty $checksumAsset "size" 0)
        if ($manifestAssetSize -le 0 -or $manifestAssetSize -gt 16777216 -or
            $zipAssetSize -le 0 -or $zipAssetSize -gt 1073741824 -or
            $checksumAssetSize -le 0 -or $checksumAssetSize -gt 4096) {
            throw "The portable incremental Release assets have unsafe or missing size metadata."
        }
        if ($CheckOnly) {
            Write-UpdateMessage "Update available: $Tag (portable incremental assets are present)."
            return
        }

        $downloadRoot = Join-Path (Join-Path $updatesRoot "downloads") $Tag
        New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
        $manifestPath = Join-Path $downloadRoot $manifestName
        $zipPath = Join-Path $downloadRoot $zipName
        $checksumPath = Join-Path $downloadRoot $checksumName
        Write-UpdateMessage "Downloading and validating the portable app-layer update..."
        Receive-ReleaseAsset -Asset $manifestAsset -Destination $manifestPath | Out-Null
        $validated = Get-ValidatedManifest -Path $manifestPath -ExpectedTag $Tag
        if ($validated.AssetName -ne $zipName) {
            throw "The manifest payload name does not match the expected Release asset."
        }
        try {
            Assert-RuntimeCompatibility -Root $Root -BuildInfo $buildInfo `
                -Compatibility $validated.RuntimeCompatibility
        }
        catch {
            $releasePage = [string](Get-ObjectProperty $Release "html_url" "https://github.com/$($script:Repository)/releases/latest")
            throw "$($_.Exception.Message) Full package: $releasePage"
        }
        Assert-FreeSpace -Root $Root -RequiredBytes ([Int64](($validated.TotalSize * 3) + $validated.PayloadSize + 67108864))
        Receive-ReleaseAsset -Asset $checksumAsset -Destination $checksumPath | Out-Null
        Receive-ReleaseAsset -Asset $zipAsset -Destination $zipPath | Out-Null
        $actualZipHash = Get-Sha256 $zipPath
        if ($actualZipHash -ne $validated.PayloadSha256 -or
            (Get-Item -LiteralPath $zipPath).Length -ne $validated.PayloadSize) {
            throw "The portable update ZIP does not match its manifest."
        }
        $checksumText = [System.IO.File]::ReadAllText($checksumPath, [System.Text.Encoding]::UTF8).Trim()
        if ($checksumText -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$' -or
            $Matches[1].ToLowerInvariant() -ne $actualZipHash -or
            ([System.IO.Path]::GetFileName($Matches[2].Trim())) -ne $zipName) {
            throw "The portable update ZIP failed its published checksum file."
        }

        $stageRoot = Join-Path (Join-Path $updatesRoot "staging") ([Guid]::NewGuid().ToString("N"))
        try {
            Expand-VerifiedUpdateArchive -ZipPath $zipPath -Destination $stageRoot `
                -ValidatedManifest $validated
            Assert-StagedReleaseMetadata -StageRoot $stageRoot -ValidatedManifest $validated
            $plan = Get-PortableChangePlan -Root $Root -StageRoot $stageRoot `
                -ValidatedManifest $validated
            $wasRunning = Get-PortableRunningState $Root
            if ($wasRunning) {
                Write-UpdateMessage "Stopping this portable instance safely..."
                Stop-PortableInstance $Root
            }
            $transaction = $null
            try {
                Write-UpdateMessage "Applying verified program files; user data and bundled runtime are excluded."
                $transaction = Install-PortablePayload -Root $Root -StageRoot $stageRoot -Plan $plan `
                    -OldVersion $currentVersionText -NewVersion ([string]$validated.Version) -WasRunning $wasRunning
                [void](Invoke-PortableLauncher -Root $Root -Arguments @("doctor"))
                if ($wasRunning -and -not $SkipRestart) {
                    Write-UpdateMessage "Restarting XP-Gacha and waiting for its health check..."
                    Start-PortableInstance $Root
                }
                Complete-PortableTransaction $transaction
                Write-UpdateMessage "Portable update completed: $Tag."
                if (-not $wasRunning) {
                    Write-UpdateMessage "The package was stopped before updating, so it was left stopped."
                }
            }
            catch {
                $applyError = $_
                if ($null -eq $transaction) {
                    $pendingJournal = Join-Path $Root "updates/state.json"
                    if (Test-Path -LiteralPath $pendingJournal -PathType Leaf) {
                        $transaction = [pscustomobject]@{
                            State = Read-JsonFile $pendingJournal
                            JournalPath = $pendingJournal
                        }
                    }
                }
                if ($null -ne $transaction) {
                    try {
                        Stop-PortableInstance $Root
                    }
                    catch { }
                    Write-UpdateMessage "Update verification failed; restoring the previous program files..."
                    Undo-PortableTransaction -State $transaction.State `
                        -JournalPath $transaction.JournalPath -ExpectedRoot $Root
                    [void](Invoke-PortableLauncher -Root $Root -Arguments @("doctor"))
                }
                if ($wasRunning -and -not $SkipRestart) {
                    Start-PortableInstance $Root
                }
                throw $applyError
            }
        }
        finally {
            Remove-VerifiedTree -Path $stageRoot -Parent (Join-Path $updatesRoot "staging")
        }
    }
    finally {
        if ($null -ne $lock) {
            $lock.Dispose()
        }
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-GitCapture {
    param([string]$Root, [string[]]$Arguments, [switch]$AllowFailure)
    $output = @(& git -C $Root @Arguments 2>&1)
    $code = $LASTEXITCODE
    if (-not $AllowFailure -and $code -ne 0) {
        throw "git $($Arguments -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return [pscustomobject]@{ Code = $code; Text = (($output | ForEach-Object { [string]$_ }) -join "`n").Trim() }
}

function Test-SourceAppRunning {
    param([string]$Root)
    $output = @(& docker compose -f (Join-Path $Root "docker-compose.yml") ps --status running -q app 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose status check failed. Start Docker Desktop or use -SkipRestart."
    }
    return [bool](($output -join "").Trim())
}

function Invoke-SourceDockerRefresh {
    param([string]$Root)
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found. Start Docker Desktop before completing the source update."
    }
    $powershell = Join-Path $PSHOME "powershell.exe"
    if (-not (Test-Path -LiteralPath $powershell)) {
        $powershell = "powershell.exe"
    }
    & $powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "scripts/start.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Docker rebuild/start failed. The source rebuild marker was preserved for a retry."
    }
}

function Invoke-SourceUpdate {
    param([string]$Root, [string]$Tag)
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git was not found. The source updater requires the Git checkout that created this folder."
    }
    $gitLockPath = (Invoke-GitCapture -Root $Root -Arguments @("rev-parse", "--git-path", "xp-gacha-update.lock")).Text
    if ([System.IO.Path]::IsPathRooted($gitLockPath)) {
        $lockPath = [System.IO.Path]::GetFullPath($gitLockPath)
    }
    else {
        $lockPath = [System.IO.Path]::GetFullPath((Join-Path $Root $gitLockPath))
    }
    $rebuildMarkerGitPath = (Invoke-GitCapture -Root $Root -Arguments @(
        "rev-parse", "--git-path", "xp-gacha-rebuild-pending.json"
    )).Text
    if ([System.IO.Path]::IsPathRooted($rebuildMarkerGitPath)) {
        $rebuildMarkerPath = [System.IO.Path]::GetFullPath($rebuildMarkerGitPath)
    }
    else {
        $rebuildMarkerPath = [System.IO.Path]::GetFullPath((Join-Path $Root $rebuildMarkerGitPath))
    }
    $lock = Acquire-UpdateLock $lockPath
    try {
        $origin = (Invoke-GitCapture -Root $Root -Arguments @("remote", "get-url", "origin")).Text
        $acceptedOrigins = @(
            "https://github.com/ink-bai5844/xp-gacha.git",
            "git@github.com:ink-bai5844/xp-gacha.git",
            "ssh://git@github.com/ink-bai5844/xp-gacha.git"
        )
        if ($origin.ToLowerInvariant() -notin $acceptedOrigins) {
            throw "The source checkout origin is not the official XP-Gacha repository: '$origin'"
        }
        $branch = (Invoke-GitCapture -Root $Root -Arguments @("symbolic-ref", "--quiet", "--short", "HEAD") -AllowFailure)
        if ($branch.Code -ne 0 -or $branch.Text -ne "main") {
            throw "Automatic source updates require the attached 'main' branch."
        }
        $status = (Invoke-GitCapture -Root $Root -Arguments @("status", "--porcelain", "--untracked-files=all")).Text
        if ($status) {
            throw "The source checkout has local changes or untracked files. Commit, move, or remove them before updating."
        }
        $temporaryRef = "refs/xp-gacha-updater/$Tag"
        Write-UpdateMessage "Fetching Release tag $Tag from the official repository..."
        [void](Invoke-GitCapture -Root $Root -Arguments @(
            "fetch", "--quiet", "--force", "--no-tags", $script:RepositoryUrl,
            "+refs/tags/$Tag`:$temporaryRef"
        ))
        $head = (Invoke-GitCapture -Root $Root -Arguments @("rev-parse", "HEAD")).Text
        $target = (Invoke-GitCapture -Root $Root -Arguments @("rev-parse", "$temporaryRef`^{}" )).Text
        $pendingMarker = $null
        if (Test-Path -LiteralPath $rebuildMarkerPath -PathType Leaf) {
            $pendingMarker = Read-JsonFile $rebuildMarkerPath
            if ([string](Get-ObjectProperty $pendingMarker "targetCommit" "") -ne $head) {
                throw "A source rebuild marker belongs to a different commit; inspect '$rebuildMarkerPath'."
            }
        }
        Write-UpdateMessage "Source commits: $($head.Substring(0, 12)) -> $($target.Substring(0, 12)) ($Tag)"
        if ($head -eq $target) {
            if ($null -ne $pendingMarker) {
                if ($CheckOnly -or $SkipRestart) {
                    Write-UpdateMessage "Source files are current, but a Docker rebuild is still pending."
                    return
                }
                Write-UpdateMessage "Retrying the pending Docker rebuild and health check..."
                Invoke-SourceDockerRefresh $Root
                Remove-Item -LiteralPath $rebuildMarkerPath -Force
                Write-UpdateMessage "Source update and pending Docker rebuild completed: $Tag."
                return
            }
            Write-UpdateMessage "Already up to date."
            return
        }
        $headIsAncestor = (Invoke-GitCapture -Root $Root -Arguments @("merge-base", "--is-ancestor", $head, $target) -AllowFailure).Code -eq 0
        $targetIsAncestor = (Invoke-GitCapture -Root $Root -Arguments @("merge-base", "--is-ancestor", $target, $head) -AllowFailure).Code -eq 0
        if ($targetIsAncestor) {
            if ($null -ne $pendingMarker -and -not $CheckOnly -and -not $SkipRestart) {
                Write-UpdateMessage "Retrying the pending Docker rebuild for the newer local source commit..."
                Invoke-SourceDockerRefresh $Root
                Remove-Item -LiteralPath $rebuildMarkerPath -Force
            }
            elseif ($null -ne $pendingMarker) {
                Write-UpdateMessage "The local source is newer, but its Docker rebuild is still pending."
            }
            Write-UpdateMessage "This source checkout is ahead of the latest stable Release; no downgrade was performed."
            return
        }
        if (-not $headIsAncestor) {
            throw "The local main branch and Release tag have diverged; automatic update was refused."
        }
        if ($CheckOnly) {
            Write-UpdateMessage "Update available: $Tag (fast-forward source update)."
            return
        }

        if (-not $SkipRestart -and -not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "Docker was not found. Start Docker Desktop, or pass -SkipRestart to update code only."
        }

        if ($null -ne $pendingMarker) {
            Remove-Item -LiteralPath $rebuildMarkerPath -Force
            Write-UpdateMessage "Superseding the older pending Docker rebuild with $Tag."
        }

        $wasRunning = $false
        if (-not $SkipRestart) {
            $wasRunning = Test-SourceAppRunning $Root
        }
        if ($wasRunning) {
            Write-JsonFile -Path $rebuildMarkerPath -Value ([ordered]@{
                schemaVersion = 1
                tag = $Tag
                previousCommit = $head
                targetCommit = $target
                createdAtUtc = [DateTime]::UtcNow.ToString("o")
            })
        }
        try {
            [void](Invoke-GitCapture -Root $Root -Arguments @("merge", "--ff-only", $temporaryRef))
        }
        catch {
            Remove-Item -LiteralPath $rebuildMarkerPath -Force -ErrorAction SilentlyContinue
            throw
        }
        $updatedHead = (Invoke-GitCapture -Root $Root -Arguments @("rev-parse", "HEAD")).Text
        if ($updatedHead -ne $target) {
            throw "Git reported success but did not reach the Release commit."
        }
        Write-UpdateMessage "Source files updated to $Tag."
        if ($wasRunning -and -not $SkipRestart) {
            Write-UpdateMessage "Rebuilding and health-checking the running Docker application..."
            Invoke-SourceDockerRefresh $Root
            Remove-Item -LiteralPath $rebuildMarkerPath -Force
        }
        elseif (-not $wasRunning) {
            Write-UpdateMessage "The Docker application was stopped before updating, so it was left stopped."
        }
        Write-UpdateMessage "Source update completed: $Tag."
    }
    finally {
        if ($null -ne $lock) {
            $lock.Dispose()
        }
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    }
}

function Assert-SelfTest {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw "Self-test failed: $Message"
    }
}

function Invoke-UpdaterSelfTest {
    Assert-SelfTest ((ConvertTo-ReleaseVersion "v0.2.10") -gt (ConvertTo-ReleaseVersion "0.2.9")) "semantic version ordering"
    Assert-SelfTest (Test-ManagedRelativePath "server/main.py") "managed Python path"
    Assert-SelfTest (Test-ManagedRelativePath "web/dist/assets/app.js") "managed web path"
    Assert-SelfTest (-not (Test-ManagedRelativePath "../outside.txt")) "zip traversal rejection"
    Assert-SelfTest (-not (Test-ManagedRelativePath "mysql/data/ibdata1")) "MySQL protection"
    Assert-SelfTest (-not (Test-ManagedRelativePath "dictionaries/STOP_TAGS.txt")) "dictionary protection"
    Assert-SelfTest (-not (Test-ManagedRelativePath "portable-settings.env")) "settings protection"
    Assert-SelfTest (-not (Test-ManagedRelativePath "data_processing/result.jsonl")) "generated task data protection"

    $testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("xp-gacha-updater-selftest-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
    try {
        $manifestPath = Join-Path $testRoot "XP-Gacha-v1.2.3-portable-win64-update.json"
        $manifestFiles = @()
        foreach ($relative in @(
            "BUILD-INFO.json", "SHA256SUMS.txt", "portable_launcher.py",
            "Update XP-Gacha.cmd", "tools/update_xp_gacha.ps1", "web/dist/index.html",
            ("README_" + [char]0x4FBF + [char]0x643A + [char]0x7248 + ".md")
        )) {
            $manifestFiles += [ordered]@{ path = $relative; sha256 = ("0" * 64); size = 0 }
        }
        Write-JsonFile -Path $manifestPath -Value ([ordered]@{
            schema = 1
            product = "XP-Gacha"
            version = "1.2.3"
            platform = "windows-x64"
            updateKind = "portable-app-layer"
            packageAsset = "XP-Gacha-v1.2.3-portable-win64-update.zip"
            packageSha256 = ("a" * 64)
            packageSize = 1
            sourceCommit = ("f" * 40)
            sourceDirty = $false
            runtimeCompatibility = [ordered]@{
                requirementsLockSha256 = ("b" * 64)
                pythonVersion = "3.13.15"
                pythonArchiveSha256 = ("c" * 64)
                mysqlVersion = "8.4.11"
                mysqlArchiveSha256 = ("d" * 64)
                visualCppRuntimeInstallerSha256 = ("e" * 64)
            }
            files = $manifestFiles
            replaceDirectories = @("web/dist")
        })
        $parsedManifest = Get-ValidatedManifest -Path $manifestPath -ExpectedTag "v1.2.3"
        Assert-SelfTest ($parsedManifest.AssetName -eq "XP-Gacha-v1.2.3-portable-win64-update.zip") "flat builder manifest contract"

        $lockPath = Join-Path $testRoot "updates/update.lock"
        $firstLock = Acquire-UpdateLock $lockPath
        $secondRejected = $false
        try {
            try { $null = Acquire-UpdateLock $lockPath } catch { $secondRejected = $true }
        }
        finally {
            $firstLock.Dispose()
        }
        Assert-SelfTest $secondRejected "exclusive update lock"

        $package = Join-Path $testRoot "package"
        $stage = Join-Path $testRoot "stage"
        foreach ($directory in @("web/dist", "data", "dictionaries", "tools")) {
            New-Item -ItemType Directory -Path (Join-Path $package $directory) -Force | Out-Null
            New-Item -ItemType Directory -Path (Join-Path $stage $directory) -Force | Out-Null
        }
        [System.IO.File]::WriteAllText((Join-Path $package "app.py"), "old", $script:Utf8NoBom)
        [System.IO.File]::WriteAllText((Join-Path $package "web/dist/index.html"), "old-web", $script:Utf8NoBom)
        [System.IO.File]::WriteAllText((Join-Path $package "data/user.txt"), "keep-data", $script:Utf8NoBom)
        [System.IO.File]::WriteAllText((Join-Path $package "dictionaries/custom.txt"), "keep-dictionary", $script:Utf8NoBom)
        [System.IO.File]::WriteAllText((Join-Path $package "portable-settings.env"), "SECRET=keep", $script:Utf8NoBom)
        [System.IO.File]::WriteAllText((Join-Path $package "notes.txt"), "keep-unknown", $script:Utf8NoBom)
        $oldAppHash = Get-Sha256 (Join-Path $package "app.py")
        $oldWebHash = Get-Sha256 (Join-Path $package "web/dist/index.html")
        [System.IO.File]::WriteAllText((Join-Path $package "SHA256SUMS.txt"),
            "$oldAppHash  app.py`n$oldWebHash  web/dist/index.html`n", $script:Utf8NoBom)

        [System.IO.File]::WriteAllText((Join-Path $stage "app.py"), "new", $script:Utf8NoBom)
        [System.IO.File]::WriteAllText((Join-Path $stage "web/dist/index.html"), "new-web", $script:Utf8NoBom)
        $appEntry = [pscustomobject]@{ path = "app.py"; sha256 = Get-Sha256 (Join-Path $stage "app.py"); size = 3 }
        $webEntry = [pscustomobject]@{ path = "web/dist/index.html"; sha256 = Get-Sha256 (Join-Path $stage "web/dist/index.html"); size = 7 }
        $map = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::OrdinalIgnoreCase)
        $map.Add($appEntry.path, $appEntry)
        $map.Add($webEntry.path, $webEntry)
        $validated = [pscustomobject]@{ FileMap = $map; Files = @($appEntry, $webEntry) }
        $plan = Get-PortableChangePlan -Root $package -StageRoot $stage -ValidatedManifest $validated
        $transaction = Install-PortablePayload -Root $package -StageRoot $stage -Plan $plan `
            -OldVersion "0.0.1" -NewVersion "0.0.2" -WasRunning $false
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "app.py"))) -eq "new") "program file application"
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "web/dist/index.html"))) -eq "new-web") "web directory application"
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "data/user.txt"))) -eq "keep-data") "data preservation"
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "portable-settings.env"))) -eq "SECRET=keep") "secret preservation"
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "notes.txt"))) -eq "keep-unknown") "unknown file preservation"
        Undo-PortableTransaction -State $transaction.State -JournalPath $transaction.JournalPath `
            -ExpectedRoot $package
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "app.py"))) -eq "old") "program rollback"
        Assert-SelfTest (([System.IO.File]::ReadAllText((Join-Path $package "web/dist/index.html"))) -eq "old-web") "web rollback"

        $committedJournal = Join-Path $package "updates/state.json"
        Write-JsonFile -Path $committedJournal -Value ([ordered]@{
            schemaVersion = 1
            phase = "committed"
            packageRoot = $package
            oldVersion = "0.0.1"
            newVersion = "0.0.2"
            backupRoot = Join-Path $package "updates/backups/committed-test"
            files = @()
            directories = @()
        })
        Recover-PendingPortableTransaction $package
        Assert-SelfTest (-not (Test-Path -LiteralPath $committedJournal)) "committed journal finalization"
        Assert-SelfTest (Test-Path -LiteralPath (Join-Path $package "updates/last-update.json")) "last update record"
    }
    finally {
        Remove-VerifiedTree -Path $testRoot -Parent ([System.IO.Path]::GetTempPath())
    }
    Write-UpdateMessage "Updater self-tests passed."
}

try {
    if ($SelfTest) {
        Invoke-UpdaterSelfTest
        exit 0
    }
    if (-not $PackageRoot) {
        $PackageRoot = Split-Path -Parent $PSScriptRoot
    }
    $root = [System.IO.Path]::GetFullPath($PackageRoot).TrimEnd('\', '/')
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "XP-Gacha root does not exist: $root"
    }
    $isPortable = (Test-Path -LiteralPath (Join-Path $root "BUILD-INFO.json") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $root "runtime/python/python.exe") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $root "portable_launcher.py") -PathType Leaf)
    $isSource = (Test-Path -LiteralPath (Join-Path $root ".git")) -and
        (Test-Path -LiteralPath (Join-Path $root "docker-compose.yml") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $root "scripts/start.ps1") -PathType Leaf)
    if ($isPortable -eq $isSource) {
        throw "Could not uniquely identify this folder as an XP-Gacha source checkout or portable package."
    }
    Write-UpdateMessage "Checking the latest stable GitHub Release..."
    $release = Get-ReleaseMetadata
    $tag = Assert-StableRelease $release
    if ($isPortable) {
        Invoke-PortableUpdate -Root $root -Release $release -Tag $tag
    }
    else {
        Invoke-SourceUpdate -Root $root -Tag $tag
    }
    exit 0
}
catch {
    [Console]::Error.WriteLine("[XP-Gacha Update] ERROR: $($_.Exception.Message)")
    if ($env:XP_GACHA_UPDATE_DEBUG) {
        [Console]::Error.WriteLine($_.ScriptStackTrace)
    }
    exit 1
}
