[CmdletBinding()]
param(
    [string]$OutputRoot = "..\XP-Gacha-Releases",
    [string]$BuildPython = "python",
    [switch]$SkipFrontendBuild,
    [switch]$SkipVerification,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$PythonVersion = "3.13.15"
$PythonArchiveName = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonArchiveName"
$PythonSha256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"

$MySqlVersion = "8.4.11"
$MySqlArchiveName = "mysql-$MySqlVersion-winx64.zip"
$MySqlUrl = "https://cdn.mysql.com/Downloads/MySQL-8.4/$MySqlArchiveName"
$MySqlMd5 = "2e833921898a9a030ea6bfe81bd811bc"

$VcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
$WixVersion = "3.14.1"
$WixArchiveName = "wix314-binaries.zip"
$WixUrl = "https://github.com/wixtoolset/wix3/releases/download/wix3141rtm/$WixArchiveName"
$WixSha256 = "6ac824e1642d6f7277d0ed7ea09411a508f6116ba6fae0aa5f2c7daa2ff43d31"
$TorchIndexUrl = "https://download.pytorch.org/whl/cpu"

$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$PortableRuntimeVerifier = Join-Path $PSScriptRoot "verify_portable_runtime.ps1"
$PortableUpdateBuilder = Join-Path $PSScriptRoot "build_portable_update.ps1"
if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    $OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
}
$CacheRoot = Join-Path $ProjectRoot ".portable-cache"
$DownloadRoot = Join-Path $CacheRoot "downloads"
$PortableDataDirectories = @(
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
    "tmp"
)
$DefaultDictionaryFiles = @(
    "STOP_TAGS.txt",
    "SEMANTIC_MAP.json",
    "TITLE_STOP_WORDS.txt",
    "TITLE_SEMANTIC_MAP.json"
)

function Write-Step {
    param([string]$Message)
    Write-Host "[portable-build] $Message"
}

function Assert-ChildPath {
    param(
        [string]$Path,
        [string]$Parent
    )
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $fullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $prefix = $fullParent + [System.IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside output root: $fullPath"
    }
}

function Remove-VerifiedTree {
    param(
        [string]$Path,
        [string]$Parent
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    Assert-ChildPath -Path $Path -Parent $Parent
    try {
        Remove-Item -LiteralPath $Path -Recurse -Force
        return
    }
    catch {
        $removeError = $_
        Write-Step "PowerShell could not remove a long-path tree; retrying with the Windows file copier."
        $emptyRoot = Join-Path $Parent (".portable-delete-empty-" + [Guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $emptyRoot -Force | Out-Null
        try {
            & robocopy.exe $emptyRoot $Path /MIR /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
            $robocopyExitCode = $LASTEXITCODE
            if ($robocopyExitCode -gt 7) {
                throw "robocopy cleanup failed with exit code $robocopyExitCode"
            }
            Remove-Item -LiteralPath $Path -Force
        }
        catch {
            throw "Unable to remove verified tree $Path. Initial error: $($removeError.Exception.Message). Fallback error: $($_.Exception.Message)"
        }
        finally {
            Remove-Item -LiteralPath $emptyRoot -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-LowerHash {
    param(
        [string]$Path,
        [ValidateSet("SHA256", "MD5")]
        [string]$Algorithm
    )
    return (Get-FileHash -LiteralPath $Path -Algorithm $Algorithm).Hash.ToLowerInvariant()
}

function Get-VerifiedDownload {
    param(
        [string]$Url,
        [string]$Destination,
        [ValidateSet("SHA256", "MD5")]
        [string]$Algorithm,
        [string]$ExpectedHash
    )
    $ExpectedHash = $ExpectedHash.ToLowerInvariant()
    if (Test-Path -LiteralPath $Destination) {
        $existingHash = Get-LowerHash -Path $Destination -Algorithm $Algorithm
        if ($existingHash -eq $ExpectedHash) {
            Write-Step "Using cached $(Split-Path -Leaf $Destination)."
            return
        }
        Remove-Item -LiteralPath $Destination -Force
    }

    $temporary = "$Destination.download"
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    Write-Step "Downloading $Url"
    & curl.exe --fail --location --retry 5 --retry-delay 2 --output $temporary $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
    $actualHash = Get-LowerHash -Path $temporary -Algorithm $Algorithm
    if ($actualHash -ne $ExpectedHash) {
        Remove-Item -LiteralPath $temporary -Force
        throw "$Algorithm mismatch for $(Split-Path -Leaf $Destination): expected $ExpectedHash, got $actualHash"
    }
    Move-Item -LiteralPath $temporary -Destination $Destination
}

function Get-SignedMicrosoftDownload {
    param(
        [string]$Url,
        [string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Destination)) {
        $temporary = "$Destination.download"
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        Write-Step "Downloading Microsoft Visual C++ app-local runtime source."
        & curl.exe --fail --location --retry 5 --retry-delay 2 --output $temporary $Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $Url"
        }
        Move-Item -LiteralPath $temporary -Destination $Destination
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $Destination
    if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "Microsoft") {
        throw "Microsoft Visual C++ Redistributable signature validation failed: $($signature.Status)"
    }
    return $signature
}

function Invoke-RobocopyTree {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$ExtraExcludedFiles = @()
    )
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $arguments = @(
        $Source,
        $Destination,
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:2",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
        "/XD",
        "__pycache__",
        ".pytest_cache",
        "/XF",
        "*.pyc",
        "*.pyo",
        "title_translation_results.jsonl*",
        "title_translation_failed_results.jsonl*",
        "title_words_frequency.csv",
        "aggregated_tags.txt"
    )
    $arguments += $ExtraExcludedFiles
    & robocopy.exe @arguments | Out-Null
    $robocopyCode = $LASTEXITCODE
    if ($robocopyCode -gt 7) {
        throw "Robocopy failed for $Source with exit code $robocopyCode"
    }
}

function Copy-ApplicationFiles {
    param([string]$Destination)

    $rootFiles = @(
        "app.py",
        "config.py",
        "config_docker.py",
        "config_empty.py",
        "data_pipeline.py",
        "launcher.py",
        "ui_data_processing.py",
        "utils_chat.py",
        "utils_charts.py",
        "utils_core.py",
        "utils_cv.py",
        "utils_history.py",
        "utils_nlp.py",
        "utils_online_cover.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "Update XP-Gacha.cmd"
    )
    foreach ($relativePath in $rootFiles) {
        $source = Join-Path $ProjectRoot $relativePath
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Required application file is missing: $relativePath"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $relativePath) -Force
    }

    foreach ($directory in @("Integration", "data_get", "data_processing", "server", "tools", "UI-imgs")) {
        Invoke-RobocopyTree -Source (Join-Path $ProjectRoot $directory) -Destination (Join-Path $Destination $directory)
    }
    $dictionaryDestination = Join-Path $Destination "dictionaries"
    New-Item -ItemType Directory -Path $dictionaryDestination -Force | Out-Null
    foreach ($dictionaryFile in $DefaultDictionaryFiles) {
        $dictionarySource = Join-Path (Join-Path $ProjectRoot "dictionaries") $dictionaryFile
        if (-not (Test-Path -LiteralPath $dictionarySource -PathType Leaf)) {
            throw "Required default dictionary is missing: dictionaries\$dictionaryFile"
        }
        Copy-Item -LiteralPath $dictionarySource -Destination (Join-Path $dictionaryDestination $dictionaryFile) -Force
    }
    Invoke-RobocopyTree -Source (Join-Path $ProjectRoot "web\dist") -Destination (Join-Path $Destination "web\dist")

    foreach ($template in Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "portable") -File) {
        Copy-Item -LiteralPath $template.FullName -Destination (Join-Path $Destination $template.Name) -Force
    }
}

function Expand-PythonRuntime {
    param(
        [string]$Archive,
        [string]$Destination
    )
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    $pthFile = Get-ChildItem -LiteralPath $Destination -Filter "python*._pth" -File | Select-Object -First 1
    if (-not $pthFile) {
        throw "Python embeddable _pth file was not found."
    }
    $pthContent = @(
        "python313.zip",
        ".",
        "..\..",
        "Lib",
        "Lib\site-packages",
        "import site",
        ""
    ) -join "`n"
    [System.IO.File]::WriteAllText($pthFile.FullName, $pthContent, [System.Text.UTF8Encoding]::new($false))
    New-Item -ItemType Directory -Path (Join-Path $Destination "Lib\site-packages") -Force | Out-Null
}

function Expand-MySqlRuntime {
    param(
        [string]$Archive,
        [string]$Destination,
        [string]$TemporaryRoot
    )
    $extractRoot = Join-Path $TemporaryRoot "mysql-extract"
    Remove-VerifiedTree -Path $extractRoot -Parent $TemporaryRoot
    New-Item -ItemType Directory -Path $extractRoot -Force | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $extractRoot -Force
    $sourceRoot = Get-ChildItem -LiteralPath $extractRoot -Directory | Where-Object Name -eq "mysql-$MySqlVersion-winx64" | Select-Object -First 1
    if (-not $sourceRoot) {
        throw "Unexpected MySQL ZIP layout."
    }
    Invoke-RobocopyTree -Source $sourceRoot.FullName -Destination $Destination
    Remove-VerifiedTree -Path $extractRoot -Parent $TemporaryRoot
}

function Install-PythonDependencies {
    param(
        [string]$BuildPythonPath,
        [string]$TargetPython,
        [string]$ReleaseRoot
    )
    $versionJson = & $BuildPythonPath -c "import json, struct, sys; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'bits':struct.calcsize('P')*8}))"
    if ($LASTEXITCODE -ne 0) {
        throw "Build Python could not run: $BuildPythonPath"
    }
    $version = $versionJson | ConvertFrom-Json
    if ($version.major -ne 3 -or $version.minor -ne 13 -or $version.bits -ne 64) {
        throw "Build Python must be CPython 3.13 x64."
    }

    $oldDisableVersionCheck = $env:PIP_DISABLE_PIP_VERSION_CHECK
    $oldNoCache = $env:PIP_NO_CACHE_DIR
    $oldNoUserSite = $env:PYTHONNOUSERSITE
    $oldPythonPath = $env:PYTHONPATH
    $oldPipConfigFile = $env:PIP_CONFIG_FILE
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_NO_CACHE_DIR = "0"
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONPATH = ""
    $env:PIP_CONFIG_FILE = [System.IO.Path]::GetFullPath((Join-Path $CacheRoot "pip-empty-config.ini"))
    if (-not (Test-Path -LiteralPath $env:PIP_CONFIG_FILE)) {
        [System.IO.File]::WriteAllText($env:PIP_CONFIG_FILE, "", [System.Text.UTF8Encoding]::new($false))
    }
    try {
        Write-Step "Installing the CPU-only PyTorch runtime into embedded Python."
        & $BuildPythonPath -m pip --isolated --python $TargetPython install --upgrade --only-binary=:all: --no-compile --no-warn-script-location --cache-dir (Join-Path $CacheRoot "pip") --index-url $TorchIndexUrl torch
        if ($LASTEXITCODE -ne 0) {
            throw "CPU-only PyTorch installation failed."
        }

        Write-Step "Installing all XP-Gacha Python dependencies into embedded Python."
        & $BuildPythonPath -m pip --isolated --python $TargetPython install --upgrade --upgrade-strategy only-if-needed --prefer-binary --no-compile --no-warn-script-location --cache-dir (Join-Path $CacheRoot "pip") -r (Join-Path $ProjectRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Python dependency installation failed."
        }

        & $TargetPython -X utf8 -c "import fastapi, numpy, pandas, scipy, sqlalchemy, streamlit, torch; print('embedded dependency bootstrap OK')"
        if ($LASTEXITCODE -ne 0) {
            throw "The embedded Python environment still depends on the build machine."
        }

        $locked = & $BuildPythonPath -m pip --isolated --python $TargetPython freeze
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to generate Python dependency lock file."
        }
        [System.IO.File]::WriteAllLines(
            (Join-Path $ReleaseRoot "requirements-lock.txt"),
            [string[]]$locked,
            [System.Text.UTF8Encoding]::new($false)
        )
    }
    finally {
        $env:PIP_DISABLE_PIP_VERSION_CHECK = $oldDisableVersionCheck
        $env:PIP_NO_CACHE_DIR = $oldNoCache
        $env:PYTHONNOUSERSITE = $oldNoUserSite
        $env:PYTHONPATH = $oldPythonPath
        $env:PIP_CONFIG_FILE = $oldPipConfigFile
    }
}

function Install-AppLocalVcRuntime {
    param(
        [string]$RedistInstaller,
        [string]$ReleaseRoot,
        [string]$WixArchive
    )
    $wixRoot = Join-Path $CacheRoot "tools\wix-$WixVersion"
    $dark = Join-Path $wixRoot "dark.exe"
    $bundleRoot = Join-Path $CacheRoot "vc-redist-bundle"
    $cabRoot = Join-Path $CacheRoot "vc-redist-cab"
    $runtimeNames = @(
        "concrt140.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "vcruntime140_threads.dll"
    )

    if (-not (Test-Path -LiteralPath $dark)) {
        Remove-VerifiedTree -Path $wixRoot -Parent $CacheRoot
        New-Item -ItemType Directory -Path $wixRoot -Force | Out-Null
        Write-Step "Extracting the pinned WiX Toolset build utility."
        Expand-Archive -LiteralPath $WixArchive -DestinationPath $wixRoot -Force
    }
    if (-not (Test-Path -LiteralPath $dark)) {
        throw "WiX dark.exe was not found after extracting $WixArchive."
    }

    Remove-VerifiedTree -Path $bundleRoot -Parent $CacheRoot
    Remove-VerifiedTree -Path $cabRoot -Parent $CacheRoot
    New-Item -ItemType Directory -Path $bundleRoot, $cabRoot -Force | Out-Null

    Write-Step "Extracting the signed Microsoft Visual C++ runtime for app-local deployment."
    & $dark -nologo -x $bundleRoot $RedistInstaller
    if ($LASTEXITCODE -ne 0) {
        throw "WiX bundle extraction failed with exit code $LASTEXITCODE."
    }
    $minimumCab = Join-Path $bundleRoot "AttachedContainer\packages\vcRuntimeMinimum_amd64\cab1.cab"
    if (-not (Test-Path -LiteralPath $minimumCab)) {
        throw "Visual C++ Redistributable bundle did not contain the expected x64 runtime CAB."
    }
    & expand.exe -F:* $minimumCab $cabRoot | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Visual C++ runtime CAB extraction failed with exit code $LASTEXITCODE."
    }

    $available = @{}
    foreach ($runtimeName in $runtimeNames) {
        $source = Join-Path $cabRoot "${runtimeName}_amd64"
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Required app-local Visual C++ runtime DLL was not extracted: $runtimeName"
        }
        $signature = Get-AuthenticodeSignature -LiteralPath $source
        if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "Microsoft") {
            throw "App-local Visual C++ runtime DLL signature is invalid: $runtimeName"
        }
        $available[$runtimeName] = $source
    }

    $vcNoticeRoot = Join-Path $ReleaseRoot "runtime\vc-runtime"
    New-Item -ItemType Directory -Path $vcNoticeRoot -Force | Out-Null
    foreach ($entry in $available.GetEnumerator()) {
        Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $ReleaseRoot "runtime\mysql\bin\$($entry.Key)") -Force
        Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $ReleaseRoot "runtime\python\$($entry.Key)") -Force
        Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $vcNoticeRoot $entry.Key) -Force
    }
    $license = Get-ChildItem -LiteralPath $bundleRoot -File -Recurse | Where-Object Name -Match "license|eula" | Select-Object -First 1
    if ($license) {
        Copy-Item -LiteralPath $license.FullName -Destination (Join-Path $vcNoticeRoot $license.Name) -Force
    }
}

function Invoke-PortableVerification {
    param([string]$ReleaseRoot)
    $python = Join-Path $ReleaseRoot "runtime\python\python.exe"
    $launcher = Join-Path $ReleaseRoot "portable_launcher.py"
    $oldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
    $oldNoUserSite = $env:PYTHONNOUSERSITE
    $oldUtf8 = $env:PYTHONUTF8
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    try {
        Write-Step "Verifying bundled application module resolution."
        & $PortableRuntimeVerifier -ReleaseRoot $ReleaseRoot
        Write-Step "Running bundled dependency doctor."
        & $python -X utf8 $launcher doctor
        if ($LASTEXITCODE -ne 0) {
            throw "Portable dependency doctor failed."
        }
        Write-Step "Running blank first-start MySQL and API smoke test."
        & $python -X utf8 $launcher verify
        if ($LASTEXITCODE -ne 0) {
            throw "Portable first-start smoke test failed."
        }
        Write-Step "Running provisioned database restart smoke test."
        & $python -X utf8 $launcher verify
        if ($LASTEXITCODE -ne 0) {
            throw "Portable restart smoke test failed."
        }
    }
    finally {
        $env:PYTHONDONTWRITEBYTECODE = $oldNoBytecode
        $env:PYTHONNOUSERSITE = $oldNoUserSite
        $env:PYTHONUTF8 = $oldUtf8
    }
}

function Reset-ReleaseDataDirectories {
    param([string]$ReleaseRoot)
    foreach ($relativePath in $PortableDataDirectories) {
        $target = Join-Path $ReleaseRoot $relativePath
        Remove-VerifiedTree -Path $target -Parent $ReleaseRoot
        New-Item -ItemType Directory -Path $target -Force | Out-Null
    }
    foreach ($relativePath in @(
        "data\gallery_info",
        "data\gallery_info_origin",
        "data\local_data",
        "datacache\imports",
        "models\cache",
        "mysql\data",
        "updates\backups"
    )) {
        New-Item -ItemType Directory -Path (Join-Path $ReleaseRoot $relativePath) -Force | Out-Null
    }
}

function Assert-NoProjectData {
    param([string]$ReleaseRoot)
    $forbiddenPaths = @(
        ".git",
        ".streamlit",
        ".env",
        ".env.local",
        "userdata",
        "web\node_modules"
    )
    foreach ($relativePath in $forbiddenPaths) {
        if (Test-Path -LiteralPath (Join-Path $ReleaseRoot $relativePath)) {
            throw "Data exclusion audit failed: $relativePath is present in the release root."
        }
    }
    foreach ($relativePath in $PortableDataDirectories) {
        $dataDirectory = Join-Path $ReleaseRoot $relativePath
        if (-not (Test-Path -LiteralPath $dataDirectory -PathType Container)) {
            throw "Data exclusion audit failed: blank data directory is missing: $relativePath"
        }
        $unexpectedFiles = Get-ChildItem -LiteralPath $dataDirectory -File -Recurse -Force -ErrorAction SilentlyContinue
        if ($unexpectedFiles) {
            throw "Data exclusion audit failed: $relativePath contains packaged user files."
        }
    }
    $dictionaryRoot = Join-Path $ReleaseRoot "dictionaries"
    $packagedDictionaryFiles = @(
        Get-ChildItem -LiteralPath $dictionaryRoot -File -Force -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name
    )
    $unexpectedDictionaries = @($packagedDictionaryFiles | Where-Object { $_ -notin $DefaultDictionaryFiles })
    $missingDictionaries = @($DefaultDictionaryFiles | Where-Object { $_ -notin $packagedDictionaryFiles })
    if ($unexpectedDictionaries -or $missingDictionaries) {
        throw (
            "Default dictionary audit failed. Missing: {0}; unexpected: {1}" -f
            ($missingDictionaries -join ", "),
            ($unexpectedDictionaries -join ", ")
        )
    }
    $sourcePycache = Get-ChildItem -LiteralPath $ReleaseRoot -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { -not $_.FullName.StartsWith((Join-Path $ReleaseRoot "runtime"), [System.StringComparison]::OrdinalIgnoreCase) }
    if ($sourcePycache) {
        throw "Data exclusion audit failed: source __pycache__ directories are present."
    }
}

function Write-ReleaseMetadata {
    param(
        [string]$ReleaseRoot,
        [string]$ReleaseVersion,
        [string]$PythonArchive,
        [string]$MySqlArchive,
        [string]$VcRedist,
        [string]$WixArchive,
        [System.Management.Automation.Signature]$VcSignature,
        [bool]$Verified
    )
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $sourceCommit = (& git -C $ProjectRoot rev-parse HEAD 2>$null)
        $commitExitCode = $LASTEXITCODE
        $sourceStatus = (& git -C $ProjectRoot status --porcelain 2>$null)
        $statusExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($commitExitCode -ne 0) {
        $sourceCommit = ""
    }
    $sourceDirty = $statusExitCode -ne 0 -or [bool]$sourceStatus
    $metadata = [ordered]@{
        product = "XP-Gacha"
        version = $ReleaseVersion
        platform = "windows-x64"
        dataLayout = "project-root"
        portableDataDirectories = @($PortableDataDirectories + @("dictionaries"))
        modelCacheDirectory = "models/cache"
        createdAtUtc = [DateTime]::UtcNow.ToString("o")
        sourceCommit = [string]$sourceCommit
        sourceDirty = $sourceDirty
        blankFirstStartVerified = $Verified
        runtime = [ordered]@{
            python = [ordered]@{
                version = $PythonVersion
                source = $PythonUrl
                archiveSha256 = Get-LowerHash -Path $PythonArchive -Algorithm SHA256
            }
            mysql = [ordered]@{
                version = $MySqlVersion
                source = $MySqlUrl
                archiveMd5 = Get-LowerHash -Path $MySqlArchive -Algorithm MD5
                archiveSha256 = Get-LowerHash -Path $MySqlArchive -Algorithm SHA256
            }
            pytorch = [ordered]@{
                variant = "cpu"
                index = $TorchIndexUrl
            }
            visualCppRuntime = [ordered]@{
                source = $VcRedistUrl
                installerSha256 = Get-LowerHash -Path $VcRedist -Algorithm SHA256
                signer = $VcSignature.SignerCertificate.Subject
                thumbprint = $VcSignature.SignerCertificate.Thumbprint
            }
            buildTools = [ordered]@{
                wix = [ordered]@{
                    version = $WixVersion
                    source = $WixUrl
                    archiveSha256 = Get-LowerHash -Path $WixArchive -Algorithm SHA256
                    shippedInRelease = $false
                }
            }
        }
        excludedRuntimeData = @(
            "catalogue CSV/database",
            "covers and Base64 caches",
            "history and UI preferences",
            "models and vectors",
            "logs and temporary files",
            "local secrets"
        )
    }
    $json = $metadata | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        (Join-Path $ReleaseRoot "BUILD-INFO.json"),
        $json + "`n",
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Write-FileManifest {
    param([string]$ReleaseRoot)
    $manifestPath = Join-Path $ReleaseRoot "SHA256SUMS.txt"
    $releasePrefix = [System.IO.Path]::GetFullPath($ReleaseRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $entries = [System.Collections.Generic.List[string]]::new()
    $files = Get-ChildItem -LiteralPath $ReleaseRoot -File -Recurse |
        Where-Object FullName -ne $manifestPath |
        Sort-Object FullName
    $count = 0
    foreach ($file in $files) {
        $fullFilePath = [System.IO.Path]::GetFullPath($file.FullName)
        if (-not $fullFilePath.StartsWith($releasePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Manifest entry is outside the release root: $fullFilePath"
        }
        $relative = $fullFilePath.Substring($releasePrefix.Length).Replace('\', '/')
        $hash = Get-LowerHash -Path $file.FullName -Algorithm SHA256
        $entries.Add("$hash  $relative")
        $count += 1
        if ($count % 2500 -eq 0) {
            Write-Step "Hashed $count files for the release manifest."
        }
    }
    [System.IO.File]::WriteAllLines($manifestPath, $entries, [System.Text.UTF8Encoding]::new($false))
}

New-Item -ItemType Directory -Path $OutputRoot, $DownloadRoot -Force | Out-Null

$versionFile = Get-Content -LiteralPath (Join-Path $ProjectRoot "server\__init__.py") -Raw
$versionMatch = [regex]::Match($versionFile, '__version__\s*=\s*["'']([^"'']+)["'']')
$ReleaseVersion = if ($versionMatch.Success) { $versionMatch.Groups[1].Value } else { "0.0.0" }
$sourceHead = (& git -C $ProjectRoot rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $sourceHead) {
    throw "The portable release must be built from a Git checkout."
}
$sourceHead = ([string]$sourceHead).Trim()
$sourceStatus = @(& git -C $ProjectRoot status --porcelain --untracked-files=all 2>$null)
if ($LASTEXITCODE -ne 0 -or $sourceStatus.Count -gt 0) {
    throw "The portable release requires a clean source tree. Commit the release before building."
}
$releaseTagRef = "refs/tags/v$ReleaseVersion^{}"
$releaseTagCommit = (& git -C $ProjectRoot rev-parse $releaseTagRef 2>$null)
if ($LASTEXITCODE -ne 0 -or ([string]$releaseTagCommit).Trim() -ne $sourceHead) {
    throw "Tag v$ReleaseVersion must exist and resolve to current HEAD $sourceHead before building the portable release."
}
$ReleaseName = "XP-Gacha-v$ReleaseVersion-portable-win64"
$FinalRoot = Join-Path $OutputRoot $ReleaseName
$ZipPath = Join-Path $OutputRoot "$ReleaseName.zip"
$ZipSidecarPath = "$ZipPath.sha256"
$UpdateAssetBase = "$ReleaseName-update"
$UpdateZipPath = Join-Path $OutputRoot "$UpdateAssetBase.zip"
$UpdateManifestPath = Join-Path $OutputRoot "$UpdateAssetBase.json"
$UpdateSidecarPath = "$UpdateZipPath.sha256"
$StagingRoot = Join-Path $OutputRoot (".build-" + [guid]::NewGuid().ToString("N").Substring(0, 10))

Assert-ChildPath -Path $FinalRoot -Parent $OutputRoot
Assert-ChildPath -Path $ZipPath -Parent $OutputRoot
Assert-ChildPath -Path $ZipSidecarPath -Parent $OutputRoot
Assert-ChildPath -Path $UpdateZipPath -Parent $OutputRoot
Assert-ChildPath -Path $UpdateManifestPath -Parent $OutputRoot
Assert-ChildPath -Path $UpdateSidecarPath -Parent $OutputRoot
Assert-ChildPath -Path $StagingRoot -Parent $OutputRoot

$releaseArtifacts = @(
    $FinalRoot,
    $ZipPath,
    $ZipSidecarPath,
    $UpdateZipPath,
    $UpdateManifestPath,
    $UpdateSidecarPath
)
$existingReleaseArtifacts = @($releaseArtifacts | Where-Object { Test-Path -LiteralPath $_ })
if ($existingReleaseArtifacts.Count -gt 0) {
    if (-not $Force) {
        throw "Release assets already exist for version $ReleaseVersion. Use -Force to replace the complete $ReleaseName asset set."
    }
    foreach ($artifact in $releaseArtifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            continue
        }
        if (Test-Path -LiteralPath $artifact -PathType Container) {
            Remove-VerifiedTree -Path $artifact -Parent $OutputRoot
        }
        else {
            Remove-Item -LiteralPath $artifact -Force
        }
    }
}

New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null

try {
    if (-not $SkipFrontendBuild) {
        Write-Step "Building the React frontend."
        & pnpm --dir (Join-Path $ProjectRoot "web") build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed."
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "web\dist\index.html"))) {
        throw "web/dist/index.html is missing."
    }

    Write-Step "Copying the data-free application allowlist."
    Copy-ApplicationFiles -Destination $StagingRoot

    $pythonArchive = Join-Path $DownloadRoot $PythonArchiveName
    $mysqlArchive = Join-Path $DownloadRoot $MySqlArchiveName
    $vcRedist = Join-Path $DownloadRoot "vc_redist.x64.exe"
    $wixArchive = Join-Path $DownloadRoot $WixArchiveName
    Get-VerifiedDownload -Url $PythonUrl -Destination $pythonArchive -Algorithm SHA256 -ExpectedHash $PythonSha256
    Get-VerifiedDownload -Url $MySqlUrl -Destination $mysqlArchive -Algorithm MD5 -ExpectedHash $MySqlMd5
    $vcSignature = Get-SignedMicrosoftDownload -Url $VcRedistUrl -Destination $vcRedist
    Get-VerifiedDownload -Url $WixUrl -Destination $wixArchive -Algorithm SHA256 -ExpectedHash $WixSha256

    Write-Step "Extracting CPython $PythonVersion embedded runtime."
    Expand-PythonRuntime -Archive $pythonArchive -Destination (Join-Path $StagingRoot "runtime\python")
    Write-Step "Extracting MySQL $MySqlVersion noinstall runtime."
    Expand-MySqlRuntime -Archive $mysqlArchive -Destination (Join-Path $StagingRoot "runtime\mysql") -TemporaryRoot $StagingRoot

    if (-not (Test-Path -LiteralPath $BuildPython)) {
        $resolvedBuildPython = (Get-Command $BuildPython -ErrorAction Stop).Source
    }
    else {
        $resolvedBuildPython = [System.IO.Path]::GetFullPath($BuildPython)
    }
    Install-PythonDependencies -BuildPythonPath $resolvedBuildPython -TargetPython (Join-Path $StagingRoot "runtime\python\python.exe") -ReleaseRoot $StagingRoot
    Install-AppLocalVcRuntime -RedistInstaller $vcRedist -ReleaseRoot $StagingRoot -WixArchive $wixArchive

    $verified = $false
    if (-not $SkipVerification) {
        Invoke-PortableVerification -ReleaseRoot $StagingRoot
        $verified = $true
    }
    Reset-ReleaseDataDirectories -ReleaseRoot $StagingRoot
    Assert-NoProjectData -ReleaseRoot $StagingRoot
    Write-ReleaseMetadata -ReleaseRoot $StagingRoot -ReleaseVersion $ReleaseVersion -PythonArchive $pythonArchive -MySqlArchive $mysqlArchive -VcRedist $vcRedist -WixArchive $wixArchive -VcSignature $vcSignature -Verified $verified

    Write-Step "Writing per-file SHA256 manifest."
    Write-FileManifest -ReleaseRoot $StagingRoot

    Move-Item -LiteralPath $StagingRoot -Destination $FinalRoot
    Write-Step "Creating ZIP archive."
    Push-Location $OutputRoot
    try {
        & tar.exe -a -c -f (Split-Path -Leaf $ZipPath) $ReleaseName
        if ($LASTEXITCODE -ne 0) {
            throw "ZIP creation failed."
        }
    }
    finally {
        Pop-Location
    }
    $zipHash = Get-LowerHash -Path $ZipPath -Algorithm SHA256
    [System.IO.File]::WriteAllText(
        $ZipSidecarPath,
        "$zipHash  $(Split-Path -Leaf $ZipPath)`n",
        [System.Text.UTF8Encoding]::new($false)
    )

    if (-not (Test-Path -LiteralPath $PortableUpdateBuilder -PathType Leaf)) {
        throw "Portable update builder is missing: $PortableUpdateBuilder"
    }
    Write-Step "Creating the data-safe incremental update assets."
    $updateResult = & $PortableUpdateBuilder `
        -ReleaseRoot $FinalRoot `
        -OutputRoot $OutputRoot `
        -Version $ReleaseVersion

    $releaseSize = (Get-ChildItem -LiteralPath $FinalRoot -File -Recurse | Measure-Object Length -Sum).Sum
    $zipSize = (Get-Item -LiteralPath $ZipPath).Length
    Write-Step "Release complete."
    Write-Host "Folder: $FinalRoot"
    Write-Host "ZIP:    $ZipPath"
    Write-Host ("Folder size: {0:N2} GiB" -f ($releaseSize / 1GB))
    Write-Host ("ZIP size:    {0:N2} GiB" -f ($zipSize / 1GB))
    Write-Host "ZIP SHA256:  $zipHash"
    Write-Host "Update ZIP:  $($updateResult.PackagePath)"
    Write-Host "Update JSON: $($updateResult.ManifestPath)"
    Write-Host "Update SUM:  $($updateResult.SidecarPath)"
    Write-Host "Update SHA:  $($updateResult.PackageSha256)"
}
catch {
    Write-Host "[portable-build] ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if (Test-Path -LiteralPath $StagingRoot) {
        Remove-VerifiedTree -Path $StagingRoot -Parent $OutputRoot
    }
    throw
}
