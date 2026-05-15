# scripts/install_tools.ps1
#
# Bootstrap the bundled toolchain for Fermata Masterclass on Windows x64.
#
# Downloads and installs:
#   - Python 3.11.9 (embeddable)         -> tools\python\
#   - FFmpeg (gyan release-essentials)   -> tools\ffmpeg\
#   - Eclipse Temurin JDK 21             -> tools\jre\
#   - Audiveris 5.6.2 (extracted MSI)    -> tools\audiveris\
#
# All downloads are cached in tools\downloads\ so re-runs are cheap.
# Existing valid installations are skipped (idempotent).
#
# Total download size: ~600 MB. Total disk after install: ~2 GB.
#
# Usage:
#   .\scripts\install_tools.ps1                # install everything
#   .\scripts\install_tools.ps1 -Force         # re-install even if present
#   .\scripts\install_tools.ps1 -Only ffmpeg   # install only one tool

[CmdletBinding()]
param(
    [switch] $Force,
    [ValidateSet('all', 'python', 'ffmpeg', 'jre', 'audiveris')]
    [string] $Only = 'all'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # huge perf win for Invoke-WebRequest

# ---------- paths ----------
$RepoRoot     = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ToolsDir     = Join-Path $RepoRoot 'tools'
$DownloadsDir = Join-Path $ToolsDir 'downloads'

New-Item -ItemType Directory -Force -Path $ToolsDir, $DownloadsDir | Out-Null

# ---------- versions / URLs ----------
$PythonVersion    = '3.11.9'
$PythonZipName    = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl        = "https://www.python.org/ftp/python/$PythonVersion/$PythonZipName"
$GetPipUrl        = 'https://bootstrap.pypa.io/get-pip.py'

$FfmpegZipName    = 'ffmpeg-release-essentials.zip'
$FfmpegUrl        = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'

# Temurin JDK 21 (LTS). The Adoptium API redirects to the latest GA release.
$JreZipName       = 'OpenJDK21U-jdk_x64_windows_hotspot.zip'
$JreUrl           = 'https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal/eclipse'

$AudiverisVersion = '5.6.2'
$AudiverisMsiName = "Audiveris-$AudiverisVersion-windowsConsole-x86_64.msi"
$AudiverisUrl     = "https://github.com/Audiveris/audiveris/releases/download/$AudiverisVersion/$AudiverisMsiName"

# ---------- helpers ----------
function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}
function Write-Ok([string]$msg) {
    Write-Host "    [ok] $msg" -ForegroundColor Green
}
function Write-Skip([string]$msg) {
    Write-Host "    [skip] $msg" -ForegroundColor DarkGray
}

function Download-IfMissing([string]$url, [string]$destPath) {
    if ((Test-Path $destPath) -and -not $Force) {
        $sizeMb = [math]::Round((Get-Item $destPath).Length / 1MB, 1)
        Write-Skip "$([System.IO.Path]::GetFileName($destPath)) already cached ($sizeMb MB)"
        return
    }
    Write-Host "    downloading $url"
    $tmp = "$destPath.part"
    Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    Move-Item -Force $tmp $destPath
    $sizeMb = [math]::Round((Get-Item $destPath).Length / 1MB, 1)
    Write-Ok "downloaded $([System.IO.Path]::GetFileName($destPath)) ($sizeMb MB)"
}

function Should-Run([string]$name) {
    return ($Only -eq 'all') -or ($Only -eq $name)
}

# ============================================================
# Python 3.11 embeddable
# ============================================================
function Install-Python {
    $pythonDir = Join-Path $ToolsDir 'python'
    $pythonExe = Join-Path $pythonDir 'python.exe'

    if ((Test-Path $pythonExe) -and -not $Force) {
        $ver = (cmd /c "`"$pythonExe`" --version 2>&1")
        Write-Skip "$ver already at tools\python\"
        return
    }

    Write-Step "Installing Python $PythonVersion (embeddable)"
    $zip = Join-Path $DownloadsDir $PythonZipName
    Download-IfMissing $PythonUrl $zip

    if (Test-Path $pythonDir) { Remove-Item -Recurse -Force $pythonDir }
    Expand-Archive -Path $zip -DestinationPath $pythonDir -Force

    # The embeddable distro disables `import site` by default. Enable it so
    # pip-installed packages are found.
    $pthFile = Get-ChildItem $pythonDir -Filter 'python*._pth' | Select-Object -First 1
    if ($pthFile) {
        (Get-Content $pthFile.FullName) `
            -replace '^#\s*import site', 'import site' `
            | Set-Content $pthFile.FullName
        Write-Ok "enabled site-packages in $($pthFile.Name)"
    }

    # Bootstrap pip.
    $getPip = Join-Path $DownloadsDir 'get-pip.py'
    Download-IfMissing $GetPipUrl $getPip
    & $pythonExe $getPip --no-warn-script-location | Out-Null
    Write-Ok "installed pip"

    & $pythonExe --version
}

# ============================================================
# FFmpeg (gyan release-essentials build)
# ============================================================
function Install-Ffmpeg {
    $ffmpegDir = Join-Path $ToolsDir 'ffmpeg'
    $ffmpegExe = Join-Path $ffmpegDir 'bin\ffmpeg.exe'

    if ((Test-Path $ffmpegExe) -and -not $Force) {
        $ver = (cmd /c "`"$ffmpegExe`" -version 2>&1" | Select-Object -First 1)
        Write-Skip "$ver already at tools\ffmpeg\"
        return
    }

    Write-Step "Installing FFmpeg"
    $zip = Join-Path $DownloadsDir $FfmpegZipName
    Download-IfMissing $FfmpegUrl $zip

    # Extract to a staging dir, then promote the inner ffmpeg-*-essentials_build/
    # contents up to tools\ffmpeg\.
    $staging = Join-Path $DownloadsDir '_ffmpeg_stage'
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    Expand-Archive -Path $zip -DestinationPath $staging -Force
    $inner = Get-ChildItem $staging -Directory | Select-Object -First 1
    if (-not $inner) { throw "Could not find FFmpeg root inside $zip" }

    if (Test-Path $ffmpegDir) { Remove-Item -Recurse -Force $ffmpegDir }
    Move-Item -Force $inner.FullName $ffmpegDir
    Remove-Item -Recurse -Force $staging

    cmd /c "`"$ffmpegExe`" -version 2>&1" | Select-Object -First 1
    Write-Ok "ffmpeg + ffprobe ready at tools\ffmpeg\bin\"
}

# ============================================================
# Eclipse Temurin JDK 21
# ============================================================
function Install-Jre {
    $jreDir = Join-Path $ToolsDir 'jre'
    $javaExe = Join-Path $jreDir 'bin\java.exe'

    if ((Test-Path $javaExe) -and -not $Force) {
        $ver = (cmd /c "`"$javaExe`" -version 2>&1" | Select-Object -First 1)
        Write-Skip "$ver already at tools\jre\"
        return
    }

    Write-Step "Installing Eclipse Temurin JDK 21"
    $zip = Join-Path $DownloadsDir $JreZipName
    Download-IfMissing $JreUrl $zip

    $staging = Join-Path $DownloadsDir '_jdk_stage'
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    Expand-Archive -Path $zip -DestinationPath $staging -Force
    $inner = Get-ChildItem $staging -Directory | Select-Object -First 1
    if (-not $inner) { throw "Could not find JDK root inside $zip" }

    if (Test-Path $jreDir) { Remove-Item -Recurse -Force $jreDir }
    Move-Item -Force $inner.FullName $jreDir
    Remove-Item -Recurse -Force $staging

    cmd /c "`"$javaExe`" -version 2>&1" | Select-Object -First 1
    Write-Ok "java ready at tools\jre\bin\"
}

# ============================================================
# Audiveris 5.6.2 (extract MSI without admin install)
# ============================================================
function Install-Audiveris {
    $audDir = Join-Path $ToolsDir 'audiveris'
    $audJar = Join-Path $audDir 'Audiveris\app\audiveris.jar'

    if ((Test-Path $audJar) -and -not $Force) {
        Write-Skip "Audiveris $AudiverisVersion already at tools\audiveris\"
        return
    }

    Write-Step "Installing Audiveris $AudiverisVersion"
    $msi = Join-Path $DownloadsDir $AudiverisMsiName
    Download-IfMissing $AudiverisUrl $msi

    if (Test-Path $audDir) { Remove-Item -Recurse -Force $audDir }
    New-Item -ItemType Directory -Force -Path $audDir | Out-Null

    # msiexec /a ... TARGETDIR=... performs an administrative install, which
    # extracts the MSI payload to TARGETDIR without registering the product
    # in Windows or requiring admin privileges.
    Write-Host "    extracting MSI (no admin required)..."
    $logFile = Join-Path $DownloadsDir 'audiveris_msiexec.log'
    $msiArgs = @(
        '/a',
        "`"$msi`"",
        '/qn',
        "TARGETDIR=`"$audDir`"",
        '/L*v',
        "`"$logFile`""
    )
    $proc = Start-Process -Wait -PassThru -NoNewWindow -FilePath 'msiexec.exe' -ArgumentList $msiArgs
    if ($proc.ExitCode -ne 0) {
        throw "msiexec failed with exit code $($proc.ExitCode); see $logFile"
    }

    # The administrative install writes both the original MSI and the payload
    # into TARGETDIR. Drop the redundant MSI copy to save ~50 MB.
    $extractedMsi = Join-Path $audDir $AudiverisMsiName
    if (Test-Path $extractedMsi) { Remove-Item -Force $extractedMsi }

    if (-not (Test-Path $audJar)) {
        throw "Audiveris install completed but $audJar is missing. See $logFile."
    }
    Write-Ok "Audiveris ready at $audJar"
}

# ============================================================
# main
# ============================================================
Write-Host ""
Write-Host "Fermata Masterclass — toolchain installer" -ForegroundColor White
Write-Host "Repo root : $RepoRoot"
Write-Host "Tools dir : $ToolsDir"
Write-Host "Cache dir : $DownloadsDir"

$start = Get-Date

if (Should-Run 'python')    { Install-Python }
if (Should-Run 'ffmpeg')    { Install-Ffmpeg }
if (Should-Run 'jre')       { Install-Jre }
if (Should-Run 'audiveris') { Install-Audiveris }

# Quick sanity probe
Write-Step "Verifying installation"
$python = Join-Path $ToolsDir 'python\python.exe'
$ffmpeg = Join-Path $ToolsDir 'ffmpeg\bin\ffmpeg.exe'
$java   = Join-Path $ToolsDir 'jre\bin\java.exe'
$audJar = Join-Path $ToolsDir 'audiveris\Audiveris\app\audiveris.jar'

$ok = $true
foreach ($p in @($python, $ffmpeg, $java, $audJar)) {
    if (Test-Path $p) {
        Write-Ok $p
    } else {
        Write-Host "    [missing] $p" -ForegroundColor Red
        $ok = $false
    }
}

$elapsed = [int]((Get-Date) - $start).TotalSeconds
Write-Host ""
if ($ok) {
    Write-Host "All tools installed in $elapsed s." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host '  tools\python\python.exe -m pip install -e ".[api,llm]"'
    Write-Host "  Copy-Item .env.example .env  # then edit GEMINI_API_KEY"
    Write-Host '  tools\python\python.exe -m uvicorn masterclass.apps.api.main:create_app --factory --host 127.0.0.1 --port 8770'
} else {
    Write-Host "Some tools are missing. See errors above." -ForegroundColor Red
    exit 1
}
