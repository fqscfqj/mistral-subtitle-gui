param(
    [switch]$NoPause,
    [switch]$NoOpenDist
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Get-ParentProcessName {
    try {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
        if (-not $processInfo -or -not $processInfo.ParentProcessId) {
            return $null
        }
        return (Get-Process -Id $processInfo.ParentProcessId -ErrorAction Stop).ProcessName
    }
    catch {
        return $null
    }
}

function Wait-ForExitIfNeeded {
    param(
        [string]$Message = "按 Enter 键关闭此窗口"
    )

    if ($NoPause -or -not $script:IsExplorerLaunch) {
        return
    }

    Write-Host ""
    Read-Host $Message | Out-Null
}

$script:IsExplorerLaunch = (Get-ParentProcessName) -eq "explorer"
$buildSucceeded = $false
$exitCode = 0

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$specPath = Join-Path $repoRoot "mistral_subtitle_gui.spec"
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$distExePath = Join-Path $repoRoot "dist\\MistralSubtitleStudio.exe"

$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip"
$tempRoot = Join-Path $repoRoot "build_tmp"
$zipPath = Join-Path $tempRoot "ffmpeg.zip"
$extractDir = Join-Path $tempRoot "ffmpeg_extract"
$buildAssetsDir = Join-Path $repoRoot "build_assets"
$bundledFfmpegPath = Join-Path $buildAssetsDir "ffmpeg.exe"
$bundledFfprobePath = Join-Path $buildAssetsDir "ffprobe.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "未找到虚拟环境 Python：$pythonExe"
}
if (-not (Test-Path -LiteralPath $specPath)) {
    throw "未找到 PyInstaller spec 文件：$specPath"
}
if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "未找到依赖文件：$requirementsPath"
}

try {
    Write-Host "[1/4] 安装构建依赖..."
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -r $requirementsPath pyinstaller

    Write-Host "[2/4] 下载并准备 ffmpeg / ffprobe..."
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $buildAssetsDir) {
        Remove-Item -LiteralPath $buildAssetsDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    New-Item -ItemType Directory -Path $extractDir | Out-Null
    New-Item -ItemType Directory -Path $buildAssetsDir | Out-Null

    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $ffmpegCandidate = Get-ChildItem -Path $extractDir -Recurse -File -Filter "ffmpeg.exe" |
        Where-Object { $_.FullName -match "\\bin\\ffmpeg\.exe$" } |
        Select-Object -First 1
    if (-not $ffmpegCandidate) {
        $ffmpegCandidate = Get-ChildItem -Path $extractDir -Recurse -File -Filter "ffmpeg.exe" | Select-Object -First 1
    }
    if (-not $ffmpegCandidate) {
        throw "下载包中未找到 ffmpeg.exe，可能是下载源结构已变化：$ffmpegUrl"
    }

    $ffprobeCandidate = Get-ChildItem -Path $extractDir -Recurse -File -Filter "ffprobe.exe" |
        Where-Object { $_.FullName -match "\\bin\\ffprobe\.exe$" } |
        Select-Object -First 1
    if (-not $ffprobeCandidate) {
        $ffprobeCandidate = Get-ChildItem -Path $extractDir -Recurse -File -Filter "ffprobe.exe" | Select-Object -First 1
    }
    if (-not $ffprobeCandidate) {
        throw "下载包中未找到 ffprobe.exe，可能是下载源结构已变化：$ffmpegUrl"
    }

    Copy-Item -LiteralPath $ffmpegCandidate.FullName -Destination $bundledFfmpegPath -Force
    Copy-Item -LiteralPath $ffprobeCandidate.FullName -Destination $bundledFfprobePath -Force

    Write-Host "[3/4] 运行 PyInstaller 构建..."
    Push-Location $repoRoot
    try {
        & $pythonExe -m PyInstaller --noconfirm --clean $specPath
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $distExePath)) {
        throw "构建完成但未找到目标产物：$distExePath"
    }

    $buildSucceeded = $true
    Write-Host ("[4/4] Build complete: {0}" -f $distExePath)
}
catch {
    Write-Error $_
    $exitCode = 1
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $buildAssetsDir) {
        Remove-Item -LiteralPath $buildAssetsDir -Recurse -Force
    }

    if ($buildSucceeded -and $script:IsExplorerLaunch -and -not $NoOpenDist) {
        Start-Process explorer.exe "/select,`"$distExePath`""
    }

    if ($buildSucceeded) {
        Wait-ForExitIfNeeded -Message "构建完成，按 Enter 键关闭此窗口"
    }
    else {
        Wait-ForExitIfNeeded -Message "构建失败，按 Enter 键关闭此窗口"
    }
}

exit $exitCode
