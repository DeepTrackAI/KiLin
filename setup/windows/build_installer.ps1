# Build the KiLin Windows installer: dist\KiLin-Setup-<version>.exe
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup\windows\build_installer.ps1
#   ... build_installer.ps1 -Wheel    bundle a wheel of this checkout
#
# Without -Wheel the installer installs the matching version from PyPI.
# Needs uv (0.8 or later; or UV_BINARY) and Inno Setup 6 (ISCC.exe).
# The installer is not code-signed, so Windows SmartScreen may warn.
param([switch]$Wheel)
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "Build the Windows installer on Windows."
}

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $root
$stage = Join-Path $root "dist\.kilin-windows"
try {
    $uv = $env:UV_BINARY
    if (-not $uv) {
        $found = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
        if ($found) { $uv = $found.Source }
    }
    if (-not $uv -or -not (Test-Path -LiteralPath $uv)) {
        throw "uv is needed to build the installer; install it or set UV_BINARY."
    }
    # The installer uses --torch-backend, added to uv pip in 0.8.
    $uvVersion = [version]((& $uv --version).Split(" ")[1])
    if ($uvVersion -lt [version]"0.8.0") {
        throw "uv $uvVersion is too old; use uv 0.8 or later."
    }

    # Inno Setup installs for all users (Program Files) or, chosen in its
    # installer, for the current user only (Local AppData\Programs).
    $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
    if (-not $iscc) {
        $iscc = @(
            ${env:ProgramFiles(x86)}, $env:ProgramFiles,
            (Join-Path $env:LOCALAPPDATA "Programs")
        ) | Where-Object { $_ } |
            ForEach-Object { Join-Path $_ "Inno Setup 6\ISCC.exe" } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Select-Object -First 1
    }
    if (-not $iscc) {
        throw "Inno Setup 6 is needed (https://jrsoftware.org/isinfo.php)."
    }

    $match = Select-String -Path "pyproject.toml" -Pattern '^version = "(.*)"$' |
        Select-Object -First 1
    if (-not $match) { throw "Cannot find the KiLin version in pyproject.toml." }
    $version = $match.Matches[0].Groups[1].Value

    # Build in a fresh folder; earlier installers in dist\ stay untouched.
    if (Test-Path -LiteralPath $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Path $stage | Out-Null
    Copy-Item -LiteralPath $uv -Destination (Join-Path $stage "uv.exe")
    $defines = @(
        "/DAppVersion=$version",
        "/DStageDir=$stage",
        "/DOutputDir=$(Join-Path $root 'dist')"
    )
    if ($Wheel) {
        & $uv build --wheel --out-dir $stage
        if ($LASTEXITCODE -ne 0) { throw "uv build failed. See the error above." }
        $wheels = @(Get-ChildItem -LiteralPath $stage -Filter "kilin-*.whl")
        if ($wheels.Count -ne 1) { throw "Expected exactly one KiLin wheel from uv build." }
        $defines += "/DWheelFile=$($wheels[0].Name)"
    }

    & $iscc @defines (Join-Path $PSScriptRoot "kilin.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed. See the error above." }
    Write-Host "Built dist\KiLin-Setup-$version.exe"
} finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -Recurse -Force $stage }
    Pop-Location
}
