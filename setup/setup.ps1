# Windows version of setup.sh. Run from PowerShell; no Python is needed first.
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT" -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    throw "Use 64-bit PowerShell on an x64 Windows computer. On Mac/Linux, use setup.sh."
}

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $uvFound = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    $uvCommand = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if ($env:UV_INSTALL_DIR) { $uvCommand = Join-Path $env:UV_INSTALL_DIR "uv.exe" }
    if ($uvFound) {
        $uvCommand = $uvFound.Source
    } elseif (-not (Test-Path -LiteralPath $uvCommand)) {
        Write-Host "uv not found; installing it from the official Astral installer..."
        # Fetch successfully before executing the official installer.
        $installer = Invoke-RestMethod "https://astral.sh/uv/install.ps1"
        Invoke-Expression $installer
    }
    if (-not (Test-Path -LiteralPath $uvCommand)) {
        throw "uv is not available. Check the installer output and try again."
    }

    # Windows uses PyTorch; MLX is for the Apple Silicon path in this project.
    & $uvCommand sync --extra torch
    if ($LASTEXITCODE -ne 0) { throw "Package setup failed. See the error above." }
    & ".venv\Scripts\python.exe" "scripts\check_setup.py"
    if ($LASTEXITCODE -ne 0) { throw "Backend check failed. See the error above." }

    Write-Host "Setup complete. The project environment is in .venv"
    Write-Host "No model weights were downloaded."
} finally {
    Pop-Location
}
