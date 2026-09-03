#!/bin/sh
set -eu

# Always run from the repository root, even when called from another folder.
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

# Reject unsupported machines before downloading anything.
case "$(uname -s):$(uname -m)" in
    Darwin:arm64|Linux:x86_64|Linux:aarch64) ;;
    Darwin:*) echo "Use native Apple Silicon. Intel Macs/Rosetta are not supported by these package versions."; exit 1 ;;
    *) echo "Use an Apple Silicon Mac, 64-bit Linux, or scripts/setup.ps1 on Windows."; exit 1 ;;
esac

# Reuse uv, including a previous install not yet on the terminal's PATH.
uv_install_dir=${UV_INSTALL_DIR:-"$HOME/.local/bin"}
if command -v uv >/dev/null 2>&1; then
    uv_command=$(command -v uv)
elif [ -x "$uv_install_dir/uv" ]; then
    uv_command="$uv_install_dir/uv"
else
    echo "uv not found; installing it from the official Astral installer..."
    installer_path=$(mktemp)
    trap 'rm -f "$installer_path"' 0
    # Download first: a failed download must never be executed as a script.
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer_path"
    UV_INSTALL_DIR="$uv_install_dir" sh "$installer_path"
    uv_command="$uv_install_dir/uv"
fi

if [ ! -x "$uv_command" ]; then
    echo "uv is not available. Check the installer error above and try again."
    exit 1
fi

# .python-version selects Python 3.12. pyproject.toml supplies the package versions.
# The mlx dependency's platform condition skips it on Linux.
"$uv_command" sync --extra mlx --extra torch

# Use the new environment directly, without needing terminal activation.
.venv/bin/python scripts/check_setup.py

echo "Setup complete. The project environment is in .venv"
echo "No model weights were downloaded."
