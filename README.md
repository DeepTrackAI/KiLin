# QAChat

This version sets up a local Python environment and checks the automatically chosen backend: MLX on Apple Silicon Macs, or PyTorch on supported Windows/Linux computers.

## Setup

From the QAChat folder, with internet access:

Mac/Linux:
```sh
sh scripts/setup.sh
```

Windows x64 (PowerShell):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

The setup script installs `uv` if needed, prepares Python 3.12 and the packages, then runs `scripts/check_setup.py`. On Apple Silicon it installs both backends, so you can also choose PyTorch.

After success, you get:

- `.venv/` — local Python and packages; excluded from Git.
- `uv.lock` — resolved package versions; keep in Git.
- A printed backend/device and `result: 6.0`, followed by `Setup complete`.

No model weights or API key are needed. This checks the tools; it does not yet answer questions.

Verified on an Apple Silicon Mac. Windows x64 and Linux x64/ARM64 still need real-machine installation tests; Intel Macs and Windows ARM64 are not supported by this setup.

For breakpoints and explanations, see [DEVELOPER_NOTES.md](DEVELOPER_NOTES.md). For a cell-by-cell example, open [notebooks/setup.ipynb](notebooks/setup.ipynb).
