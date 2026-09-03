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

The setup script installs `uv` if needed, prepares Python 3.12 and the packages using the shared `uv.lock`, then runs `scripts/check_setup.py`. On Apple Silicon it installs both backends, so you can also choose PyTorch.

`uv.lock` is generated but kept in Git: it records the chosen package versions, including packages required by other packages, so users can repeat our setup instead of choosing versions again. It contains no installed packages or model weights.

After success, you get:

- `.venv/` — each user's own Python environment and installed packages; created locally and excluded from Git.
- A simple calculation check: the selected backend (MLX or PyTorch) adds `2 + 3`, and the script checks that the answer is `5`. This confirms the backend can run a calculation. It prints the backend, device, and `result: 5.0`, followed by `Setup complete`.

No model weights or API key are needed. This checks the tools; it does not yet answer questions.

Verified on an Apple Silicon Mac. Windows x64 and Linux x64/ARM64 still need real-machine installation tests; Intel Macs and Windows ARM64 are not supported by this setup.

## Try the notebooks

After setup finishes, use these examples to see what works and try it yourself. Open a notebook, select Python from QAChat's `.venv`, and run cells from the top, one at a time with Shift+Enter.

- [Setup and backend check](notebooks/setup.ipynb) — see which backend and device are used. Edit `A` and `B`, then run the check: `2 + 3` should give `5`. This calls the same code as setup; no LLM is loaded.
