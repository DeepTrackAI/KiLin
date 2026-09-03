# QAChat

This version sets up a local model chat and lets you preview which repository files to use; repository search, RAG, and the GUI come later.

## One-time setup

On a new computer, run the command for your system below from the QAChat folder, with internet access. You only need this initial setup once.

Already finished setup? Go straight to [the notebooks](#start-with-the-notebooks).

Mac/Linux:

```sh
sh scripts/setup.sh
```

Windows x64 (PowerShell):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

The script installs `uv` if needed, prepares Python 3.12 and the packages using `uv.lock`, then runs the backend check. On Apple Silicon it installs both MLX and PyTorch.

Expect the selected backend/device, `result: 5.0` from the backend calculating `2 + 3`, and `Setup complete`. Setup downloads no model weights and needs no API key. Once you see `Setup complete`, continue to the notebooks below.

Verified on an Apple Silicon Mac. Windows x64 and Linux x64/ARM64 still need real-machine installation tests; Intel Macs and Windows ARM64 are not supported by this setup.

## Start with the notebooks

After setup, open these notebooks in order, select Python from QAChat's `.venv`, and run one cell at a time with Shift+Enter:

1. [Setup and backend check](notebooks/10_setup.ipynb) — choose a backend and enter two numbers. The backend calculates their sum and checks the answer, showing that it can run a calculation.
2. [Model check and download](notebooks/20_model_download.ipynb) — choose a model folder, see supported models and required packages, and check or download the files. Keep `MODEL_DIRECTORY = None` for the default folder, or enter your own path. Downloading starts only if you set `DOWNLOAD = True`.
3. [Load a model and chat](notebooks/30_model_chat.ipynb) — choose a downloaded model, load it, say hello, ask a follow-up, then release its memory. Expect real model replies. `BACKEND = "auto"` uses the engine matching the chosen model; MLX and PyTorch weights are not interchangeable.
4. [Choose repository files](notebooks/40_repository_files.ipynb) — choose a local folder, select file types or paths, and see candidate files, sizes, and reasons for skipping entries. No LLM is needed; you can open this directly after setup. It creates no files and does not read source contents or build an index.

The repository preview uses its own selection rules, not `.gitignore`. Hidden entries, common model/generated folders, and links are skipped. Add custom private folders to `EXCLUDE_PATHS`; the preview is not a secret scanner.

These notebooks call the shared code; they are examples, not separate implementations. Chat never downloads missing files. Use one model at a time: large models need several GB of free memory. Messages stay in memory unless you save notebook outputs; clear those outputs before committing.

## Where files go

- `.venv/` — Python and installed packages; excluded from Git.
- `models/<model-name>/<revision>/` — weights, tokenizer files, and model settings; inside QAChat, but excluded from Git. Download bookkeeping also stays under `models/`.
- `uv.lock` — generated package-version record, including packages required by other packages; keep it in Git so others can repeat the setup. It contains no installed packages or weights.

To choose your own storage folder, set `MODEL_DIRECTORY` in the notebook, `--directory` on a command, or `QACHAT_MODEL_HOME` in your environment. An explicit folder overrides the environment setting. Relative paths start at QAChat; absolute paths can point elsewhere, including another drive. Existing files are reused when they match the catalog's `<model-name>/<revision>/` layout.

Never commit model files. If you choose another folder inside the repository, ignore that whole folder too. Git ignore rules do not prevent cloud-sync tools from copying it.

## Optional model commands

Use `scripts/download_model.py` instead of the download notebook if you prefer the terminal. It calls the same shared downloader.

List choices, sizes, local status, and the storage folder without downloading:

```sh
.venv/bin/python scripts/download_model.py
```

Download the 8.32 GB MLX model for Apple Silicon into QAChat's `models/` folder:

```sh
.venv/bin/python scripts/download_model.py download qwen3-14b-mlx
```

Or choose your own storage folder:

```sh
.venv/bin/python scripts/download_model.py download qwen3-14b-mlx --directory "~/my-models"
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. The list also includes a smaller PyTorch model. MLX and PyTorch need different weight files.

Correct files are reused and checked before acceptance. Press Ctrl+C to stop, then rerun to retry. These public models need no API key.

## Run all tests

After setup, run from QAChat:

```sh
.venv/bin/python -m pytest -q
```

On Windows: `.venv\Scripts\python.exe -m pytest -q`.

Expect a summary ending in `passed`. Tests need no internet or downloaded LLMs; they use small stand-ins and, when PyTorch is installed, a tiny locally created CPU model. They do not prove that a full LLM fits your computer—use the chat notebook for that.
