"""List, check, and download pinned chat models. Never load an LLM here."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads(Path(__file__).with_name("model_catalog.json").read_text())
PACKAGES = {
    "mlx": ("huggingface-hub", "platformdirs", "mlx", "mlx-lm", "transformers"),
    "torch": ("huggingface-hub", "platformdirs", "torch", "transformers"),
}


def model_info(name: str) -> dict:
    """Accept only the small, reviewed list; never an arbitrary model URL."""
    if name not in CATALOG:
        raise ValueError(f"Unknown model {name!r}. Choose: {', '.join(CATALOG)}")
    return CATALOG[name]


def model_home(directory: str | Path | None = None) -> Path:
    """Choose explicit folder, environment setting, or the ignored QAChat/models."""
    chosen = directory or os.environ.get("QACHAT_MODEL_HOME") or "models"
    path = Path(chosen).expanduser()
    # Relative choices mean the same thing in notebooks and command-line use.
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def model_path(name: str, directory: str | Path | None = None) -> Path:
    info = model_info(name)
    home = model_home(directory)
    path = home / name / info["revision"]
    if not path.resolve().is_relative_to(home):
        raise ValueError("The model folder must stay inside the chosen storage folder.")
    return path


def package_versions(name: str) -> dict[str, str | None]:
    """Read installed package names/versions without importing GPU libraries."""
    packages = {}
    for package in PACKAGES[model_info(name)["backend"]]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def _matches(path: Path, expected: dict, *, verify: bool) -> bool:
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    if not verify:
        return True
    # Hub weights use SHA-256; small Git files include a header in their SHA-1.
    digest = hashlib.sha256() if expected["hash_type"] == "sha256" else hashlib.sha1()
    if expected["hash_type"] == "git_sha1":
        digest.update(f"blob {expected['size']}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected["hash"]


def check_model(
    name: str, directory: str | Path | None = None, *, verify: bool = False
) -> dict:
    """Check local files only. Full verification reads every byte; it is slower."""
    info = model_info(name)
    path = model_path(name, directory)  # Breakpoint 1: inspect the storage path.
    missing = [file["name"] for file in info["files"] if not (path / file["name"]).is_file()]
    invalid = [
        file["name"] for file in info["files"]
        if file["name"] not in missing and not _matches(path / file["name"], file, verify=verify)
    ]
    status = "missing" if not path.exists() else "incomplete"
    if not missing and not invalid:
        status = "verified" if verify else "downloaded"
    elif invalid and verify:
        status = "corrupt"
    return {
        "model": name,
        "backend": info["backend"],
        "repository": info["repo_id"],
        "revision": info["revision"],
        "path": str(path),
        "size_gb": round(sum(file["size"] for file in info["files"]) / 1_000_000_000, 2),
        "status": status,
        "missing_files": missing,
        "invalid_files": invalid,
    }


def list_models(directory: str | Path | None = None) -> list[dict]:
    """Our catalogued choices, not a search of every model on the internet."""
    return [check_model(name, directory) for name in CATALOG]


def _fetch_files(name: str, directory: Path) -> None:
    """Run only in the download child process, so stopping it stops all transfers."""
    from huggingface_hub import hf_hub_download

    info = model_info(name)
    path = model_path(name, directory)
    path.mkdir(parents=True, exist_ok=True)
    for file in info["files"]:
        target = path / file["name"]
        if _matches(target, file, verify=True):
            continue
        print(f"Downloading {file['name']} ({file['size']:,} bytes)", flush=True)
        hf_hub_download(  # Breakpoint 2: the real network download.
            repo_id=info["repo_id"], revision=info["revision"],
            filename=file["name"], local_dir=path,
            endpoint="https://huggingface.co", token=False,
            force_download=target.exists(),
        )
        if not _matches(target, file, verify=True):
            raise RuntimeError(f"File check failed: {file['name']}. Retry the download.")


def download_model(name: str, directory: str | Path | None = None) -> Path:
    """Explicit download. Reuse correct files; Ctrl+C/notebook Stop cancels it."""
    before = check_model(name, directory, verify=True)
    if before["status"] == "verified":
        return Path(before["path"])  # No network access when files are already good.
    home = model_home(directory)
    home.mkdir(parents=True, exist_ok=True)
    # Keep the download library's cache here too, without changing the user's
    # global environment or sending a saved Hugging Face token for public models.
    env = {
        **os.environ,
        "HF_HOME": str(home / ".huggingface"),
        "HF_HUB_CACHE": str(home / ".huggingface" / "hub"),
        "HF_XET_CACHE": str(home / ".huggingface" / "xet"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    command = [
        sys.executable, "-u", "-m", "app_core.models", "_fetch", name,
        "--directory", str(home),
    ]
    process = subprocess.Popen(  # Breakpoint 3: the cancellable download starts.
        command, cwd=Path(__file__).resolve().parents[1], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
        if process.wait() != 0:
            raise RuntimeError("Download did not finish. Check the message above, then retry.")
    finally:
        # Also covers KeyboardInterrupt, including the notebook Stop button.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
    report = check_model(name, directory, verify=True)
    if report["status"] != "verified":
        raise RuntimeError(f"Model files are not complete: {report['status']}. Retry the download.")
    return Path(report["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="list", choices=("list", "check", "download", "_fetch"))
    parser.add_argument("model", nargs="?")
    parser.add_argument("--directory", type=Path, help="Model root; default: QACHAT_MODEL_HOME or QAChat/models. Relative paths start at QAChat.")
    parser.add_argument("--verify", action="store_true", help="Check every file's content, not just its size.")
    args = parser.parse_args(argv)
    if args.action != "list" and args.model is None:
        parser.error("Choose a model name. Run without arguments to list choices.")
    try:
        if args.action == "_fetch":
            _fetch_files(args.model, model_home(args.directory))
            return 0
        print(f"Model storage: {model_home(args.directory)}")
        if args.action == "list":
            for report in list_models(args.directory):
                print(f"{report['model']}: {report['backend']}, {report['size_gb']} GB, {report['status']}")
            print("List/check is offline. Use download MODEL only when you want to download.")
            return 0
        if args.action == "download":
            print("Press Ctrl+C to stop. Partial files stay here for a later retry.")
            download_model(args.model, args.directory)
        report = check_model(args.model, args.directory, verify=args.verify)
        for label, value in report.items():
            print(f"{label}: {value}")
        print("Required packages:", package_versions(args.model))
        print("Downloaded files and installed packages do not yet prove a model can chat.")
        return 0 if report["status"] in {"downloaded", "verified"} else 1
    except KeyboardInterrupt:
        print("\nDownload/check stopped. No model was loaded.", file=sys.stderr)
        return 130
    except (OSError, ImportError, ValueError, RuntimeError) as error:
        print(f"Model step failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
