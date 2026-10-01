"""Prepare the QAChat environment before tutorial imports.

Functions
---------
prepare_notebook
    Find the project, verify its Python kernel, and use uv to repair missing
    declared packages.

"""

from __future__ import annotations

from importlib import invalidate_caches
from importlib.util import find_spec
from pathlib import Path
import platform
import shutil
import subprocess
import sys

_COMMON_MODULES = (
    "huggingface_hub",
    "pypdf",
    "torch",
    "transformers",
)

_MLX_MODULES = (
    "mlx",
    "mlx_embeddings",
    "mlx_lm",
)


def prepare_notebook(start: str | Path | None = None) -> Path:
    """Check or repair the environment used by one tutorial.

    Parameters
    ----------
    start: str | pathlib.Path | None, optional
        Directory at or below the QAChat root. The current directory is used
        when this value is omitted.

    Returns
    -------
    pathlib.Path
        QAChat repository root.

    Raises
    ------
    RuntimeError
        If the repaired environment is not the selected notebook kernel or
        required packages remain unavailable.

    """

    root = _project_root(Path.cwd() if start is None else Path(start))
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    expected_python = _project_python(root)
    if not _same_python(expected_python):
        _sync_environment(root)
        raise RuntimeError(
            "QAChat is ready, but this notebook uses another Python. "
            f"Select this kernel and rerun the cell: {expected_python}"
        )

    missing = _missing_modules()
    if missing:
        print("Repairing missing packages with uv:", ", ".join(missing))
        _sync_environment(root)
        invalidate_caches()
        missing = _missing_modules()

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Packages are still missing after uv sync: {names}"
        )

    print(f"QAChat environment ready: {sys.executable}")
    return root


def _project_root(start: Path) -> Path:
    """Find the nearest parent containing pyproject.toml and qachat."""

    candidate = start.expanduser().resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "qachat"
        ).is_dir():
            return directory

    raise RuntimeError(
        "You need to open notebook from inside the QAChat project. "
        f"The current directory is {candidate}."
    )


def _project_python(root: Path) -> Path:
    """Return the expected Python executable inside the project environment."""

    if platform.system() == "Windows":
        return root / ".venv/Scripts/python.exe"

    return root / ".venv/bin/python"


def _same_python(expected: Path) -> bool:
    """Return whether the notebook uses the project environment."""

    return (
        expected.exists()
        and expected.resolve() == Path(sys.executable).resolve()
    )


def _required_modules() -> tuple[str, ...]:
    """Return modules installed by the setup command on this computer."""

    modules = list(_COMMON_MODULES)

    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine().lower() == "arm64"
    )
    if is_apple_silicon:
        modules.extend(_MLX_MODULES)

    return tuple(modules)


def _missing_modules() -> tuple[str, ...]:
    """Return required modules that the current kernel cannot import."""

    return tuple(
        name for name in _required_modules() if find_spec(name) is None
    )


def _sync_environment(root: Path) -> None:
    """Run the existing setup route or uv sync in the project root."""

    uv_command = shutil.which("uv")
    if uv_command is not None:
        command = [uv_command, "sync"]
        if platform.system() != "Windows":
            command.extend(("--extra", "mlx"))
        command.extend(("--extra", "torch"))
        command.extend(("--extra", "bench"))
    elif platform.system() == "Windows":
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "setup/setup.ps1"),
        ]
    else:
        command = ["/bin/sh", str(root / "setup/setup.sh")]

    subprocess.run(command, cwd=root, check=True)
