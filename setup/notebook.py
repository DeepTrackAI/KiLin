"""Prepare the QAChat development environment for tutorial notebooks.

This auxiliary repository module prepares notebooks that run from the QAChat
source tree. It is intended for tutorials and development workflows and is not
part of the released ``qachat`` package.

The main entry point locates the QAChat repository, makes the source tree
importable, verifies that the notebook uses the project's virtual environment,
and repairs missing declared dependencies through the repository setup route.
The helper functions implement repository discovery, interpreter validation,
platform-specific dependency selection, and environment synchronization.

Key Features
------------
- **Repository discovery**

    Find the nearest parent directory containing both ``pyproject.toml`` and
    the ``qachat`` source directory.

- **Kernel validation**

    Verify that the active notebook interpreter is the Python executable from
    the repository's ``.venv`` environment.

- **Dependency checks**

    Check common tutorial dependencies and add MLX-specific dependencies on
    native Apple Silicon systems.

- **Environment repair**

    Prefer ``uv sync`` when ``uv`` is available and otherwise fall back to the
    repository setup script for the current platform.

Module Structure
----------------
Functions:

- ``prepare_notebook(start=None) -> pathlib.Path``

    Prepare a tutorial notebook environment and return the repository root.

Private helpers:

- ``_project_root(start) -> pathlib.Path``

    Locate the QAChat repository root.

- ``_project_python(root) -> pathlib.Path``

    Return the expected virtual-environment Python executable.

- ``_same_python(expected) -> bool``

    Check whether the active interpreter matches the project environment.

- ``_required_modules() -> tuple[str, ...]``

    Return the tutorial modules required on the current platform.

- ``_missing_modules() -> tuple[str, ...]``

    Return required modules that cannot be imported by the active kernel.

- ``_sync_environment(root) -> None``

    Synchronize the project environment through ``uv`` or a setup script.

Examples
--------
At the start of a tutorial notebook located inside the QAChat repository::

    from notebook import prepare_notebook

    root = prepare_notebook()

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
    "pdfminer",
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
        If the repository cannot be found, the active notebook kernel is not
        the project interpreter, or required packages remain unavailable.
    subprocess.CalledProcessError
        If an environment synchronization command fails.

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
    """Find the QAChat repository root above a starting path.

    Parameters
    ----------
    start: pathlib.Path
        Directory at or below the QAChat repository root.

    Returns
    -------
    pathlib.Path
        Nearest parent containing ``pyproject.toml`` and ``qachat``.

    Raises
    ------
    RuntimeError
        If no QAChat repository root can be found.

    """

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
    """Return the expected Python executable for the project environment.

    Parameters
    ----------
    root: pathlib.Path
        QAChat repository root.

    Returns
    -------
    pathlib.Path
        Python executable inside the repository's ``.venv`` directory.

    """

    if platform.system() == "Windows":
        return root / ".venv/Scripts/python.exe"

    return root / ".venv/bin/python"


def _same_python(expected: Path) -> bool:
    """Check whether the notebook uses the project environment.

    Parameters
    ----------
    expected: pathlib.Path
        Expected Python executable in the project environment.

    Returns
    -------
    bool
        ``True`` if the expected executable exists and matches the active
        Python interpreter.

    """

    return (
        expected.exists()
        and expected.resolve() == Path(sys.executable).resolve()
    )


def _required_modules() -> tuple[str, ...]:
    """Return tutorial modules required on the current platform.

    Returns
    -------
    tuple[str, ...]
        Common tutorial modules, with MLX modules added on Apple Silicon.

    """

    modules = list(_COMMON_MODULES)

    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine().lower() == "arm64"
    )
    if is_apple_silicon:
        modules.extend(_MLX_MODULES)

    return tuple(modules)


def _missing_modules() -> tuple[str, ...]:
    """Return required modules that the current kernel cannot import.

    Returns
    -------
    tuple[str, ...]
        Names of required modules for which no import specification is found.

    """

    return tuple(
        name for name in _required_modules() if find_spec(name) is None
    )


def _sync_environment(root: Path) -> None:
    """Synchronize the QAChat development environment.

    ``uv sync`` is preferred when ``uv`` is available. Otherwise, the
    repository's platform-specific setup script is used.

    Parameters
    ----------
    root: pathlib.Path
        QAChat repository root in which to run the synchronization command.

    Raises
    ------
    subprocess.CalledProcessError
        If the synchronization command exits with a nonzero status.

    """

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
