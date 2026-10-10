"""Launch QAChat's optional graphical interface from the command line.

``main`` parses the GUI and backend options. Without ``--gui`` it prints help.
Before a GUI launch it checks for the requested dependency modules and may
run ``uv sync`` in a development checkout to install missing extras. This
preparation may modify the environment and require network access.

The backend argument selects which dependency extra is checked or installed.
It is not forwarded to the GUI as a project or model backend setting.

Key Features
------------

- **Explicit GUI launch**

    Defer GUI imports until the caller requests the interface.

- **Environment preparation**

    Check module discoverability and invoke the checkout setup route when
    dependencies are missing.

Module Structure
----------------

- ``main``

    Launch the selected QAChat interface.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat.cli
>>> exit_code = qachat.cli.main([])

A normal GUI launch is ``qachat.cli.main(["--gui"])`` and starts an
interactive application.

"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
import importlib
from importlib.util import find_spec
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from qachat.runtime import (
    BACKEND_AUTO,
    BACKEND_CHOICES,
    BACKEND_MLX,
    BACKEND_TORCH,
    Backend,
    SelectedBackend,
)

__all__ = [
    "main",
]


def _selected_backend(backend: Backend) -> SelectedBackend:
    """Resolve Automatic to the backend used by the local platform.

    Parameters
    ----------
    backend: str
        Requested backend name: auto, mlx, or torch. Auto selects the platform
        default for dependency preparation.

    Returns
    -------
    str
        Requested explicit name, or mlx on Darwin/arm64 and torch otherwise.
        This helper does not validate package availability.

    """

    if backend != BACKEND_AUTO:
        return backend
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return BACKEND_MLX
    return BACKEND_TORCH


def _ensure_gui_environment(backend: Backend) -> None:
    """Install missing Qt and backend layers into the project environment.

    Parameters
    ----------
    backend: str
        Requested backend name: auto, mlx, or torch. Auto selects the platform
        default for dependency preparation.

    Raises
    ------
    RuntimeError
        If dependency repair is needed but the checkout profile or uv
        executable is unavailable.
    subprocess.CalledProcessError
        If the uv synchronization command fails.

    Notes
    -----
    Discoverability checks do not import or validate the dependencies.
    Successful synchronization can install packages and access the network.

    """

    modules = {
        BACKEND_MLX: ("PySide6", "mlx_embeddings", "mlx_lm"),
        BACKEND_TORCH: ("PySide6", "torch", "transformers"),
    }
    selected = _selected_backend(backend)
    if all(find_spec(module) is not None for module in modules[selected]):
        return

    uv = shutil.which("uv")
    project = Path(__file__).resolve().parents[1]
    if uv is None or not (project / "pyproject.toml").is_file():
        raise RuntimeError(
            "The GUI dependencies are missing. From the QAChat repository, "
            f"run: uv sync --extra gui --extra {selected}"
        )

    print(f"Preparing the QAChat GUI ({selected.upper()})…")
    subprocess.run(
        [uv, "sync", "--extra", "gui", "--extra", selected],
        cwd=project,
        check=True,
    )
    importlib.invalidate_caches()


def main(arguments: Sequence[str] | None = None) -> int:
    """Launch the selected QAChat interface.

    Parameters
    ----------
    arguments: Sequence[str] | None, optional
        Command-line arguments without the executable name. None reads the real
        command line. Defaults to ``None``.

    Returns
    -------
    int
        Interface exit code, or zero after displaying help.

    Notes
    -----
    The backend choice is used for dependency preparation and is not passed to
    the GUI application. Argument parsing may terminate with SystemExit for
    help or invalid options; GUI startup returns its own exit code.

    """

    parser = ArgumentParser(
        prog="qachat",
        description="Run QAChat locally against your own project.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the desktop interface",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_AUTO,
        help="backend to prepare for the desktop interface",
    )
    options = parser.parse_args(arguments)
    if not options.gui:
        parser.print_help()
        return 0

    _ensure_gui_environment(options.backend)

    from qachat.gui.app import main as gui_main

    # Do not forward QAChat's own --gui option to Qt.
    return gui_main([sys.argv[0]])
