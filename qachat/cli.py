"""Provide the small command-line entry point for QAChat.

Functions
---------
main
    Parse the launch option and open the requested interface.

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


def _selected_backend(backend: str) -> str:
    """Resolve Automatic to the backend used by the local platform."""

    if backend != "auto":
        return backend
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"


def _ensure_gui_environment(backend: str) -> None:
    """Install missing Qt and backend layers into the project environment."""

    modules = {
        "mlx": ("PySide6", "mlx_embeddings", "mlx_lm"),
        "torch": ("PySide6", "torch", "transformers"),
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
    arguments: collections.abc.Sequence[str] | None, optional
        Command-line arguments without the executable name. None reads
        the real command line.

    Returns
    -------
    int
        Interface exit code, or zero after displaying help.

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
        choices=("auto", "mlx", "torch"),
        default="auto",
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
