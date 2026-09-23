"""Start the optional desktop interface without changing QAChat imports.

Functions
---------
main
    Create the Qt application or explain which optional dependency is absent.

"""

from __future__ import annotations

from collections.abc import Sequence
import sys


def main(arguments: Sequence[str] | None = None) -> int:
    """Start the desktop window, or explain how to install Qt.

    Parameters
    ----------
    arguments: collections.abc.Sequence[str] | None, optional
        Arguments passed to Qt. None uses the process command line.

    Returns
    -------
    int
        Qt's exit code, or one when the GUI dependency is missing.

    """

    try:
        from PySide6.QtWidgets import QApplication

        from qachat.gui.window import ChatWindow
    except ImportError as error:
        print(
            f"GUI dependency missing: {error}. Install with "
            "uv sync --extra gui --extra mlx (or --extra torch).",
            file=sys.stderr,
        )
        return 1

    app = QApplication(list(arguments) if arguments is not None else sys.argv)
    app.setApplicationName("QAChat")
    window = ChatWindow()
    window.show()
    return app.exec()
