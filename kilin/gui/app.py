"""Create and run KíLín's Qt desktop application.

``main`` imports the optional Qt components, builds the main window, and runs
the application event loop. Missing GUI dependencies produce an explanatory
message and a nonzero return code rather than triggering installation. Before
the first window, a user account that has not entered the passkey is asked
for it (``kilin.gui._activation``). The window module owns communication with
the separate engine process.

Key Features
------------

- **Lazy Qt imports**

    Keep optional GUI imports inside the application startup path.

- **Explicit event loop**

    Return the Qt application exit status to a command-line entry point.

Module Structure
----------------

- ``main``

    Start the desktop window, or explain how to install Qt.

Examples
--------
>>> import kilin.gui.app
>>> exit_code = kilin.gui.app.main(["kilin"])

This example opens a desktop window and returns after the application exits.

"""

from __future__ import annotations

from collections.abc import Sequence
import sys

__all__ = [
    "main",
]


def main(arguments: Sequence[str] | None = None) -> int:
    """Start the desktop window, or explain how to install Qt.

    Parameters
    ----------
    arguments: Sequence[str] | None, optional
        Arguments passed to Qt. None uses the process command line. Defaults to
        ``None``.

    Returns
    -------
    int
        Qt's exit code, one when the GUI dependency is missing, or zero
        when the user quits the passkey dialog.

    """

    try:
        from PySide6.QtWidgets import QApplication

        from kilin.gui._activation import ask_passkey
        from kilin.gui.window import ChatWindow
    except ImportError as error:
        from kilin.gui._paths import gui_install_command
        from kilin.runtime import BACKEND_AUTO, choose_backend

        print(
            f"GUI dependency missing: {error}. Install with "
            f"{gui_install_command(choose_backend(BACKEND_AUTO))}.",
            file=sys.stderr,
        )
        return 1

    app = QApplication(list(arguments) if arguments is not None else sys.argv)
    app.setApplicationName("KíLín")
    if not ask_passkey():
        return 0
    window = ChatWindow()
    window.show()
    return app.exec()
