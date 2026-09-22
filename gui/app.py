"""Start the optional desktop interface without changing QAChat imports.

Functions
---------
main
    Create the Qt application or explain which optional dependency is absent.

"""

from __future__ import annotations

import sys


def main() -> int:
    """Start the desktop window, or explain how to install Qt.

    Returns
    -------
    int
        Qt's exit code, or one when the GUI dependency is missing.

    """

    try:
        from PySide6.QtWidgets import QApplication

        from gui.window import ChatWindow
    except ImportError as error:
        print(
            f"GUI dependency missing: {error}. Install with "
            "uv sync --extra gui --extra mlx (or --extra torch).",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("QAChat")
    window = ChatWindow()
    window.show()
    return app.exec()
