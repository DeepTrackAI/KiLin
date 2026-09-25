"""Check the installed QAChat command without opening a real window."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from qachat.cli import main


class TestCommandLine(unittest.TestCase):
    """The command exposes a short, explicit desktop launch option."""

    def test_no_option_displays_help(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            result = main([])

        self.assertEqual(result, 0)
        self.assertIn("--gui", output.getvalue())

    @patch("qachat.cli._ensure_gui_environment")
    @patch("qachat.gui.app.main", return_value=7)
    def test_gui_option_opens_desktop_interface(
        self, gui_main, ensure_environment
    ) -> None:
        result = main(["--gui", "--backend", "mlx"])

        self.assertEqual(result, 7)
        ensure_environment.assert_called_once_with("mlx")
        gui_main.assert_called_once()
        self.assertEqual(len(gui_main.call_args.args[0]), 1)


if __name__ == "__main__":
    unittest.main()
