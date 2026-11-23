"""Verify shared Node harness semantics and source extraction boundaries.

These tests protect the adapter independently of a particular GUI feature.
The feature tests continue to assert their own rendered/event behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from tests.kilin.gui._frontend_support import extract_function, run_node


class TestFrontendSupport(unittest.TestCase):
    """Keep extraction and Node invocation compatible with older tests."""

    def test_extract_returns_original_top_level_function(self) -> None:
        source = (
            "const before = 1;\n"
            "function example(value) {\n"
            "  if (value) { return true; }\n"
            "  return false;\n"
            "}\n"
            "const after = 2;\n"
        )
        self.assertEqual(
            extract_function(source, "example"),
            source[source.index("function example(") :
                   source.index("\n}\n") + 3],
        )
        self.assertNotIn("const after", extract_function(source, "example"))

    def test_missing_declaration_or_terminator_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_function("function present() {\n}\n", "missing")
        with self.assertRaises(ValueError):
            extract_function("function present() { return 2; }", "present")

    @patch("tests.kilin.gui._frontend_support.subprocess.run")
    @patch("tests.kilin.gui._frontend_support.shutil.which")
    def test_node_runner_preserves_process_options(self, which, run) -> None:
        which.return_value = "/path/to/node"
        completed = subprocess.CompletedProcess(
            ["/path/to/node", "-e", "example"], 1, "", "Error"
        )
        run.return_value = completed
        self.assertIs(run_node("example"), completed)
        which.assert_called_once_with("node")
        run.assert_called_once_with(
            ["/path/to/node", "-e", "example"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=None,
        )

    @patch("tests.kilin.gui._frontend_support.subprocess.run")
    @patch("tests.kilin.gui._frontend_support.shutil.which")
    def test_custom_environment_and_timeout_are_forwarded(
        self, which, run
    ) -> None:
        which.return_value = "/path/to/node"
        options = {"TEST_CONTEXT": "isolated"}
        run_node("example", timeout=4, env=options)
        run.assert_called_once_with(
            ["/path/to/node", "-e", "example"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
            env=options,
        )

    @patch("tests.kilin.gui._frontend_support.subprocess.run")
    @patch("tests.kilin.gui._frontend_support.shutil.which")
    def test_node_runner_forwards_script_arguments(
        self, which, run
    ) -> None:
        which.return_value = "/path/to/node"
        args = ("path with spaces/app.js", "reason_codes.json")
        run_node("example", args=args)
        run.assert_called_once_with(
            ["/path/to/node", "-e", "example", *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=None,
        )

    @patch("tests.kilin.gui._frontend_support.shutil.which")
    def test_missing_node_has_explicit_failure(self, which) -> None:
        which.return_value = None
        with self.assertRaisesRegex(FileNotFoundError, "Node.js"):
            run_node("example")


if __name__ == "__main__":
    unittest.main()
