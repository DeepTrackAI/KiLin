"""Test cross-platform execution of isolated frontend JavaScript.

The adapter sends potentially large Node scripts on standard input while
retaining the argument layout used by existing frontend tests.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import shutil
import subprocess
import unittest
from unittest.mock import patch

from tests.kilin.gui._frontend_support import run_node


class TestFrontendSupport(unittest.TestCase):
    """Keep Node harness transport independent of command-line limits."""

    @patch("tests.kilin.gui._frontend_support.subprocess.run")
    @patch("tests.kilin.gui._frontend_support.shutil.which")
    def test_script_is_stdin_not_command_argument(
        self, which, execute
    ) -> None:
        """Preserve arguments, timeout, environment, and error handling."""

        which.return_value = "node"
        script = "console.log('sent by stdin');"
        execute.return_value = subprocess.CompletedProcess([], 0, "OK", "")
        result = run_node(
            script, args=("item",), timeout=7, env={"KILIN_TEST": "1"}
        )
        self.assertEqual(result.stdout, "OK")
        argv = execute.call_args.args[0]
        self.assertEqual(argv[0:2], ["node", "-e"])
        self.assertNotIn(script, argv)
        self.assertEqual(argv[-1], "item")
        self.assertEqual(execute.call_args.kwargs["input"], script)
        self.assertEqual(execute.call_args.kwargs["timeout"], 7)
        self.assertEqual(
            execute.call_args.kwargs["env"], {"KILIN_TEST": "1"}
        )
        self.assertFalse(execute.call_args.kwargs["check"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
    def test_long_script_runs_with_expected_node_arguments(self) -> None:
        """Exercise a script longer than the Windows command-line cap."""

        script = (
            "const assert = require('node:assert/strict');\n"
            "assert.equal(process.argv[1], 'item');\n"
            + "//" + "x" * 40_000 + "\n"
            "console.log('long harness passed');\n"
        )
        result = run_node(script, args=("item",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("long harness passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
