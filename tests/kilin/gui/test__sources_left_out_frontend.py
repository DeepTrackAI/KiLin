"""Exercise the reminder about files left out of added folders.

Run the production ``sourcesLeftOutMessage`` function in Node and check that
the ``sources_left_out`` event shows it. Rendering is covered only by the
Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_APP = _ROOT / "kilin/gui/web/app.js"


class TestSourcesLeftOutFrontend(unittest.TestCase):
    """Tell the user which added files are not indexed."""

    def test_event_shows_the_reminder(self):
        """The sources_left_out event shows the message."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('data.event === "sources_left_out"')
        end = source.index("} else if (data.event", start)
        self.assertIn(
            "alert(sourcesLeftOutMessage(data.files));", source[start:end]
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_message_names_files_and_counts_the_rest(self):
        """Up to ten files are named with a reason; the rest are counted."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index("function sourcesLeftOutMessage(")
        function = source[start : source.index("\n}\n", start) + 3]
        script = r"""
const assert = require("node:assert/strict");
const selectionReasonCodes = { FILE_TOO_LARGE: "file_too_large" };
"""
        script += function
        script += r"""
assert.equal(
  sourcesLeftOutMessage([
    { path: "notes/data.json", reason_code: "file_type_not_selected" },
    { path: "notes/big.txt", reason_code: "file_too_large" },
  ]),
  [
    "KíLín left out 2 files it does not index in this project:",
    "",
    "notes/data.json (file type not included)",
    "notes/big.txt (too large)",
    "",
    "The other files were added.",
  ].join("\n"),
);
const many = Array.from({ length: 12 }, (_, index) => ({
  path: `notes/${index}.json`, reason_code: "file_type_not_selected",
}));
const message = sourcesLeftOutMessage(many);
assert.ok(message.startsWith("KíLín left out 12 files"));
assert.ok(message.includes("notes/9.json"));
assert.ok(!message.includes("notes/10.json"));
assert.ok(message.includes("and 2 more"));
console.log("left out passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("left out passed", result.stdout)
