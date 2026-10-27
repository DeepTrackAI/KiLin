"""Exercise clearing a stale error banner without native Qt.

Run the production ``clearError`` function in Node with a minimal DOM
adapter, and check that the ``opened`` and ``conversation_opened`` events
use it: once a project or conversation opens, an error from an earlier
attempt (such as a missing workspace folder) no longer applies. Rendering
is covered only by the Qt/WebEngine smoke test and manual checks.

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


class TestErrorBannerFrontend(unittest.TestCase):
    """Hide an error once it no longer applies."""

    def test_opened_events_clear_the_error(self):
        """Opening a project or a conversation calls clearError."""

        source = _APP.read_text(encoding="utf-8")
        for event in ("opened", "conversation_opened"):
            with self.subTest(event=event):
                start = source.index(f'data.event === "{event}"')
                end = source.index("} else if (data.event", start)
                self.assertIn("clearError();", source[start:end])

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_clear_error_keeps_newer_status_text(self):
        """The Settings copy of the error is cleared, newer text is not."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index("function clearError(")
        function = source[start : source.index("\n}\n", start) + 3]
        start = source.index("let shownErrorStatus")
        state = source[start : source.index(";", start) + 1]
        script = r"""
const assert = require("node:assert/strict");
function element() {
  const classes = new Set();
  return {
    textContent: "",
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
    },
  };
}
const status = element();
const error = element();
const byId = (id) => {
  assert.equal(id, "model-download-status");
  return status;
};
"""
        script += "\n".join([state, function])
        script += r"""
const shown = "Choose a workspace folder for a project without sources.";
status.textContent = shown;
shownErrorStatus = shown;
clearError();
assert.ok(error.classList.contains("hidden"));
assert.equal(status.textContent, "");

error.classList.remove("hidden");
shownErrorStatus = shown;
status.textContent = "Model settings applied.";
clearError();
assert.ok(error.classList.contains("hidden"));
assert.equal(status.textContent, "Model settings applied.");

status.textContent = "";
clearError();
assert.equal(status.textContent, "");
console.log("error banner passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("error banner passed", result.stdout)
