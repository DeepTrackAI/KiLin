"""Check the browser's reporting of safely preserved project files.

Run the production stop-event handler with lightweight Node page substitutes.
Python tests also exercise window handlers when PySide6 is available.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


class TestStopCleanupFrontend(unittest.TestCase):
    """Explain retained workspaces and never claim they were deleted."""

    def test_stop_displays_preserved_paths(self) -> None:
        """Display preserved paths without interpreting them as markup."""

        source = _APP.read_text(encoding="utf-8")
        handler = extract_function(source, "onEvent")
        if shutil.which("node") is None:
            self.assertIn('data.preserved_projects || []', handler)
            self.assertIn('error.textContent =', handler)
            return
        script = r"""
const assert = require("node:assert/strict");
const calls = [];
const fields = {};
function element() {
  return { textContent: "", value: "code", classList: {
    add() {}, remove() {}, toggle() {},
  }};
}
const byId = (id) => (fields[id] ??= element());
const error = element();
const chat = element();
const shell = element();
const setup = element();
let setupCompletionVisible = true;
let downloadingModel = null;
let installingModels = false;
let resumeAfterStop = false;
let ready = true;
const bridge = {
  requestSettings(type, workspace) {
    calls.push(["settings", type, workspace]);
  },
  listSavedProjects(workspace) { calls.push(["projects", workspace]); },
};
function restoreUnansweredQuestion() {}
function updateFloatingHeader() {}
function showWorkspace() {}
function clearReferences() {}
function setBusy() {}
""" + handler + r"""
onEvent(JSON.stringify({event: "stopped", preserved_projects: [
  "/tmp/finished<item>", "/tmp/partial",
]}));
assert.equal(byId("connection").textContent,
  "Stopped · project files preserved");
assert.match(error.textContent, /\/tmp\/finished<item>/);
assert.match(error.textContent, /\/tmp\/partial/);
assert.deepEqual(calls, [["settings", "code", ""], ["projects", ""]]);
assert.equal(byId("connection").textContent.includes("removed"), false);
// An empty claim is the only kind of directory the engine may remove.
calls.length = 0;
onEvent(JSON.stringify({event: "stopped", removed_projects: ["/tmp/empty"]}));
assert.equal(byId("connection").textContent,
  "Stopped · empty preparation folder removed");
assert.deepEqual(calls, [["settings", "code", ""], ["projects", ""]]);
console.log("safe Stop messages passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("safe Stop messages passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
