"""Exercise KíLín setup as the first step before any project.

While required models are missing and can be installed, Settings stays
open on KíLín setup, Prepare project is disabled and projects cannot be
resumed. While they download, Install becomes Stop downloading and nothing
else can start. Run the production page functions in Node with stubs;
downloads and readiness are tested in the engine.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


def _function(source: str, name: str) -> str:
    """Return one top-level function's source text."""

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


class TestSetupLockFrontend(unittest.TestCase):
    """Setup comes first; a download can be stopped and started again."""

    def test_setup_guards_project_opening(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        submit = source[source.index(
            'byId("project-form").addEventListener("submit"'
        ):]
        self.assertIn(
            "if (busy || !bridge || needsSetup()) return;",
            submit[: submit.index("\n});\n")],
        )
        self.assertEqual(source.count("needsSetup()) return;"), 3)
        self.assertIn(
            'byId("open").disabled = value || !bridge || !kilinReady;',
            _function(source, "setBusy"),
        )
        self.assertIn(
            "showSetupLock();", _function(source, "setBusy")
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_settings_stay_on_setup_until_it_is_done(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (id) => (fields[id] = fields[id] || {
  id, textContent: "", disabled: false, title: "", open: false,
});
let busy = false;
const bridge = {};
let kilinReady = false;
let readinessEnvironmentReady = true;
let readinessMissingCount = 1;
let installingModels = false;
const calls = [];
const openModels = () => calls.push("open settings");
const hideModels = () => calls.push("hide");
const document = { activeElement: null };
let applyingModels = false;
let closeAfterModelsApply = false;
"""
        for name in (
            "setupLocked", "showSetupLock", "needsSetup", "closeModels"
        ):
            script += _function(source, name)
        script += r"""
// Models missing: Install is offered; Close and Prepare project wait.
showSetupLock();
assert.equal(
  byId("install-required").textContent, "Install required models",
);
assert.equal(byId("install-required").disabled, false);
assert.equal(byId("close-models").disabled, true);
assert.equal(byId("open").disabled, true);
closeModels();
assert.deepEqual(calls, []);
assert.match(byId("model-download-status").textContent, /Finish KíLín setup/);
// Resuming a project shows setup instead.
assert.equal(needsSetup(), true);
assert.deepEqual(calls, ["open settings"]);
// Downloading: Install becomes Stop and stays clickable; nothing else.
busy = true;
installingModels = true;
showSetupLock();
assert.equal(byId("install-required").textContent, "Stop downloading");
assert.equal(byId("install-required").disabled, false);
assert.equal(byId("use-installed").disabled, true);
// Stopped: back to missing models, Install offered again.
busy = false;
installingModels = false;
showSetupLock();
assert.equal(
  byId("install-required").textContent, "Install required models",
);
assert.equal(byId("close-models").disabled, true);
// A missing runtime cannot be fixed here: Settings can close, but no
// project can be prepared.
readinessEnvironmentReady = false;
showSetupLock();
assert.equal(byId("close-models").disabled, false);
assert.equal(byId("open").disabled, true);
// Ready: everything opens up.
kilinReady = true;
readinessEnvironmentReady = true;
readinessMissingCount = 0;
calls.length = 0;
showSetupLock();
assert.equal(byId("close-models").disabled, false);
assert.equal(byId("open").disabled, false);
assert.equal(needsSetup(), false);
closeModels();
assert.deepEqual(calls, ["hide"]);
console.log("setup lock passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("setup lock passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
