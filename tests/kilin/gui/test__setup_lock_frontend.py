"""Exercise the new-project setup gate without blocking saved projects.

Missing defaults prevent new-project preparation, but Settings is
always dismissible and saved projects can be resumed. While models
download, Install becomes Stop downloading. Production JavaScript
functions are exercised in Node without Qt or a browser.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function,
    run_node,
)

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestSetupLockFrontend(unittest.TestCase):
    """Keep new-project setup separate from saved-project access."""

    def test_setup_guards_project_opening(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        submit = source[source.index(
            'byId("project-form").addEventListener("submit"'
        ):]
        self.assertIn(
            "if (busy || !bridge || needsNewProjectSetup()) return;",
            submit[: submit.index("\n});\n")],
        )
        self.assertEqual(
            source.count("needsNewProjectSetup()) return;"), 1
        )
        self.assertNotIn(
            "needsNewProjectSetup()",
            extract_function(source, "renderSavedProjects"),
        )
        self.assertNotIn(
            "needsNewProjectSetup()",
            extract_function(source, "renderSidebarProjects"),
        )
        self.assertIn(
            'byId("open").disabled = value || !bridge || !kilinReady;',
            extract_function(source, "setBusy"),
        )
        self.assertIn(
            "showSetupLock();", extract_function(source, "setBusy")
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
let folderConfirmed = true;
let termsAccepted = true;
const showSetupConfirmations = () => {};
const showWelcome = () => calls.push("welcome");
const drawAttention = (element) => calls.push(`attention ${element.id}`);
"""
        for name in (
            "showSetupLock", "needsNewProjectSetup", "closeModels",
            "remindSetupConfirmations", "pointToProjectsFolder",
        ):
            script += extract_function(source, name)
        script += r"""
// Models missing: Install is offered; only Prepare project waits.
showSetupLock();
assert.equal(
  byId("install-required").textContent, "Install required models",
);
assert.equal(byId("install-required").disabled, false);
assert.equal(byId("close-models").disabled, false);
assert.equal(byId("open").disabled, true);
closeModels();
assert.deepEqual(calls, ["hide"]);
// Only new-project creation asks for the missing defaults.
assert.equal(needsNewProjectSetup(), true);
assert.deepEqual(calls, ["hide", "open settings"]);
assert.match(byId("model-download-status").textContent, /new project/);
// Downloading: Install becomes Stop and stays clickable; nothing else.
busy = true;
installingModels = true;
showSetupLock();
assert.equal(byId("install-required").textContent, "Stop downloading");
assert.equal(byId("install-required").disabled, false);
assert.equal(byId("use-installed").disabled, true);
assert.equal(byId("close-models").disabled, false);
// A pending settings save still defers closing until acknowledgement.
applyingModels = true;
closeAfterModelsApply = false;
closeModels();
assert.equal(closeAfterModelsApply, true);
assert.deepEqual(calls, ["hide", "open settings"]);
applyingModels = false;
// Stopped: back to missing models, Install offered again.
busy = false;
installingModels = false;
showSetupLock();
assert.equal(
  byId("install-required").textContent, "Install required models",
);
assert.equal(byId("close-models").disabled, false);
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
assert.equal(needsNewProjectSetup(), false);
closeModels();
assert.deepEqual(calls, ["hide"]);
// Without a confirmed folder or agreed terms, Close points to them.
calls.length = 0;
folderConfirmed = false;
termsAccepted = false;
closeModels();
// The terms come first: their welcome dialog opens again.
// Nothing in Settings is highlighted for the terms.
assert.deepEqual(calls, ["welcome"]);
assert.equal(byId("first-run-setup").open, true);
assert.equal(byId("model-download-status").textContent,
  "Confirm the projects folder and accept the terms of use to continue.");
termsAccepted = true;
calls.length = 0;
closeModels();
// The Confirm button and the message are highlighted.
assert.deepEqual(calls, [
  "attention confirm-workspace", "attention model-download-status",
]);
assert.equal(byId("model-download-status").textContent,
  "Confirm the projects folder to continue.");
folderConfirmed = true;
calls.length = 0;
closeModels();
assert.deepEqual(calls, ["hide"]);
console.log("setup lock passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("setup lock passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
