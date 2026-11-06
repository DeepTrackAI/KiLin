"""Verify backend preference feedback and setup readiness in the GUI.

Exercise production JavaScript event branches without a Qt/WebEngine runtime.
The tests cover save acknowledgements, failed saves, stale readiness, and
unsupported explicit backend messages. No model is downloaded or loaded.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


class TestBackendPreferenceFrontend(unittest.TestCase):
    """Treat the last acknowledged backend as the saved selection."""

    def test_preference_acknowledgement_and_failed_save(self) -> None:
        """Invalidate stale readiness and restore only failed selections."""

        source = _APP.read_text(encoding="utf-8")
        handler = extract_function(source, "onEvent")
        if shutil.which("node") is None:
            self.assertIn('data.event === "backend_preference_saved"',
                          handler)
            self.assertIn('data.event === "backend_preference_failed"',
                          handler)
            self.assertIn("confirmedBackendPreference", handler)
            return
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
function element() {
  return {
    value: "", textContent: "", open: false,
    classList: { add() {}, remove() {}, toggle() {} },
  };
}
const byId = (name) => (fields[name] ??= element());
const error = element();
let ready = false;
let kilinReady = true;
let readinessEnvironmentReady = true;
let readinessMissingCount = 0;
let confirmedBackendPreference = "auto";
let renders = 0;
let locks = 0;
const showModelFields = () => { renders += 1; };
const showSetupLock = () => { locks += 1; };
""" + handler + r"""
byId("backend").value = "torch";
onEvent(JSON.stringify({
  event: "backend_preference_saved", backend: "torch",
}));
assert.equal(confirmedBackendPreference, "torch");
assert.equal(kilinReady, false);
assert.equal(readinessEnvironmentReady, false);
assert.equal(readinessMissingCount, 0);
assert.equal(byId("readiness-status").textContent, "Checking");
assert.equal(locks, 1);
// A failed second choice must restore the confirmed one.
byId("backend").value = "mlx";
onEvent(JSON.stringify({
  event: "backend_preference_failed", backend: "mlx",
  message: "Could not save the backend preference: disk full",
}));
assert.equal(byId("backend").value, "torch");
assert.equal(renders, 1);
assert.match(error.textContent, /disk full/);
// An older failure must not undo a newer pending selection.
byId("backend").value = "auto";
onEvent(JSON.stringify({
  event: "backend_preference_failed", backend: "mlx",
  message: "Old failed request",
}));
assert.equal(byId("backend").value, "auto");
assert.equal(renders, 1);
console.log("backend preference feedback passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend preference feedback passed", result.stdout)

    def test_unsupported_backend_readiness_is_not_reported_ready(
        self,
    ) -> None:
        """Display the exact backend restriction without a fallback."""

        source = _APP.read_text(encoding="utf-8")
        handler = extract_function(source, "onEvent")
        if shutil.which("node") is None:
            self.assertIn("data.unavailable_reason", handler)
            self.assertIn('data.event === "readiness"', handler)
            return
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
function element() {
  return {
    textContent: "", open: false,
    classList: {add() {}, remove() {}, toggle() {}},
    replaceChildren() {this.children = [];},
    append(node) {this.children.push(node);},
    focus() {},
  };
}
const byId = (name) => (fields[name] ??= element());
const document = {createElement() {return element();}};
let requiredModels = [];
let readinessEnvironmentReady = true;
let installCommand = "";
let readinessMissingCount = 0;
let cachedModels = new Set();
let kilinReady = true;
let folderConfirmed = true;
let termsAccepted = true;
let termsVersion = "";
let acceptingTerms = false;
let confirmedBackendPreference = "mlx";
const showSetupLock = () => {};
const refreshModelStatus = () => {};
const renderTerms = () => {};
const clearSetupReminder = () => {};
""" + handler + r"""
onEvent(JSON.stringify({
  event: "readiness", backend: "mlx", ready: false,
  environment_ready: false,
  unavailable_reason: "`mlx` requires a native Apple Silicon Mac.",
  required_models: [], missing_models: [], cached_models: [],
  replacements: {}, can_use_installed: false,
  folder_confirmed: true, terms_accepted: true,
}));
assert.equal(kilinReady, false);
assert.equal(readinessEnvironmentReady, false);
assert.equal(byId("readiness-steps").children[0].textContent,
  "`mlx` requires a native Apple Silicon Mac.");
assert.equal(byId("readiness-status").textContent, "1 missing");
assert.equal(byId("readiness-summary").textContent,
  "`mlx` requires a native Apple Silicon Mac.");
console.log("unsupported backend readiness passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unsupported backend readiness passed", result.stdout)

    def test_setup_needs_the_folder_and_terms_after_the_models(
        self,
    ) -> None:
        """Installed models alone do not finish KíLín setup."""

        source = _APP.read_text(encoding="utf-8")
        handler = extract_function(source, "onEvent")
        if shutil.which("node") is None:
            self.assertIn("data.folder_confirmed", handler)
            self.assertIn("data.terms_accepted", handler)
            return
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
function element() {
  return {
    textContent: "", open: false, children: [], dataset: {},
    classList: {add() {}, remove() {}, toggle() {}},
    replaceChildren() {this.children = [];},
    append(...nodes) {this.children.push(...nodes);},
    focus() {},
  };
}
const byId = (name) => (fields[name] ??= element());
const document = {createElement() {return element();}};
let requiredModels = [];
let readinessEnvironmentReady = true;
let installCommand = "";
let readinessMissingCount = 0;
let cachedModels = new Set();
let kilinReady = true;
let folderConfirmed = true;
let termsAccepted = true;
let termsVersion = "";
let acceptingTerms = false;
let confirmedBackendPreference = "mlx";
const shownTerms = [];
const showSetupLock = () => {};
const refreshModelStatus = () => {};
const renderTerms = (text) => shownTerms.push(text);
const clearSetupReminder = () => {};
let closed = 0;
const closeModels = () => { closed += 1; };
let pointed = 0;
const pointToProjectsFolder = () => { pointed += 1; };
let welcomes = 0;
const showWelcome = () => { welcomes += 1; };
""" + handler + r"""
const readiness = (fields) => onEvent(JSON.stringify({
  event: "readiness", backend: "mlx", ready: true,
  environment_ready: true, unavailable_reason: "",
  required_models: ["m"], missing_models: [], cached_models: ["m"],
  replacements: {}, can_use_installed: false,
  projects_folder: "/Projects", terms: "As is.", terms_version: "v1",
  ...fields,
}));
readiness({folder_confirmed: false, terms_accepted: false});
assert.equal(kilinReady, false);
assert.equal(termsVersion, "v1");
// The terms come first, in the welcome dialog.
assert.equal(welcomes, 1);
assert.deepEqual(shownTerms, ["As is."]);
const steps = byId("readiness-steps").children.map((step) => step.textContent);
assert.deepEqual(steps.slice(2), [
  "Confirm the projects folder below",
  "Accept the terms of use",
]);
assert.equal(byId("readiness-status").textContent, "2 missing");
assert.equal(byId("readiness-summary").textContent,
  "Confirm the projects folder and accept the terms of use to finish setup.");
// Accepting the terms in the dialog points to the folder next.
acceptingTerms = true;
readiness({folder_confirmed: false, terms_accepted: true});
assert.equal(pointed, 1);
assert.equal(acceptingTerms, false);
readiness({folder_confirmed: true, terms_accepted: false});
assert.equal(kilinReady, false);
assert.equal(byId("readiness-summary").textContent,
  "Accept the terms of use to finish setup.");
assert.equal(closed, 0);
readiness({folder_confirmed: true, terms_accepted: true});
assert.equal(kilinReady, true);
// The last step finished setup: it collapses and Settings closes, once.
assert.equal(byId("first-run-setup").open, false);
assert.equal(closed, 1);
readiness({folder_confirmed: true, terms_accepted: true});
assert.equal(closed, 1);
assert.equal(byId("readiness-steps").children[2].textContent,
  "Projects folder: /Projects");
assert.equal(byId("readiness-status").textContent, "Ready");
assert.equal(byId("readiness-summary").textContent, "KíLín is ready.");
console.log("setup confirmations passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("setup confirmations passed", result.stdout)

    def test_backend_selection_uses_qt_preference_slot(self) -> None:
        """Project backend selection sends no unintended engine command."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index(
            'byId("backend").addEventListener("change", () => {'
        )
        handler = source[start:source.index("\n});", start) + 4]
        if shutil.which("node") is None:
            self.assertIn("setBackendPreference", handler)
            self.assertIn("if (!ready)", handler)
            return
        script = r"""
const assert = require("node:assert/strict");
const calls = [];
const field = {value: "torch", addEventListener(name, handler) {
  assert.equal(name, "change");
  this.change = handler;
}};
const byId = (id) => {assert.equal(id, "backend"); return field;};
let ready = false;
const bridge = {
  setBackendPreference(value) {calls.push(value);},
};
const saveModelFields = () => {};
const showModelFields = () => {};
""" + handler + r"""
field.change();
assert.deepEqual(calls, ["torch"]);
ready = true;
field.change();
assert.deepEqual(calls, ["torch"]);
console.log("backend selection dispatch passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend selection dispatch passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
