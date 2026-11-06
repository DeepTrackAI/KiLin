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
let readinessMissingCount = 0;
let cachedModels = new Set();
let kilinReady = true;
let confirmedBackendPreference = "mlx";
const showSetupLock = () => {};
const refreshModelStatus = () => {};
""" + handler + r"""
onEvent(JSON.stringify({
  event: "readiness", backend: "mlx", ready: false,
  environment_ready: false,
  unavailable_reason: "`mlx` requires a native Apple Silicon Mac.",
  required_models: [], missing_models: [], cached_models: [],
  replacements: {}, can_use_installed: false,
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
