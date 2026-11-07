"""Exercise acknowledgement-driven terms acceptance without a Qt runtime.

The production JavaScript handlers run in Node with a minimal DOM and bridge.
The tests cover saving, retry, outdated terms, and out-of-order readiness.
The Qt bridge's own signal contract is verified by ``test__protocol.py``.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestTermsAcceptanceFrontend(unittest.TestCase):
    """Do not treat readiness as proof that terms were persisted."""

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_retry_and_version_change(self) -> None:
        """Exercise modal retry, explicit acknowledgement, and stale events."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        functions = "\n".join(extract_function(source, name) for name in (
            "setWelcomeError", "showWelcome", "showSetupConfirmations",
            "onEvent",
        ))
        marker = 'byId("accept-terms").addEventListener("click", () => {'
        start = source.index(marker)
        click_handler = source[start : source.index("\n});", start) + 4]
        script = r"""
const assert = require("node:assert/strict");
const elements = {};
function element(id = "") {
  const classes = new Set();
  const node = {
    id, textContent: "", open: false, checked: false, disabled: false,
    children: [], dataset: {}, calls: {}, title: "",
    classList: {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      toggle(name, value) {
        const add = value === undefined ? !classes.has(name) : value;
        if (add) classes.add(name);
        else classes.delete(name);
      },
      contains(name) { return classes.has(name); },
    },
    replaceChildren() { this.children = []; },
    append(...nodes) { this.children.push(...nodes); },
    addEventListener(name, callback) { this.calls[name] = callback; },
    closest() { return { classList: {toggle() {}} }; },
    focus() {},
    showModal() { this.open = true; },
    close() { this.open = false; this.closed += 1; },
    closed: 0,
  };
  return node;
}
const byId = (name) => (elements[name] ??= element(name));
const document = { createElement: () => element() };
byId("welcome-error").classList.add("hidden");
byId("terms-agree").checked = true;
byId("welcome").open = true;
const error = element();
let ready = false;
let busy = false;
let requiredModels = [];
let readinessEnvironmentReady = true;
let installCommand = "";
let readinessMissingCount = 0;
let cachedModels = new Set();
let kilinReady = false;
let folderConfirmed = false;
let termsAccepted = false;
let termsVersion = "v1";
let acceptingTerms = false;
let pendingTermsVersion = "";
let acknowledgedTermsVersion = "";
let confirmedBackendPreference = "auto";
const calls = [];
const bridge = {
  acceptTerms(version) { calls.push(["accept", version]); },
  requestReadiness() { calls.push(["readiness"]); },
};
const refreshModelStatus = () => {};
const renderTerms = (text) => calls.push(["render", text]);
const clearSetupReminder = () => {};
const closeModels = () => calls.push(["close models"]);
const pointToProjectsFolder = () => calls.push(["point to folder"]);
const showSetupLock = () => showSetupConfirmations();
""" + functions + "\n" + click_handler + r"""
const click = () => byId("accept-terms").calls.click();
const notify = (data) => onEvent(JSON.stringify(data));
const readiness = (patch = {}) => notify({
  event: "readiness", backend: "mlx", ready: true,
  environment_ready: true, required_models: [], missing_models: [],
  cached_models: [], can_use_installed: false, replacements: {},
  folder_confirmed: false, projects_folder: "/Projects",
  terms_accepted: false, terms_version: "v1", terms: "Old terms",
  ...patch,
});
showSetupConfirmations();
assert.equal(byId("accept-terms").disabled, false);
click();
assert.equal(acceptingTerms, true);
assert.equal(pendingTermsVersion, "v1");
assert.equal(byId("accept-terms").disabled, true);
assert.equal(byId("accept-terms").textContent, "Saving agreement…");
click();
assert.deepEqual(calls, [["accept", "v1"]]);
// A readiness event is not an acknowledgement and must not clear the box.
readiness();
assert.equal(acceptingTerms, true);
assert.equal(byId("terms-agree").checked, true);
assert.equal(byId("welcome").open, true);
notify({event: "terms_acceptance_failed", version: "wrong",
  reason: "save_failed", message: "Stale failure"});
assert.equal(acceptingTerms, true);
notify({event: "terms_acceptance_failed", version: "v1",
  reason: "save_failed", message: "Could not save: read-only"});
assert.equal(acceptingTerms, false);
assert.equal(pendingTermsVersion, "");
assert.equal(byId("terms-agree").checked, true);
assert.equal(byId("accept-terms").disabled, false);
assert.equal(byId("accept-terms").textContent, "Continue");
assert.equal(byId("welcome-error").classList.contains("hidden"), false);
assert.match(byId("welcome-error").textContent, /read-only/);
assert.equal(byId("welcome").open, true);
// Readiness and opening the dialog again must not clear the user's choice.
readiness();
assert.equal(byId("terms-agree").checked, true);
click();
assert.equal(acceptingTerms, true);
assert.equal(byId("welcome-error").classList.contains("hidden"), true);
assert.deepEqual(calls.filter((entry) => entry[0] === "accept"), [
  ["accept", "v1"], ["accept", "v1"],
]);
notify({event: "terms_acceptance_saved", version: "wrong"});
assert.equal(acceptingTerms, true);
notify({event: "terms_acceptance_saved", version: "v1"});
assert.equal(acceptingTerms, false);
assert.equal(termsAccepted, true);
assert.equal(byId("welcome").open, false);
assert.equal(byId("welcome").closed, 1);
assert.equal(kilinReady, false);
// The prior readiness reply can arrive after the acknowledgement.
readiness();
assert.equal(termsAccepted, true);
assert.equal(acknowledgedTermsVersion, "v1");
assert.equal(byId("welcome").open, false);
readiness({terms_accepted: true});
assert.equal(acknowledgedTermsVersion, "");
assert.deepEqual(calls.filter((entry) => entry[0] === "point to folder"), [
  ["point to folder"],
]);
// An outdated submission must require a fresh checkbox and terms text.
termsAccepted = false;
termsVersion = "v1";
byId("welcome").open = true;
byId("terms-agree").checked = true;
showSetupConfirmations();
click();
notify({event: "terms_acceptance_failed", version: "v1",
  reason: "outdated_version", current_version: "v2",
  message: "The terms changed"});
assert.equal(acceptingTerms, false);
assert.equal(termsVersion, "");
assert.equal(byId("terms-agree").checked, false);
assert.equal(byId("accept-terms").disabled, true);
assert.equal(byId("welcome-error").textContent, "The terms changed");
assert.deepEqual(calls[calls.length - 1], ["readiness"]);
readiness({terms_version: "v2", terms: "New terms"});
assert.equal(termsVersion, "v2");
assert.equal(byId("terms-agree").checked, false);
assert.equal(byId("accept-terms").disabled, true);
assert.deepEqual(calls[calls.length - 1], ["render", "New terms"]);
click();
assert.equal(acceptingTerms, false);
byId("terms-agree").checked = true;
showSetupConfirmations();
click();
assert.deepEqual(calls[calls.length - 1], ["accept", "v2"]);
notify({event: "terms_acceptance_saved", version: "v2"});
assert.equal(termsAccepted, true);
assert.equal(byId("welcome").open, false);
console.log("terms acceptance retry and version change passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "terms acceptance retry and version change passed",
            result.stdout,
        )

    def test_error_is_inside_accessible_modal(self) -> None:
        """Make the save error visible without the page's generic banner."""

        markup = (_WEB / "index.html").read_text(encoding="utf-8")
        welcome = markup.split('<dialog id="welcome"', 1)[1].split(
            "</dialog>", 1
        )[0]
        self.assertIn('id="welcome-error"', welcome)
        self.assertIn('role="alert"', welcome)
        self.assertIn('class="welcome-error hidden"', welcome)
        stylesheet = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".welcome-error", stylesheet)


if __name__ == "__main__":
    unittest.main()
