"""Keep saved-project access independent of new-project setup.

Exercise the production JavaScript project-list handlers and failed-resume
recovery in Node without loading Qt, a model, or the application GUI.
The fallback assertions retain useful coverage when Node is unavailable.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


class TestSavedProjectSetupFrontend(unittest.TestCase):
    """Saved project opening must not depend on creation defaults."""

    def test_resume_buttons_work_with_missing_global_models(self) -> None:
        """Both project lists dispatch Resume, not new-project setup."""

        source = _WEB.read_text(encoding="utf-8")
        home = extract_function(source, "renderSavedProjects")
        sidebar = extract_function(source, "renderSidebarProjects")
        if shutil.which("node") is None:
            self.assertIn("bridge.resumeProject(project.project)", home)
            self.assertIn("bridge.resumeProject(project.project)", sidebar)
            self.assertNotIn("needsNewProjectSetup()", home + sidebar)
            return
        script = r"""
const assert = require("node:assert/strict");
class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.handlers = {};
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, callback) { this.handlers[name] = callback; }
  setAttribute() {}
  click() { this.handlers.click?.(); }
}
const fields = {};
const byId = (id) => (fields[id] ??= new Element());
const document = {createElement: (tag) => new Element(tag)};
const calls = [];
let busy = false;
let ready = false;
let kilinReady = false;
let savedProjects = [];
let activeProjectTypeId = "";
const projectTypes = {};
const bridge = {resumeProject(path) { calls.push(path); }};
const sidebarPref = (_, fallback) => fallback;
const sidebarProjectOrder = (items) => items;
const setBusy = (state) => { busy = state; };
const openModels = () => { throw Error("Resume opened Settings"); };
""" + home + sidebar + r"""
const projects = [
  {name: "Prior", type_id: "literature", project: "/projects/prior"},
  {name: "Next", type_id: "code", project: "/projects/next"},
];
renderSavedProjects(projects);
const homeButtons = byId("saved-projects").children[0].children[1].children;
assert.equal(homeButtons.length, 2);
homeButtons[0].click();
assert.deepEqual(calls, ["/projects/prior"]);
assert.equal(busy, true);
// Busy state prevents a second launch; it is not a setup gate.
homeButtons[1].click();
assert.deepEqual(calls, ["/projects/prior"]);
busy = false;
renderSidebarProjects(projects, "/projects/prior");
const rows = byId("project-list").children;
rows[0].children[0].click();
assert.deepEqual(calls, ["/projects/prior"]);
rows[1].children[0].click();
assert.deepEqual(calls, ["/projects/prior", "/projects/next"]);
// The project button never checks kilinReady (still false).
assert.equal(kilinReady, false);
console.log("saved project resume bypass passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "saved project resume bypass passed", result.stdout
        )

    def test_failed_switch_does_not_display_a_closed_project(self) -> None:
        """Post-release errors return to launch; pre-release errors do not."""

        source = _WEB.read_text(encoding="utf-8")
        handler = extract_function(source, "onEvent")
        launch = extract_function(source, "returnToLaunch")
        if shutil.which("node") is None:
            self.assertIn("if (data.resume_released) returnToLaunch();",
                          handler)
            self.assertIn('bridge.closeProject();', launch)
            return
        script = r"""
const assert = require("node:assert/strict");
class Element {
  constructor() {
    this.value = "";
    this.classList = {
      hidden: new Set(),
      add: (name) => this.classList.hidden.add(name),
      remove: (name) => this.classList.hidden.delete(name),
    };
  }
}
const fields = {};
const byId = (id) => (fields[id] ??= new Element());
const setup = new Element();
const chat = new Element();
const shell = new Element();
const calls = [];
const shown = [];
const bridge = {
  closeProject() { calls.push("close"); },
  requestSettings() { calls.push("settings"); },
  listSavedProjects() { calls.push("projects"); },
};
let ready = true;
let busy = false;
let openProjectPath = "/projects/old";
const answerSettingsSave = {
  pending: null, queued: null, uncertain: false,
};
let branchDraft = null;
let applyingIndexing = false;
let attachedAsk = null;
let activeTracePath = "";
const selectedSources = [];
const previewRoots = new Map();
const previewSkipped = new Map();
const checkedFiles = new Map();
const renderPreview = () => {};
const showWorkspace = () => {};
const clearReferences = () => {};
const showProjectSettings = () => {};
const updateFloatingHeader = () => {};
const setBusy = (value) => { busy = value; };
const refreshSavedProject = () => calls.push("refresh");
const showError = (message) => shown.push(message);
const clearAskingConversation = () => {};
const showIndexingFields = () => {};
""" + launch + handler + r"""
// Invalid destination profile: the previous session was not released.
onEvent(JSON.stringify({event: "error", resume_released: false,
  message: "Invalid saved project"}));
assert.equal(ready, true);
assert.equal(openProjectPath, "/projects/old");
assert.deepEqual(calls, []);
assert.deepEqual(shown, ["Invalid saved project"]);
// Missing model after the old session was released: no stale chat UI.
onEvent(JSON.stringify({event: "error", resume_released: true,
  message: "Saved model not installed"}));
assert.equal(ready, false);
assert.equal(openProjectPath, "");
assert.equal(chat.classList.hidden.has("hidden"), true);
assert.equal(setup.classList.hidden.has("hidden"), false);
assert.deepEqual(calls, ["close", "settings", "refresh"]);
assert.deepEqual(shown, [
  "Invalid saved project", "Saved model not installed",
]);
assert.equal(busy, false);
console.log("resume recovery passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resume recovery passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
