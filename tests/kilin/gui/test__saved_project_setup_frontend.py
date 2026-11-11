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
    this.dataset = {};
  }
  querySelector() { return null; }
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

    def test_type_change_preserves_saved_project_list(self) -> None:
        """Changing creation type must not hide resumable projects."""

        source = _WEB.read_text(encoding="utf-8")
        home = extract_function(source, "renderSavedProjects")
        refresh = extract_function(source, "refreshSavedProject")
        marker = 'byId("type-id").addEventListener("change", () => {'
        start = source.index(marker)
        change = source[start:source.index("\n});", start) + 4]
        if shutil.which("node") is None:
            self.assertNotIn("hideResume()", change)
            self.assertIn('host.classList.toggle("hidden"', home)
            return
        script = r"""
const assert = require("node:assert/strict");
class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.handlers = {};
    this.dataset = {};
    this.value = "";
    this.className = "";
    this.scrollTop = 0;
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name);
        else classes.delete(name);
      },
    };
  }
  querySelector(selector) {
    for (const child of this.children) {
      if (`.${child.className}` === selector) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, handler) { this.handlers[name] = handler; }
}
const fields = {};
const byId = (id) => (fields[id] ??= new Element());
const document = {createElement: (tag) => new Element(tag)};
const requests = [];
const bridge = {
  requestSettings(type, path) { requests.push(["settings", type, path]); },
  listSavedProjects(path) { requests.push(["projects", path]); },
  resumeProject(path) { requests.push(["resume", path]); },
};
const hideResume = () => byId("saved-projects").classList.add("hidden");
let ready = false;
let busy = false;
let savedProjects = [];
const projectTypes = {
  literature: {label: "Literature"}, code: {label: "Code"},
};
""" + home + refresh + change + r"""
const projects = [
  {name: "Papers", type_id: "literature", project: "/projects/papers"},
  {name: "Scripts", type_id: "code", project: "/projects/scripts"},
];
const host = byId("saved-projects");
host.classList.add("hidden");
renderSavedProjects(projects);
assert.equal(host.classList.contains("hidden"), false);
const list = host.querySelector(".saved-project-list");
assert.equal(list.children.length, 2);
byId("type-id").value = "code";
byId("type-id").handlers.change();
assert.equal(host.classList.contains("hidden"), false);
assert.deepEqual(requests, [
  ["settings", "code", ""], ["projects", ""],
]);
renderSavedProjects(projects.map((project) => ({...project})));
assert.equal(host.classList.contains("hidden"), false);
assert.equal(host.querySelector(".saved-project-list"), list);
// An unchanged refresh must also restore a previously hidden list.
host.classList.add("hidden");
renderSavedProjects(projects);
assert.equal(host.classList.contains("hidden"), false);
assert.equal(host.querySelector(".saved-project-list"), list);
// An empty list is correctly hidden, including on an unchanged refresh.
renderSavedProjects([]);
assert.equal(host.classList.contains("hidden"), true);
renderSavedProjects([]);
assert.equal(host.classList.contains("hidden"), true);
console.log("project type preserves saved project list");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "project type preserves saved project list", result.stdout
        )

    def test_refreshing_the_same_projects_keeps_the_list_scroll(
        self,
    ) -> None:
        """The periodic refresh leaves an unchanged list in place."""

        source = _WEB.read_text(encoding="utf-8")
        home = extract_function(source, "renderSavedProjects")
        if shutil.which("node") is None:
            self.assertIn("host.dataset.shown === shown", home)
            return
        script = r"""
const assert = require("node:assert/strict");
class Element {
  constructor() {
    this.children = [];
    this.className = "";
    this.scrollTop = 0;
    this.dataset = {};
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  querySelector(selector) {
    for (const child of this.children) {
      if (`.${child.className}` === selector) return child;
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener() {}
}
const fields = {};
const byId = (id) => (fields[id] ??= new Element());
const document = {createElement: () => new Element()};
let savedProjects = [];
const projectTypes = {literature: {label: "Literature collection"}};
""" + home + r"""
const projects = [
  {name: "One", type_id: "literature", project: "/projects/one"},
  {name: "Two", type_id: "literature", project: "/projects/two"},
];
const host = byId("saved-projects");
renderSavedProjects(projects);
const list = host.querySelector(".saved-project-list");
list.scrollTop = 120;
// The refresh brings the same projects: nothing is rebuilt.
renderSavedProjects(projects.map((project) => ({...project})));
assert.equal(host.querySelector(".saved-project-list"), list);
assert.equal(list.scrollTop, 120);
// A new project rebuilds the list at the same scroll position.
renderSavedProjects([...projects,
  {name: "Three", type_id: "literature", project: "/projects/three"}]);
const rebuilt = host.querySelector(".saved-project-list");
assert.notEqual(rebuilt, list);
assert.equal(rebuilt.children.length, 3);
assert.equal(rebuilt.scrollTop, 120);
console.log("saved project scroll passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saved project scroll passed", result.stdout)

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
