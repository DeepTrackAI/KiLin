"""Exercise unavailable saved projects using production browser functions.

Node.js supplies a minimal DOM adapter; no native Qt runtime is needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest


_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("node"), "Node.js is not available")
class TestProjectAvailabilityFrontend(unittest.TestCase):
    """Keep unavailable rows visible without attempting to open them."""

    def test_project_lists_and_empty_catalog(self):
        source = (_ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        functions = []
        for name in (
            "renderProjectTypes", "renderSavedProjects",
            "renderSidebarProjects", "showSettings",
        ):
            start = source.index(f"function {name}(")
            end = source.index("\n}\n", start) + 3
            functions.append(source[start:end])
        script = r'''
const assert = require("node:assert/strict");
class Element {
  constructor() {
    this.children = []; this.handlers = {}; this.dataset = {};
    this.value = ""; this.textContent = ""; this.scrollTop = 0;
    this.classList = {toggle() {}};
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(name, fn) { this.handlers[name] = fn; }
  setAttribute() {}
  querySelector() { return this.children[0]?.children[1]; }
}
const fields = {};
const byId = (id) => (fields[id] ??= new Element());
const document = {createElement: () => new Element()};
let projectTypes = Object.create(null);
let savedProjects = [], openProjectPath = "/current";
let activeProjectTypeId = "";
let ready = false, busy = false, profile = {old: true};
const calls = [], errors = [];
const requests = [];
const bridge = {resumeProject: (path) => calls.push(path),
                requestSettings: (...args) => requests.push(args)};
const setBusy = (value) => { busy = value; };
const showError = (reason) => errors.push(reason);
const sidebarPref = (_, fallback) => fallback;
const sidebarProjectOrder = (projects) => projects;
const openModels = () => {};
let menu;
const showWorkspaceMenu = (_, __, actions) => { menu = actions; };
'''
        script += "\n".join(functions)
        script += r'''
const good = {project: "/good", name: "Good", type_id: "good",
              available: true, unavailable_reason: null};
const missing = {project: "/lost", name: "Lost", type_id: "lost",
                 available: false, unavailable_reason: "Tool missing"};
renderProjectTypes([{type_id: "good", label: "Available"}]);
assert.equal(byId("type-id").value, "good");
renderSavedProjects([good, missing]);
const host = byId("saved-projects");
let rows = host.children[0].children[1].children;
assert.equal(rows.length, 2);
assert.equal(rows[1].children[1].textContent, "lost (unavailable)");
assert.equal(rows[1].title, "Tool missing");
rows[1].handlers.click();
assert.deepEqual(errors, ["Tool missing"]);
assert.deepEqual(calls, []);
assert.equal(busy, false);
rows[0].handlers.click();
assert.deepEqual(calls, ["/good"]);
busy = false;
renderSidebarProjects([good, missing], "/current");
const sidebar = byId("project-list").children[1].children[0];
assert.equal(sidebar.children[1].textContent, "lost (unavailable)");
sidebar.handlers.click();
sidebar.handlers.contextmenu({preventDefault() {}});
assert.equal(menu[0][0], "Why unavailable?");
menu[0][1]();
assert.deepEqual(errors, ["Tool missing", "Tool missing", "Tool missing"]);
assert.deepEqual(calls, ["/good"]);
assert.equal(busy, false);
// An availability change must invalidate the launch-list render cache.
renderSavedProjects([good, {...missing, unavailable_reason: "New reason"}]);
rows = host.children[0].children[1].children;
assert.equal(rows[1].title, "New reason");
showSettings({profile: null,
              project_types: [{type_id: "good", label: "Available"}],
              project_type_error: "Old selection"});
assert.deepEqual(requests, [["good", ""]]);
assert.equal(profile, null);
showSettings({profile: null, project_types: [],
              project_type_error: "No project types"});
assert.equal(profile, null);
assert.equal(byId("type-id").value, "");
assert.equal(byId("type-id").disabled, true);
assert.equal(byId("type-id").children.length, 0);
assert.equal(host.children[0].children[1].children.length, 2);
assert.equal(errors.at(-1), "No project types");
'''
        result = subprocess.run(
            [shutil.which("node"), "-e",
             "eval(require('node:fs').readFileSync(0, 'utf8'))"],
            input=script, capture_output=True, text=True, encoding="utf-8",
            timeout=15, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
