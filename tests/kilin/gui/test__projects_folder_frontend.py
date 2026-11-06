"""Exercise the remembered projects folder in Settings.

The projects folder holds the projects. Run the production page functions
in Node with stubs: the field shows the folder in use, Use default is
offered only when a folder was chosen, a missing folder is explained, a
switch is offered and allowed only on the start screen (read-only from a
project), and an empty project gets a free name inside it. Saving and the
engine restart are tested on the Qt side.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function as _function,
    run_node,
)

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestProjectsFolderFrontend(unittest.TestCase):
    """Remember the projects folder; Use default goes back."""

    def test_projects_folder_field_holds_the_projects(self):
        """Read-only field with Browse and Use default, in KíLín setup."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('id="first-run-setup"')
        project = page[start : page.index("</details>", start)]
        for part in (
            '<input id="workspace-path" type="text" readonly>',
            'id="browse-workspace"',
            'id="default-workspace"',
            'id="workspace-note"',
        ):
            self.assertIn(part, project)
        self.assertNotIn("projects-folder", page)
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("bridge.requestProjectsFolder();", source)
        # The open project's own path comes from the engine, not the field.
        self.assertNotIn('byId("workspace-path").value = data.project', source)
        self.assertIn(
            "const root = openProjectPath.replace(",
            _function(source, "copyToSources"),
        )
        start = source.index('byId("project-form").addEventListener(')
        create = source[start : source.index("\n});\n", start)]
        # A name set in Settings wins; a taken one is refused first.
        self.assertIn(
            "const destination = name ? projectPath(name) : "
            'included.length ? "" : emptyProjectPath();',
            create,
        )
        self.assertLess(
            create.index("already exists"), create.index("bridge.openProject(")
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_folder_is_shown_and_switched_only_on_the_start_screen(self):
        """Show, explain a missing folder, refresh after a switch."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const parts = {};
const byId = (id) => (parts[id] = parts[id] || {
  id, value: "", title: "", textContent: "", disabled: false,
  hidden: new Set(),
  classList: { toggle(name, on) {
    parts[id].hidden[on ? "add" : "delete"](name);
  } },
});
const shown = (id) => !byId(id).hidden.has("hidden");
const calls = [];
const bridge = new Proxy({}, {
  get: (target, name) => (...args) => calls.push(name),
});
const error = {
  textContent: "", hidden: true,
  classList: { remove() { error.hidden = false; } },
};
let busy = false;
let ready = false;
let switchingProjectsFolder = false;
let savedProjects = [];
let projectsFolderInfo = null;
let openProjectPath = "";
const setBusy = (value) => { busy = value; };
const refreshSavedProject = () => calls.push("refreshSavedProject");
const showSetupConfirmations = () => {};
"""
        for name in (
            "showProjectsFolder",
            "showProjectSettings",
            "projectName",
            "changeProjectsFolder",
            "emptyProjectPath",
            "projectPath",
        ):
            script += _function(source, name)
        script += r"""
// The default folder: nothing to go back to.
showProjectsFolder({ folder: "/repo/sandbox", saved: "", missing: false });
assert.equal(byId("workspace-path").value, "/repo/sandbox");
assert.equal(byId("default-workspace").disabled, true);
// Nothing to explain, so no note.
assert.ok(!shown("workspace-note"));
assert.equal(byId("browse-workspace").disabled, false);
assert.deepEqual(calls, []);

// A chosen folder that is not mounted is explained.
showProjectsFolder({
  folder: "/repo/sandbox", saved: "/Cloud/Projects", missing: true,
});
assert.equal(byId("default-workspace").disabled, false);
assert.equal(byId("workspace-note").textContent,
  "/Cloud/Projects is not available, so KíLín uses the default folder.");
assert.ok(shown("workspace-note"));

// Use default on the start screen: Qt saves and restarts the engine.
changeProjectsFolder(true);
assert.deepEqual(calls, ["useDefaultProjectsFolder"]);
assert.equal(busy, true);
// After the restart the start screen lists that folder's projects.
showProjectsFolder({ folder: "/repo/sandbox", saved: "", missing: false });
assert.equal(busy, false);
assert.deepEqual(calls.slice(1),
  ["requestSettings", "requestReadiness", "refreshSavedProject"]);

// Browse asks Qt for a folder; nothing is busy until one is chosen.
calls.length = 0;
changeProjectsFolder(false);
assert.deepEqual(calls, ["chooseProjectsFolder"]);
assert.equal(busy, false);

// With a project open the folder is read-only: the buttons stay but are
// disabled, saying where to change it; a switch is refused with a notice.
calls.length = 0;
switchingProjectsFolder = false;
ready = true;
showProjectSettings();
assert.equal(byId("browse-workspace").disabled, true);
assert.equal(byId("default-workspace").disabled, true);
assert.equal(byId("browse-workspace").title,
  "Change it from the start screen.");
changeProjectsFolder(true);
assert.deepEqual(calls, []);
assert.equal(error.hidden, false);
assert.match(error.textContent, /start screen/);
// An empty project gets a free name inside the projects folder.
byId("workspace-path").value = "/Users/me/Desktop/";
savedProjects = [{ name: "TestKiLin" }, { name: "new-project" }];
assert.equal(emptyProjectPath(), "/Users/me/Desktop/new-project-2");
savedProjects = [];
assert.equal(emptyProjectPath(), "/Users/me/Desktop/new-project");
// A name set in Settings goes in the same folder.
assert.equal(projectPath("review"), "/Users/me/Desktop/review");
console.log("projects folder passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("projects folder passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
