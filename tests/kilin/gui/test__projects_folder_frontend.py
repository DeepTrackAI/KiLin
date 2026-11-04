"""Exercise the remembered workspace folder in Settings.

The Workspace folder holds the projects. Run the production page functions
in Node with stubs: the field shows the folder in use, Use default is
offered only when a folder was chosen, a missing folder is explained, a
switch is allowed only on the start screen, and an empty project gets a
free name inside it. Saving and the engine restart are tested on the Qt
side.

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
    """Remember the workspace folder; Use default goes back."""

    def test_workspace_folder_field_holds_the_projects(self):
        """Read-only field with Browse and Use default; never one project."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('data-settings-tab="project"')
        project = page[start : page.index("</section>", start)]
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
        self.assertIn(
            'const destination = included.length ? "" : emptyProjectPath();',
            create,
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
});
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
const setBusy = (value) => { busy = value; };
const refreshSavedProject = () => calls.push("refreshSavedProject");
"""
        for name in (
            "showProjectsFolder",
            "changeProjectsFolder",
            "emptyProjectPath",
        ):
            script += _function(source, name)
        script += r"""
// The default folder: nothing to go back to.
showProjectsFolder({ folder: "/repo/sandbox", saved: "", missing: false });
assert.equal(byId("workspace-path").value, "/repo/sandbox");
assert.equal(byId("default-workspace").disabled, true);
assert.match(byId("workspace-note").textContent, /holds your projects/);
assert.deepEqual(calls, []);

// A chosen folder that is not mounted is explained.
showProjectsFolder({
  folder: "/repo/sandbox", saved: "/Cloud/Projects", missing: true,
});
assert.equal(byId("default-workspace").disabled, false);
assert.equal(byId("workspace-note").textContent,
  "/Cloud/Projects is not available, so KíLín uses the default folder.");

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

// With a project open the switch is refused with a notice.
calls.length = 0;
switchingProjectsFolder = false;
ready = true;
changeProjectsFolder(true);
assert.deepEqual(calls, []);
assert.equal(error.hidden, false);
assert.match(error.textContent, /start screen/);
// An empty project gets a free name inside the workspace folder.
byId("workspace-path").value = "/Users/me/Desktop/";
savedProjects = [{ name: "TestKiLin" }, { name: "new-project" }];
assert.equal(emptyProjectPath(), "/Users/me/Desktop/new-project-2");
savedProjects = [];
assert.equal(emptyProjectPath(), "/Users/me/Desktop/new-project");
console.log("projects folder passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("projects folder passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
