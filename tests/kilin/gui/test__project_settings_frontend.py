"""Exercise the project's name, Delete project and saving in Settings.

The General tab of the Project region shows the open project's name,
which can be changed in place, and Delete project, which closes the
project and moves it to the deleted folder after a confirmation. On the
start screen the name is kept for the next new project. With a project
open, every other change is saved as soon as it is made: Indexing values
(which rebuild the index), answer numbers, models and search depth. A
refused or cancelled change is put back. Run the production page
functions in Node with stubs. Moving the folder and rebuilding are tested
in the core and the engine.

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


class TestProjectSettingsFrontend(unittest.TestCase):
    """Name, rename or delete the project from the Project region."""

    def test_name_and_delete_sit_in_the_general_tab(self):
        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('data-settings-tab="general"')
        project = page[start : page.index("</section>", start)]
        self.assertIn('<input id="project-title" type="text"', project)
        self.assertIn('id="delete-project"', project)
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'byId("project-title").addEventListener("change", '
            "renameProjectFromSettings);",
            source,
        )
        self.assertIn(
            'byId("delete-project").addEventListener("click", '
            "deleteOpenProject);",
            source,
        )
        # Settings opens on the General tab and redraws when opened.
        self.assertIn('let settingsTab = "general";', source)
        self.assertIn(
            "showProjectSettings();", extract_function(source, "openModels")
        )
        # Opening a project, or Settings with one open, shows that
        # project's own values, never edits made on the start screen.
        start = source.index('data.event === "opened"')
        opened = source[start : source.index("} else if", start)]
        self.assertIn("showIndexingFields();", opened)
        self.assertIn('byId("backend").value = profile.backend;', opened)
        self.assertIn(
            "if (ready && profile) showIndexingFields();",
            extract_function(source, "openModels"),
        )
        # The message under the tabs answers only the last action.
        self.assertIn(
            "clearSettingsStatus();", extract_function(source, "openModels")
        )
        self.assertIn("clearSettingsStatus(); showSettingsTab(", source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_rename_and_delete_the_open_project(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const parts = {};
const byId = (id) => (parts[id] = parts[id] || {
  id, value: "", textContent: "", classes: new Set(),
  classList: { toggle(name, on) {
    parts[id].classes[on ? "add" : "delete"](name);
  } },
});
const calls = [];
const bridge = new Proxy({}, {
  get: (target, name) => (...args) => calls.push([name, ...args]),
});
let busy = false;
let ready = false;
let openProjectPath = "";
let projectsFolderInfo = null;
let answer = false;
const setBusy = (value) => { busy = value; };
const askKilin = async () => answer;
const hideModels = () => calls.push(["hideModels"]);
const showSetupConfirmations = () => {};
const returnToLaunch = () => {
  calls.push(["returnToLaunch"]);
  ready = false;
  openProjectPath = "";
};
"""
        for name in (
            "showProjectSettings",
            "projectName",
            "renameProjectFromSettings",
        ):
            script += extract_function(source, name)
        script += "async " + extract_function(source, "deleteOpenProject")
        script += r"""
(async () => {
  // Start screen: no project to delete; a typed name is kept for the
  // next new project, not sent as a rename.
  showProjectSettings();
  assert.ok(byId("delete-project").classes.has("hidden"));
  byId("project-title").value = "review";
  renameProjectFromSettings();
  showProjectSettings();
  assert.equal(byId("project-title").value, "review");
  assert.deepEqual(calls, []);

  // An open project shows its folder's name.
  ready = true;
  openProjectPath = "/Users/me/KiLin/papers/";
  showProjectSettings();
  assert.ok(!byId("delete-project").classes.has("hidden"));
  assert.equal(byId("project-title").value, "papers");

  // An empty or unchanged name is put back and nothing is sent.
  for (const value of ["  ", "papers"]) {
    byId("project-title").value = value;
    renameProjectFromSettings();
    assert.equal(byId("project-title").value, "papers");
  }
  assert.deepEqual(calls, []);
  // A new name renames the project.
  byId("project-title").value = " review ";
  renameProjectFromSettings();
  assert.equal(busy, true);
  assert.deepEqual(calls, [["renameProject", "review"]]);
  busy = false;

  // Cancel keeps the project.
  calls.length = 0;
  await deleteOpenProject();
  assert.deepEqual(calls, []);
  // Delete closes it first, then asks to move that folder.
  answer = true;
  await deleteOpenProject();
  assert.deepEqual(calls, [
    ["hideModels"], ["returnToLaunch"],
    ["deleteProject", "/Users/me/KiLin/papers/"],
  ]);
  console.log("project settings passed");
})().catch((error) => { console.error(error); process.exit(1); });
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("project settings passed", result.stdout)


    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_indexing_edits_rebuild_the_open_project(self):
        """Only an open project's changed values are sent, once valid."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (id) => (fields[id] = fields[id] || { value: "" });
let ready = false;
let profile = {
  extensions: [".md", ".txt", ".html"], max_file_bytes: 1000,
  target_tokens: 300, max_tokens: 400, overlap_tokens: 40,
};
const supportedExtensions = new Set([".md", ".txt"]);
const fill = () => {
  byId("extensions").value = ".html";
  byId("max-file-bytes").value = "1000";
  byId("target-tokens").value = "300";
  byId("max-tokens").value = "400";
  byId("overlap-tokens").value = "40";
};
"""
        script += r"""
const calls = [];
const bridge = {
  applyIndexingSettings: (json) => calls.push(JSON.parse(json)),
};
let busy = false;
let applyingIndexing = false;
const showIndexingCapabilities = () => {};
const setBusy = (value) => { busy = value; };
let answer = true;
const asked = [];
const askKilin = async (title) => { asked.push(title); return answer; };
"""
        for name in (
            "fileTypes",
            "indexingEdits",
            "indexingProblem",
            "showIndexingFields",
            "keepNewProjectValue",
        ):
            script += extract_function(source, name)
        script += "async " + extract_function(source, "saveIndexingEdits")
        script += r"""
(async () => {
fill();
// Nothing to rebuild on the start screen: the values are for new projects.
byId("target-tokens").value = "350";
assert.equal(indexingEdits(), null);
// But a value Prepare project would refuse is refused at once.
const overlap = byId("overlap-tokens");
overlap.dataset = { before: "40" };
overlap.value = "500";
await saveIndexingEdits({ target: overlap });
assert.equal(overlap.value, "40");
assert.match(byId("model-download-status").textContent, /Overlap/);
assert.deepEqual(calls, []);
assert.deepEqual(asked, []);
byId("target-tokens").value = "300";
// An open project: unchanged values send nothing.
ready = true;
fill();
assert.equal(indexingEdits(), null);
// Only the changed values are sent; the field lists the extra types.
byId("target-tokens").value = "350";
byId("extensions").value = ".html, .css";
assert.deepEqual(indexingEdits(), {
  extensions: [".md", ".txt", ".html", ".css"], target_tokens: 350,
});
assert.equal(indexingProblem(), "");
// Cancelling the rebuild puts the value back.
answer = false;
await saveIndexingEdits();
assert.deepEqual(asked, ["Rebuild the index?"]);
assert.deepEqual(calls, []);
assert.equal(byId("target-tokens").value, 300);
// Agreeing saves the entered values and rebuilds.
answer = true;
byId("target-tokens").value = "350";
byId("extensions").value = ".html, .css";
await saveIndexingEdits();
assert.equal(busy, true);
assert.match(byId("model-download-status").textContent, /Applying/);
assert.doesNotMatch(byId("model-download-status").textContent, /Saved/);
assert.deepEqual(calls, [{
  extensions: [".md", ".txt", ".html", ".css"], target_tokens: 350,
}]);
// A refused value is put back without asking.
busy = false;
calls.length = 0;
asked.length = 0;
fill();
byId("overlap-tokens").value = "350";
assert.match(indexingProblem(), /Overlap/);
await saveIndexingEdits();
assert.deepEqual(calls, []);
assert.deepEqual(asked, []);
assert.equal(byId("overlap-tokens").value, 40);
assert.match(byId("model-download-status").textContent, /Overlap/);
console.log("indexing edits passed");
})().catch((error) => { console.error(error); process.exit(1); });
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("indexing edits passed", result.stdout)
        # Saved on Enter or leaving the field, not on each arrow click or
        # when Settings closes.
        self.assertIn("saveOnEnter(byId(id), saveIndexingEdits);", source)
        self.assertNotIn(
            'addEventListener("change", saveIndexingEdits)', source
        )
        self.assertNotIn(
            "indexingEdits", extract_function(source, "closeModels")
        )


    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_answer_numbers_are_saved_when_entered(self):
        """Temperature and the token limit save at once; bad ones revert."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'byId(id).addEventListener("change", saveAnswerNumbers);', source
        )
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (id) => (fields[id] = fields[id] || { value: "" });
const calls = [];
const bridge = {
  saveAnswerSettings: (json) => calls.push(JSON.parse(json)),
};
let ready = true;
let busy = false;
let openProjectPath = "/saved/project";
let savedAnswerNumbers = { temperature: 0, max_new_tokens: null };
let savedAnswerThinking = true;
const answerSettingsSave = {
  nextId: 0, pending: null, queued: null, uncertain: false,
};
const updateSendButton = () => {};
const renderEffortPicker = () => {};
const error = { textContent: "", classList: { remove: () => {} } };
"""
        for name in (
            "keepNewProjectValue", "confirmedAnswerSettings",
            "sameAnswerSettings", "displayAnswerSettings",
            "queueAnswerSettings", "matchesAnswerSettingsSave",
            "answerSettingsSaved", "saveAnswerNumbers",
        ):
            script += extract_function(source, name)
        script += r"""
byId("temperature").value = "0.7";
byId("max-new-tokens").value = "";
byId("thinking").checked = true;
saveAnswerNumbers();
assert.deepEqual(calls, [{
  request_id: 1, project: "/saved/project",
  settings: { temperature: 0.7, thinking: true, max_new_tokens: null },
}]);
assert.equal(savedAnswerNumbers.temperature, 0);
answerSettingsSaved({
  request_id: 1, project: "/saved/project", temperature: 0.7,
  max_new_tokens: null, thinking: true,
});
assert.equal(savedAnswerNumbers.temperature, 0.7);
// A refused value is put back to the saved one.
byId("temperature").value = "-1";
saveAnswerNumbers();
assert.equal(calls.length, 1);
assert.equal(byId("temperature").value, 0.7);
// On the start screen they wait for the new project: checked at once,
// but nothing is sent.
ready = false;
const field = byId("temperature");
field.dataset = { before: "0.7" };
field.value = "0.2";
saveAnswerNumbers({ target: field });
assert.equal(calls.length, 1);
assert.equal(
  byId("model-download-status").textContent, "Kept for the new project."
);
// A refused start-screen value is put back, as Prepare project would.
field.value = "-3";
saveAnswerNumbers({ target: field });
assert.equal(field.value, "0.7");
assert.match(byId("model-download-status").textContent, /temperature/);
assert.equal(calls.length, 1);
console.log("answer numbers passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("answer numbers passed", result.stdout)


    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_model_changes_save_when_made(self):
        """Saved as made; refused ones go back; Close only closes."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = extract_function(source, "renderModelsPanel")
        self.assertEqual(render.count("saveModelEdits();"), 3)
        self.assertEqual(render.count("saveOnEnter("), 2)
        self.assertIn(
            "saveModelEdits();",
            extract_function(source, "restoreDefaultModels"),
        )
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (id) => (fields[id] = fields[id] || { value: "" });
let busy = false;
let ready = true;
let pending = true;
let sent = true;
let applyingModels = false;
let closeAfterModelsApply = false;
let folderConfirmed = true;
let termsAccepted = true;
const calls = [];
const modelEditsPending = () => pending;
const applyModelEdits = async () => {
  calls.push("apply");
  byId("model-download-status").textContent = sent ? "" : "Refused.";
  return sent;
};
const DEFAULT_LEVEL = 2;
let draftLevels = [];
let draftDepth = [];
let draftRouter = "";
const levels = [1, 2, 3].map((n) => ({ name: `L${n}`, model: `m${n}` }));
const cloneLevels = (items) => items.map((item) => ({ ...item }));
const currentLevels = () => levels;
const currentDepth = () => [5, 8, 10, 15, 20];
const currentRouter = () => "router";
let modelsBaseline = { embedding_model: "embed", query_format: "qwen3" };
const renderModelsPanel = () => calls.push("render");
const hideModels = () => calls.push("hide");
const document = { activeElement: null };
let activeBackend = "mlx";
let savedLevels = {};
let savedRouters = {};
let savedDepth = null;
const levelProblem = (items) => (items.some((item) => !item.name.trim())
  ? "Give effort level 1 a name." : "");
const saveModelFields = () => calls.push("fields");
const currentModelSettings = () => ({ depth: savedDepth });
const refreshRestoreButtons = () => {};
// KíLín setup is done, so Settings can close.
const setupLocked = () => false;
"""
        for name in (
            "showSavedModels", "closeModels", "keepNewProjectModels"
        ):
            script += extract_function(source, name)
        script += "async " + extract_function(source, "saveModelEdits")
        script += r"""
(async () => {
// A change that is sent stays as entered.
draftDepth = [5, 9, 10, 15, 20];
await saveModelEdits();
assert.deepEqual(calls, ["apply"]);
assert.deepEqual(draftDepth, [5, 9, 10, 15, 20]);
// A refused or cancelled one puts the saved values back, keeping why.
calls.length = 0;
sent = false;
byId("embedding-model").value = "other";
await saveModelEdits();
assert.deepEqual(calls, ["apply", "render"]);
assert.deepEqual(draftDepth, [5, 8, 10, 15, 20]);
assert.equal(byId("embedding-model").value, "embed");
assert.equal(byId("chat-model").value, "m3");
assert.equal(byId("model-download-status").textContent, "Refused.");
// Start screen: kept on the page for the new project; nothing is sent
// to the engine, so no preferences are written.
calls.length = 0;
ready = false;
draftLevels = cloneLevels(levels);
draftLevels[0].name = "Quick";
draftDepth = [2, 3, 4, 5, 6];
draftRouter = "small";
await saveModelEdits();
assert.deepEqual(calls, ["fields"]);
assert.equal(savedLevels.mlx[0].name, "Quick");
assert.equal(savedRouters.mlx, "small");
assert.deepEqual(savedDepth, [2, 3, 4, 5, 6]);
assert.equal(
  byId("model-download-status").textContent, "Kept for the new project."
);
// An empty level name is refused there too, and the values go back.
calls.length = 0;
draftLevels[0].name = " ";
await saveModelEdits();
assert.deepEqual(calls, ["render"]);
assert.match(byId("model-download-status").textContent, /name/);
ready = true;
// Close never applies; a save still running closes Settings afterwards.
calls.length = 0;
closeModels();
assert.deepEqual(calls, ["hide"]);
calls.length = 0;
applyingModels = true;
closeModels();
assert.deepEqual(calls, []);
assert.equal(closeAfterModelsApply, true);
console.log("model saves passed");
})().catch((error) => { console.error(error); process.exit(1); });
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model saves passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
