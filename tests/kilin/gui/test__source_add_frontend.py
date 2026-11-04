"""Exercise the dialog shown before sources are added to an open project.

Run the production ``sourceAddSummary`` and ``sourceAddReason`` functions in
Node, and check that dropping sources asks the engine to check them, that
the ``sources_checked`` event opens the dialog, and that Add sends the same
sources with the choices for copies already in the project. Rendering is
covered only by the Qt/WebEngine smoke test and manual checks.

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

_ROOT = Path(__file__).resolve().parents[3]
_APP = _ROOT / "kilin/gui/web/app.js"


class TestSourceAddFrontend(unittest.TestCase):
    """Show what adding sources would copy, then add on confirmation."""

    def test_drop_checks_first_and_add_confirms(self):
        """Check with the backend first; add only on explicit confirmation."""

        source = _APP.read_text(encoding="utf-8")
        if not shutil.which("node"):
            add_sources = _function(source, "addSources")
            self.assertIn(
                "bridge.checkProjectSources(JSON.stringify(paths));",
                add_sources,
            )
            self.assertNotIn("bridge.addProjectSources", add_sources)
            start = source.index('byId("source-add-confirm")')
            confirm = source[start:source.index("});", start)]
            self.assertIn("bridge.addProjectSources(", confirm)
            return
        start = source.index('byId("source-add-confirm").addEventListener')
        handler = source[start:source.index("});", start) + 3]
        script = r"""
const assert = require("node:assert/strict");
const calls = [];
let busy = false;
let ready = true;
let selectedSources = [];
let sourceAddPaths = ["/tmp/a.py"];
const error = { classList: { add(value) {
  calls.push(["hide-error", value]);
}}};
let bridge = {
  checkProjectSources(payload) {
    calls.push(["check", JSON.parse(payload)]);
  },
  addProjectSources(paths, decisions) {
    calls.push(["add", JSON.parse(paths), JSON.parse(decisions)]);
  },
};
function setBusy(value, label) {
  busy = value;
  calls.push(["busy", value, label]);
}
function previewSources() { calls.push(["preview"]); }
let confirm = null;
const elements = {
  "source-add": { close() { calls.push(["close"]); } },
  "source-add-confirm": {
    addEventListener(event, handler) {
      assert.equal(event, "click");
      confirm = handler;
    },
  },
  "source-add-files": {
    querySelectorAll(selector) {
      assert.equal(selector, "input:checked");
      return [{ dataset: { path: "sources/a.py" }, value: "replace" }];
    },
  },
};
function byId(id) { assert.ok(elements[id]); return elements[id]; }
"""
        script += _function(source, "addSources")
        script += handler
        script += r"""
addSources(["/tmp/a.py"]);
assert.deepEqual(calls.at(-1), ["check", ["/tmp/a.py"]]);
assert.equal(calls.filter((call) => call[0] === "add").length, 0);
assert.equal(typeof confirm, "function");
busy = false;  // The source-check reply has arrived.
confirm();
assert.deepEqual(calls.at(-1), ["add", ["/tmp/a.py"],
  { "sources/a.py": "replace" }]);
assert.equal(calls.filter((call) => call[0] === "add").length, 1);
// Busy operations cannot trigger another check or an Add.
addSources(["/tmp/b.py"]);
confirm();
assert.equal(calls.filter((call) => call[0] === "add").length, 1);
// A new project collects candidates for preview, without engine writes.
busy = false;
ready = false;
addSources([" a.py ", "a.py", ""]);
assert.deepEqual(selectedSources, ["a.py"]);
assert.deepEqual(calls.at(-1), ["preview"]);
assert.equal(calls.filter((call) => call[0] === "add").length, 1);
console.log("source confirmation passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source confirmation passed", result.stdout)

    def test_sources_checked_opens_selection_dialog(self):
        """Keep the event-to-dialog dispatch contract visible."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('data.event === "sources_checked"')
        end = source.index('} else if (data.event', start)
        self.assertIn("showSourceAdd(data);", source[start:end])

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_summary_and_reasons(self):
        """Count files to add, already added, changed and left out."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const selectionReasonCodes = {
  FILE_TOO_LARGE: "file_too_large", EMPTY_FILE: "empty_file"
};
"""
        script += _function(source, "sourceAddPending")
        script += _function(source, "sourceAddSummary")
        script += _function(source, "sourceAddReason")
        script += r"""
const none = { added: [], changed: [], deleted: [], edited: [] };
const left = [
  { path: "notes/data.json", reason_code: "file_type_not_selected" },
  { path: "notes/big.txt", reason_code: "file_too_large" },
];
assert.equal(
  sourceAddSummary({
    add: [
      { name: "notes", kind: "folder", files: 11, state: "new" },
      { name: "a.md", kind: "file", files: 1, state: "renamed" },
    ],
    files: left, skipped_folders: [], changes: none,
  }),
  "KíLín will add 12 files. 2 files are left out because this project "
    + "does not index them.",
);
assert.equal(
  sourceAddSummary({
    add: [{ name: "a.md", kind: "file", files: 1, state: "new" }],
    files: [], skipped_folders: [], changes: none,
  }),
  "KíLín will add 1 file.",
);
// Sources already in the project are named, with what changed since.
assert.equal(
  sourceAddSummary({
    add: [{ name: "README.md", kind: "file", files: 1, state: "existing" }],
    files: [], skipped_folders: [], changes: none,
  }),
  "README.md is already in the project, with no changes.",
);
assert.equal(
  sourceAddSummary({
    add: [
      { name: "GAUDI", kind: "folder", files: 9, state: "existing" },
      { name: "b.md", kind: "file", files: 1, state: "new" },
    ],
    files: [], skipped_folders: [],
    changes: { ...none, changed: ["sources/GAUDI/a.py"],
      added: ["sources/GAUDI/new.py"], edited: ["sources/GAUDI/c.py"] },
  }),
  "KíLín will add 1 file. GAUDI is already in the project; 2 files "
    + "changed since it was copied.",
);
assert.equal(
  sourceAddSummary({
    add: [], files: left, skipped_folders: ["ops"], changes: none,
  }),
  "Nothing can be added: this project does not index these files.",
);
assert.equal(sourceAddReason(left[0]), "File type not included");
assert.equal(sourceAddReason(left[1]), "Too large");
assert.equal(sourceAddReason(
  { path: "empty.txt", reason_code: "empty_file" }
), "Empty file");
console.log("source add passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source add passed", result.stdout)
