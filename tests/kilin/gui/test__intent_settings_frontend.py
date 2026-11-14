"""Characterize project-local Intents and indexing settings UI in Node.

The tests exercise serialization and project identity without Qt, while
engine tests check core validation and index recovery.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestIntentSettingsFrontend(unittest.TestCase):

    def test_editor_panel_and_protocol_are_connected(self):
        page = (_WEB / "index.html").read_text(encoding="utf-8")
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn('data-tab="intents">Intents</button>', page)
        self.assertIn('data-settings-tab="intents"', page)
        self.assertIn('id="intent-settings"', page)
        self.assertIn('bridge.requestProjectBehavior()', source)
        self.assertIn('bridge.saveProjectBehavior(', source)
        self.assertIn('data.event === "project_behavior"', source)
        self.assertIn('save.textContent = "Save settings"', source)
        self.assertIn('restore.textContent = "Restore defaults"', source)
        self.assertNotIn(
            'innerHTML', extract_function(source, 'renderProjectBehavior')
        )
        for key in (
            'code-graph', 'document-headers', 'save-index-capabilities',
            'reset-index-capabilities', 'index-capabilities-status',
        ):
            self.assertIn(f'id="{key}"', page)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_editor_values_and_revision_aware_saving(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r'''
const assert = require("node:assert/strict");
let openProjectPath = "/one";
let ready = true;
let projectBehaviorData = {project: "/one", revision: "rev"};
let behaviorSave = {nextId: 0, pending: null};
const calls = [];
const bridge = { saveProjectBehavior(...args) {calls.push(args);} };
function settingLabel(key) { return key; }
function field(path, kind, value) {
  return { dataset: {behaviorPath: path, behaviorKind: kind},
    value: String(value), checked: Boolean(value),
    checkValidity() {return true;} };
}
const controls = [
  field("description", "text", "Personal"),
  field("thinking", "select", "off"),
  field("retrieval.mode", "select", "search"),
  field("retrieval.strategy", "select", "hybrid"),
  field("retrieval.max_sources", "number", "3"),
  field("shortcuts", "lines", "^hello$\n^hi$"),
  field("_guidance_mode", "select", "inherit"),
  field("answer_guidance", "multiline", "Ignored"),
];
const base = {description: "Base", thinking: null,
  answer_guidance: null, shortcuts: [], retrieval: {
    mode: "search", strategy: "source_first", max_sources: 3,
    passages_per_source: 2, max_passages: 8,
    evidence_tokens: 5000, coverage: "bounded"}};
const status = {textContent: "", classList: {remove() {}, add() {}}};
const editor = {dataset: {initial: JSON.stringify(base),
    behaviorScope: "intent", intentId: "question_answering"},
  querySelectorAll() {return controls;},
  querySelector(selector) {
    if (selector.includes("_guidance_mode")) return controls[6];
    if (selector === ".behavior-status") return status;
    return null;
  }};
'''
        script += extract_function(source, 'behaviorValues')
        script += extract_function(source, 'saveBehaviorEditor')
        script += r'''
const values = behaviorValues(editor);
assert.equal(values.thinking, false);
assert.equal(values.answer_guidance, null);
assert.deepEqual(values.shortcuts, ["^hello$", "^hi$"]);
assert.equal(values.retrieval.strategy, "hybrid");
assert.equal(values.retrieval.max_sources, undefined);
saveBehaviorEditor(editor, values);
assert.deepEqual(calls[0], ["/one", "intent", "question_answering",
  JSON.stringify(values), "rev", 1]);
saveBehaviorEditor(editor, values);
assert.equal(calls.length, 1);
behaviorSave.pending = null;
openProjectPath = "/different";
saveBehaviorEditor(editor, values);
assert.equal(calls.length, 1);
console.log("intent editor passed");
'''
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("intent editor passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_indexing_capabilities_are_confirmed_and_serialized(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r'''
const assert = require("node:assert/strict");
const fields = {
  "code-graph": {value: "off"},
  "document-headers": {checked: true},
  "index-capabilities-status": {textContent: ""},
};
const byId = (id) => fields[id];
let ready = true;
let busy = false;
let applyingIndexing = false;
let openProjectPath = "/one";
let projectBehaviorData = {project: "/one", revision: "rev",
  capabilities: {values: {code_graph: "auto", document_headers: false},
    defaults: {code_graph: "auto", document_headers: false}}};
let behaviorSave = {nextId: 0, pending: null};
const calls = [];
const bridge = {saveProjectBehavior(...args) {calls.push(args);}};
let approved = false;
async function askKilin() { return approved; }
function setBusy(value) {busy = value;}
function showIndexingCapabilities() {}
'''
        script += extract_function(source, 'indexingCapabilityValues')
        script += extract_function(source, 'saveIndexCapabilities').replace(
            'function saveIndexCapabilities(',
            'async function saveIndexCapabilities(', 1,
        )
        script += r'''
(async () => {
  assert.deepEqual(indexingCapabilityValues(),
    {code_graph: false, document_headers: true});
  await saveIndexCapabilities();
  assert.equal(calls.length, 0);
  approved = true;
  await saveIndexCapabilities();
  assert.deepEqual(calls[0], ["/one", "capabilities", "",
    JSON.stringify({code_graph: false, document_headers: true}),
    "rev", 1]);
  assert.equal(busy, true);
  assert.equal(applyingIndexing, true);
  await saveIndexCapabilities();
  assert.equal(calls.length, 1);
  console.log("indexing capabilities passed");
})().catch((error) => {console.error(error); process.exitCode = 1;});
'''
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("indexing capabilities passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
