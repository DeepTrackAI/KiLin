"""Exercise conversation-memory controls with production frontend functions.

Node.js runs the actual feedback, policy, provenance and navigation handlers
without Qt. Native WebEngine interaction remains covered by optional smoke
checks rather than simulated here.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"
_HTML = Path(__file__).resolve().parents[3] / "kilin/gui/web/index.html"


class TestConversationMemoryFrontend(unittest.TestCase):
    """Verify safe message reactions and source-independent navigation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load only production UI assets."""

        cls.source = _WEB.read_text(encoding="utf-8")
        cls.html = _HTML.read_text(encoding="utf-8")

    def test_settings_and_protocol_contracts(self) -> None:
        """Ensure visible fields and engine events retain their exact IDs."""

        for name in (
            "memory-enabled", "memory-questions", "memory-answers",
            "memory-max-hits", "memory-max-tokens",
            "memory-exclude-disliked", "save-memory-policy",
            "reset-memory-policy", "sync-memory-index",
            "memory-index-status",
        ):
            self.assertIn(f'id="{name}"', self.html)
        for event in (
            "conversation_feedback", "message_feedback", "memory_status",
            "memory_sync_started", "memory_synced", "memory_sync_failed",
        ):
            self.assertIn(f'data.event === "{event}"', self.source)
        self.assertIn('"Earlier conversation memory (not source evidence)"',
                      self.source)
        self.assertIn('"intent_memory"', self.source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_feedback_save_and_correlated_acknowledgement(self) -> None:
        """Ignore wrong replies and restore controls on rejected writes."""

        script = r"""
const assert = require("node:assert/strict");
let ready = true, busy = false, openProjectPath = "/workspace";
const turnUuid = "0123456789abcdef0123456789abcdef";
const classes = new Set();
const button = {
  dataset: {turnUuid, feedbackRole: "assistant", feedbackAction: "liked"},
  classList: {
    toggle: (name, value) => value ? classes.add(name) : classes.delete(name),
    contains: (name) => classes.has(name),
  },
  setAttribute(name, value) { this[name] = value; },
  closest: (name) => name === ".turn"
    ? {dataset: {conversationId: "c-1"}} : row,
};
const status = {textContent: ""};
const row = {
  classList: {toggle: () => {}},
  querySelectorAll: () => [button],
  querySelector: (name) => name === ".message-feedback-status"
    ? status : button,
};
const conversation = {
  querySelectorAll: (name) => name === ".message-feedback"
    ? [button] : name === ".message-actions" ? [row] : [],
};
const calls = [];
const bridge = {
  saveMessageFeedback: (value) => calls.push(JSON.parse(value)),
  requestConversationFeedback: (project) => calls.push({reload: project}),
};
const error = {textContent: "", classList: {remove: () => {}}};
let projectBehaviorData = {intents: []};
let feedbackData = {project: "/workspace",
  revision: "a".repeat(64), entries: {}};
let feedbackSave = {nextId: 0, pending: null};
"""
        for name in (
            "messageFeedback", "memoryIntentNever", "refreshFeedbackControls",
            "setMessageFeedback", "receiveFeedback", "receiveFeedbackSave",
        ):
            script += extract_function(self.source, name)
        script += r"""
refreshFeedbackControls();
assert.equal(button.disabled, false);
setMessageFeedback(button);
assert.equal(calls.length, 1);
assert.deepEqual(calls[0], {
  project: "/workspace", conversation_id: "c-1", turn_uuid: turnUuid,
  role: "assistant", saved: false, reaction: "liked",
  revision: "a".repeat(64), request_id: 1,
});
assert.equal(button.disabled, true);
receiveFeedbackSave({project: "/other", request_id: 1,
  turn_uuid: turnUuid, role: "assistant", error: "wrong project"});
assert.notEqual(feedbackSave.pending, null);
receiveFeedback({project: "/workspace", revision: "outdated", entries: {}});
assert.equal(feedbackData.revision, "a".repeat(64));
receiveFeedbackSave({project: "/workspace", request_id: 1,
  turn_uuid: turnUuid, role: "assistant", revision: "b".repeat(64),
  entries: {[turnUuid]: {assistant: {saved: false, reaction: "liked"}}}});
assert.equal(feedbackSave.pending, null);
assert.equal(button.disabled, false);
assert.equal(button["aria-pressed"], "true");
assert.equal(status.textContent, "Saved.");
setMessageFeedback(button);
assert.equal(calls[1].reaction, "neutral");
receiveFeedbackSave({project: "/workspace", request_id: 2,
  turn_uuid: turnUuid, role: "assistant", error: "stale revision"});
assert.deepEqual(calls[2], {reload: "/workspace"});
assert.equal(status.textContent, "stale revision");
busy = true;
refreshFeedbackControls();
assert.equal(button.disabled, true);
busy = false;
button.dataset.feedbackRole = "user";
button.dataset.feedbackAction = "disliked";
refreshFeedbackControls();
assert.match(button.title, /question.*memory indexing/);
button.dataset.feedbackRole = "assistant";
refreshFeedbackControls();
assert.match(button.title, /answer.*when enabled, context/);
console.log("conversation feedback contract passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("contract passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_project_memory_settings_use_dedicated_scope(self) -> None:
        """Save memory separately from the document-indexing capabilities."""

        script = r"""
const assert = require("node:assert/strict");
let ready = true, busy = false, openProjectPath = "/workspace";
const controls = {};
const byId = (name) => (controls[name] ||= {
  checked: false, _value: "", textContent: "", disabled: false,
  set value(raw) {this._value = String(raw);},
  get value() {return this._value;},
  checkValidity: () => true,
  classList: {add: () => {}, remove: () => {}},
});
const calls = [];
const bridge = {saveProjectBehavior: (...values) => calls.push(values)};
let behaviorSave = {nextId: 0, pending: null};
let displayedMemoryProject = "", displayedMemoryRevision = "";
let memorySyncPending = false;
const values = {enabled: true, questions: "all", answers: "selected",
  max_hits: 4, max_tokens: 1024, exclude_disliked_answers: true};
let projectBehaviorData = {
  project: "/workspace", revision: "revision-1",
  memory: {values, defaults: {...values, enabled: false}},
};
"""
        for name in (
            "saveMemoryBehavior", "memoryPolicyValues", "showMemoryPolicy",
            "saveMemoryPolicy",
        ):
            script += extract_function(self.source, name)
        script += r"""
showMemoryPolicy();
assert.equal(byId("memory-enabled").checked, true);
assert.equal(byId("memory-max-tokens").value, "1024");
byId("memory-max-hits").value = "6";
saveMemoryPolicy();
assert.equal(calls.length, 1);
assert.equal(calls[0][0], "/workspace");
assert.equal(calls[0][1], "memory");
assert.equal(calls[0][2], "");
assert.equal(JSON.parse(calls[0][3]).max_hits, 6);
assert.equal(calls[0][4], "revision-1");
assert.equal(behaviorSave.pending.scope, "memory");
showMemoryPolicy();
assert.equal(byId("memory-max-hits").value, "6");
busy = true;
saveMemoryPolicy();
assert.equal(calls.length, 1);
console.log("memory settings scope passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scope passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_memory_navigation_never_becomes_a_source_citation(self) -> None:
        """Use recorded origins for conversation links, not evidence IDs."""

        script = r"""
const assert = require("node:assert/strict");
const elements = [];
const document = {createElement: (kind) => {
  const element = {kind, children: [], disabled: false, textContent: "",
    listeners: {}, append(...items) {this.children.push(...items);},
    addEventListener(name, callback) {this.listeners[name] = callback;}};
  elements.push(element);
  return element;
}};
let lastConversationData = {conversations: [
  {conversation_id: "c-1", name: "Original brainstorming"},
]};
let busy = false, bridge = {}, reading = null, activeConversationId = "c-2";
let memoryNavigation = null;
const answer = {children: [], append(item) {this.children.push(item);}};
const openMemoryOrigin = (origin) => {memoryNavigation = origin;};
"""
        script += extract_function(self.source, "appendMemoryProvenance")
        script += r"""
appendMemoryProvenance(answer, {
  parameters: {memory_references: [{label: "M1", role: "assistant",
    origins: [{conversation_id: "c-1", turn_number: 2}]}]},
});
assert.equal(answer.children.length, 1);
const panel = answer.children[0];
assert.equal(panel.className, "memory-provenance");
assert.match(panel.children[0].textContent, /not source evidence/);
const button = panel.children[1];
assert.equal(button.disabled, false);
assert.match(button.textContent, /Original brainstorming/);
button.listeners.click({stopPropagation: () => {}});
assert.deepEqual(memoryNavigation, {conversation_id: "c-1", turn_number: 2});
lastConversationData.conversations = [];
appendMemoryProvenance(answer, {
  parameters: {memory_references: [{label: "M1",
    origins: [{conversation_id: "c-1", turn_number: 2}]}]},
});
assert.equal(answer.children[1].children[1].disabled, true);
assert.equal(elements.filter((element) => element.kind === "sup").length, 0);
console.log("memory provenance separate passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("separate passed", result.stdout)
