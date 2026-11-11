"""Verify the sidebar's transient answering-conversation indicator.

Execute the production sidebar and busy-state functions in Node with a
minimal DOM adapter. The status must identify the writer even while a
reader views a different conversation, then clear when answering ends.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestGeneratingConversationFrontend(unittest.TestCase):
    """Keep the running answer visible in the conversation list."""

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_sidebar_marks_only_the_answering_conversation(self):
        """Preserve the marker across redraws and clear it on stop."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
class Element {
  constructor() {
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.classes = new Set();
    this.classList = {
      add: (name) => this.classes.add(name),
      remove: (name) => this.classes.delete(name),
      contains: (name) => this.classes.has(name),
      toggle: (name, value) => {
        if (value === undefined ? !this.classes.has(name) : value) {
          this.classes.add(name);
        } else {
          this.classes.delete(name);
        }
      },
    };
    this.textContent = "";
  }
  set className(value) {
    this.classes = new Set(value.split(/\s+/).filter(Boolean));
  }
  get className() { return [...this.classes].join(" "); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener() {}
}
const fields = {};
const byId = (id) => (fields[id] ??= new Element());
const document = {
  createElement: () => new Element(),
  querySelectorAll(selector) {
    if (selector === ".conversation-item") {
      return byId("conversation-list").children.map(
        (row) => row.children[0]);
    }
    return [];
  },
};
const activity = new Element();
const unreadConversations = new Set();
let busy = false;
let ready = true;
let kilinReady = true;
let bridge = {};
let activeConversationId = "a";
let answeringConversationId = "";
let openProjectPath = "/project";
let askingConversation = null;
let reading = null;
let lastConversationData = null;
const sidebarPref = (_, fallback) => fallback;
const sidebarConversationOrder = (items) => items;
const updateChatSummary = () => {};
const showSetupLock = () => {};
const syncSourceDropTarget = () => {};
const updateSendButton = () => {};
const showNativeSourceDrag = () => {};
const renderModelsPanel = () => {};
const focusComposer = () => {};
const makeReorderable = () => {};
"""
        for name in (
            "syncGeneratingConversations", "renderConversations", "setBusy",
            "showAskingConversation", "clearAskingConversation",
            "provisionalTitle",
        ):
            script += extract_function(source, name)
        script += r"""
const data = {
  active_conversation_id: "a",
  conversations: [
    {conversation_id: "a", name: "Analysis", empty: false},
    {conversation_id: "b", name: "Review", empty: false},
  ],
};
function rows() {
  return Object.fromEntries(document.querySelectorAll(".conversation-item")
    .map((button) => [button.dataset.conversationId, button]));
}
function marked() {
  return Object.entries(rows()).filter(([, item]) =>
    item.classList.contains("generating")).map(([id]) => id);
}
renderConversations(data);
assert.deepEqual(marked(), []);
// Other background work must not be confused with answering.
setBusy(true, "Indexing");
assert.deepEqual(marked(), []);
setBusy(false);
// Starting an answer changes the exact writer's appearance.
answeringConversationId = "a";
activity.classList.add("answer-progress");
setBusy(true, "Answering");
assert.deepEqual(marked(), ["a"]);
assert.equal(rows().a.attributes["aria-label"],
  "Analysis (generating answer)");
assert.equal(rows().b.attributes["aria-label"], "Review");
// Reading another chat makes b selected but does not move the spinner.
reading = {id: "b", name: "Review"};
renderConversations(data);
assert.deepEqual(marked(), ["a"]);
assert.equal(rows().b.classList.contains("active"), true);
assert.equal(rows().a.classList.contains("active"), false);
// Completion, cancellation, and failure all reset through setBusy(false).
setBusy(false);
assert.deepEqual(marked(), []);
assert.equal(rows().a.attributes["aria-label"], "Analysis");
assert.equal(answeringConversationId, "");
// The first question makes an empty chat visible before it is saved.
reading = null;
renderConversations({
  active_conversation_id: "a",
  conversations: [{conversation_id: "a", name: "Conversation 1", empty: true}],
});
assert.equal(Object.keys(rows()).length, 0);
answeringConversationId = "a";
activity.classList.add("answer-progress");
showAskingConversation("Summarize this paper?");
setBusy(true, "Answering");
assert.deepEqual(marked(), ["a"]);
assert.equal(rows().a.classList.contains("provisional"), true);
setBusy(false);
clearAskingConversation();
assert.deepEqual(marked(), []);
assert.equal(Object.keys(rows()).length, 0);
console.log("sidebar generating indicator passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "sidebar generating indicator passed", result.stdout
        )

    def test_generation_state_and_styles_are_local_to_gui(self):
        """Use the existing answer lifecycle and reduced-motion policy."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(
            "syncGeneratingConversations();",
            extract_function(source, "setBusy"),
        )
        self.assertIn(
            "syncGeneratingConversations();",
            extract_function(source, "renderConversations"),
        )
        for name in ("completePendingTurn", "showError"):
            with self.subTest(name=name):
                self.assertIn(
                    "setBusy(false);", extract_function(source, name)
                )
        self.assertIn(
            'data.event === "answer_cancelled"',
            extract_function(source, "onEvent"),
        )
        self.assertIn(".conversation-item.generating", styles)
        self.assertIn("animation: send-waiting", styles)
        self.assertIn(
            "prefers-reduced-motion: reduce", styles
        )


if __name__ == "__main__":
    unittest.main()
