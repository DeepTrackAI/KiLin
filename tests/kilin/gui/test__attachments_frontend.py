"""Exercise files attached to a chat message.

Run the production attachment functions in Node with stubs: attached files
are added to the project's sources before the question is asked, the
question lists their ``sources/`` paths, and the question bubble shows them
as chips that open the workspace tree. Rendering is covered only by the
Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


def _function(source: str, name: str) -> str:
    """Return one top-level function's source text."""

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


def _constant(source: str, name: str) -> str:
    """Return one top-level constant's source line."""

    start = source.index(f"const {name} =")
    return source[start : source.index("\n", start) + 1]


class TestAttachmentsFrontend(unittest.TestCase):
    """Attached files go into sources, then the question is asked."""

    def test_send_adds_attachments_before_asking(self):
        """Check, add without the dialog, ask after the refresh."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('byId("question-form").addEventListener(')
        send = source[start : source.index("\n});\n", start)]
        self.assertIn(
            "if (!ask.attachments.length) return sendQuestion(ask);", send
        )
        self.assertIn(
            "bridge.checkProjectSources(JSON.stringify(ask.attachments));",
            send,
        )
        self.assertNotIn("bridge.askQuestion(", send)
        events = _function(source, "onEvent")
        self.assertIn(
            "if (attachedAsk) return addAttachedSources(data);", events
        )
        start = events.index('data.event === "opened"')
        opened = events[start : events.index("data.event ===", start + 1)]
        self.assertIn("sendQuestion(ask);", opened)
        # The References header names the question without the file list.
        self.assertIn(
            "Sources for: ${splitAttached(turn.question).question}",
            _function(source, "selectTurn"),
        )
        start = events.index('data.event === "error"')
        failed = events[start:]
        self.assertIn('byId("question").value = attachedAsk.question;', failed)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_attachments_flow_and_chips(self):
        """Paths sent, left-out files named, chips shown and clickable."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const calls = [];
const bridge = new Proxy({}, {
  get: (target, name) => (...args) => calls.push([name, ...args]),
});
const busyStates = [];
const setBusy = (value, message) => busyStates.push([value, message]);
const pending = [];
const addPendingTurn = (text) => pending.push(text);
const showAskingConversation = () => {};
const error = {
  textContent: "", hidden: true,
  classList: { remove() { error.hidden = false; } },
};
let attachedAsk = null;
let activeConversationId = "c1";
let answeringConversationId = "";
function element(tag) {
  return {
    tag, children: [], className: "", title: "", type: "", listeners: {},
    append(...items) { this.children.push(...items); },
    set textContent(value) { this.children = [String(value)]; },
    get textContent() {
      return this.children.map((child) => (
        typeof child === "string" ? child : child.textContent)).join("");
    },
    addEventListener(type, listener) { this.listeners[type] = listener; },
  };
}
var document = { createElement: element };
const revealed = [];
const revealWorkspacePath = (path) => revealed.push(path);
"""
        script += _constant(source, "ATTACHED_LIST")
        for name in (
            "splitAttached",
            "fillQuestion",
            "sendQuestion",
            "addAttachedSources",
        ):
            script += _function(source, name)
        script += r"""
const ask = (attachments) => ({
  question: "What does it cite?", attachments, files: [], leftOut: [],
  effort: "{}", settings: { temperature: 0.2 },
});

// New and already-added files: copy the new one, then ask.
attachedAsk = ask(["/x/CITATION.cff", "/x/old.pdf", "/x/data", "/x/e.txt"]);
addAttachedSources({
  sources: ["/x/CITATION.cff", "/x/old.pdf", "/x/data", "/x/e.txt"],
  add: [
    { name: "CITATION.cff", kind: "file", state: "new" },
    { name: "old.pdf", kind: "file", state: "existing" },
    { name: "data", kind: "folder", state: "new" },
  ],
  files: [{ path: "e.txt" }], skipped_folders: [],
});
assert.equal(calls.length, 1);
assert.equal(calls[0][0], "addProjectSources");
assert.equal(calls[0][2], "{}");
assert.deepEqual(attachedAsk.files,
  ["sources/CITATION.cff", "sources/old.pdf", "sources/data/"]);
// After the refresh the page asks with the sources/ paths listed.
sendQuestion(attachedAsk);
const [, text, , settings] = calls[1];
assert.equal(calls[1][0], "askQuestion");
assert.equal(text, "What does it cite?\n\nAttached files:\n"
  + "- sources/CITATION.cff\n- sources/old.pdf\n- sources/data/");
assert.equal(JSON.parse(settings)._title_question, "What does it cite?");
assert.equal(JSON.parse(settings).temperature, 0.2);
assert.equal(pending[0], text);
// A file that could not be added is named, not silently dropped.
assert.equal(error.hidden, false);
assert.match(error.textContent, /could not add e\.txt/);

// Only files already in sources: ask at once, nothing copied.
calls.length = 0;
error.hidden = true;
attachedAsk = ask(["/x/old.pdf"]);
addAttachedSources({
  sources: ["/x/old.pdf"],
  add: [{ name: "old.pdf", kind: "file", state: "existing" }],
  files: [], skipped_folders: [],
});
assert.equal(attachedAsk, null);
assert.deepEqual(calls.map(([name]) => name), ["askQuestion"]);
assert.equal(error.hidden, true);

// Splitting saved questions, also from older chats.
assert.deepEqual(splitAttached("Plain?"), { question: "Plain?", files: [] });
assert.deepEqual(splitAttached(text).files,
  ["sources/CITATION.cff", "sources/old.pdf", "sources/data/"]);
assert.deepEqual(
  splitAttached("Old?\n\nAttached paths to consider:\n- /Users/me/a.pdf"),
  { question: "Old?", files: ["/Users/me/a.pdf"] });

// Chips: sources/ files are buttons that reveal; old paths only a name.
const bubble = element("div");
fillQuestion(bubble, `${text}`);
assert.equal(bubble.children[0], "What does it cite?");
const chips = bubble.children[1].children;
assert.equal(bubble.children[1].className, "question-attachments");
assert.deepEqual(chips.map((chip) => chip.textContent),
  ["CITATION.cff", "old.pdf", "data"]);
assert.ok(chips.every((chip) => chip.tag === "button"));
chips[0].listeners.click();
assert.deepEqual(revealed, ["sources/CITATION.cff"]);
const old = element("div");
fillQuestion(old, "Old?\n\nAttached paths to consider:\n- /Users/me/a.pdf");
const [chip] = old.children[1].children;
assert.equal(chip.tag, "span");
assert.equal(chip.textContent, "a.pdf");
assert.equal(chip.title, "/Users/me/a.pdf");
console.log("attachments passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("attachments passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
