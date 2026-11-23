"""Read other chats while an answer is generated.

Qt reads a saved chat's file without the busy engine; the page shows it
read-only, keeps the running answer in its own chat, and marks that chat
unread when the answer finishes meanwhile. The page logic runs in Node with
stub elements; rendering is covered only by manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function as _function,
    run_node,
)

from kilin.conversation_format import new_conversation_session

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


def _branch(source: str, event: str) -> str:
    """Return the ``onEvent`` branch handling one event."""

    start = source.index(f'data.event === "{event}"')
    return source[start : source.index("data.event ===", start + 1)]


class TestSavedConversationTurns(unittest.TestCase):
    """Qt reads a saved chat from its file, never from the engine."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.read = staticmethod(importlib.import_module(
                "kilin.gui.window"
            ).saved_conversation_turns)
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("PySide6"):
                raise unittest.SkipTest("PySide6 is not installed") from error
            raise

    def _save(self, root: Path, name: str, state: dict) -> None:
        """Write one conversation file in the canonical format."""

        traces = root / "logs" / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": {
            "05_conversations": new_conversation_session(state)
        }}
        (traces / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_turns_of_the_requested_chat_with_their_timing(self):
        """Keep each timeline stored within the matching saved turn."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing = {"elapsed_ms": 1200, "steps": [{
                "phase": "generating", "label": "Generating answer",
                "ms": 1200,
            }]}
            first = {
                "timestamp": "t1", "question": "Q1", "answer": "A1",
                "timeline": timing,
            }
            second = {"timestamp": "t2", "question": "Q2", "answer": "A2"}
            self._save(root, "conversation_1.json", {
                "conversation_id": "other", "turns": [second],
            })
            self._save(root, "conversation_2.json", {
                "conversation_id": "wanted", "turns": [first],
            })
            turns = self.read(root, "wanted")
            self.assertFalse(
                (root / "logs" / "answer_timelines.json").exists()
            )
        self.assertEqual(turns, [first])

    def test_missing_or_unreadable_chats_give_none(self):
        """An unknown ID or a broken file is not shown as an empty chat."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(self.read(root, "wanted"))
            traces = root / "logs" / "traces"
            traces.mkdir(parents=True)
            (traces / "conversation_bad.json").write_text("{", "utf-8")
            self.assertIsNone(self.read(root, "wanted"))
            self._save(root, "conversation_3.json", {
                "conversation_id": "wanted", "turns": "not a list",
            })
            self.assertIsNone(self.read(root, "wanted"))


class TestReadWhileAnsweringFrontend(unittest.TestCase):
    """Show another chat read-only; keep the answer where it belongs."""

    def test_chats_stay_clickable_while_answering(self):
        """Only the chat list stays enabled; delete and projects do not."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        busy = _function(source, "setBusy")
        self.assertIn(
            'querySelectorAll(".conversation-item").forEach((item) => '
            "{ item.disabled = (value && !answering) || !ready; });",
            busy,
        )
        self.assertIn(
            '".conversation-delete, .project-item, .project-settings, '
            '.message-branch"',
            busy,
        )

    def test_reading_never_reaches_the_engine_or_its_chat(self):
        """No engine lookups or branches from the chat being read."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            "if (busy || reading) return;",
            _function(source, "branchFromMessage"),
        )
        reference = _function(source, "openReference")
        self.assertLess(
            reference.index("if (reading) {"),
            reference.index("bridge?.previewReference("),
        )
        for event in ("opened", "conversation_opened"):
            with self.subTest(event=event):
                self.assertIn("reading = null;", _branch(source, event))
        # Stop belongs to the chat being answered: reading shows a
        # disabled Send.
        send = _function(source, "updateSendButton")
        self.assertIn("const stop = answering && !reading;", send)
        self.assertIn(
            'byId("send").disabled = !ready || (busy && !stop)', send
        )
        self.assertIn('Boolean(answerSettingsSave.pending)', send)
        self.assertIn('answerSettingsSave.uncertain', send)

    def test_answer_finished_while_reading_marks_its_chat_unread(self):
        """No scroll or selection in the chat being read; mark unread."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        complete = _function(source, "completePendingTurn")
        self.assertIn(
            "unreadConversations.add(activeConversationId);", complete
        )
        self.assertLess(
            complete.index("if (reading?.live) {\n    renderConversations"),
            complete.index("selectTurn(current.item, turn);"),
        )
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".conversation-item.unread", styles)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_read_another_chat_and_come_back(self):
        """Ask Qt for the chat, show it, then restore the answering one."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const asked = [];
const bridge = { viewConversation: (id) => asked.push(id) };
let answeringConversationId = "a";
let activeConversationId = "a";
let reading = null;
let currentTurnCount = 3;
let lastConversationData = { conversations: [] };
const unreadConversations = new Set(["a"]);
let shown = ["answer in a"];
const conversation = {
  get childNodes() { return shown; },
  replaceChildren(...nodes) { shown = nodes; },
};
const conversationScroll = { scrollTop: 40 };
const addTurn = (turn) => shown.push(turn.question);
const clearReferences = () => {};
const highlightMemoryOrigin = () => {};
const renderConversations = () => {};
const updateChatSummary = (count) => { currentTurnCount = count; };
const updateSendButton = () => {};
"""
        for name in (
            "readConversation", "showReadConversation", "returnToAnswer",
        ):
            script += _function(source, name)
        script += r"""
// The answering chat itself opens as usual.
assert.equal(readConversation({ conversation_id: "a" }), false);
// Another chat is asked for once, from Qt.
assert.equal(readConversation({ conversation_id: "b", name: "B" }), true);
readConversation({ conversation_id: "b", name: "B" });
assert.deepEqual(asked, ["b"]);
showReadConversation([{ question: "Q in b" }]);
assert.deepEqual(shown, ["Q in b"]);
assert.equal(currentTurnCount, 1);
// Clicking the answering chat goes back, and it is no longer unread.
assert.equal(readConversation({ conversation_id: "a" }), true);
assert.deepEqual(shown, ["answer in a"]);
assert.equal(currentTurnCount, 3);
assert.equal(conversationScroll.scrollTop, 40);
assert.equal(reading, null);
assert.equal(unreadConversations.has("a"), false);
// Once the answer is done, other chats open through the engine again.
answeringConversationId = "";
assert.equal(readConversation({ conversation_id: "b", name: "B" }), false);
console.log("reading passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reading passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
