"""Exercise the Copy, Branch from here, and time row under each message.

Run the production ``messageTime`` and ``answerCopyText`` functions in Node,
and check that saved and newly answered turns get the row under both the
question and the answer, shown on hover or keyboard focus. Rendering and the
clipboard are covered only by the Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import os
from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function as _function,
    run_node,
)

_ROOT = Path(__file__).resolve().parents[3]
_APP = _ROOT / "kilin/gui/web/app.js"
_STYLES = _ROOT / "kilin/gui/web/styles.css"


class TestMessageActionsFrontend(unittest.TestCase):
    """Copy a message or see when it was sent."""

    def test_every_answered_turn_gets_the_row(self):
        """Saved and newly answered turns, under question and answer."""

        source = _APP.read_text(encoding="utf-8")
        self.assertIn(
            "addMessageActions(item, turn);", _function(source, "addTurn")
        )
        self.assertIn(
            "addMessageActions(current.item, turn);",
            _function(source, "completePendingTurn"),
        )
        actions = _function(source, "addMessageActions")
        self.assertIn("question.after(row);", actions)
        self.assertIn(
            "answer.after(messageActions(answerCopyText(turn), "
            'turn.timestamp, "answer", ',
            actions,
        )

    def test_every_message_can_be_branched_from(self):
        """Branch from here sits in each row; the composer has none."""

        source = _APP.read_text(encoding="utf-8")
        row = _function(source, "messageActions")
        self.assertIn('branch.title = "Branch from here";', row)
        self.assertIn("row.append(copy, branch);", row)
        actions = _function(source, "addMessageActions")
        self.assertIn("branchFromMessage(item, false, asked)", actions)
        self.assertIn('branchFromMessage(item, true, "")', actions)
        # Busy KíLín disables every row's Branch button.
        self.assertIn(".message-branch", _function(source, "setBusy"))
        page = (_ROOT / "kilin/gui/web/index.html").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("branch-chat", page)
        self.assertNotIn("branch-chat", source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_branch_keeps_turns_up_to_the_message(self):
        """An answer keeps its turn; a question keeps those before it."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const sent = [];
const bridge = {
  branchConversation: (count) => sent.push(count),
  newConversation: () => sent.push("new"),
};
const box = { value: "", focus() {} };
const byId = (id) => {
  assert.equal(id, "question");
  return box;
};
let busy = false;
let branchDraft = null;
let reading = null;
const busyCalls = [];
const setBusy = (value, message) => {
  busy = value;
  busyCalls.push(message);
};
const turn = (classes) => ({ classes });
const first = turn("turn");
const stopped = turn("turn stopped");
const second = turn("turn");
let shown = [first, stopped, second];
const conversation = {
  querySelectorAll(selector) {
    assert.equal(selector, ".turn:not(.pending):not(.stopped)");
    return shown.filter((item) => item !== stopped);
  },
  querySelector(selector) {
    assert.equal(selector, ".turn");
    return shown[0] || null;
  },
};
"""
        script += _function(source, "openEmptyChat")
        script += _function(source, "branchFromMessage")
        script += r"""
// From the second answer: keep both turns.
branchFromMessage(second, true, "");
assert.deepEqual(sent, [2]);
assert.equal(branchDraft, "");
assert.deepEqual(busyCalls, ["Branching this conversation…"]);
// Busy: nothing more is sent.
branchFromMessage(first, true, "");
assert.deepEqual(sent, [2]);
busy = false;
// From the second question: keep the first turn, put the question back.
branchFromMessage(second, false, "Why?");
assert.deepEqual(sent, [2, 1]);
assert.equal(branchDraft, "Why?");
busy = false;
// From the first question: nothing to keep, so an empty chat, which the
// engine reuses if one is unused, with the question to edit.
branchFromMessage(first, false, "Hello");
assert.deepEqual(sent, [2, 1, "new"]);
assert.equal(branchDraft, "Hello");
busy = false;
// A stopped answer was never saved and cannot be branched from.
branchFromMessage(stopped, true, "");
assert.deepEqual(sent, [2, 1, "new"]);
// New chat on a chat with no turns stays there: no new file.
shown = [];
openEmptyChat("Again");
assert.deepEqual(sent, [2, 1, "new"]);
assert.equal(box.value, "Again");
console.log("branch from message passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("branch from message passed", result.stdout)

    def test_copy_uses_qt_in_the_app(self):
        """Qt WebEngine blocks the page's clipboard; the bridge copies."""

        source = _APP.read_text(encoding="utf-8")
        copy = _function(source, "copyToClipboard")
        self.assertLess(
            copy.index("bridge.copyText(text);"),
            copy.index("navigator.clipboard"),
        )

    def test_row_shows_on_hover_and_focus(self):
        """Hidden until the message is hovered or the row has focus."""

        styles = _STYLES.read_text(encoding="utf-8")
        start = styles.index(".message-actions {")
        self.assertIn("opacity: 0", styles[start : styles.index("}", start)])
        for selector in (
            ".user-message:hover + .message-actions",
            ".answer-message:hover + .message-actions",
            ".message-actions:focus-within",
        ):
            self.assertIn(selector, styles)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_time_and_copy_text(self):
        """Short times, and copied answers use the shown citations."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
"""
        script += source[
            source.index("const MESSAGE_MONTHS") : source.index(
                "\n", source.index("const MESSAGE_MONTHS")
            )
            + 1
        ]
        for name in ("messageTime", "splitThinking", "answerCopyText"):
            script += _function(source, name)
        script += r"""
const now = new Date("2026-10-28T16:00:00Z");
assert.equal(messageTime("2026-10-28T14:05:09+00:00", now), "14:05");
assert.equal(messageTime("2026-03-02T09:30:00Z", now), "2 Mar 09:30");
assert.equal(
  messageTime("2025-10-28T23:59:00Z", now), "28 Oct 2025 23:59"
);
assert.equal(messageTime("", now), "");
assert.equal(messageTime("not a time", now), "");
// Copied answers drop thinking and use the numbers the user sees.
const labelsForTurn = () => ({
  labels: new Map([[143, 1], [7, 2], [422, 3]]), cited: 2,
});
assert.equal(
  answerCopyText({
    answer: "<think>plan</think>\nPipelines [143]. Both [143, 7]. "
      + "Year [2024].",
  }),
  "Pipelines [1]. Both [1, 2]. Year [2024].",
);
console.log("message actions passed");
"""
        result = run_node(script, env={**os.environ, "TZ": "UTC"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("message actions passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
