"""Exercise what the chat shows when an answer fails or is stopped.

Run the production ``restoreUnansweredQuestion``, ``paintPendingTurn`` and
``appendAnswerText`` functions in Node with stub page elements: a failed or
stopped answer puts its question back in the message box, Stop keeps moving
until KíLín is ready again, and thinking that arrives finished in one piece
is not labelled "Thought for 0s". Rendering is covered only by the
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


def _branch(source: str, event: str) -> str:
    """Return the ``onEvent`` branch handling one event."""

    start = source.index(f'data.event === "{event}"')
    return source[start : source.index("data.event ===", start + 1)]


def _node(script: str) -> subprocess.CompletedProcess:
    """Run a script in Node and return the finished process."""

    return subprocess.run(
        [shutil.which("node"), "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


class TestAnswerRecoveryFrontend(unittest.TestCase):
    """Keep the question, and say only what is known about thinking."""

    def test_question_is_kept_until_the_answer_arrives(self):
        """Asking remembers it; an answer forgets it."""

        source = _APP.read_text(encoding="utf-8")
        self.assertIn(
            "unansweredQuestion = ask.question;",
            _function(source, "sendQuestion"),
        )
        self.assertIn(
            'unansweredQuestion = "";',
            _function(source, "completeAfterAnswerStages"),
        )

    def test_failure_stop_and_restart_put_the_question_back(self):
        """An error, Stop, and a backend restart each restore it."""

        source = _APP.read_text(encoding="utf-8")
        failed = _function(source, "showError")
        start = failed.index("if (pendingTurn) {")
        self.assertIn(
            "restoreUnansweredQuestion();",
            failed[start : failed.index("}", start)],
        )
        for event in ("stopped", "backend_restarted"):
            with self.subTest(event=event):
                self.assertIn(
                    "restoreUnansweredQuestion();", _branch(source, event)
                )

    def test_stop_keeps_moving_until_kilin_is_ready(self):
        """Stop shows "Stopping…" and spins until the answer has ended."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('byId("send").addEventListener("click"')
        stop = source[start : source.index("});", start)]
        self.assertIn('byId("send").classList.add("waiting");', stop)
        self.assertIn("pendingTurn.stopping = true;", stop)
        # "Stopping" shows once, in the summary line: the "Generating" lines
        # go, and later steps cannot bring them back.
        self.assertIn("resetAnswerStages();", stop)
        self.assertIn(
            'pendingTurn.label.closest(".answer-phase-header")?.remove();',
            stop,
        )
        self.assertIn("pendingTurn.detail.remove();", stop)
        self.assertIn(
            "if (!pendingTurn || pendingTurn.stopping) return;",
            _function(source, "showTurnPhase"),
        )
        busy = _function(source, "setBusy")
        self.assertIn('byId("send").classList.remove("waiting");', busy)
        # "Stopping…" stays until the project is open again; the reopening
        # steps stay hidden, as during the answer.
        stopped = _branch(source, "stopped")
        self.assertNotIn("keepStoppedTurn(", stopped)
        self.assertNotIn('remove("answer-progress")', stopped)
        for event in ("opened", "conversation_opened"):
            with self.subTest(event=event):
                self.assertIn("finishStopping();", _branch(source, event))
        finish = _function(source, "finishStopping")
        self.assertIn("if (!pendingTurn?.stopping) return;", finish)
        self.assertIn("keepStoppedTurn(pendingTurn);", finish)
        styles = (_APP.parent / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".send-button.waiting span", styles)
        self.assertIn("@keyframes send-waiting", styles)

    def test_stop_is_clean_and_finishes_without_a_restart(self):
        """Stop sends cancel_answer; answer_cancelled ends it in place."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('byId("send").addEventListener("click"')
        stop = source[start : source.index("});", start)]
        self.assertIn("bridge.cancelAnswer();", stop)
        self.assertNotIn("bridge.stopOperation();", stop)
        cancelled = _branch(source, "answer_cancelled")
        for step in (
            "resumeAfterStop = false;",
            "finishStopping();",
            "stoppedTurn = null;",
            "restoreUnansweredQuestion();",
            "setBusy(false);",
        ):
            with self.subTest(step=step):
                self.assertIn(step, cancelled)
        self.assertNotIn("resumeProjectConversation", cancelled)

    def test_a_stopped_answer_keeps_its_steps(self):
        """The steps stay, ending with the Stop, like a "Took …" line."""

        source = _APP.read_text(encoding="utf-8")
        kept = _function(source, "keepStoppedTurn")
        self.assertIn("renderTimeline(record,", kept)
        self.assertIn('stop.textContent = "Stopped at your request";', kept)
        self.assertIn("`Stopped after ${formatDuration(elapsed)}`", kept)
        self.assertNotIn("not saved", kept)

    def test_a_stopped_first_answer_is_not_replaced_by_the_welcome(self):
        """Reopening shows the stopped answer, not the empty-chat screen."""

        source = _APP.read_text(encoding="utf-8")
        for event in ("opened", "conversation_opened"):
            with self.subTest(event=event):
                branch = _branch(source, event)
                self.assertIn(
                    "if (!conversation.children.length) "
                    "renderConversationEmpty();",
                    branch,
                )
                self.assertNotIn(
                    "if (!data.turns.length) renderConversationEmpty();",
                    branch,
                )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_stopping_label_cycles_its_dots(self):
        """"Stopping" replaces the timer and keeps animating."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
let clock = 0;
const performance = { now: () => clock };
const usageSum = () => 0;
let pendingTurn = {
  phase: "Stopping", stopping: true, started: 0, elapsedMs: 5000,
  label: { textContent: "" }, timer: { textContent: "" },
};
"""
        script += _function(source, "paintPendingTurn")
        script += r"""
const seen = new Set();
for (clock = 0; clock < 1600; clock += 400) {
  paintPendingTurn();
  seen.add(pendingTurn.timer.textContent);
}
assert.deepEqual(
  [...seen], ["Stopping", "Stopping.", "Stopping..", "Stopping..."]
);
console.log("stopping passed");
"""
        result = _node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stopping passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_restore_never_replaces_a_new_question(self):
        """Fill an empty box once; leave text the user typed."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const box = { value: "" };
const byId = (id) => {
  assert.equal(id, "question");
  return box;
};
let unansweredQuestion = "Why?";
"""
        script += _function(source, "restoreUnansweredQuestion")
        script += r"""
restoreUnansweredQuestion();
assert.equal(box.value, "Why?");
// Restored once: a second failure does not bring it back again.
box.value = "";
restoreUnansweredQuestion();
assert.equal(box.value, "");
// A question typed while the answer ran is kept.
unansweredQuestion = "Old";
box.value = "New question";
restoreUnansweredQuestion();
assert.equal(box.value, "New question");
console.log("restore passed");
"""
        result = _node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("restore passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_thinking_received_whole_shows_no_time(self):
        """Only thinking the page saw happen gets "Thought for"."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
let clock = 0;
const performance = { now: () => clock };
function element() {
  return {
    textContent: "", open: false, className: "", scrollTop: 0,
    scrollHeight: 0, classList: { add() {}, remove() {} },
    children: [], append(...items) { this.children.push(...items); },
    prepend(item) { this.children.unshift(item); },
    replaceChildren() { this.children = []; },
  };
}
const document = { createElement: () => element() };
function thoughtsBlock(text, label, open) {
  const block = element();
  const summary = element();
  summary.textContent = label;
  const body = element();
  block.open = open;
  block.querySelector = (selector) => (
    selector === "summary" ? summary : body
  );
  return block;
}
const resetAnswerStages = () => {};
const markTimelineStep = () => {};
const paintPendingTurn = () => {};
const renderMarkdown = () => {};
const conversationScroll = {
  scrollHeight: 0, scrollTop: 0, clientHeight: 0,
};
let pendingTurn = null;
"""
        for name in ("formatDuration", "splitThinking", "appendAnswerText"):
            script += _function(source, name)
        script += r"""
const summary = () => pendingTurn.thoughts.querySelector("summary");
// A tool's model call: the whole thinking block arrives at once.
pendingTurn = { answer: element() };
appendAnswerText("<think>Model call 1:\nPlan</think>\n\n");
assert.equal(summary().textContent, "Show thinking");
assert.equal(pendingTurn.thinkingMs, undefined);
appendAnswerText("The answer.");
assert.equal(summary().textContent, "Show thinking");
// Streamed thinking: the page sees it start and end.
pendingTurn = { answer: element() };
appendAnswerText("<think>Let me");
clock = 12000;
appendAnswerText(" check.</think>The answer.");
assert.equal(summary().textContent, "Thought for 12s");
assert.equal(pendingTurn.thinkingMs, 12000);
console.log("thinking passed");
"""
        result = _node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("thinking passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
