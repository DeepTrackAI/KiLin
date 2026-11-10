"""Exercise the step list shown while an answer is being made.

Run the production ``timelineSteps`` function in Node to verify the
live, display-only activity list. Completed turns must render the core's
persisted timing record rather than saving another browser-side timeline.

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


class TestLiveStepsFrontend(unittest.TestCase):
    """Follow the route of an answer while it is made."""

    def test_pending_answer_lists_steps_live(self):
        """A folded list from the start, refreshed by the ticker."""

        source = _APP.read_text(encoding="utf-8")
        pending = _function(source, "addPendingTurn")
        self.assertIn('document.createElement("details")', pending)
        self.assertNotIn("durationLine.open = true;", pending)
        self.assertIn(
            "item.append(questionNode, durationLine, answer);", pending
        )
        paint = _function(source, "paintPendingTurn")
        self.assertIn(
            "appendTimelineRows(list, timelineSteps(pendingTurn.timeline, "
            "performance.now()), true);",
            paint,
        )
        # Streaming no longer creates a second time line.
        self.assertNotIn(
            "current.durationLine = document.createElement",
            _function(source, "appendAnswerText"),
        )

    def test_live_and_saved_lists_share_one_renderer(self):
        """The saved "Took …" list uses the same rows."""

        source = _APP.read_text(encoding="utf-8")
        self.assertIn(
            "appendTimelineRows(list, record?.steps || []);",
            _function(source, "renderTimeline"),
        )
        complete = _function(source, "completePendingTurn")
        self.assertIn(
            "renderTimeline(turn.timeline || null, turn);", complete,
        )
        self.assertNotIn("timelineRecord", source)
        self.assertNotIn("saveAnswerTimeline", source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_completed_rows_show_core_measured_durations(self):
        """Show persisted labels and durations, not browser estimates."""
        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const document = { createElement: type => ({
  type, children: [], textContent: "", className: "",
  append(...nodes) { this.children.push(...nodes); },
}) };
const usageSum = () => 0;
const formatDuration = ms => `${Math.round(ms / 1000)}s`;
const formatTokens = n => String(n);
"""
        script += _function(source, "appendTimelineRows")
        script += _function(source, "renderTimeline")
        script += r"""
const record = {elapsed_ms: 4200, steps: [
  {phase: 'routing', label: 'Identifying intent', ms: 1200},
  {phase: 'generating', label: 'Generating answer', ms: 3000},
]};
const line = renderTimeline(record, {parameters: {}});
assert.equal(line.children[0].textContent, 'Took 4s');
assert.deepEqual(line.children[1].children.map(row => [
  row.children[0].textContent, row.children[1].textContent,
]), [
  ['Identifying intent', '1.2s'],
  ['Generating answer', '3.0s'],
]);
console.log('completed core timing passed');
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("completed core timing passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_token_row_shows_saved_model_settings(self):
        """Use the completed turn settings, not current GUI selections."""
        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const document = { createElement: type => ({
  type, children: [], textContent: "", className: "", title: "",
  append(...nodes) { this.children.push(...nodes); },
}) };
const usageSum = usage => (usage?.input_tokens || 0)
  + (usage?.output_tokens || 0);
const formatDuration = ms => `${Math.round(ms / 1000)}s`;
const formatTokens = n => String(n);
"""
        script += _function(source, "appendTimelineRows")
        script += _function(source, "renderTimeline")
        script += r"""
const usage = { input_tokens: 5127, output_tokens: 545 };
const record = { elapsed_ms: 36000, steps: [] };
const savedTurn = { parameters: {
  usage, chat_model: "Qwen/Qwen3-8B-MLX-4bit",
  thinking: true, max_new_tokens: 8192,
} };
const labels = turn => renderTimeline(record, turn)
  .children[1].children.map(row => row.children[0].textContent);
assert.deepEqual(labels(savedTurn), [
  "Tokens: 5,127 read, 545 written",
  "Model: Qwen/Qwen3-8B-MLX-4bit (thinking on, 8,192 max tokens)",
]);
const line = renderTimeline(record, savedTurn);
assert.equal(line.children[0].textContent, "Took 36s · 5672 tokens");
assert.equal(line.children[1].children.length, 2);
assert.deepEqual(labels({ parameters: {
  usage, chat_model: "other-model", thinking: false,
  max_new_tokens: 1024,
} }), [
  "Tokens: 5,127 read, 545 written",
  "Model: other-model (thinking off, 1,024 max tokens)",
]);
assert.deepEqual(labels({ parameters: { usage } }),
  ["Tokens: 5,127 read, 545 written"]);
assert.deepEqual(labels({ parameters: {
  usage, chat_model: "old-model",
} }), [
  "Tokens: 5,127 read, 545 written",
  "Model: old-model",
]);
assert.equal(renderTimeline(null, { parameters: {
  chat_model: "old-model", thinking: true,
} }), null);
console.log("saved model settings passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saved model settings passed", result.stdout)

    def test_no_frontend_activity_inference(self) -> None:
        """Only core activity messages may append to the live timeline."""
        source = _APP.read_text(encoding="utf-8")
        self.assertIn(
            'data.event === "answer_activity"', source,
        )
        handler = _function(source, "markTimelineStep")
        self.assertIn("activity.index", handler)
        self.assertIn("label: activity.label", handler)
        self.assertNotIn("intentName", source)
        self.assertNotIn("describeHistory", source)
        self.assertNotIn("markTimelineStep(detail)", source)
        self.assertNotIn('markTimelineStep("Waiting', source)
        self.assertNotIn('markTimelineStep("Generating', source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_step_durations(self):
        """Each step lasts until the next; the last until the end."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
"""
        script += _function(source, "timelineSteps")
        script += r"""
const timeline = [
  { label: "Identifying intent", at: 100 },
  { label: "Reading the whole paper (part 1 of 4)", at: 3700 },
];
assert.deepEqual(timelineSteps(timeline, 33100), [
  { label: "Identifying intent", ms: 3600 },
  { label: "Reading the whole paper (part 1 of 4)", ms: 29400 },
]);
assert.deepEqual(timelineSteps([], 10), []);
console.log("live steps passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("live steps passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
