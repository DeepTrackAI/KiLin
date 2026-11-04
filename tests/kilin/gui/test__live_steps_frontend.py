"""Exercise the step list shown while an answer is being made.

Run the production ``timelineSteps`` function in Node and check that the
pending answer lists its steps live with the same row renderer as the saved
"Took …" list. Rendering is covered only by the Qt/WebEngine smoke test and
manual checks.

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
        self.assertIn(
            "timelineSteps(current.timeline || [], current.started + "
            "totalMs)",
            _function(source, "timelineRecord"),
        )

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
  { label: "Starting", at: 0 },
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
