"""Exercise how answer citations are numbered for display.

Run the production ``citationLabels`` and ``referencedSources`` functions in
Node and check that the answer links and the References panel show the same
labels, that the panel lists only the cited sources, and that links still
open the evidence by its own number. Rendering is covered only by the
Qt/WebEngine smoke test and manual checks.

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


class TestCitationLabelsFrontend(unittest.TestCase):
    """Show cited sources as 1, 2, 3… in reading order."""

    def test_answer_and_references_share_the_labels(self):
        """Both show the label; links and previews keep the evidence number."""

        source = _APP.read_text(encoding="utf-8")
        render = _function(source, "renderAnswer")
        self.assertIn(
            "turnCitationLabels.set(turn, "
            "{ labels, cited: new Set(markers.flat()).size });",
            render,
        )
        self.assertIn("link.textContent = String(labels.get(number));", render)
        self.assertIn("selectTurn(item, turn, number);", render)
        select = _function(source, "selectTurn")
        self.assertIn("const { labels, cited } = labelsForTurn(turn);", select)
        self.assertIn(
            "const sources = referencedSources(evidence, labels, cited);",
            select,
        )
        self.assertIn("for (const source of sources) {", select)
        self.assertIn(
            "number.textContent = "
            "String(labels.get(Number(source.citation)));",
            select,
        )
        self.assertIn(
            "button.dataset.citation = String(source.citation);", select
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_labels_follow_reading_order(self):
        """Number citations by first use; list only the cited sources."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
"""
        script += _function(source, "citationLabels")
        script += _function(source, "referencedSources")
        script += r"""
const evidence = [143, 227, 286, 288, 306, 422, 500]
  .map((citation) => ({ citation }));
const labels = citationLabels(
  [[143], [143], [227], [286], [286, 288], [306], [422], [288]], evidence,
);
assert.deepEqual([...labels], [
  [143, 1], [227, 2], [286, 3], [288, 4], [306, 5], [422, 6], [500, 7],
]);
// Nothing cited: the saved evidence order.
assert.deepEqual(
  [...citationLabels([], [{ citation: 9 }, { citation: 4 }])],
  [[9, 1], [4, 2]],
);
// The panel lists only the cited sources, in label order.
const read = [5, 9, 7, 3].map((citation) => ({ citation }));
const cited = citationLabels([[7], [3, 7]], read);
assert.deepEqual(
  referencedSources(read, cited, 2).map((source) => source.citation),
  [7, 3],
);
// An answer citing nothing lists everything it read.
assert.deepEqual(
  referencedSources(read, citationLabels([], read), 0)
    .map((source) => source.citation),
  [5, 9, 7, 3],
);
console.log("citation labels passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("citation labels passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
