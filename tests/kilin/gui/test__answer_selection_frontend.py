"""Verify that selecting answer text does not open the side panels.

Exercise the production answer-selection and citation handlers in Node.
Saved and newly completed answers must share their selection behavior;
normal text interaction must not override the user's panel choice.

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
_STYLES = _ROOT / "kilin/gui/web/styles.css"


class TestAnswerSelectionFrontend(unittest.TestCase):
    """Keep text selection separate from explicit reference navigation."""

    def test_saved_and_completed_answers_share_one_handler(self):
        """Both render paths use the same selection policy."""

        source = _APP.read_text(encoding="utf-8")
        saved = _function(source, "addTurn")
        completed = _function(source, "completePendingTurn")
        self.assertIn("bindAnswerSelection(answer, item, turn);", saved)
        self.assertIn(
            "bindAnswerSelection(current.answer, current.item, turn);",
            completed,
        )
        self.assertNotIn('answer.addEventListener("click"', saved)
        self.assertNotIn("chooseSidePanels(true)", saved)
        self.assertNotIn("chooseSidePanels(true)", completed)
        styles = _STYLES.read_text(encoding="utf-8")
        self.assertIn(".answer-message {", styles)
        self.assertIn("white-space: pre-wrap; cursor: text;", styles)
        self.assertIn("font: inherit; line-height: normal;", styles)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_plain_clicks_and_text_selection_do_not_open_panels(self):
        """Only a deliberate keyboard activation may open hidden panels."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const shown = [];
const selected = [];
let open = false;
let hasTextSelection = false;
const shell = {
  classList: { contains: (name) => {
    assert.equal(name, "sidebar-open");
    return open;
  } },
};
const window = {
  getSelection: () => ({ isCollapsed: !hasTextSelection }),
};
const chooseSidePanels = (visible) => {
  shown.push(visible);
  open = visible;
};
const selectTurn = (item, turn) => selected.push([item, turn]);
const answer = {
  events: {},
  setAttribute(name, value) { this[name] = value; },
  addEventListener(name, callback) { this.events[name] = callback; },
};
const item = { id: "answer-article" };
const turn = { answer: "Example response" };
const textTarget = { closest: () => null };
const nestedControl = { closest: () => ({ tagName: "BUTTON" }) };
"""
        script += _function(source, "bindAnswerSelection")
        script += r"""
bindAnswerSelection(answer, item, turn);
assert.equal(answer.tabIndex, 0);
assert.equal(answer["aria-label"],
  "Select this answer to view its references");
// A plain click in a hidden answer does not open the panels.
answer.events.click({ target: textTarget });
assert.deepEqual(shown, []);
assert.deepEqual(selected, []);
// When panels are already open, a click can inspect this answer.
open = true;
answer.events.click({ target: textTarget });
assert.deepEqual(selected, [[item, turn]]);
assert.deepEqual(shown, []);
// Dragging to highlight text does not change the selected answer.
hasTextSelection = true;
answer.events.click({ target: textTarget });
assert.equal(selected.length, 1);
assert.deepEqual(shown, []);
hasTextSelection = false;
// Buttons, external links and details headings are not answer clicks.
answer.events.click({ target: nestedControl });
assert.equal(selected.length, 1);
assert.deepEqual(shown, []);
// A text selection with the panels closed cannot reopen them.
open = false;
hasTextSelection = true;
answer.events.click({ target: textTarget });
assert.deepEqual(shown, []);
// Explicit keyboard activation of the focused answer still works.
hasTextSelection = false;
let prevented = 0;
const key = (name, target = answer) => {
  answer.events.keydown({ key: name, target,
    preventDefault: () => { prevented += 1; } });
};
key("Enter", nestedControl);
key("ArrowDown");
assert.equal(prevented, 0);
assert.deepEqual(shown, []);
key("Enter");
assert.deepEqual(shown, [true]);
assert.equal(prevented, 1);
assert.equal(selected.length, 2);
assert.deepEqual(selected[1], [item, turn]);
key(" ");
assert.deepEqual(shown, [true, true]);
assert.equal(prevented, 2);
console.log("answer selection preserves panel choice");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preserves panel choice", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_citation_click_still_opens_panels(self):
        """A linked numeric citation explicitly opens its source preview."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const shown = [];
const selected = [];
const chooseSidePanels = (visible) => shown.push(visible);
const selectTurn = (...args) => selected.push(args);
const splitThinking = (answer) => ({ answer, thinking: null });
const citationLabels = () => new Map([[4, 1]]);
const appendMemoryProvenance = () => {};
const turnCitationLabels = new WeakMap();
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.events = {}; }
  append(...children) { this.children.push(...children); }
  addEventListener(name, callback) { this.events[name] = callback; }
  setAttribute(name, value) { this[name] = value; }
}
const document = {
  createElement: (tag) => new Element(tag),
  createTextNode: (value) => value,
};
// Feed a real citation marker to the production renderAnswer callback.
const renderMarkdown = (host, text, linkCitation) => {
  const link = linkCitation("[4]");
  if (link) host.append(link);
};
const answer = new Element("div");
const item = { dataset: { turnIndex: "0" } };
const turn = { answer: "A claim [4]", evidence: [{ citation: 4 }] };
"""
        script += _function(source, "renderAnswer")
        script += r"""
renderAnswer(answer, turn, item);
assert.deepEqual(shown, []);
assert.deepEqual(selected, []);
const body = answer.children[0];
const superscript = body.children[0];
const button = superscript.children[0];
assert.equal(button.tag, "button");
assert.equal(button["aria-label"], "Show reference 1");
let stopped = false;
button.events.click({ stopPropagation: () => { stopped = true; } });
assert.equal(stopped, true);
assert.deepEqual(shown, [true]);
assert.deepEqual(selected, [[item, turn, 4]]);
console.log("citation opens its reference");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("citation opens its reference", result.stdout)


if __name__ == "__main__":
    unittest.main()
