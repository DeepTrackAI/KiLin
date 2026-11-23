"""Exercise the expanding chat editor in the desktop frontend.

Run the production resizing and draft helpers with a simulated textarea in
Node.js. Check the browser-owned height cap and keyboard contract in the
packaged HTML/CSS; native Qt/WebEngine rendering is verified separately.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import re
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestComposerAutogrowFrontend(unittest.TestCase):
    """Keep longer drafts visible without growing beyond the chat layout."""

    def test_editor_keeps_its_minimum_and_maximum_height(self):
        """The two-line input grows up to 250 px, then scrolls."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        start = styles.index(".composer textarea {")
        rule = styles[start : styles.index("}", start)]

        self.assertIn('id="question" rows="2"', page)
        for declaration in (
            "min-height: 54px", "max-height: 250px",
            "overflow-y: auto", "resize: none",
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(declaration, rule)

    def test_all_programmatic_drafts_use_the_resizing_helper(self):
        """Restored and cleared text gets the same sizing as typed text."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        # The setter alone assigns the editor value programmatically.
        self.assertEqual(
            re.findall(r'byId\("question"\)\.value\s*=', source),
            ['byId("question").value ='],
        )
        self.assertNotIn("question.value =", source)
        for call in (
            "setQuestionDraft(question);",
            "setQuestionDraft(draft);",
            "setQuestionDraft(text);",
            "setQuestionDraft(branchDraft);",
            "setQuestionDraft(attachedAsk.question);",
            'setQuestionDraft("");',
            "setQuestionDraft(ask.question);",
        ):
            with self.subTest(call=call):
                self.assertIn(call, source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_height_follows_text_and_available_width(self):
        """Grow, shrink, and reflow without observing height recursively."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        start = source.index('const questionInput = byId("question");')
        end = source.index("conversationScroll.addEventListener", start)
        script = r"""
const assert = require("node:assert/strict");
const heights = [];
const style = {
  set height(value) { heights.push(value); this.current = value; },
  get height() { return this.current; },
};
const box = {
  value: "", scrollHeight: 54, offsetWidth: 360, style, listeners: {},
  addEventListener(name, callback) { this.listeners[name] = callback; },
};
let submitted = 0;
const form = { requestSubmit() { submitted++; } };
const byId = (id) => id === "question" ? box : form;
let notifyResize;
class ResizeObserver {
  constructor(callback) { notifyResize = callback; }
  observe(element) { assert.equal(element, box); }
}
"""
        for name in ("resizeQuestionInput", "setQuestionDraft"):
            script += extract_function(source, name)
        script += source[start:end]
        script += r"""
assert.deepEqual(heights.slice(-2), ["auto", "54px"]);
assert.equal(box.listeners.input, resizeQuestionInput);

// Typing and pasting both trigger the same native input event.
box.value = "Two lines";
box.scrollHeight = 86;
box.listeners.input();
assert.deepEqual(heights.slice(-2), ["auto", "86px"]);

// A restored question also grows without waiting for a new keystroke.
box.scrollHeight = 135;
setQuestionDraft("A restored multiline question");
assert.equal(box.value, "A restored multiline question");
assert.deepEqual(heights.slice(-2), ["auto", "135px"]);

// Deleting text and sending contract the editor immediately.
box.scrollHeight = 54;
box.listeners.input();
assert.deepEqual(heights.slice(-2), ["auto", "54px"]);
setQuestionDraft("");
assert.equal(box.value, "");
assert.equal(style.height, "54px");

// A narrowed center panel changes wrapping even without a window resize.
box.offsetWidth = 200;
box.scrollHeight = 148;
notifyResize();
assert.deepEqual(heights.slice(-2), ["auto", "148px"]);
const previous = heights.length;
notifyResize();
assert.equal(heights.length, previous);
box.offsetWidth = 360;
box.scrollHeight = 54;
notifyResize();
assert.equal(style.height, "54px");

// Shift+Enter remains a newline; bare Enter submits the form.
let prevented = 0;
const press = (shiftKey) => box.listeners.keydown({
  key: "Enter", shiftKey, preventDefault() { prevented++; },
});
press(true);
assert.equal(prevented, 0);
assert.equal(submitted, 0);
press(false);
assert.equal(prevented, 1);
assert.equal(submitted, 1);
console.log("chat composer autoresize passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("chat composer autoresize passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
