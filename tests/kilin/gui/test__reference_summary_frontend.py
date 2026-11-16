"""Check the saved turn summary in the References panel.

Exercise the production selection handler using a small Node DOM adapter.
The full stored router summary and citation statistics are presentation
only; source ordering, citation numbers, and saved turns remain unchanged.

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


class TestReferenceSummaryFrontend(unittest.TestCase):
    """Display summaries separately from the existing evidence count."""

    def test_header_uses_saved_representation(self):
        """Use the persisted summary and the one-based turn number."""

        source = _APP.read_text(encoding="utf-8")
        select = _function(source, "selectTurn")
        self.assertIn("turn.parameters?.representation?.summary", select)
        self.assertIn("Number(item.dataset.turnIndex) + 1", select)
        self.assertIn('line.className = "reference-statistics";', select)
        self.assertNotIn("Sources for:", select)
        styles = _STYLES.read_text(encoding="utf-8")
        self.assertIn(
            ".references-panel .reference-statistics { display: block;",
            styles,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_summary_and_statistics_for_saved_and_new_turns(self):
        """Keep summaries complete, counts separate, and fallbacks safe."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this._text = "";
    this.className = "";
    this.dataset = {};
    this.events = {};
    this.classes = new Set();
    this.classList = {
      add: (name) => this.classes.add(name),
      remove: (name) => this.classes.delete(name),
    };
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  get textContent() {
    return this._text + this.children.map((child) =>
      child.textContent).join("");
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this._text = "";
    this.children = [...children];
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelector() { return null; }
}
const elements = {
  "references-for": new Element("p"),
  "references-list": new Element("div"),
};
const byId = (name) => elements[name];
const document = { createElement: (tag) => new Element(tag) };
let selectedTurn = null;
let previewsCleared = 0;
const clearSourcePreview = () => { previewsCleared += 1; };
const openReference = () => { throw Error("Preview opened unexpectedly"); };
"""
        start = source.index("const ATTACHED_LIST =")
        script += source[start : source.index("\n", start) + 1]
        for name in (
            "splitAttached",
            "citationLabels",
            "referencedSources",
            "labelsForTurn",
            "selectTurn",
            "clearReferences",
        ):
            if name == "labelsForTurn":
                script += "const turnCitationLabels = new WeakMap();\n"
            script += _function(source, name)
        script += r"""
const heading = elements["references-for"];
const list = elements["references-list"];
const evidence = Array.from({length: 8}, (_, index) => ({
  citation: index + 11, path: "paper.md", start_line: 2, end_line: 5,
}));
const completeSummary = "S".repeat(200);
const saved = {
  question: "A long original question: " + "details ".repeat(100),
  evidence,
  parameters: { representation: { turn_id: 5, summary: completeSummary } },
};
turnCitationLabels.set(saved, {
  labels: citationLabels([[11], [13], [16]], evidence), cited: 3,
});
const firstItem = new Element("article");
firstItem.dataset.turnIndex = "4";
selectTurn(firstItem, saved);
assert.equal(heading._text, `Turn 5 — ${completeSummary}`);
assert.equal(heading.children.length, 1);
assert.equal(heading.children[0].className, "reference-statistics");
assert.equal(heading.children[0].textContent, "3 cited of 8 passages read");
assert.deepEqual(
  list.children.map((entry) => entry.children[0].dataset.citation),
  ["11", "13", "16"],
);
assert.deepEqual(list.children.map((entry) =>
  entry.children[0].children[0].textContent), ["1", "2", "3"]);
assert.ok(firstItem.classes.has("selected"));

// Selecting another answer replaces the previous summary and count.
const recent = {
  question: "What changed?",
  evidence: [{citation: 9, path: "code.py", start_line: 1, end_line: 2}],
  parameters: {
    representation: { turn_id: 2, summary: "Updated the tests." },
  },
};
turnCitationLabels.set(recent, {
  labels: citationLabels([[9]], recent.evidence), cited: 1,
});
const secondItem = new Element("article");
secondItem.dataset.turnIndex = "1";
selectTurn(secondItem, recent);
assert.equal(heading.textContent, "Turn 2 — Updated the tests.");
assert.equal(heading.children.length, 0);
assert.ok(!firstItem.classes.has("selected"));
assert.ok(secondItem.classes.has("selected"));
assert.equal(list.children.length, 1);

// Existing turns without a representation use only a short question excerpt.
const longQuestion = "What is known? " + "Context ".repeat(60);
const old = {
  question: longQuestion + "\n\nAttached files:\n- sources/data.pdf",
  evidence: [],
  parameters: {},
};
const thirdItem = new Element("article");
thirdItem.dataset.turnIndex = "0";
selectTurn(thirdItem, old);
assert.ok(heading._text.startsWith("Turn 1 — What is known?"));
assert.ok(heading._text.length <= "Turn 1 — ".length + 201);
assert.ok(heading._text.endsWith("…"));
assert.ok(!heading._text.includes("sources/data.pdf"));
assert.equal(heading.children[0].textContent,
  "No sources saved for this answer.");
assert.equal(list.children.length, 0);

// Invalid summaries also fall back to a question, never object coercion.
selectTurn(thirdItem, {
  question: "Fallback question?", evidence: [],
  parameters: { representation: { summary: { bad: true } } },
});
assert.equal(heading._text, "Turn 1 — Fallback question?");
clearReferences();
assert.equal(heading.textContent, "Select an answer to see its sources.");
assert.equal(heading.children.length, 0);
assert.equal(list.children.length, 0);
assert.equal(selectedTurn, null);
assert.equal(previewsCleared, 5);
console.log("reference summary display passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reference summary display passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
