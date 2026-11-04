"""Exercise how answers show Markdown tables.

Run the production Markdown renderer in Node with a minimal stub DOM: a
header row followed by a divider row becomes a table, with column
alignment, inline formatting and citations in cells. Text that only looks
like a table stays a paragraph. Rendering is covered only by the
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

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


def _constant(source: str, name: str) -> str:
    """Return one top-level constant's source line."""

    start = source.index(f"const {name} =")
    return source[start : source.index("\n", start) + 1]


class TestMarkdownTablesFrontend(unittest.TestCase):
    """Show Markdown tables in answers as tables."""

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_tables_render_as_tables(self):
        """Header, alignment, cells, citations; non-tables stay text."""

        source = _APP.read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
// A minimal DOM: elements with tag names, children and text.
function element(tag) {
  return {
    tag, children: [], style: {}, className: "",
    append(...items) { this.children.push(...items); },
    set textContent(value) { this.children = [String(value)]; },
    get textContent() {
      return this.children.map((child) => (
        typeof child === "string" ? child : child.textContent)).join("");
    },
  };
}
const document = { createElement: element };
const find = (node, tag) => node.children.filter(
  (child) => typeof child !== "string" && child.tag === tag);
"""
        for name in (
            "MARKDOWN_FENCE",
            "MARKDOWN_MATH",
            "MARKDOWN_HEADING",
            "MARKDOWN_RULE",
            "MARKDOWN_QUOTE",
            "MARKDOWN_BULLET",
            "MARKDOWN_NUMBER",
            "MARKDOWN_TABLE_DIVIDER",
            "MARKDOWN_INLINE",
        ):
            script += _constant(source, name)
        for name in (
            "markdownTableCells",
            "isMarkdownTableStart",
            "isMarkdownBlockStart",
            "renderMarkdown",
            "appendInlineMarkdown",
        ):
            script += _function(source, name)
        script += r"""
const host = element("div");
renderMarkdown(host, [
  "Compared models:",
  "| Model | Size | Notes |",
  "|:------|-----:|:-----:|",
  "| **Qwen3** | 8B | fast [1] |",
  "| `a\\|b` | 14B |",
  "",
  "After the table.",
].join("\n"), (marker) => {
  const link = element("a");
  link.textContent = marker;
  return link;
});
// A table may follow a paragraph line directly.
assert.deepEqual(host.children.map((child) => child.tag),
  ["p", "div", "p"]);
assert.equal(host.children[0].textContent, "Compared models:");
const frame = host.children[1];
assert.equal(frame.className, "answer-table");
const [table] = find(frame, "table");
const [head] = find(table, "thead");
const [body] = find(table, "tbody");
const headers = find(find(head, "tr")[0], "th");
assert.deepEqual(headers.map((cell) => cell.textContent),
  ["Model", "Size", "Notes"]);
assert.deepEqual(headers.map((cell) => cell.style.textAlign),
  ["left", "right", "center"]);
const rows = find(body, "tr").map((row) => find(row, "td"));
assert.equal(rows.length, 2);
assert.deepEqual(rows[0].map((cell) => cell.textContent),
  ["Qwen3", "8B", "fast[1]"]);
assert.equal(find(rows[0][0], "strong").length, 1);
assert.equal(find(rows[0][2], "a").length, 1);
// An escaped pipe stays in its cell; a short row gets an empty cell.
assert.deepEqual(rows[1].map((cell) => cell.textContent),
  ["a|b", "14B", ""]);
assert.equal(find(rows[1][0], "code").length, 1);

// Pipes without a matching divider row are ordinary text.
const plain = element("div");
renderMarkdown(plain, "a | b\n| c | d |\n---");
assert.deepEqual(plain.children.map((child) => child.tag), ["p", "hr"]);
const uneven = element("div");
renderMarkdown(uneven, "| a | b |\n|---|");
assert.deepEqual(uneven.children.map((child) => child.tag), ["p"]);
console.log("markdown tables passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("markdown tables passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
