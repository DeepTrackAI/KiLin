"""Exercise how answers show LaTeX maths.

Run the production Markdown renderer in Node with the bundled KaTeX and a
minimal stub DOM: inline and display maths become MathML, prices and code
stay text, and LaTeX that KaTeX cannot read keeps its original text.
Drawing the MathML is covered only by the Qt/WebEngine smoke test and
manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_WEB = _ROOT / "kilin/gui/web"


def _function(source: str, name: str) -> str:
    """Return one top-level function's source text."""

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


def _constant(source: str, name: str) -> str:
    """Return one top-level constant's source line."""

    start = source.index(f"const {name} =")
    return source[start : source.index("\n", start) + 1]


class TestMathFrontend(unittest.TestCase):
    """Show LaTeX maths in answers as maths."""

    def test_katex_ships_with_the_gui(self):
        """Bundled, loaded before app.js, licensed, and packaged."""

        vendor = _WEB / "vendor"
        self.assertTrue((vendor / "katex.min.js").is_file())
        self.assertIn(
            "MIT License",
            (vendor / "KATEX-LICENSE").read_text(encoding="utf-8"),
        )
        page = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertLess(
            page.index('<script src="vendor/katex.min.js" defer>'),
            page.index('<script src="app.js" defer>'),
        )
        project = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for pattern in (
            '"web/vendor/*.js",',
            '"web/vendor/KATEX-LICENSE",',
            '"web/vendor/README.md",',
        ):
            self.assertIn(pattern, project)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_maths_render_as_mathml(self):
        """Inline and display maths; prices, code, and errors stay text."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const bundled = require(%r);
// KaTeX writes into a real element in the app; here it records the markup.
const katex = {
  render(tex, node, options) {
    node.options = options;
    node.html = bundled.renderToString(tex, options);
  },
};
function element(tag) {
  return {
    tag, children: [], style: {}, className: "", html: "",
    append(...items) { this.children.push(...items); },
    set textContent(value) { this.children = [String(value)]; },
    get textContent() {
      return this.children.map((child) => (
        typeof child === "string" ? child : child.textContent)).join("");
    },
  };
}
// var, not const: KaTeX checks document when it loads, before this line.
var document = { createElement: element };
const nodes = (host) => host.children.filter(
  (child) => typeof child !== "string");
const render = (text) => {
  const host = element("div");
  renderMarkdown(host, text);
  return host;
};
const inline = (text) => nodes(render(text).children[0]);
""" % str(_WEB / "vendor/katex.min.js")
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
            "mathNode",
            "renderMarkdown",
            "appendInlineMarkdown",
        ):
            script += _function(source, name)
        script += r"""
// Inline maths, both delimiters; underscores inside are not italics.
let [math] = inline("Energy $E = mc^2$ here.");
assert.equal(math.className, "answer-math");
assert.equal(math.options.output, "mathml");
assert.equal(math.options.displayMode, false);
assert.match(math.html, /<math/);
assert.match(math.html, /<msup>/);
let found = inline("Terms \\(a_1 + b_1\\) and $x_i$.");
assert.deepEqual(found.map((node) => node.className),
  ["answer-math", "answer-math"]);
assert.match(found[0].html, /<msub>/);
// Single-character maths, and maths beside a citation.
found = inline("Let $x$ be fixed [1].");
assert.equal(found.length, 1);
// Prices and escaped dollars stay text; so does code.
for (const text of [
  "It costs $5 and $10.", "From $5-$10 each.", "Cost: 5$ or $ 7 $.",
  "Literal \\$x\\$ here.",
]) {
  assert.deepEqual(inline(text), [], text);
}
[math] = inline("Run `echo $HOME$` now.");
assert.equal(math.tag, "code");

// Display maths: one line, several lines, and \[ … \].
let host = render("Before:\n$$\n\\frac{a}{b}\n$$\nAfter.");
assert.deepEqual(host.children.map((child) => child.tag),
  ["p", "div", "p"]);
const block = host.children[1];
assert.equal(block.className, "answer-math display");
assert.equal(block.options.displayMode, true);
assert.match(block.html, /<mfrac>/);
host = render("\\[ x^2 + y^2 = z^2 \\]");
assert.equal(host.children[0].className, "answer-math display");
host = render("$$ \\sum_{i=1}^n i $$");
assert.match(host.children[0].html, /<munderover>|<msubsup>/);

// LaTeX KaTeX cannot read keeps its original text.
[math] = inline("Broken $\\frac{a$ here.");
assert.equal(math.html, "");
assert.equal(math.textContent, "$\\frac{a$");
// An unclosed $$ (or one still streaming) stays text and is not lost.
host = render("$$ x^2\nMore text.");
assert.deepEqual(host.children.map((child) => child.textContent),
  ["$$ x^2", "More text."]);
// Links are refused: KaTeX runs without trust and makes no href.
[math] = inline("Link $\\href{javascript:alert(1)}{x}$.");
assert.match(math.html, /<math/);
assert.doesNotMatch(math.html, /href=/);
console.log("maths passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("maths passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
