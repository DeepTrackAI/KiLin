"""Test that report links never request the network while rendering.

The inline renderer creates link-like buttons without remote ``href``
attributes. Only clicking calls the native bridge to open a browser.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function,
    run_node,
)

_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("node"), "Node.js is not available")
class TestExternalLinksFrontend(unittest.TestCase):
    """Run the actual inline renderer with a minimal DOM adapter."""

    def test_external_links_are_click_only(self) -> None:
        source = (_ROOT / "kilin/gui/web/app.js").read_text(
            encoding="utf-8"
        )
        start = source.index("const MARKDOWN_INLINE = ")
        expression = source[start:source.index("\n", start)]
        function = extract_function(source, "appendInlineMarkdown")
        script = r"""
const assert = require("node:assert/strict");
const opened = [];
const bridge = { openExternalLink: (url) => opened.push(url) };
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.events = {};
  }
  append(...children) { this.children.push(...children); }
  addEventListener(name, callback) { this.events[name] = callback; }
}
const document = { createElement: (tag) => new Element(tag) };
"""
        script += expression + "\n" + function
        script += r"""
const host = new Element("p");
appendInlineMarkdown(host,
  "[Search Google Scholar]"
  + "(https://scholar.google.com/scholar?q=Smith) [3]", null);
assert.equal(opened.length, 0);
const button = host.children.find((child) => child instanceof Element);
assert.equal(button.tag, "button");
assert.equal(button.className, "external-link");
assert.equal(button.href, undefined);
assert.equal(button.textContent, "Search Google Scholar");
let prevented = false;
let stopped = false;
button.events.click({
  preventDefault: () => { prevented = true; },
  stopPropagation: () => { stopped = true; },
});
assert.equal(opened.length, 1);
assert.equal(opened[0], "https://scholar.google.com/scholar?q=Smith");
assert.equal(prevented, true);
assert.equal(stopped, true);
const unsafe = new Element("p");
appendInlineMarkdown(unsafe, "[Bad](javascript:alert(1))", null);
assert.equal(unsafe.children.some((node) => node instanceof Element), false);
console.log("report links require explicit clicks");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("require explicit clicks", result.stdout)


if __name__ == "__main__":
    unittest.main()
