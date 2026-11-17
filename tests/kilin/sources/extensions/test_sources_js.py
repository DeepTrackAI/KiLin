"""Unit tests for JavaScript declaration boundaries.

JavaScript is never executed and no JavaScript runtime is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.sources.extensions import sources_js
from kilin.sources.sources import SourceFile

SOURCE = """import { api } from "./api.js";

/**
 * Add two numbers.
 */
export function add(a, b) {
  return a + b;
}
init();
const twice = (n) => n * 2;
export const load = async (
  name,
) => {
  return api(name);
};
// Keeps the projects.
class Store extends Base {
  #items = [];
  constructor() {
    super();
  }
  async save(name) {
    if (name) {
      return add(1, 2);
    }
  }
}
const LIMIT = 3;
"""


def _units(text: str = SOURCE):
    return sources_js.extract_source_units(SourceFile(Path("a.js"), text))


class TestJavaScriptExtraction(unittest.TestCase):
    """Verify declarations, attached comments and the context between."""

    def test_declarations_and_context_cover_the_file(self) -> None:
        units = _units()
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [
                ("js_module_context", None),
                ("js_function", "add"),
                ("js_module_context", None),
                ("js_function", "twice"),
                ("js_function", "load"),
                ("js_class", "Store"),
                ("js_module_context", None),
            ],
        )
        self.assertEqual("".join(unit.text for unit in units), SOURCE)

    def test_comment_above_a_declaration_stays_with_it(self) -> None:
        add = next(unit for unit in _units() if unit.symbol == "add")
        self.assertTrue(add.text.startswith("/**"))
        self.assertEqual((add.start_line, add.end_line), (3, 8))

    def test_class_methods_are_children(self) -> None:
        store = next(unit for unit in _units() if unit.symbol == "Store")
        self.assertTrue(store.text.startswith("// Keeps the projects."))
        self.assertEqual(
            [child.symbol for child in store.children],
            [None, "Store.constructor", "Store.save"],
        )
        self.assertEqual(
            "".join(child.text for child in store.children), store.text
        )

    def test_statements_are_not_methods(self) -> None:
        store = next(unit for unit in _units() if unit.symbol == "Store")
        self.assertNotIn("Store.if", [c.symbol for c in store.children])

    def test_empty_file(self) -> None:
        self.assertEqual(_units(""), [])


if __name__ == "__main__":
    unittest.main()
