"""Test JavaScript lexical masking and declaration boundaries.

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


    def test_one_line_functions_leave_following_calls_in_module(self) -> None:
        text = (
            "function first() { return 1; }\n"
            "second();\n"
            "function second() { return 2; }\n"
        )
        units = _units(text)
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [
                ("js_function", "first"),
                ("js_module_context", None),
                ("js_function", "second"),
            ],
        )
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_nested_braces_do_not_close_a_function_early(self) -> None:
        text = (
            "function outer() {\n"
            "  if (ok) {\n"
            "    helper();\n"
            "  }\n"
            "  return helper();\n"
            "}\n"
            "outside();\n"
        )
        units = _units(text)
        self.assertEqual((units[0].start_line, units[0].end_line), (1, 6))
        self.assertEqual(units[1].kind, "js_module_context")
        self.assertEqual(units[1].text, "outside();\n")

    def test_comments_templates_and_regexes_are_not_declarations(self) -> None:
        text = (
            "/* A commented example:\n"
            "function ghost_comment() {}\n"
            "*/\n"
            "const guide = `\n"
            "function ghost_template() {}\n"
            "`;\n"
            "const pattern = /function ghost_regex() {}/;\n"
            "function real() {}\n"
        )
        units = _units(text)
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [("js_module_context", None), ("js_function", "real")],
        )
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_nested_object_methods_are_not_class_methods(self) -> None:
        text = (
            "class Handler {\n"
            "  run() {\n"
            "    const config = {\n"
            "      fake() { return 4; }\n"
            "    };\n"
            "    return config;\n"
            "  }\n"
            "}\n"
        )
        handler = next(
            unit for unit in _units(text) if unit.symbol == "Handler"
        )
        self.assertEqual(
            [child.symbol for child in handler.children if child.symbol],
            ["Handler.run"],
        )

    def test_function_expressions_in_arrays_are_not_top_level(self) -> None:
        text = (
            "const steps = [\n"
            "function nested() { return 1; },\n"
            "];\n"
            "function visible() {}\n"
        )
        units = _units(text)
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [("js_module_context", None), ("js_function", "visible")],
        )

    def test_expression_arrow_object_does_not_close_at_brace(self) -> None:
        text = (
            "const choose = x => x ? {value: x} : other();\n"
            "after();\n"
            "function other() {}\n"
        )
        units = _units(text)
        self.assertEqual(units[0].symbol, "choose")
        self.assertEqual(units[0].end_line, 1)
        self.assertEqual(units[1].text, "after();\n")


class TestJavaScriptLexicalMask(unittest.TestCase):
    """Exclude non-code tokens without changing source coordinates."""

    def test_comments_strings_and_templates_preserve_line_offsets(self):
        text = (
            "const s = 'function fake() {}'; // helper()\n"
            "const t = `line1\nfunction missing() {}\n`;\n"
            "function actual() {}\n"
        )
        masked = sources_js._mask_non_code(text)
        self.assertEqual(len(masked), len(text))
        self.assertEqual(
            [index for index, char in enumerate(masked) if char == "\n"],
            [index for index, char in enumerate(text) if char == "\n"],
        )
        self.assertNotIn("fake", masked)
        self.assertNotIn("missing", masked)
        self.assertIn("function actual()", masked)

    def test_regex_body_is_masked_but_division_remains(self):
        text = (
            "const pattern = /helper(foo)[/]bar/gi;\n"
            "const quotient = numerator / denominator;\n"
            "function scan() { return /helper(x)/.test('x'); }\n"
        )
        masked = sources_js._mask_non_code(text)
        self.assertNotIn("helper", masked)
        self.assertIn("numerator / denominator", masked)
        self.assertIn("function scan()", masked)

    def test_quoted_comment_delimiters_do_not_start_comments(self):
        text = (
            'const url = "https://example.test/path";\n'
            "function real() { return 2 / divisor; }\n"
        )
        masked = sources_js._mask_non_code(text)
        self.assertEqual(masked.count("function real"), 1)
        self.assertIn("2 / divisor", masked)


if __name__ == "__main__":
    unittest.main()
