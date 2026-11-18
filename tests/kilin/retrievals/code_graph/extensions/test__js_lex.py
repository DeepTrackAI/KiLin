"""Verify offset-preserving masking for JavaScript structure and calls.

This scanner is intentionally smaller than a JavaScript parser. Tests
cover the literal forms needed to avoid fictitious retrieval entities.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.sources.extensions._js_lex import mask_non_code


class TestJavaScriptLexicalMask(unittest.TestCase):
    """Exclude non-code tokens without changing source coordinates."""

    def test_comments_strings_and_templates_preserve_line_offsets(self):
        text = (
            "const s = 'function fake() {}'; // helper()\n"
            "const t = `line1\nfunction missing() {}\n`;\n"
            "function actual() {}\n"
        )
        masked = mask_non_code(text)
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
        masked = mask_non_code(text)
        self.assertNotIn("helper", masked)
        self.assertIn("numerator / denominator", masked)
        self.assertIn("function scan()", masked)

    def test_quoted_comment_delimiters_do_not_start_comments(self):
        text = (
            'const url = "https://example.test/path";\n'
            "function real() { return 2 / divisor; }\n"
        )
        masked = mask_non_code(text)
        self.assertEqual(masked.count("function real"), 1)
        self.assertIn("2 / divisor", masked)


if __name__ == "__main__":
    unittest.main()
