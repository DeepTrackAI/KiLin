"""Tests for model-specific retrieval query formats."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

import unittest

from qachat.query_formats import format_qwen3_query


class TestQueryFormats(unittest.TestCase):

    def test_format_qwen3_query(self) -> None:
        query = format_qwen3_query(
            question="Where is image segmentation implemented?",
            instruction="Retrieve relevant software repository code.",
        )

        self.assertEqual(
            query,
            "Instruct: Retrieve relevant software repository code.\n"
            "Query: Where is image segmentation implemented?",
        )

    def test_invalid_text_is_rejected(self) -> None:
        invalid = (
            ("", "Retrieve code."),
            ("Where is the code?", "  "),
            (None, "Retrieve code."),
            ("Where is the code?", None),
        )
        for question, instruction in invalid:
            with self.subTest(
                question=question,
                instruction=instruction,
            ):
                with self.assertRaises(ValueError):
                    format_qwen3_query(question, instruction)


if __name__ == "__main__":
    unittest.main()
