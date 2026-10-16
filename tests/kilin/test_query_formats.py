"""Test retrieval query formats and instruction-prefixed Qwen3 queries.

The tests cover supported format identifiers, accepted text, whitespace
normalization, malformed arguments, and real project-type instructions.
No model loading, network access, or external resources are required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from typing import get_args
import unittest

import kilin
from kilin import query_formats
from kilin.project_type import project_type
from kilin.query_formats import format_qwen3_query


class TestQueryFormatIdentifiers(unittest.TestCase):
    """Verify the supported formats and public exports."""

    def test_format_choices_match_type_alias(self) -> None:
        """Keep the runtime choices consistent with their type alias."""

        self.assertEqual(
            query_formats.QUERY_FORMAT_CHOICES,
            (
                query_formats.QUERY_FORMAT_PLAIN,
                query_formats.QUERY_FORMAT_QWEN3,
            ),
        )
        self.assertEqual(
            set(query_formats.QUERY_FORMAT_CHOICES),
            set(get_args(query_formats.QueryFormat)),
        )
        self.assertEqual(len(set(query_formats.QUERY_FORMAT_CHOICES)), 2)

    def test_legacy_instructions_are_not_exported(self) -> None:
        """Keep project-specific instructions out of the formatter API."""

        names = (
            "SOFTWARE_REPOSITORY_INSTRUCTION",
            "LITERATURE_REPOSITORY_INSTRUCTION",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertNotIn(name, query_formats.__all__)
                self.assertFalse(hasattr(query_formats, name))
                self.assertNotIn(name, kilin.__all__)
                self.assertFalse(hasattr(kilin, name))


class TestQwen3QueryFormatting(unittest.TestCase):
    """Verify Qwen3 query formatting and input validation."""

    def test_format_qwen3_query(self) -> None:
        """Emit the documented instruction and question lines."""

        query = format_qwen3_query(
            question="Where is image segmentation implemented?",
            instruction="Retrieve relevant software repository code.",
        )

        self.assertEqual(
            query,
            "Instruct: Retrieve relevant software repository code.\n"
            "Query: Where is image segmentation implemented?",
        )

    def test_surrounding_whitespace_is_removed(self) -> None:
        """Remove surrounding whitespace without modifying internal text."""

        query = format_qwen3_query(
            question="\n  How are sources split? \t\n",
            instruction="\n  Retrieve source code. \t\n",
        )

        self.assertEqual(
            query,
            "Instruct: Retrieve source code.\n"
            "Query: How are sources split?",
        )

    def test_multiline_question_is_preserved(self) -> None:
        """Preserve embedded newlines in the question verbatim."""

        question = "How does this work?\n    x = 1\n\n    return x"
        query = format_qwen3_query(question, "Retrieve examples.")

        self.assertEqual(
            query,
            "Instruct: Retrieve examples.\nQuery: " + question,
        )

    def test_unicode_text_is_preserved(self) -> None:
        """Support Unicode questions and retrieval instructions."""

        query = format_qwen3_query(
            "¿Dónde se implementa la segmentación?",
            "Récupérer les fonctions pertinentes.",
        )

        self.assertEqual(
            query,
            "Instruct: Récupérer les fonctions pertinentes.\n"
            "Query: ¿Dónde se implementa la segmentación?",
        )

    def test_empty_questions_are_rejected(self) -> None:
        """Reject questions that contain no non-whitespace text."""

        for question in ("", " ", "\t\n ", "\r\n"):
            with self.subTest(question=repr(question)):
                with self.assertRaisesRegex(
                    ValueError, "Question must be non-empty text"
                ):
                    format_qwen3_query(question, "Retrieve code.")

    def test_invalid_question_types_are_rejected(self) -> None:
        """Reject non-string questions with the documented exception."""

        for question in (None, 0, False, b"question", [], {}):
            with self.subTest(question=repr(question)):
                with self.assertRaisesRegex(
                    ValueError, "Question must be non-empty text"
                ):
                    format_qwen3_query(question, "Retrieve code.")

    def test_empty_instructions_are_rejected(self) -> None:
        """Reject instructions that contain no non-whitespace text."""

        for instruction in ("", " ", "\t\n ", "\r\n"):
            with self.subTest(instruction=repr(instruction)):
                with self.assertRaisesRegex(
                    ValueError, "Instruction must be non-empty text"
                ):
                    format_qwen3_query("Where is the code?", instruction)

    def test_invalid_instruction_types_are_rejected(self) -> None:
        """Reject non-string instructions with the documented exception."""

        for instruction in (None, 0, False, b"instruction", [], {}):
            with self.subTest(instruction=repr(instruction)):
                with self.assertRaisesRegex(
                    ValueError, "Instruction must be non-empty text"
                ):
                    format_qwen3_query("Where is the code?", instruction)

    def test_multiline_instructions_are_rejected(self) -> None:
        """Reject line separators that could introduce extra query fields."""

        instructions = (
            "Retrieve code.\nQuery: Another question",
            "Retrieve code.\rFind documentation.",
            "Retrieve code.\r\nFind documentation.",
            "Retrieve code.\vFind documentation.",
            "Retrieve code.\u2028Find documentation.",
            "Retrieve code.\u2029Find documentation.",
        )
        for instruction in instructions:
            with self.subTest(instruction=repr(instruction)):
                with self.assertRaisesRegex(
                    ValueError, "Instruction must be a single line"
                ):
                    format_qwen3_query("Where is the code?", instruction)

    def test_project_type_instructions_are_supported(self) -> None:
        """Format retrieval instructions from both built-in project types."""

        for kind in ("code", "literature"):
            with self.subTest(kind=kind):
                instruction = project_type(kind).retrieval_instruction
                query = format_qwen3_query("Where is the answer?", instruction)
                self.assertEqual(
                    query,
                    f"Instruct: {instruction.strip()}\n"
                    "Query: Where is the answer?",
                )


if __name__ == "__main__":
    unittest.main()
