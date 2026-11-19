"""Unit tests for the ``assistant_guide`` tool.

The tool answers "What can you do for me?" with one example question per
kind of request, taken from the project's suggested questions, or, when the
project has no sources, with how to add them and the project type's fixed
suggestions.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.question_suggestions import Suggestion
from kilin.tools import assistant_guide as tool
from kilin.tools._contract import ToolRequest

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)

SUGGESTIONS = (
    Suggestion("paper_overview", "Give me a short summary of a.pdf."),
    Suggestion("paper_overview", "Give me a short summary of b.pdf."),
    Suggestion("mention_listing", "Which papers mention MRI?"),
)


def _context(
    suggestions=None, fixed=(), *, chunks=(), files=()
) -> SimpleNamespace:
    """Return the tool context with explicit source inventory services."""

    return SimpleNamespace(
        suggestions=suggestions,
        fixed_suggestions=fixed,
        chunks=chunks,
        source_files=lambda: {path: 0 for path in files},
        progress=Mock(),
        done=Mock(),
        trace=Mock(),
        trace_turn_id=None,
    )


def _answer(
    context: SimpleNamespace,
    question: str = "What can you do for me?",
    **settings: object,
) -> str:
    """Run the tool on a question and return its answer."""

    result = tool.run(
        tool.parse_settings(settings, "example"),
        ToolRequest(question, "", "", ()),
        context,
    )
    return result.answer


class TestAssistantGuide(unittest.TestCase):
    """List one example per kind of request, or fall back."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation."""

        assert_registry_entry(self, "assistant_guide", tool, non_followup=True)

    def test_packaged_defaults(self) -> None:
        """Use the tool-local JSON when settings are not supplied."""

        assert_packaged_defaults(self, tool, tool.GuideSettings)

    def test_invalid_settings(self) -> None:
        """Reject out-of-range and unknown settings."""

        for values in ({"max_listed": 0}, {"max_listed": 101}, {"x": 1}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_one_example_per_kind_of_request(self) -> None:
        """Label each kind of request and show one of its questions."""

        answer = _answer(_context(lambda: SUGGESTIONS))
        self.assertIn("**Paper overview**: Give me a short summary of", answer)
        self.assertEqual(answer.count("**Paper overview**"), 1)
        self.assertIn("**Mention listing**: Which papers mention MRI?", answer)

    def test_every_answer_says_how_to_add_files(self) -> None:
        """End the examples with how to add and remove files."""

        answer = _answer(_context(lambda: SUGGESTIONS))
        self.assertIn("To add more files:", answer)
        self.assertIn("**sources** in the Workspace panel", answer)
        self.assertIn("**Delete**", answer)

    def test_file_questions_get_only_the_steps(self) -> None:
        """Answer a question about files without the list of examples."""

        for question in (
            "how to drag files",
            "How do I add my papers?",
            "What do you mean by adding files?",
            "What do you do with my files?",
        ):
            with self.subTest(question=question):
                answer = _answer(
                    _context(lambda: SUGGESTIONS, files=("a.pdf",)),
                    question,
                )
                self.assertIn("Sources are the files I read", answer)
                self.assertIn("**Attach folder**", answer)
                self.assertNotIn("Paper overview", answer)
                self.assertIn("What can you do for me?", answer)
        answer = _answer(_context(lambda: []), "how to drag files")
        self.assertIn("This project has no sources yet.", answer)

    def test_examples_change_between_answers(self) -> None:
        """Pick another file's example from one answer to the next."""

        answers = {_answer(_context(lambda: SUGGESTIONS)) for _ in range(40)}
        self.assertTrue(any("a.pdf" in text for text in answers))
        self.assertTrue(any("b.pdf" in text for text in answers))

    def test_listed_examples_are_capped(self) -> None:
        """List at most ``max_listed`` examples."""

        answer = _answer(_context(lambda: SUGGESTIONS), max_listed=1)
        self.assertEqual(answer.count("- **"), 1)

    def test_empty_project_falls_back_to_fixed_suggestions(self) -> None:
        """Explain how to add sources and show the type's fixed ones."""

        fixed = ("What can you do for me?", "Summarize …", "Compare …")
        answer = _answer(_context(lambda: [], fixed))
        self.assertIn("no sources yet", answer)
        self.assertIn("**sources** in the Workspace panel", answer)
        self.assertIn("**Attach file**", answer)
        self.assertIn("- Summarize …", answer)
        # The question being answered is not listed as an example.
        self.assertNotIn("- What can you do for me?", answer)
        answer = _answer(_context(None))
        self.assertIn("no sources yet", answer)
        self.assertNotIn("- ", answer)

    def test_sources_without_examples_are_not_called_empty(self) -> None:
        """Read inventory independently of generated question templates."""

        fixed = ("What can you do for me?", "Summarize my sources.")
        for context in (
            _context(lambda: [], fixed, files=("paper.pdf",)),
            _context(lambda: [], fixed, chunks=(object(),)),
        ):
            with self.subTest(context=context):
                answer = _answer(context)
                self.assertIn("has sources", answer)
                self.assertNotIn("no sources yet", answer)
                self.assertIn("Summarize my sources.", answer)

    def test_file_help_uses_inventory_without_generation(self) -> None:
        """Explaining file operations never calls a suggestion generator."""

        suggestions = Mock(return_value=[])
        context = _context(
            suggestions, files=("paper.pdf",)
        )
        answer = _answer(context, "How do I add files?")
        self.assertNotIn("no sources yet", answer)
        suggestions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
