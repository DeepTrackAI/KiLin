"""Test the writing_check tool package without live model backends.

Check the registry entry and intent, settings validation, and full runs
on a small LaTeX project with a stub model: the part checked, the plan
and its quick and cancelled versions, the saved report, failed calls, a
part the document does not have, and Word files read through their
source reader.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from docx import Document as WordDocument

from kilin.answering import _cued_intent
from kilin.project_type import project_type
from kilin.tools import writing_check as tool
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_WRITING_CHECK
from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)
from tests.kilin.tools._shared._support import chunk

DRAFT = r"""\documentclass{article}
\begin{document}
\section{Introduction}
Single-particle tracking (SPT) reveals the the diffusion of molecules.
\section{Methods}
We tracks particles in order to estimate diffusion \cite{a}.

The SPT data set was analysed with a novel pipeline.
\section{Results}
The dataset showed that the dataset is large.
\end{document}
"""


def _model(system: str, prompt: str, tokens: int | None) -> str:
    """Answer like a model: a brief, or the changes for a paragraph."""
    if "describe documents" in system:
        return "A biophysics article on particle tracking."
    checked = '"""\nWe tracks' in prompt  # The paragraph, not context.
    if checked and "Find the grammar errors" in prompt:
        return ("ORIGINAL: We tracks\nSUGGESTION: We tracked\n"
                "REASON: agreement\n")
    if checked:
        return ("ORIGINAL: in order to estimate\nSUGGESTION: to estimate\n"
                "REASON: concise\nORIGINAL: particles\nSUGGESTION: 12 "
                "particles\nREASON: invented\n")
    return "NONE"


class TestWritingCheckPackage(unittest.TestCase):

    def test_registry_uses_package_entry_points(self) -> None:
        assert_registry_entry(self, "writing_check", tool, non_followup=True)

    def test_packaged_defaults_are_documented(self) -> None:
        assert_packaged_defaults(self, tool, tool.WritingCheckSettings)

    def test_levels(self) -> None:
        from kilin.tools.writing_check.tool import _levels
        for level, question, expected in (
            ("auto", "check the grammar of draft.tex", ("grammar",)),
            ("auto", "check the wording of the methods",
             ("grammar", "style")),
            ("auto", "make the results clearer", ("grammar", "style")),
            ("all", "check draft.tex", ("grammar", "style")),
            ("style", "check draft.tex", ("style",)),
        ):
            with self.subTest(level=level, question=question):
                self.assertEqual(_levels(level, question), expected)

    def test_invalid_project_type_settings(self) -> None:
        for values in (
            {"level": "everything"}, {"unknown": 1}, {"chat_findings": 201},
            {"confirmation": "sometimes"}, {"quick_paragraphs": 0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_literature_intent_uses_the_tool(self) -> None:
        intent = project_type("literature").intents["writing_check"]
        self.assertEqual(intent.tool.name, TOOL_WRITING_CHECK)

    def test_writing_questions_are_cued_in_both_project_types(self) -> None:
        for type_id in ("literature", "writing"):
            project = project_type(type_id)
            self.assertEqual(
                project.intents["writing_check"].tool.name,
                TOOL_WRITING_CHECK,
            )
            for question in (
                "check the grammar of the methods of draft.tex",
                "proofread the methods section",
                "check the English of the second paragraph",
                "are there typos in my manuscript?",
            ):
                with self.subTest(type_id=type_id, question=question):
                    self.assertEqual(_cued_intent(project, question),
                                     "writing_check")
            for question in (
                "check the references of draft.tex",
                "which papers discuss writing style?",
                "summarize the paper",
            ):
                with self.subTest(type_id=type_id, question=question):
                    self.assertNotEqual(_cued_intent(project, question),
                                        "writing_check")
        # Rewriting stays with the writing assistant's own intent.
        writing = project_type("writing")
        for question in ("rewrite this abstract",
                         "proofread this: The cells was moving."):
            with self.subTest(question=question):
                self.assertEqual(_cued_intent(writing, question), "writing")
        self.assertIn(".tex", writing.extensions)


class TestWritingCheckRun(unittest.TestCase):

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "sources").mkdir()
        (self.root / "sources" / "draft.tex").write_text(
            DRAFT, encoding="utf-8"
        )

    def _context(self, choice: str = "continue", **extra) -> SimpleNamespace:
        values = dict(
            project_root=self.root,
            source_root=self.root / "sources",
            chunks=[chunk("d1", DRAFT, path="draft.tex")],
            source_files=lambda: {Path("draft.tex"): len(DRAFT)},
            rank=lambda text: [],
            retrieval_count=5,
            progress=Mock(), done=Mock(),
            ask_model=Mock(side_effect=_model),
            ask_router=Mock(return_value="NONE"),
            confirm=Mock(return_value=choice),
            choose=None,
            trace=SimpleNamespace(emit=Mock()), trace_turn_id=None,
            card=None,
        )
        values.update(extra)
        return SimpleNamespace(**values)

    def _run(self, question: str, context=None, **settings):
        context = context or self._context()
        request = ToolRequest(question, question)
        return tool.run(
            tool.WritingCheckSettings(**settings), request, context
        ), context

    def test_whole_document(self) -> None:
        result, context = self._run("check the grammar and style of "
                                    "draft.tex")
        answer = result.answer
        self.assertTrue(answer.startswith(
            "**Writing check of draft.tex** (the whole document)"
        ))
        for line in (
            "- line 4 · grammar: “the the” → “the” (the word is written "
            "twice)",
            "- line 6 · grammar: “We tracks” → “We tracked” (agreement)",
            "- line 6 · style: “in order to estimate” → “to estimate” "
            "(concise)",
            "- line 8 · consistency: “data set” → “dataset”",
            "- line 8 · style: “novel”",
        ):
            with self.subTest(line=line):
                self.assertIn(line, answer)
        self.assertNotIn("12 particles", answer)
        self.assertIn(
            "1 suggested change was dropped (1 numbers changed); they are "
            "listed in the report.", answer,
        )
        self.assertIn("draft_writing_check.md", answer)
        # One brief call (no card), then grammar and style per paragraph.
        self.assertEqual(context.ask_model.call_count, 1 + 4 * 2)
        report = (self.root / "outputs" / "draft_writing_check.md")
        text = report.read_text(encoding="utf-8")
        self.assertIn("## 2 Methods", text)
        self.assertIn("| 6 | grammar | We tracks | We tracked | agreement |",
                      text)
        self.assertEqual(result.summary["findings"],
                         {"grammar": 2, "consistency": 1, "style": 2})
        self.assertEqual(result.summary["dropped_reasons"],
                         {"numbers changed": 1})
        self.assertIn("## Dropped suggestions", text)
        self.assertIn("| 6 | style | particles | 12 particles | it changes "
                      "numbers |", text)
        # Progress names each paragraph once, grammar then style.
        steps = [c.args[0] for c in context.progress.call_args_list
                 if c.args[0].startswith("Checking ") and "paragraph" in
                 c.args[0]]
        self.assertEqual(steps[:2], ["Checking grammar of paragraph 1 of 4",
                                     "Checking style of paragraph 1 of 4"])

    def test_part_and_card(self) -> None:
        card = Mock(return_value={"title": "Tracking", "aim": "Diffusion."})
        context = self._context(card=card)
        result, _ = self._run("proofread the methods section",
                              context, level="grammar")
        self.assertIn("(section 2 Methods)", result.answer)
        self.assertNotIn("the the", result.answer)
        # With a card, no brief call; grammar only, two paragraphs.
        self.assertEqual(context.ask_model.call_count, 2)
        prompt = context.ask_model.call_args_list[0].args[1]
        self.assertIn("Title: Tracking", prompt)
        card.assert_called_once_with(Path("draft.tex"))

    def test_quick_and_cancelled_plans(self) -> None:
        context = self._context("quick")
        result, _ = self._run("check the grammar of draft.tex", context,
                              confirmation="always", quick_paragraphs=1)
        context.confirm.assert_called_once()
        self.assertEqual(context.ask_model.call_count, 1 + 1)
        self.assertIn("Quick version: grammar of the first paragraph of 4.",
                      result.answer)
        context = self._context("cancel")
        result, _ = self._run("check the grammar of draft.tex", context,
                              confirmation="always")
        self.assertEqual(context.ask_model.call_count, 0)
        self.assertIn("The model checks were cancelled", result.answer)
        self.assertIn("“the the”", result.answer)

    def test_failed_calls_give_a_partial_result(self) -> None:
        context = self._context(
            ask_model=Mock(side_effect=RuntimeError("down")),
            card=Mock(return_value={"title": "T"}),
        )
        result, _ = self._run("check the grammar of draft.tex", context,
                              level="grammar")
        self.assertIn("“the the”", result.answer)
        self.assertEqual(result.summary.get("status"), "incomplete")

    def test_missing_section(self) -> None:
        result, _ = self._run("check the discussion section of draft.tex")
        self.assertEqual(
            result.answer,
            "There is no section called “discussion”. The headings of "
            "draft.tex are: 1 Introduction; 2 Methods; 3 Results. Name "
            "one, for example: check the grammar of section 1.",
        )
        choose = Mock(return_value=1)
        result, context = self._run(
            "check the discussion section of draft.tex",
            self._context(choose=choose), level="grammar",
        )
        self.assertIn("(section 2 Methods)", result.answer)
        self.assertEqual(choose.call_args.args[1][-1], "The whole document")

    def test_document_without_sections(self) -> None:
        (self.root / "sources" / "notes.txt").write_text(
            "We tracks the the particles.\n", encoding="utf-8"
        )
        context = self._context(
            chunks=[chunk("n1", "notes", path="notes.txt")],
            source_files=lambda: {Path("notes.txt"): 1},
            card=Mock(return_value={"title": "Notes"}),
        )
        result, _ = self._run("check the grammar of notes.txt", context)
        report = (self.root / "outputs" / "notes_writing_check.md")
        self.assertIn("## Findings", report.read_text(encoding="utf-8"))
        self.assertNotIn("**", result.answer.split("\n", 2)[2])

    def test_quoted_text_not_in_the_document(self) -> None:
        result, context = self._run(
            "check the grammar of this sentence: The results was clear "
            "and robust.", level="grammar",
        )
        self.assertIn("(the text in the question)", result.answer)
        self.assertEqual(context.ask_model.call_count, 1 + 1)

    def test_unselected_document(self) -> None:
        context = self._context(source_files=lambda: {})
        result, _ = self._run("check draft.tex", context)
        self.assertIn("could not read draft.tex", result.answer)

    def test_word_document(self) -> None:
        word = WordDocument()
        word.add_heading("Methods", level=1)
        word.add_paragraph("We tracks particles in order to estimate "
                           "diffusion.")
        word.save(self.root / "sources" / "draft.docx")
        context = self._context(
            chunks=[chunk("w1", "Methods", path="draft.docx")],
            source_files=lambda: {Path("draft.docx"): 1},
        )
        result, _ = self._run("check the grammar of draft.docx", context,
                              level="grammar")
        self.assertIn("**Methods**", result.answer)
        self.assertIn("“We tracks” → “We tracked”", result.answer)


if __name__ == "__main__":
    unittest.main()
