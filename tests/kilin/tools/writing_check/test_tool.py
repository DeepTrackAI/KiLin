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
from kilin.tools._contract import EarlierTurn, ToolRequest
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
    """Answer like a model: a brief, rewrites, or a paragraph's changes."""
    if "rewrite text" in system:
        return (
            "OPTION: We tracked particles to estimate diffusion [cite].\n"
            "OPTION: We tracked particles in order to...\n"
            "OPTION: We tracked particles with remarkable new hardware to "
            "estimate diffusion [cite].\n"
        )
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
            {"max_model_calls": 0}, {"max_model_calls": 501},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_literature_intent_uses_the_tool(self) -> None:
        intent = project_type("literature").intents["writing_check"]
        self.assertEqual(intent.tool.name, TOOL_WRITING_CHECK)

    def test_writing_questions_are_cued_in_the_project_types(self) -> None:
        for type_id in ("literature", "writing", "reviewing"):
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
                "suggest a better phrasing for line 75",
                "split the sentence at line 75",
                "line 75 · style: “DeepTrack2 complements platforms like "
                "BiaPy” (47 words; consider splitting it)\n\n"
                "suggest a better phrasing",
            ):
                with self.subTest(type_id=type_id, question=question):
                    self.assertEqual(_cued_intent(project, question),
                                     "writing_check")
            for question in (
                "check the references of draft.tex",
                "which papers discuss writing style?",
                "summarize the paper",
                "what does line 3 of the table show?",
            ):
                with self.subTest(type_id=type_id, question=question):
                    self.assertNotEqual(_cued_intent(project, question),
                                        "writing_check")
        # The reviewer's own text stays with review_writing.
        reviewing = project_type("reviewing")
        for question in (
            "proofread my comments",
            "rewrite my report",
            "check the grammar of my comments: The author claim that",
            "check the English of my review",
            "any typos in my report?",
            "rephrase my comment on line 75 of the manuscript",
        ):
            with self.subTest(question=question):
                self.assertEqual(_cued_intent(reviewing, question),
                                 "review_writing")
        for question in (
            "check the wording of my assessment",
            "check the style of my evaluation",
            "check the grammar in our critique",
            "check the English of my feedback",
            "check the spelling of my recommendation",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    _cued_intent(reviewing, question), "review_writing"
                )
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

    def _run(self, question: str, context=None, earlier=(), **settings):
        context = context or self._context()
        request = ToolRequest(question, question, earlier=earlier)
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
        # The paragraphs behind the findings are the answer's evidence.
        self.assertIn("(agreement) [2]", answer)
        self.assertEqual(
            [(item["citation"], item["start_line"], item["end_line"])
             for item in result.evidence],
            [(1, 4, 4), (2, 6, 6), (3, 8, 8)],
        )
        self.assertIn("\\cite{a}", result.evidence[1]["text"])
        self.assertIn(
            "1 suggested change was dropped (1 numbers changed); they are "
            "listed in the report.", answer,
        )
        self.assertIn("draft_writing_check.md", answer)
        self.assertIn("scientific meaning", answer)
        # One brief call (no card), then grammar and style per paragraph.
        self.assertEqual(context.ask_model.call_count, 1 + 4 * 2)
        self.assertEqual(result.summary["model_calls"], 9)
        self.assertEqual(result.summary["model_checks"], 8)
        context.done.assert_called_once_with(
            "Checked 4 paragraphs", evidence_count=3
        )
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

    def test_rewrite_follow_up(self) -> None:
        first, _ = self._run("check the grammar of draft.tex")
        earlier = (EarlierTurn(
            1, "check the grammar of draft.tex", answer=first.answer,
            paths=tuple(item["path"] for item in first.evidence),
        ),)
        context = self._context()
        result, context = self._run(
            "suggest a better phrasing for line 6", context, earlier
        )
        answer = result.answer
        self.assertTrue(answer.startswith(
            "**Rewriting line 6 of draft.tex**\n\nText: “We tracks "
            "particles in order to estimate diffusion [cite].” [1]"
        ))
        self.assertIn(
            "1. “We tracked particles to estimate diffusion [cite].”",
            answer,
        )
        self.assertNotIn("2. ", answer)
        for why in ("cut short with an ellipsis",
                    "adds words the paragraph does not have"):
            with self.subTest(why=why):
                self.assertIn(why, answer)
        self.assertEqual(context.ask_model.call_count, 1)
        self.assertEqual(result.evidence[0]["start_line"], 6)
        self.assertEqual(result.summary["rewrites"], 1)

    def test_rewrite_without_an_earlier_check(self) -> None:
        result, _ = self._run("rephrase line 6 of draft.tex")
        self.assertIn("**Rewriting line 6 of draft.tex**", result.answer)
        result, _ = self._run("rephrase line 2 of draft.tex")
        self.assertEqual(result.answer,
                         "Line 2 of draft.tex holds no prose to rewrite.")

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

    def test_absolute_call_cap_without_confirmation(self) -> None:
        """Stop at a hard limit even in a script without callbacks.

        """

        context = self._context(confirm=None)
        result, _ = self._run(
            "check grammar and style of draft.tex", context,
            level="all", max_model_calls=2,
        )
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertEqual(result.summary["model_calls"], 2)
        self.assertEqual(result.summary["model_checks"], 1)
        self.assertEqual(result.summary["omitted_model_checks"], 7)
        self.assertEqual(result.summary["status"], "partial")
        self.assertIn("7 model checks were not run", result.answer)
        self.assertIn("Unreviewed paragraphs", result.answer)
        plan = context.trace.emit.call_args_list[0].kwargs["data"]
        self.assertEqual(plan["batches"], 9)
        self.assertEqual(plan["batches_to_run"], 2)
        self.assertTrue((self.root / "outputs" /
                         "draft_writing_check.md").exists())

    def test_one_call_is_used_for_review_not_optional_brief(self) -> None:
        """A tight budget prioritizes checking the requested prose.

        """

        context = self._context(confirm=None)
        result, _ = self._run(
            "check the grammar of draft.tex", context,
            level="grammar", max_model_calls=1,
        )
        self.assertEqual(context.ask_model.call_count, 1)
        self.assertNotIn("Describing the document", [
            call.args[0] for call in context.progress.call_args_list
        ])
        self.assertEqual(result.summary["model_checks"], 1)
        self.assertEqual(result.summary["model_calls"], 1)
        self.assertEqual(result.summary["omitted_model_checks"], 3)

    def test_model_failures_still_use_the_budget(self) -> None:
        """Count failed requests and do not retry past the limit.

        """

        context = self._context(
            confirm=None, card=Mock(return_value={"title": "Draft"}),
            ask_model=Mock(side_effect=RuntimeError("offline")),
        )
        result, _ = self._run(
            "check the grammar of draft.tex", context,
            level="grammar", max_model_calls=2,
        )
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertEqual(result.summary["model_calls"], 2)
        self.assertEqual(result.summary["omitted_model_checks"], 2)
        self.assertEqual(result.summary["status"], "incomplete")

    def test_parent_latex_file_is_not_silently_checked_as_empty(self) -> None:
        """Keep referenced files behind explicit source selection.

        """

        (self.root / "sources" / "main.tex").write_text(
            "\\begin{document}\n\\input{chapter}\n"
            "\\end{document}\n", encoding="utf-8"
        )
        context = self._context(
            chunks=[chunk("main", "text", path="main.tex")],
            source_files=lambda: {Path("main.tex"): 1},
        )
        result, _ = self._run("check the grammar of main.tex", context)
        self.assertIn(
            "does not assemble multi-file manuscripts", result.answer
        )
        context.ask_model.assert_not_called()

    def test_quoted_text_not_in_the_document(self) -> None:
        result, context = self._run(
            "check the grammar of this sentence: The results was clear "
            "and robust.", level="grammar",
        )
        self.assertIn("(the text in the question)", result.answer)
        self.assertEqual(result.evidence, ())
        self.assertEqual(context.ask_model.call_count, 1 + 1)

    def test_question_text_is_never_attributed_to_a_source(self) -> None:
        """Do not cite the selected file for user-provided words.

        """

        result, context = self._run(
            "check the grammar of this: The the results was clear and "
            "robust.", level="grammar",
        )
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.summary["text_origin"], "question")
        self.assertIn("text supplied in the question", result.answer)
        self.assertNotIn("[1]", result.answer)
        self.assertNotIn("line 1", result.answer)
        report = self.root / "outputs" / "question_writing_check.md"
        self.assertTrue(report.exists())
        self.assertIn("text supplied in the question", report.read_text())
        context.done.assert_called_once_with(
            "Checked 1 paragraphs", evidence_count=0
        )

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
        self.assertIn("extracted-text line", result.answer)
        self.assertIn("not native PDF or Word", result.answer)
        self.assertEqual(context.done.call_args.kwargs["evidence_count"],
                         len(result.evidence))


if __name__ == "__main__":
    unittest.main()
