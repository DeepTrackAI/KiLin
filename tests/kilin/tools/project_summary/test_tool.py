"""Test the project_summary tool without a live model.

Check its settings, templates and routing, the parts of the report written
by code, and full runs on a small project with a stub model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.answering import _cued_intent
from kilin.project_type import project_type, with_project_routing
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_REGISTRY
from kilin.tools.project_summary import (
    SHIPPED_TEMPLATES_DIR,
    TEMPLATE_CHOOSE,
    ProjectSummarySettings,
    choose_template,
    parse_settings,
    project_templates,
    routing,
    run,
    workspace_resources,
)
from kilin.tools.project_summary._documents import ProjectDocument
from kilin.tools.project_summary._mapping import Check, DocumentNotes
from kilin.tools.project_summary._reduce import (
    agreement_problem,
    chosen_numbers,
    citation_problem,
    clean_citations,
    coverage_problem,
    fitted_headline,
    support_problem,
    documents_line,
    documents_list,
    note_groups,
    per_document_markdown,
)
from kilin.tools._shared.report_templates import TemplateSection
from tests.kilin.tools._shared._support import chunk

LAB = """---
title: Lab Overview
description: What our papers say together.
max_words: 200
scope: project
---

# Headline (max 80 characters)

**Documents:** {documents}

## Themes (3-40 words)

The themes the documents share.

## Models (per document)

When: the document trains a deep-learning model.
Do: name the architecture (max 6 words).

## Datasets (per document, table)

When: the document uses a public dataset.
Do: | Dataset | Size |

## Start Here (1-2 items)

The best first reads, chosen from {key_documents}.
"""
PAPERS = {
    "a.md": "Paper A. We trained a U-Net on the public BBBC dataset of 40 "
            "images.",
    "b.md": "Paper B. A review of microscopy segmentation methods.",
    "c.md": "Paper C. We trained a ResNet classifier on private data.",
}


def _model(system: str, prompt: str, tokens: int | None) -> str:
    """Answer like a model, by the kind of prompt.

    Parameters
    ----------
    system: str
        The system prompt.
    prompt: str
        The user prompt.
    tokens: int | None
        Unused.

    Returns
    -------
    str
        Notes and checks for a document, a section, chosen numbers or a
        headline.

    """

    if "Document [1]" in prompt:
        return (
            "NOTE 1: Segments nuclei with deep learning.\n"
            "CHECK A: YES\nQUOTE A: trained a U-Net\nENTRY A: U-Net\n"
            "CHECK B: YES\nQUOTE B: the public BBBC dataset\n"
            "ENTRY B: BBBC | 40 images\n"
        )
    if "Document [2]" in prompt:
        return (
            "NOTE 1: Reviews segmentation methods.\n"
            "CHECK A: NO\nQUOTE A: A review\nCHECK B: NO\nQUOTE B: x\n"
        )
    if "Document [3]" in prompt:
        return (
            "NOTE 1: Classifies cells.\n"
            "CHECK A: YES\nQUOTE A: we trained a transformer\n"
            "ENTRY A: Transformer\nCHECK B: NO\nQUOTE B: private data\n"
        )
    if "Condense" in prompt:
        return "Deep learning for nuclei [1, 3]. Reviews [2]."
    if "NUMBERS" in prompt:
        return "NUMBERS: 2"
    if "Write the headline" in prompt:
        return "Deep learning dominates our microscopy papers."
    return "## Themes\n\nDeep learning for nuclei and cells [1, 3], " \
        "reviewed in [2]."


class _Project(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.templates = self.root / "templates"
        self.templates.mkdir()
        (self.templates / "lab_overview.md").write_text(LAB)
        self.chunks = [
            chunk(name[0], text, path=name) for name, text in PAPERS.items()
        ]

    def tearDown(self) -> None:
        self.folder.cleanup()

    def _run(self, question="write a lab overview of the project",
             model=_model, confirm=None, **settings):
        context = SimpleNamespace(
            project_root=self.root, chunks=self.chunks, progress=Mock(),
            done=Mock(), trace=SimpleNamespace(emit=Mock()),
            trace_turn_id=None, ask_model=Mock(side_effect=model),
            confirm=confirm, choose=None,
            card=lambda path: {"title": "Study A"} if path.name == "a.md"
            else None,
        )
        values = {"confirmation": "never", **settings}
        result = run(ProjectSummarySettings(**values),
                     ToolRequest(question, question), context)
        return result, context


class TestRun(_Project):

    def test_report(self) -> None:
        result, context = self._run()
        report = (self.root / result.summary["report"]).read_text()
        self.assertEqual(result.summary["report"],
                         "outputs/lab_overview_project_report.md")
        self.assertTrue(report.startswith(
            "# Deep learning dominates our microscopy papers\n\n"
            "**Documents:** 3 documents, listed at the end of the report "
            "[1–3].\n\n## Themes\n\nDeep learning for nuclei and cells "
            "[1, 3], reviewed in [2]."
        ))
        self.assertIn(
            "## Models\n\n1 of 3 documents. Unclear: [3] (the quoted words "
            "are not in the document).\n\n- **Study A**: U-Net [1]",
            report,
        )
        self.assertIn(
            "## Datasets\n\n1 of 3 documents.\n\n| Document | Dataset | "
            "Size |\n| --- | --- | --- |\n| Study A [1] | BBBC | 40 images |",
            report,
        )
        self.assertIn("## Start Here\n\n- b.md [2]", report)
        self.assertTrue(report.rstrip().endswith(
            "**Documents**\n\n[1] Study A (`a.md`)  \n[2] b.md  \n[3] c.md"
        ))
        self.assertTrue(result.summary["valid"])
        self.assertEqual([e["citation"] for e in result.evidence],
                         [1, 2, 3])
        self.assertEqual(result.evidence[0]["path"], Path("a.md"))
        self.assertIn("*Saved to `outputs/lab_overview_project_report.md`",
                      result.answer)
        # Three documents, one section, one choice, one headline.
        self.assertEqual(context.ask_model.call_count, 6)

    def test_failed_document_and_groups(self) -> None:
        def model(system, prompt, tokens):
            if "Document [2]" in prompt:
                raise RuntimeError("model stopped")
            return _model(system, prompt, tokens)
        with patch(
            "kilin.tools.project_summary._configuration.GROUP_TOKENS", 1
        ):
            result, context = self._run(model=model)
        report = (self.root / result.summary["report"]).read_text()
        self.assertEqual(result.summary["failed"], [2])
        self.assertIn("Not read: [2].", report)
        self.assertIn("Not read: [2].", result.answer)
        prompts = [c.args[1] for c in context.ask_model.call_args_list]
        self.assertEqual(sum("Condense" in p for p in prompts), 2)

    def test_citations_are_repaired_or_removed(self) -> None:
        def model(system, prompt, tokens):
            if "Section: Themes" in prompt:
                return "Networks are studied [n1], and plants [n3]."
            if "has a problem" in prompt:
                return "Networks are studied [1, 3], and plants [2]."
            return _model(system, prompt, tokens)
        result, context = self._run(model=model)
        report = (self.root / result.summary["report"]).read_text()
        self.assertIn("## Themes\n\nNetworks are studied [1, 3], and "
                      "plants [2].", report)
        fix = next(c.args[1] for c in context.ask_model.call_args_list
                   if "has a problem" in c.args[1])
        self.assertIn("[n1], [n3]", fix)
        self.assertTrue(result.summary["valid"])

        def stubborn(system, prompt, tokens):
            if "Section: Themes" in prompt or "has a problem" in prompt:
                return "Document 1 studies networks [n1] and plants [2]."
            return _model(system, prompt, tokens)
        result, _ = self._run(model=stubborn)
        report = (self.root / result.summary["report"]).read_text()
        self.assertIn("## Themes\n\nDocument 1 studies networks and plants "
                      "[2].", report)
        self.assertIn("Themes: 1 citation(s) of no document removed.",
                      result.answer)
        self.assertIn("names documents in the text", result.answer)
        self.assertFalse(result.summary["valid"])

    def test_citation_of_the_wrong_document_is_repaired(self) -> None:
        def model(system, prompt, tokens):
            if "Section: Themes" in prompt:
                return ("U-Net models segment nuclei [1], and transformers "
                        "classify cells [2].")
            if "has a problem" in prompt:
                return ("U-Net models segment nuclei [1], and transformers "
                        "classify cells [3].")
            return _model(system, prompt, tokens)
        result, context = self._run(model=model)
        fix = next(c.args[1] for c in context.ask_model.call_args_list
                   if "has a problem" in c.args[1])
        self.assertIn("“and transformers classify cells” cites [2], whose "
                      "notes do not mention it; it may be [3]", fix)
        report = (self.root / result.summary["report"]).read_text()
        self.assertIn("classify cells [3].", report)

    def test_headline_and_coverage(self) -> None:
        (self.templates / "lab_overview.md").write_text(LAB.replace(
            "## Themes (3-40 words)", "## Themes (3-40 words, all documents)"
        ).replace("(max 80 characters)", "(max 40 characters)"))

        def model(system, prompt, tokens):
            if "Section: Themes" in prompt:
                return "Nuclei are segmented with deep learning [1]."
            if "has a problem" in prompt:
                return "Nuclei are segmented [1], cells classified [3], " \
                    "and methods reviewed [2]."
            if "Write the headline" in prompt:
                return "Deep learning dominates the segmentation of nuclei " \
                    "in our papers"
            if "shorter headline" in prompt:
                return "Deep learning segments nuclei and the cells we study"
            return _model(system, prompt, tokens)
        result, context = self._run(model=model)
        fix = next(c.args[1] for c in context.ask_model.call_args_list
                   if "has a problem" in c.args[1])
        self.assertIn("leaves out [2], [3]", fix)
        report = (self.root / result.summary["report"]).read_text()
        self.assertIn("and methods reviewed [2].", report)
        # Still too long after asking: cut at a word, without a final
        # "and the".
        self.assertTrue(report.startswith(
            "# Deep learning segments nuclei\n"
        ))

    def test_cancel_and_quick(self) -> None:
        result, context = self._run(confirm=Mock(return_value="cancel"),
                                    confirmation="always")
        self.assertEqual(result.answer,
                         "The Lab Overview was not written (cancelled).")
        context.ask_model.assert_not_called()
        plan = context.confirm.call_args.args[0]
        self.assertEqual((plan.items, plan.batches), (3, 6))
        self.assertIn("3 documents of the project, 1 with a card",
                      plan.scope)
        result, _ = self._run(confirm=Mock(return_value="quick"),
                              confirmation="always")
        self.assertEqual(result.summary["choice"], "quick")

    def test_no_documents_or_templates(self) -> None:
        self.chunks = []
        result, _ = self._run()
        self.assertIn("no indexed documents", result.answer)
        (self.templates / "lab_overview.md").unlink()
        result, _ = self._run()
        self.assertIn("no project report template", result.answer)


class TestTemplates(_Project):

    def test_settings_and_registry(self) -> None:
        self.assertEqual(parse_settings({}, "x"), ProjectSummarySettings())
        for bad in ({"template": "../x"}, {"document_tokens": 10},
                    {"unknown": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_settings(bad, "x")
        self.assertIn("project_summary", TOOL_REGISTRY)
        shipped = {r.destination for r in workspace_resources(
            ProjectSummarySettings())}
        self.assertEqual(shipped, {"templates/project_overview.md",
                                   "templates/literature_table.md"})

    def test_shipped_templates(self) -> None:
        for path in SHIPPED_TEMPLATES_DIR.glob("*.md"):
            shutil.copy(path, self.templates)
        found = {name: template for name, template, error in
                 project_templates(self.templates)}
        self.assertEqual(set(found),
                         {"lab_overview", "literature_table",
                          "project_overview"})
        self.assertTrue(all(found.values()))

    def test_choose(self) -> None:
        self.assertEqual(
            choose_template("write the lab overview", self.templates,
                            TEMPLATE_CHOOSE).name, "lab_overview",
        )
        self.assertEqual(
            choose_template("summarize the project", self.templates,
                            TEMPLATE_CHOOSE).name, "lab_overview",
        )
        shutil.copy(SHIPPED_TEMPLATES_DIR / "project_overview.md",
                    self.templates)
        choose = Mock(return_value=1)
        self.assertEqual(
            choose_template("summarize the project", self.templates,
                            TEMPLATE_CHOOSE, choose).name,
            "project_overview",
        )
        with self.assertRaisesRegex(ValueError, "Which project report"):
            choose_template("summarize the project", self.templates,
                            TEMPLATE_CHOOSE)

    def test_routing(self) -> None:
        found = routing(ProjectSummarySettings(), self.root)
        self.assertIn("Lab Overview (lab_overview): What our papers say "
                      "together", found.description)
        cues = [re.compile(cue, re.I) for cue in found.cues]
        self.assertTrue(any(c.search("write a lab overview") for c in cues))
        for type_id in ("literature", "writing"):
            project = with_project_routing(project_type(type_id), self.root)
            for question, intent in (
                ("write a lab overview of the project",
                 "templated_project_report"),
                ("summarize the whole project", "templated_project_report"),
                ("make a literature table of all the papers",
                 "templated_project_report"),
                ("wrap up everything in the collection",
                 "templated_project_report"),
                ("how many papers are in this project?", None),
            ):
                with self.subTest(type_id=type_id, question=question):
                    self.assertEqual(_cued_intent(project, question),
                                     intent)


class TestReduce(unittest.TestCase):

    def test_code_written_parts(self) -> None:
        documents = [
            ProjectDocument(n, Path(f"{n}.md"), f"Doc {n}", (), None)
            for n in (1, 2, 3, 4)
        ]
        section = TemplateSection("Models", when="x", do="y",
                                  source="per_document")
        notes = {
            1: DocumentNotes(1, checks={"Models": Check("yes", entry="A")}),
            2: DocumentNotes(2, checks={"Models": Check("no")}),
            3: DocumentNotes(3, checks={"Models": Check(
                "unclear", why="no answer")}),
            4: DocumentNotes(4, failure="stopped"),
        }
        self.assertEqual(
            per_document_markdown(section, documents, notes),
            "1 of 4 documents. Unclear: [3] (no answer). Not read: [4]."
            "\n\n- **Doc 1**: A [1]",
        )
        notes[1] = DocumentNotes(1, checks={"Models": Check("no")})
        self.assertEqual(per_document_markdown(section, documents, notes),
                         "None of the 4 documents. Unclear: [3] (no "
                         "answer). Not read: [4].")
        self.assertEqual(documents_line(documents[:1]),
                         "1 document, listed at the end of the report [1].")
        self.assertIn("[4] Doc 4 (`4.md`)", documents_list(documents))
        self.assertEqual(chosen_numbers("I pick 4 and 2", 4, 1), [4])
        for content, cite, problem in (
            ("Shown [1, 3] and [2-3].", True, ""),
            ("Shown [n1] and [n2].", True, "[n1], [n2]"),
            ("Shown [6].", True, "[6]"),
            ("Document 4 and Documents 1 and 2 agree.", False,
             "names documents in the text"),
            ("Shown without citations.", True, "cites no document"),
            ("Shown without citations.", False, ""),
        ):
            with self.subTest(content=content):
                found = citation_problem(content, 5, cite)
                if problem:
                    self.assertIn(problem, found)
                else:
                    self.assertEqual(found, "")
        self.assertEqual(
            clean_citations("A [n1], b [2] and c [7].", 5),
            ("A, b [2] and c.", 2),
        )
        # A real report's Overview: [4] is BRAPH 2, the genomics is [5].
        texts = {
            1: "BRAPH: A graph theory software for the analysis of brain "
               "connectivity; freeware MATLAB, network measures",
            2: "Computational memory capacity predicts aging; reservoir "
               "computing on brain connectivity, Cam-CAN",
            3: "Deep-Learning Investigation of Vibrational Raman Spectra "
               "for Plant-Stress Analysis; variational autoencoder",
            4: "BRAPH 2 framework for network analyses in neurosciences; "
               "Genesis compiler, distributions, brain connectivity",
            5: "Genomic results for psychiatric disorders and brain cell "
               "types; genome-wide association studies, schizophrenia",
        }
        overview = (
            "The table covers graph theory software for brain connectivity "
            "[1], computational memory capacity in aging [2], deep learning "
            "for plant stress analysis [3], and genomic analysis of "
            "psychiatric disorders [4]."
        )
        self.assertIn(
            "“and genomic analysis of psychiatric disorders” cites [4], "
            "whose notes do not mention it; it may be [5]",
            support_problem(overview, texts),
        )
        self.assertEqual(
            support_problem(overview.replace("[4]", "[5]"), texts), ""
        )
        # Too few distinctive words to judge, or no better document.
        self.assertEqual(support_problem("As shown [4].", texts), "")
        self.assertEqual(
            support_problem("Quantum chromodynamics lattices [4].", texts),
            "",
        )
        # A real report's "Agreements and Disagreements".
        self.assertIn(
            "cites only [1]",
            agreement_problem(
                "The documents agree on the use of non-parametric "
                "permutation tests [1], and on memory capacity [2]."
            ),
        )
        self.assertIn("cites only [4]", agreement_problem(
            "They agree that X [1, 2]. However, they disagree on the "
            "applications of BRAPH 2 [4]."
        ))
        for text in ("They agree that thresholds matter [1, 4].",
                     "They concur on reproducibility [2-4].",
                     "BRAPH 2 differs from BRAPH [1] in its GUI [4].",
                     "No document disagrees with another."):
            with self.subTest(text=text):
                self.assertEqual(agreement_problem(text), "")
        self.assertEqual(coverage_problem("A [1, 3] and [2].", [1, 2, 3]),
                         "")
        self.assertIn("leaves out [3], [5]",
                      coverage_problem("A [1, 4] and [2].", [1, 2, 3, 4, 5]))
        self.assertEqual(
            fitted_headline("BRAPH tools enable network analysis of brain "
                            "connectivity in neurological conditions "
                            "through graph theory and statistics", 110),
            "BRAPH tools enable network analysis of brain connectivity in "
            "neurological conditions through graph theory",
        )
        self.assertEqual(fitted_headline("Short.", 20), "Short")
        groups = note_groups([(1, "a " * 30), (2, "b"), (3, "c")], 10)
        self.assertEqual([[n for n, _ in g] for g in groups], [[1], [2, 3]])


if __name__ == "__main__":
    unittest.main()
