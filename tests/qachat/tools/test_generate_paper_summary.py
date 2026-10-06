"""Check generate_paper_summary: paper choice, pieces, fixes, saving."""

from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from qachat.report_templates import load_template, report_skeleton
from qachat.repository import scan_repository
from qachat.tools import ToolRequest
from qachat.tools.generate_paper_summary import (
    SummarySettings,
    _content,
    choose_template,
    parse_settings,
    reference_list,
    run,
    summary_file_text,
    ungrounded,
)
from qachat.tools.load_target import (
    load_target,
    save_target_file,
    target_folder,
)
from tests.qachat.tools.helpers import chunk, context

_SUMMARY = load_template("research_summary")
_BRIEFING = load_template("research_briefing")


def _words(count: int) -> str:
    return " ".join(["word"] * (count - 1)) + " end [1]."


_CITATION = "A paper: We used Python 3.11 and it works [1]."


def _fitting(name: str) -> str:
    """Content that fits a briefing section's limit."""

    if name == "This is a summary of":
        return _CITATION
    section = next(s for s in _BRIEFING.sections if s.name == name)
    if section.items:
        return "- First paper [1]\n- Second paper [1]"
    return _words(sum(section.words) // 2)


class TestGeneratePaperSummary(unittest.TestCase):
    """The model writes pieces; Python checks, fixes once, and saves."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        target = target_folder(self.root)
        (target / "my_paper.md").write_text(
            "# A paper\n\nWe used Python 3.11.\n\n## Results\n\nIt works.\n"
        )

    def summarize(self, question, write, settings=None, chunks=()):
        """Run the tool; ``write(kind, section)`` returns each reply."""

        calls = []

        def model(system, prompt, tokens):
            if "take notes" in system:
                calls.append(("notes", None))
                return "- Uses Python 3.11 [1]"
            if "write headlines" in system:
                calls.append(("headline", None))
                return write("headline", None)
            name = re.search(r'"(.+?)" section', prompt).group(1)
            kind = "fix" if "revise one section" in system else "section"
            calls.append((kind, name))
            return write(kind, name)

        built, events = context(chunks, model=model, root=self.root)
        result = run(
            settings or SummarySettings(),
            ToolRequest(question, question),
            built,
        )
        return result, calls, events

    def test_target_paper_is_written_checked_and_saved(self) -> None:
        result, calls, _ = self.summarize(
            "summarize my paper with the template",
            lambda kind, name: _CITATION
            if name == "Citation"
            else f"## {name}\nText [1].",
        )
        self.assertTrue(result.summary["valid"], result.summary["problems"])
        self.assertEqual(result.summary["repairs"], 0)
        self.assertEqual(result.summary["paper"], "target/my_paper.md")
        self.assertEqual(result.summary["saved"], "target/my_paper_summary.md")
        saved = (self.root / "target/my_paper_summary.md").read_text()
        self.assertTrue(saved.startswith("# Research Summary\n\n*Source:"))
        self.assertNotIn("[1]", saved)  # Passage numbers stay in QAChat.
        self.assertNotIn("## Citation\n## Citation", saved)
        # Checks are kept with the turn; the answer stays a clean document.
        self.assertEqual(result.summary["problems"], [])
        self.assertNotIn("Template check", result.answer)
        # One part of notes, then one call per section.
        self.assertEqual(len(calls), 1 + len(_SUMMARY.sections))

    def test_pdf_is_saved_next_to_the_markdown(self) -> None:
        written = []

        def write_pdf(markdown, path, label=""):
            written.append((path.name, label, markdown.splitlines()[0]))
            path.write_bytes(b"%PDF")
            return 1

        def model(system, prompt, tokens):
            if "take notes" in system:
                return "- Uses Python 3.11 [1]"
            return _CITATION

        built, _ = context(model=model, root=self.root, write_pdf=write_pdf)
        result = run(
            SummarySettings(), ToolRequest("my paper", "my paper"), built
        )
        self.assertEqual(
            result.summary["saved_pdf"], "target/my_paper_summary.pdf"
        )
        self.assertEqual(
            written,
            [
                (
                    "my_paper_summary.pdf",
                    "Research Summary",
                    "# Research Summary",
                )
            ],
        )
        self.assertIn("and `target/my_paper_summary.pdf`", result.answer)

    def test_briefing_section_outside_its_limit_is_fixed_once(self) -> None:
        def write(kind, name):
            if kind == "headline":
                return "# Plants show stress before it is visible"
            if kind == "section" and name == "The Reason":
                return _words(20)
            return _fitting(name)

        settings = SummarySettings(template="research_briefing")
        result, calls, events = self.summarize("my paper", write, settings)
        self.assertTrue(result.summary["valid"], result.summary["problems"])
        self.assertEqual(result.summary["repairs"], 1)
        self.assertIn(("fix", "The Reason"), calls)
        fixing = [m for _, m, _ in events if m.startswith("Fixing:")]
        self.assertEqual(
            fixing,
            ["Fixing: The Reason has 20 words; it should have 120-150."],
        )
        self.assertTrue(
            result.answer.startswith(
                "# Plants show stress before it is visible"
            )
        )

    def test_still_outside_after_a_fix_is_reported(self) -> None:
        def write(kind, name):
            if kind == "headline":
                return "x" * 120
            if name == "The Reason":
                return _words(20)
            return _fitting(name)

        settings = SummarySettings(template="research_briefing")
        result, calls, _ = self.summarize("my paper", write, settings)
        self.assertFalse(result.summary["valid"])
        self.assertEqual(calls.count(("headline", None)), 2)
        problems = " ".join(result.summary["problems"])
        self.assertIn("The headline has 120 characters", problems)
        self.assertIn("The Reason has 20 words", problems)
        self.assertNotIn("The Reason has 20 words", result.answer)

    def test_references_must_come_from_the_reference_list(self) -> None:
        paper = [
            chunk("repo/p.pdf", ["Title. Smith, A. et al.", "Methods."]),
            chunk(
                "repo/p.pdf",
                [
                    "References",
                    "1. Jones, B. Raman spectroscopy of leaves. Plant J. 2019",
                    "2. Lee, C. Deep learning for spectra. Nat. Methods 2021",
                ],
                start=3,
            ),
        ]
        references = reference_list(tuple(paper))
        self.assertTrue(references.startswith("1. Jones"))
        real = "- Jones, B. Raman spectroscopy of leaves. Plant J. 2019"
        invented = "- Unsupervised analysis reveals plant stress responses"
        self.assertEqual(ungrounded(real, references, items=True), [])
        self.assertEqual(
            ungrounded(f"{real}\n{invented}", references, items=True), [2]
        )

        def write(kind, name):
            if kind == "headline":
                return "Leaves under stress"
            if name == "Related Reviews and Literature":
                if kind == "section":
                    return f"{real}\n{invented}"
                return (
                    f"{real}\n- Lee, C. Deep learning for spectra. "
                    "Nat. Methods 2021"
                )
            if name == "This is a summary of":
                return "Smith, A. et al. Title."
            return _fitting(name)

        settings = SummarySettings(template="research_briefing")
        result, calls, _ = self.summarize(
            "turn p.pdf into a research briefing", write, settings, paper
        )
        self.assertIn(("fix", "Related Reviews and Literature"), calls)
        self.assertTrue(result.summary["valid"], result.summary["problems"])
        self.assertNotIn("Unsupervised analysis", result.answer)

    def test_reference_numbers_are_not_passage_citations(self) -> None:
        section = next(
            s for s in _BRIEFING.sections if s.source == "references"
        )
        self.assertEqual(
            _content("- [2] Jones, B. Leaves. 2019\n- [5] Lee", section),
            "- Jones, B. Leaves. 2019\n- Lee",
        )

    def test_that_paper_follows_the_author_named_earlier(self) -> None:
        chunks = [
            chunk("repo/smith2020.pdf", ["Smith studies plants."]),
            chunk("repo/README.md", ["See Smith et al. 2020 for plants."]),
        ]
        asked = []
        built, _ = context(
            chunks,
            model=lambda system, prompt, tokens: "Text [1].",
            root=self.root,
            confirm=lambda plan: asked.append(plan) or "continue",
        )
        # "that paper" carries the earlier question as the user's words.
        request = ToolRequest(
            "can you turn that paper into a research summary",
            "can you turn that paper into a research summary "
            "what is smith's paper about?",
        )
        (self.root / "target" / "my_paper.md").unlink()
        result = run(SummarySettings(), request, built)
        self.assertEqual(result.summary["paper"], "repo/smith2020.pdf")
        self.assertEqual(result.summary["reason"], "named source file")
        self.assertEqual(asked, [])  # Named, so no confirmation needed.

    def test_a_guess_prefers_pdfs_and_is_always_confirmed(self) -> None:
        from qachat.retrievals import HybridResult

        readme = chunk("repo/README.md", ["Deep plants and spectra."])
        paper = chunk("repo/deep2024.pdf", ["Spectra of plants."])
        ranked = [
            HybridResult(score=0.5, chunk=readme, channels=("x",)),
            HybridResult(score=0.1, chunk=paper, channels=("x",)),
        ]
        asked = []
        built, _ = context(
            [readme, paper],
            model=lambda system, prompt, tokens: "Text [1].",
            rank=lambda text: ranked,
            root=self.root,
            confirm=lambda plan: asked.append(plan) or "cancel",
        )
        (self.root / "target" / "my_paper.md").unlink()
        question = "turn the plant spectra work into a research summary"
        result = run(
            SummarySettings(), ToolRequest(question, question), built
        )
        self.assertEqual(result.summary["paper"], "repo/deep2024.pdf")
        self.assertEqual(len(asked), 1)
        self.assertIn("chose this paper", " ".join(asked[0].details))
        self.assertIn("Cancelled", result.answer)

    def test_briefing_citation_is_one_line_under_the_headline(self) -> None:
        settings = SummarySettings(template="research_briefing")

        def write(kind, name):
            if kind == "headline":
                return "Plants show stress early"
            return _fitting(name)

        result, _, _ = self.summarize("my paper", write, settings)
        self.assertIn(
            "# Plants show stress early\n\n**This is a summary of:** A paper",
            result.answer,
        )
        self.assertNotIn("## This is a summary of", result.answer)

    def test_source_paper_named_in_the_question(self) -> None:
        chunks = [chunk("repo/smith2020.pdf", ["Smith used R."])]
        result, _, _ = self.summarize(
            "turn smith2020.pdf into a research summary",
            lambda kind, name: "Text [1].",
            chunks=chunks,
        )
        self.assertEqual(result.summary["paper"], "repo/smith2020.pdf")
        self.assertEqual(result.summary["reason"], "named source file")
        self.assertTrue((self.root / "target/smith2020_summary.md").exists())

    def test_template_is_chosen_from_the_question(self) -> None:
        settings = parse_settings(
            {
                "template": "research_briefing",
                "templates": ["research_summary"],
            },
            "x",
        )
        self.assertEqual(
            choose_template("make a research summary of it", settings),
            "research_summary",
        )
        self.assertEqual(
            choose_template("summarize my paper", settings),
            "research_briefing",
        )

    def test_settings_and_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown report template"):
            parse_settings({"template": "nope"}, "x")
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            parse_settings({"summary_tokens": 9}, "x")
        with self.assertRaisesRegex(FileNotFoundError, "my_paper.md"):
            load_target(self.root, "missing.pdf")
        with self.assertRaisesRegex(ValueError, "cannot be read"):
            load_target(self.root, "image.png")

    def test_saving_never_overwrites(self) -> None:
        first = save_target_file(self.root, "a_summary.md", "one")
        second = save_target_file(self.root, "a_summary.md", "two")
        self.assertEqual(first.as_posix(), "target/a_summary.md")
        self.assertEqual(second.as_posix(), "target/a_summary-2.md")
        self.assertEqual((self.root / first).read_text(), "one")

    def test_target_folder_is_never_indexed(self) -> None:
        sources = self.root / "sources"
        sources.mkdir()
        (sources / "paper.md").write_text("# Indexed\n")
        report = scan_repository(sources, extensions=(".md",))
        self.assertEqual(report["selected_files"], ["paper.md"])
        self.assertTrue((self.root / "target/my_paper.md").exists())

    def test_file_text_keeps_title_first(self) -> None:
        text = summary_file_text(
            report_skeleton(_SUMMARY) + "\nx [3]", Path("p.pdf")
        )
        self.assertTrue(
            text.startswith("# Research Summary\n\n*Source: p.pdf")
        )
        self.assertNotIn("[3]", text)


if __name__ == "__main__":
    unittest.main()
