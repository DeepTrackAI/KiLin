"""Check generate_paper_summary: paper choice, pieces, fixes, saving."""

from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.tools import EarlierTurn, ToolRequest
from qachat.tools.generate_paper_summary import (
    SummarySettings,
    _content,
    _fit_total,
    _trim_words,
    choose_template,
    paper_citation,
    parse_settings,
    reference_entries,
    run,
    summary_file_text,
)
from qachat.tools.outputs import (
    load_output,
    output_folder,
    save_output_file,
)
from qachat.tools.report_template import load_template, templates_folder
from tests.qachat.tools.helpers import chunk, context

_BRIEFING = load_template("research_briefing")
# A user's own template, as saved in a project's templates/ folder.
_LAB_SUMMARY = """# Lab Summary

## Background
Why the work was done.

## Findings
What it found.
"""
_JONES = "Jones, B. Raman spectroscopy of leaves. Plant J. 12, 3 (2019)."
_LEE = "Lee, C. Deep learning for spectra. Nat. Methods 18, 9 (2021)."


def _words(count: int) -> str:
    return " ".join(["word"] * (count - 1)) + " end [1]."


def _fitting(name: str) -> str:
    """Content that fits a briefing section's limit."""

    section = next(s for s in _BRIEFING.sections if s.name == name)
    return _words(sum(section.words) // 2)


class TestGeneratePaperSummary(unittest.TestCase):
    """The model writes pieces; Python checks, fixes, copies, and saves."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (output_folder(self.root) / "my_paper.md").write_text(
            "# A paper\n\nWe used Python 3.11.\n\n## Results\n\nIt works.\n\n"
            f"## References\n\n1. {_JONES}\n2. {_LEE}\n"
        )
        (templates_folder(self.root) / "lab_summary.md").write_text(
            _LAB_SUMMARY
        )

    def summarize(self, question, write, settings=None, chunks=()):
        """Run the tool; ``write(kind, section)`` returns each reply."""

        calls = []

        def model(system, prompt, tokens):
            if "take notes" in system:
                calls.append(("notes", None))
                return "- Uses Python 3.11 for leaf spectroscopy [1]"
            if "write headlines" in system:
                calls.append(("headline", None))
                return write("headline", None)
            name = re.search(r'"(.+?)" section', prompt).group(1)
            kind = (
                "fix"
                if "revise one section" in system
                else "references"
                if "choose references" in system
                else "section"
            )
            calls.append((kind, name))
            return write(kind, name)

        built, events = context(chunks, model=model, root=self.root)
        result = run(
            settings or SummarySettings(),
            ToolRequest(question, question),
            built,
        )
        return result, calls, events

    def briefing(self, write, question="my paper"):
        """Summarize my_paper.md as a briefing; references pick 1 and 2."""

        def reply(kind, name):
            if kind == "references":
                return "2, 1"
            return write(kind, name)

        return self.summarize(question, reply)

    def test_outputs_paper_is_written_checked_and_saved(self) -> None:
        result, calls, _ = self.summarize(
            "summarize my paper with the lab summary template",
            lambda kind, name: f"## {name}\nText [1].",
        )
        self.assertTrue(result.summary["valid"], result.summary["problems"])
        self.assertEqual(result.summary["template"], "lab_summary")
        self.assertEqual(result.summary["paper"], "outputs/my_paper.md")
        self.assertEqual(
            result.summary["saved"], "outputs/my_paper_summary.md"
        )
        saved = (self.root / "outputs/my_paper_summary.md").read_text()
        self.assertTrue(saved.startswith("# Lab Summary\n\n## Background"))
        self.assertTrue(saved.rstrip().endswith("against the paper.*"))
        # The answer says where the files are, then shows the file itself.
        self.assertTrue(
            result.answer.startswith(
                "Done: the lab summary of my_paper.md is saved as "
                "`outputs/my_paper_summary.md`."
            )
        )
        self.assertIn(saved.strip(), result.answer)
        self.assertNotIn("[1]", saved)  # Passage numbers stay in QAChat.
        self.assertNotIn("## Background\n## Background", saved)
        self.assertEqual(result.summary["problems"], [])
        # One part of notes, then one call per section.
        self.assertEqual(len(calls), 3)

    def test_pdf_is_saved_next_to_the_markdown(self) -> None:
        written = []

        def write_pdf(markdown, path, label=""):
            written.append((path.name, label, markdown.splitlines()[0]))
            path.write_bytes(b"%PDF")
            return 1

        def model(system, prompt, tokens):
            if "take notes" in system:
                return "- Uses Python 3.11 [1]"
            return "Text [1]."

        built, _ = context(model=model, root=self.root, write_pdf=write_pdf)
        question = "my paper as a lab summary"
        result = run(
            SummarySettings(), ToolRequest(question, question), built
        )
        self.assertEqual(
            result.summary["saved_pdf"], "outputs/my_paper_summary.pdf"
        )
        self.assertEqual(
            written,
            [("my_paper_summary.pdf", "Lab Summary", "# Lab Summary")],
        )
        self.assertIn("and `outputs/my_paper_summary.pdf`", result.answer)

    def test_briefing_section_outside_its_limit_is_fixed(self) -> None:
        def write(kind, name):
            if kind == "headline":
                return "# Plants show stress before it is visible"
            if kind == "section" and name == "The Reason":
                return _words(20)
            return _fitting(name)

        result, calls, events = self.briefing(write)
        self.assertTrue(result.summary["valid"], result.summary["problems"])
        self.assertEqual(result.summary["repairs"], 1)
        self.assertIn(("fix", "The Reason"), calls)
        fixing = [m for _, m, _ in events if m.startswith("Fixing:")]
        self.assertEqual(
            fixing,
            ["Fixing: The Reason has 20 words; it should have 120-150."],
        )
        self.assertIn(
            "\n# Plants show stress before it is visible\n", result.answer
        )

    def test_still_outside_after_a_fix_is_reported(self) -> None:
        def write(kind, name):
            if kind == "headline":
                return "x" * 120
            if name == "The Reason":
                return _words(20)
            return _fitting(name)

        result, calls, _ = self.briefing(write)
        self.assertFalse(result.summary["valid"])
        self.assertEqual(calls.count(("headline", None)), 2)
        problems = " ".join(result.summary["problems"])
        self.assertIn("The headline has 120 characters", problems)
        self.assertIn("The Reason has 20 words", problems)
        self.assertNotIn("The Reason has 20 words", result.answer)

    def test_citation_and_references_are_copied_never_written(self) -> None:
        def write(kind, name):
            return "Plants under stress" if kind == "headline" else (
                _fitting(name)
            )

        result, calls, _ = self.briefing(write)
        # The model only chose entry numbers; the entries are verbatim.
        self.assertIn(f"- {_LEE}\n- {_JONES}", result.answer)
        self.assertIn(("references", "Related Reviews and Literature"), calls)
        self.assertNotIn(("section", "This is a summary of"), calls)
        self.assertIn(
            "# Plants under stress\n\n**This is a summary of:** my_paper.md "
            "(the file has no title metadata).",
            result.answer,
        )
        self.assertNotIn("## This is a summary of", result.answer)

    def test_reference_numbers_outside_the_list_are_ignored(self) -> None:
        def write(kind, name):
            if kind == "references":
                return "There is no good match, maybe 9."
            return "Leaves" if kind == "headline" else _fitting(name)

        result, _, _ = self.summarize("my paper", write)
        # Too few valid numbers: the closest entries to the notes fill in.
        section = result.answer.split("## Related Reviews and Literature")[1]
        self.assertEqual(section.count("\n- "), 2)
        self.assertIn(_JONES, section)
        self.assertNotIn("maybe", section)

    def test_reference_entries_are_split_as_printed(self) -> None:
        paper = (
            chunk("p.pdf", ["Title. Smith, A. et al.", "Methods."]),
            chunk(
                "p.pdf",
                [
                    "References",
                    "1. Jones, B. Raman spectro-",
                    "scopy of leaves. Plant J. 12, 3 (2019).",
                    "[2] Lee, C. Deep learning for spectra. Nat. Methods",
                    "18, 9 (2021).",
                    "Page 12",
                ],
                start=3,
            ),
        )
        self.assertEqual(
            reference_entries(paper),
            [_JONES, _LEE],
        )
        self.assertEqual(reference_entries((paper[0],)), [])

    def test_running_heads_stay_out_of_reference_entries(self) -> None:
        head = ["I.-u. Tiunn et al.", "SoftwareX 35 (2026) 102791"]
        paper = tuple(
            chunk("p.pdf", head + [f"Body text of page {n}."], start=n * 10)
            for n in range(1, 4)
        ) + (
            chunk(
                "p.pdf",
                ["References", "1. Jones, B. Raman spectroscopy of",
                 *head, "leaves. Plant J. 12, 3 (2019).",
                 f"2. {_LEE}"],
                start=50,
            ),
        )
        self.assertEqual(reference_entries(paper), [_JONES, _LEE])

    def test_missing_reference_list_is_said_not_invented(self) -> None:
        (self.root / "outputs" / "my_paper.md").write_text(
            "# A paper\n\nIt works.\n"
        )
        result, calls, _ = self.summarize(
            "my paper", lambda kind, name: "T" if kind == "headline" else (
                _fitting(name)
            )
        )
        self.assertNotIn(("references", "Related Reviews and Literature"),
                         calls)
        self.assertIn("reference list could not be read", result.answer)
        self.assertFalse(result.summary["valid"])

    def test_citation_comes_from_metadata(self) -> None:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.add_metadata(
            {
                "/Title": "Deep spectra of stressed plants",
                "/Author": "Anoop C. Patil; Yu-Wei Chang",
            }
        )
        pdf = self.root / "paper.pdf"
        with pdf.open("wb") as file:
            writer.write(file)
        first = chunk(
            "paper.pdf",
            ["https://doi.org/10.1000/xyz123", "Accepted: 6 March 2025"],
        )
        self.assertEqual(
            paper_citation(pdf, (first,)),
            "Patil, A. C. et al. Deep spectra of stressed plants. "
            "https://doi.org/10.1000/xyz123 (2025).",
        )
        self.assertEqual(
            paper_citation(self.root / "missing.pdf", ()),
            "missing.pdf (the file has no title metadata).",
        )

    def paper_for(self, chunks, request, **options):
        """Run the tool up to the plan; return the summary and the asks."""

        asked = []
        built, _ = context(
            chunks,
            model=lambda system, prompt, tokens: "Text [1].",
            root=self.root,
            confirm=lambda plan: asked.append(plan) or "cancel",
            **options,
        )
        (self.root / "outputs" / "my_paper.md").unlink(missing_ok=True)
        return run(SummarySettings(), request, built), asked

    def test_this_follows_the_chain_back_to_the_named_paper(self) -> None:
        # The user's conversation: walk through Tiunn's paper, summarize
        # "that" (evidence split over two papers), then "turn this into a
        # research briefing". Patil's paper matches the words best.
        tiunn, mijalkov = Path("repo/tiunn26.pdf"), Path("repo/mijalkov25.pdf")
        chunks = [
            chunk("repo/tiunn26.pdf", ["BRAPH 2 software."]),
            chunk("repo/mijalkov25.pdf", ["Memory capacity."]),
            chunk("repo/patil25deep.pdf", ["Research on plant stress."]),
        ]
        request = ToolRequest(
            "turn this into research briefing",
            "turn this into research briefing",
            earlier=(
                EarlierTurn(4, "can you turn that into a summary report",
                            (tiunn,) * 7 + (mijalkov,) * 7),
                EarlierTurn(3, "can you walk me through Tiunn's paper",
                            (tiunn,) * 36),
            ),
        )
        result, _ = self.paper_for(chunks, request)
        self.assertEqual(result.summary["paper"], "repo/tiunn26.pdf")
        self.assertEqual(result.summary["reason"], "named in turn 3")

    def test_this_means_the_paper_an_earlier_turn_used(self) -> None:
        chunks = [
            chunk("repo/a.pdf", ["A."]),
            chunk("repo/b.pdf", ["B."]),
        ]
        request = ToolRequest(
            "make a briefing of it",
            "make a briefing of it",
            earlier=(
                EarlierTurn(2, "what are the main results?",
                            (Path("repo/b.pdf"),) * 5 + (Path("repo/a.pdf"),)),
            ),
        )
        result, asked = self.paper_for(chunks, request)
        self.assertEqual(result.summary["paper"], "repo/b.pdf")
        self.assertEqual(
            result.summary["reason"], "the file discussed in turn 2"
        )

    def test_unclear_paper_is_asked_not_guessed(self) -> None:
        from qachat.retrievals import HybridResult

        papers = [chunk(f"repo/{n}.pdf", ["Spectra."]) for n in "ab"]
        ranked = [
            HybridResult(score=0.5, chunk=c, channels=("x",)) for c in papers
        ]
        questions = []

        def choose(question, options):
            questions.append((question, options))
            return 1

        request = ToolRequest(
            "turn this into a research briefing",
            "turn this into a research briefing",
        )
        result, _ = self.paper_for(
            papers, request, rank=lambda text: ranked, choose=choose
        )
        (question, options), = questions
        self.assertEqual(
            question, "Which paper should I turn into a research briefing?"
        )
        self.assertEqual(
            options,
            ("a.pdf (best match for the question)",
             "b.pdf (matches the question)"),
        )
        self.assertEqual(result.summary["paper"], "repo/b.pdf")
        cancelled, _ = self.paper_for(
            papers, request, rank=lambda text: ranked,
            choose=lambda question, options: None,
        )
        # A cancel asks the user to name the paper, listing candidates.
        self.assertIn("Tell me which one you mean", cancelled.answer)
        self.assertIn("a.pdf, b.pdf", cancelled.answer)
        # Even one candidate is asked about, not assumed.
        asked = []
        single, _ = self.paper_for(
            papers[:1], request, rank=lambda text: ranked[:1],
            choose=lambda question, options: asked.append(options) or 0,
        )
        self.assertEqual(asked, [("a.pdf (best match for the question)",)])
        self.assertEqual(single.summary["paper"], "repo/a.pdf")

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
        (self.root / "outputs" / "my_paper.md").unlink()
        question = "turn the plant spectra work into a research briefing"
        result = run(
            SummarySettings(), ToolRequest(question, question), built
        )
        self.assertEqual(result.summary["paper"], "repo/deep2024.pdf")
        self.assertEqual(len(asked), 1)
        self.assertIn("chose this paper", " ".join(asked[0].details))
        self.assertIn("Cancelled", result.answer)

    def test_source_paper_named_in_the_question(self) -> None:
        chunks = [chunk("repo/smith2020.pdf", ["Smith used R."])]
        result, _, _ = self.summarize(
            "turn smith2020.pdf into a lab summary",
            lambda kind, name: "Text [1].",
            chunks=chunks,
        )
        self.assertEqual(result.summary["paper"], "repo/smith2020.pdf")
        self.assertEqual(result.summary["reason"], "named in the question")
        self.assertTrue(
            (self.root / "outputs/smith2020_summary.md").exists()
        )

    def test_template_is_chosen_from_the_question(self) -> None:
        folder = self.root / "templates"
        self.assertEqual(
            choose_template(
                "make a lab summary of it", folder, "research_briefing"
            ).name,
            "lab_summary",
        )
        self.assertEqual(
            choose_template("summarize it", folder, "research_briefing").name,
            "research_briefing",
        )
        # A deleted default falls back to the shipped template.
        (folder / "research_briefing.md").unlink()
        self.assertEqual(
            choose_template("summarize it", folder, "research_briefing").name,
            "research_briefing",
        )
        # A broken template is reported when it is asked for.
        (folder / "broken.md").write_text("## A (sixty words)\n")
        with self.assertRaisesRegex(ValueError, "broken, line 1"):
            choose_template("use the broken template", folder, "lab_summary")
        result, _, _ = self.summarize(
            "my paper with the broken template", lambda kind, name: "x"
        )
        self.assertIn("cannot read the limit", result.answer)

    def test_settings_and_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "template file name"):
            parse_settings({"template": "../nope"}, "x")
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            parse_settings({"templates": ["a"]}, "x")
        with self.assertRaisesRegex(FileNotFoundError, "my_paper.md"):
            load_output(self.root, "missing.pdf")
        with self.assertRaisesRegex(ValueError, "cannot be read"):
            load_output(self.root, "image.png")

    def test_saving_never_overwrites(self) -> None:
        first = save_output_file(self.root, "a_summary.md", "one")
        second = save_output_file(self.root, "a_summary.md", "two")
        self.assertEqual(first.as_posix(), "outputs/a_summary.md")
        self.assertEqual(second.as_posix(), "outputs/a_summary-2.md")
        self.assertEqual((self.root / first).read_text(), "one")

    def test_outputs_are_never_indexed_and_target_moves(self) -> None:
        sources = self.root / "sources"
        sources.mkdir()
        (sources / "paper.md").write_text("# Indexed\n")
        report = scan_repository(sources, extensions=(".md",))
        self.assertEqual(report["selected_files"], ["paper.md"])
        with TemporaryDirectory() as other:
            old = Path(other) / "target"
            old.mkdir()
            (old / "draft.md").write_text("Draft")
            folder = output_folder(Path(other))
            self.assertEqual((folder / "draft.md").read_text(), "Draft")
            self.assertFalse(old.exists())

    def test_long_sections_lose_trailing_sentences(self) -> None:
        text = "One two three. Four five six. Seven eight nine [2]."
        self.assertEqual(_trim_words(text, 6), "One two three. Four five six.")
        self.assertEqual(_trim_words(text, 1), "One two three.")
        written = [
            ("The Reason", " ".join(["Long sentence here."] * 60)),
            ("The Future", "Short."),
        ]
        fitted = _fit_total(written, _BRIEFING)
        self.assertLessEqual(
            sum(len(text.split()) for _, text in fitted), 750
        )
        self.assertEqual(fitted[1], ("The Future", "Short."))

    def test_inline_section_is_one_line_without_its_label(self) -> None:
        section = next(s for s in _BRIEFING.sections if s.inline)
        reply = (
            "**This is a summary of:** Patil, A. C. et al. Deep spectra.\n"
            "More text the model added."
        )
        self.assertEqual(
            _content(reply, section), "Patil, A. C. et al. Deep spectra."
        )

    def test_file_text_keeps_title_first(self) -> None:
        text = summary_file_text(
            "# Lab Summary\n## Background\nx [3]", Path("p.pdf")
        )
        self.assertTrue(text.startswith("# Lab Summary\n## Background"))
        self.assertTrue(text.rstrip().endswith("against the paper.*"))
        self.assertNotIn("[3]", text)


if __name__ == "__main__":
    unittest.main()
