"""Check one-page report PDFs and the template guides in projects."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pypdf import PdfReader

from qachat.gui.engine import ChatEngine
from qachat.gui.report_pdf import markdown_to_html, write_report_pdf
from qachat.report_templates import (
    GUIDE_MARKER,
    load_template,
    template_guide,
)


def _report(words: int) -> str:
    body = " ".join(["word"] * words)
    return f"# A headline\n\n*Source: p.pdf.*\n\n## The Reason\n{body}\n"


class TestReportPdf(unittest.TestCase):
    """Reports fit one A4 page by shrinking the font, within limits."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_briefing_length_fits_one_page(self) -> None:
        path = self.root / "brief.pdf"
        pages = write_report_pdf(_report(750), path, label="Briefing")
        self.assertEqual(pages, 1)
        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 1)
        text = " ".join(reader.pages[0].extract_text().split())
        self.assertIn("A headline", text)
        self.assertIn("BRIEFING", text.upper())

    def test_too_long_reports_more_pages(self) -> None:
        pages = write_report_pdf(_report(3000), self.root / "long.pdf")
        self.assertGreater(pages, 1)

    def test_markdown_is_escaped_and_styled(self) -> None:
        html = markdown_to_html(
            "# T <b>\n## Section\nSome **bold** and *it*.\n- one\n- two"
        )
        self.assertIn("T &lt;b&gt;", html)
        self.assertIn('<p class="section">Section</p>', html)
        self.assertIn("<b>bold</b>", html)
        self.assertIn("<i>it</i>", html)
        self.assertIn("<ul><li>one</li><li>two</li></ul>", html)


class TestTemplateGuides(unittest.TestCase):
    """Literature projects get a readable guide to each report template."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.engine = ChatEngine(self.root, lambda event: None)

    def test_guide_matches_the_template(self) -> None:
        guide = template_guide(load_template("research_briefing"))
        self.assertTrue(guide.startswith(GUIDE_MARKER))
        self.assertIn("At most 90 characters.", guide)
        self.assertIn("## 4. The Discovery\n\n*200-300 words.*", guide)
        self.assertIn("Chosen from the paper's own reference list.", guide)
        self.assertIn("At most 750 words", guide)

    def test_guides_are_written_and_user_files_kept(self) -> None:
        self.engine._write_template_guides(self.root, "literature")
        folder = self.root / "target" / "templates"
        names = sorted(path.name for path in folder.iterdir())
        self.assertEqual(
            names,
            [
                "research_briefing_template.md",
                "research_briefing_template.pdf",
                "research_summary_template.md",
                "research_summary_template.pdf",
            ],
        )
        mine = folder / "research_summary_template.md"
        mine.write_text("My own notes\n")
        self.engine._write_template_guides(self.root, "literature")
        self.assertEqual(mine.read_text(), "My own notes\n")

    def test_code_projects_get_no_guides(self) -> None:
        self.engine._write_template_guides(self.root, "code")
        self.assertFalse((self.root / "target" / "templates").exists())


if __name__ == "__main__":
    unittest.main()
