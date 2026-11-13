"""Unit tests for Word ``.docx`` reading.

Generated files cover individual edge cases. The stable ``sample.docx``
fixture is a LibreOffice conversion of ``sample.doc``, with explicit title
and author properties. It verifies real-document extraction, Unicode,
reading order, metadata, and citable source coordinates.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.repository import scan_repository
from kilin.sources._word_metadata import read_word_metadata
from kilin.sources.sources import (
    SourceFile,
    extract_source_units,
    read_repository_files,
)

try:
    from docx import Document
    from kilin.sources.extensions import sources_docx
except ModuleNotFoundError as error:
    if error.name not in {"docx", "lxml"}:
        raise
    Document = None
    sources_docx = None


FIXTURE = Path(__file__).parent / "fixtures" / "sample.docx"


@unittest.skipIf(sources_docx is None, "Optional Word reader unavailable")
class TestDocxReading(unittest.TestCase):
    """Verify Word text extraction, heading sections, and errors."""

    def setUp(self) -> None:
        """Create a temporary folder for generated Word files."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def _read(self, document: object, name: str = "notes.docx") -> str:
        """Save a generated document and return its extracted text."""

        path = self.root / name
        document.save(str(path))
        return sources_docx.read_source_file(path, Path(name)).text

    def test_paragraphs_keep_reading_order(self) -> None:
        """Join nonblank paragraphs with blank lines, in document order."""

        document = Document()
        document.add_paragraph("First paragraph.")
        document.add_paragraph("")
        document.add_paragraph("  Second paragraph.  ")
        self.assertEqual(
            self._read(document), "First paragraph.\n\nSecond paragraph.\n"
        )

    def test_headings_become_sections(self) -> None:
        """Turn Title and Heading styles into cited Markdown sections."""

        document = Document()
        document.add_heading("A Study of Sleep", level=0)
        document.add_paragraph("Short abstract.")
        document.add_heading("Methods", level=1)
        document.add_paragraph("We measured sleep.")
        document.add_heading("Participants", level=2)
        document.add_paragraph("Forty adults.")
        text = self._read(document)
        self.assertEqual(
            text,
            (
                "# A Study of Sleep\n\nShort abstract.\n\n# Methods\n\n"
                "We measured sleep.\n\n## Participants\n\nForty adults.\n"
            ),
        )
        units = sources_docx.extract_source_units(
            SourceFile(Path("notes.docx"), text)
        )
        self.assertEqual(
            [unit.section for unit in units],
            ["A Study of Sleep", "Methods", "Participants"],
        )
        self.assertEqual({unit.kind for unit in units}, {"markdown_section"})
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_list_items_are_marked(self) -> None:
        """Start list items with a dash, indented by nesting level."""

        document = Document()
        document.add_paragraph("Apples", style="List Bullet")
        document.add_paragraph("Pears", style="List Number")
        nested = document.add_paragraph("Green pears")
        numbering = nested._p.get_or_add_pPr().get_or_add_numPr()
        numbering.get_or_add_ilvl().val = 1
        numbering.get_or_add_numId().val = 1
        self.assertEqual(
            self._read(document), "- Apples\n\n- Pears\n\n  - Green pears\n"
        )

    def test_builtin_list_styles_and_indented_non_list(self) -> None:
        """Respect the levels of built-in numbered and bulleted styles."""

        document = Document()
        for style, text in (
            ("List Paragraph", "Indented, not a bullet"),
            ("List Bullet", "First"),
            ("List Bullet 2", "Nested"),
            ("List Bullet 3", "Third level"),
            ("List Number", "One"),
            ("List Number 2", "Two"),
            ("List Continue", "Continuation, not a bullet"),
        ):
            document.add_paragraph(text, style=style)
        self.assertEqual(
            self._read(document),
            (
                "Indented, not a bullet\n\n- First\n\n"
                "  - Nested\n\n    - Third level\n\n"
                "- One\n\n  - Two\n\n"
                "Continuation, not a bullet\n"
            ),
        )

    def test_explicit_numbering_zero_disables_style_list(self) -> None:
        """Honor a paragraph's explicit suppression of style numbering."""

        document = Document()
        paragraph = document.add_paragraph("Plain", style="List Bullet")
        numbering = paragraph._p.get_or_add_pPr().get_or_add_numPr()
        numbering.get_or_add_numId().val = 0
        self.assertEqual(self._read(document), "Plain\n")

    def test_stable_word_fixture_reads_without_legacy_dependency(self) -> None:
        """Read the converted sample even when olefile is unavailable."""

        with patch.dict(sys.modules, {"olefile": None}):
            source = sources_docx.read_source_file(
                FIXTURE, Path("sample.docx")
            )
        self.assertIn("Study title\n\nFirst paragraph", source.text)
        self.assertIn("café and naïve — “quotes”, α and β", source.text)
        self.assertIn("Name | Value\nalpha | 1", source.text)
        self.assertIn("See the website for more.", source.text)
        self.assertIn("\\# not a heading", source.text)
        self.assertEqual(
            read_word_metadata(FIXTURE),
            ("A Sample Study of Word Documents", "Ada Lovelace"),
        )

    def test_stable_fixture_keeps_citable_line_ranges(self) -> None:
        """Match stored passage coordinates to the fixture's text lines."""

        source = sources_docx.read_source_file(
            FIXTURE, Path("sample.docx")
        )
        units = sources_docx.extract_source_units(source)
        self.assertTrue(units)
        lines = source.text.splitlines(keepends=True)
        self.assertEqual("".join(unit.text for unit in units), source.text)
        for unit in units:
            self.assertEqual(
                "".join(lines[unit.start_line - 1 : unit.end_line]),
                unit.text,
            )

    def test_tables_become_rows_in_reading_order(self) -> None:
        """Write each row on one line, between the paragraphs around it."""

        document = Document()
        document.add_paragraph("Before the table.")
        table = document.add_table(rows=3, cols=2)
        table.cell(0, 0).text = "Name"
        table.cell(0, 1).text = "Age"
        table.cell(1, 0).text = "Ada"
        table.cell(1, 1).text = "36"
        table.cell(2, 0).merge(table.cell(2, 1)).text = "Merged note"
        document.add_paragraph("After the table.")
        self.assertEqual(
            self._read(document),
            (
                "Before the table.\n\nName | Age\nAda | 36\nMerged note\n\n"
                "After the table.\n"
            ),
        )

    def test_body_text_cannot_start_a_false_section(self) -> None:
        """Escape body lines that Markdown would read as a heading."""

        document = Document()
        document.add_heading("Results", level=1)
        document.add_paragraph("# not a heading")
        text = self._read(document)
        self.assertIn("\\# not a heading", text)
        units = sources_docx.extract_source_units(
            SourceFile(Path("notes.docx"), text)
        )
        self.assertEqual([unit.section for unit in units], ["Results"])

    def test_empty_document_has_no_units(self) -> None:
        """Return empty text and no passages for a document without text."""

        text = self._read(Document())
        self.assertEqual(text, "")
        self.assertEqual(
            sources_docx.extract_source_units(
                SourceFile(Path("notes.docx"), text)
            ),
            [],
        )

    def test_damaged_file_is_rejected_clearly(self) -> None:
        """Report a file that is not a Word document by its name."""

        path = self.root / "broken.docx"
        path.write_bytes(b"not a zip file")
        with self.assertRaisesRegex(
            ValueError, "Cannot read Word file.*broken.docx"
        ):
            sources_docx.read_source_file(path, Path("broken.docx"))

    def test_scanned_docx_is_indexed_with_citable_lines(self) -> None:
        """Select, read, and split a .docx like any other source file."""

        document = Document()
        document.add_heading("Introduction", level=1)
        document.add_paragraph("Sleep matters.")
        document.save(str(self.root / "paper.docx"))
        report = scan_repository(self.root, extensions=[".docx"])
        self.assertEqual(report["selected_files"], ["paper.docx"])
        files = read_repository_files(report)
        units = extract_source_units(files)
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual(unit.path, Path("paper.docx"))
        self.assertEqual(unit.section, "Introduction")
        lines = files[0].text.splitlines(keepends=True)
        self.assertEqual(
            "".join(lines[unit.start_line - 1 : unit.end_line]), unit.text
        )


class TestMissingWordReader(unittest.TestCase):
    """Verify the message shown when python-docx is not installed."""

    def test_missing_package_names_the_fix(self) -> None:
        """Say which package to install, and keep the missing module name."""

        module = "kilin.sources.extensions.sources_docx"
        with patch.dict(sys.modules, {"docx": None, module: None}):
            del sys.modules[module]
            with self.assertRaisesRegex(
                ModuleNotFoundError, "needs the python-docx package"
            ) as caught:
                import_module(module)
        self.assertEqual(caught.exception.name, "docx")


if __name__ == "__main__":
    unittest.main()
