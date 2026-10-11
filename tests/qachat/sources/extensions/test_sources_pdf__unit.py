"""Deterministic unit tests for PDF text extraction and page provenance.

PDF parser boundaries are mocked to test fallback and error contracts without
requiring external documents or depending on parser layout heuristics.

"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qachat.sources.sources import SourceFile

try:
    from qachat.sources.extensions import sources_pdf
except ModuleNotFoundError as error:
    if error.name not in {"pdfminer", "pypdf"}:
        raise
    sources_pdf = None


def _page(text: str | None) -> SimpleNamespace:
    """Return a fake PDF page with one text-extraction method.

    Parameters
    ----------
    text: str | None
        Page text returned by the mocked PDF parser.

    Returns
    -------
    types.SimpleNamespace
        Object implementing ``extract_text``.

    """

    return SimpleNamespace(extract_text=Mock(return_value=text))


@unittest.skipIf(sources_pdf is None, "Optional PDF readers unavailable")
class TestPdfReading(unittest.TestCase):
    """Verify PDF reader selection, fallback, and error handling."""

    def setUp(self) -> None:
        """Restore the PDF parser logger after each reading test."""

        logger = logging.getLogger("pdfminer")
        self.addCleanup(logger.setLevel, logger.level)

    def test_layout_extraction_is_preferred_over_plain_text(self) -> None:
        """Use valid layout pages rather than the pypdf text fallback."""

        page = _page("Wrong fallback text")
        reader = SimpleNamespace(is_encrypted=False, pages=[page])
        with (
            patch.object(sources_pdf, "PdfReader", return_value=reader),
            patch.object(
                sources_pdf, "_layout_pages", return_value=["Layout text"]
            ),
        ):
            file = sources_pdf.read_source_file(
                Path("paper.pdf"), Path("papers/paper.pdf")
            )
        self.assertEqual(file.path, Path("papers/paper.pdf"))
        self.assertEqual(file.text, "Layout text")
        page.extract_text.assert_not_called()

    def test_plain_fallback_preserves_blank_page_positions(self) -> None:
        """Keep form-feed page separators and original page labels."""

        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[_page("First\n"), _page(""), _page("Third\n")],
        )
        with (
            patch.object(sources_pdf, "PdfReader", return_value=reader),
            patch.object(sources_pdf, "_layout_pages", return_value=None),
        ):
            file = sources_pdf.read_source_file(
                Path("paper.pdf"), Path("paper.pdf")
            )
        self.assertEqual(file.text, "First\n\f\fThird\n")
        units = sources_pdf.extract_source_units(file)
        self.assertEqual([unit.section for unit in units],
                         ["Page 1", "Page 3"])
        self.assertEqual([unit.text for unit in units],
                         ["First\n", "Third\n"])

    def test_unparsable_layout_falls_back_to_plain_text(self) -> None:
        """Use pypdf when the layout parser raises an exception."""

        reader = SimpleNamespace(
            is_encrypted=False, pages=[_page("Plain (cid:12)text.")]
        )
        with (
            patch.object(sources_pdf, "PdfReader", return_value=reader),
            patch.object(
                sources_pdf, "_pdfminer_text", side_effect=ValueError("bad")
            ),
        ):
            file = sources_pdf.read_source_file(
                Path("paper.pdf"), Path("paper.pdf")
            )
        self.assertEqual(file.text, "Plain text.")

    def test_layout_page_count_mismatch_uses_plain_fallback(self) -> None:
        """Avoid assigning incorrect page provenance to layout output."""

        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[_page("First"), _page("Second")],
        )
        with (
            patch.object(sources_pdf, "PdfReader", return_value=reader),
            patch.object(sources_pdf, "_pdfminer_text", return_value="Only\f"),
        ):
            file = sources_pdf.read_source_file(
                Path("paper.pdf"), Path("paper.pdf")
            )
        self.assertEqual(file.text, "First\fSecond")

    def test_reader_os_and_pdf_errors_are_translated(self) -> None:
        """Report unreadable PDFs as ValueError with source provenance."""

        for error in (OSError("read"), sources_pdf.PdfReadError("bad")):
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    sources_pdf, "PdfReader", side_effect=error
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Cannot read PDF.*paper.pdf"
                    ):
                        sources_pdf.read_source_file(
                            Path("paper.pdf"), Path("paper.pdf")
                        )

    def test_encrypted_pdf_is_rejected(self) -> None:
        """Do not attempt to read encrypted documents."""

        reader = SimpleNamespace(is_encrypted=True, pages=[])
        with patch.object(sources_pdf, "PdfReader", return_value=reader):
            with self.assertRaisesRegex(ValueError, "Encrypted PDF"):
                sources_pdf.read_source_file(
                    Path("locked.pdf"), Path("locked.pdf")
                )

    def test_image_only_pdf_reports_ocr_requirement(self) -> None:
        """Reject PDFs without any extractable text."""

        reader = SimpleNamespace(
            is_encrypted=False, pages=[_page(None), _page("  \n")]
        )
        with (
            patch.object(sources_pdf, "PdfReader", return_value=reader),
            patch.object(sources_pdf, "_layout_pages", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "need OCR"):
                sources_pdf.read_source_file(
                    Path("scan.pdf"), Path("scan.pdf")
                )

    def test_plain_text_extraction_error_is_translated(self) -> None:
        """Expose pypdf page extraction failures as ValueError."""

        page = SimpleNamespace(
            extract_text=Mock(side_effect=sources_pdf.PdfReadError("bad"))
        )
        reader = SimpleNamespace(is_encrypted=False, pages=[page])
        with (
            patch.object(sources_pdf, "PdfReader", return_value=reader),
            patch.object(sources_pdf, "_layout_pages", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "Cannot extract PDF"):
                sources_pdf.read_source_file(
                    Path("paper.pdf"), Path("paper.pdf")
                )

    def test_unicode_surrogates_and_unmapped_glyphs_are_cleaned(self) -> None:
        """Preserve valid Unicode while repairing malformed glyph text."""

        raw = "Valid \ud835\udcbf and broken \ud835 (cid:14)text."
        clean = sources_pdf._clean_extracted_text(raw)
        self.assertEqual(clean, "Valid 𝒿 and broken � text.")
        self.assertNotIn("(cid:", clean)

    def test_repeated_blank_lines_are_normalized(self) -> None:
        """Trim trailing spaces and collapse excessive blank lines."""

        text = "First   \n\n\n\nSecond  \n"
        self.assertEqual(
            sources_pdf._clean_extracted_text(text),
            "First\n\nSecond\n",
        )


@unittest.skipIf(sources_pdf is None, "Optional PDF readers unavailable")
class TestPdfUnits(unittest.TestCase):
    """Verify page-unit coordinates independently of PDF parsing."""

    def test_page_numbers_skip_blank_pages_without_renumbering(self) -> None:
        """Preserve original page numbers in section labels."""

        file = SourceFile(Path("paper.pdf"), "One\nTwo\n\f\f\fFour")
        units = sources_pdf.extract_source_units(file)
        self.assertEqual([unit.section for unit in units],
                         ["Page 1", "Page 4"])
        self.assertEqual([unit.kind for unit in units],
                         ["pdf_page", "pdf_page"])
        self.assertEqual(
            [(unit.start_line, unit.end_line) for unit in units],
            [(1, 2), (1, 1)],
        )
        self.assertEqual([unit.text for unit in units],
                         ["One\nTwo\n", "Four"])

    def test_empty_and_blank_pages_produce_no_units(self) -> None:
        """Skip empty documents and whitespace-only pages."""

        for text in ("", "\f", " \n\f \n"):
            with self.subTest(text=text):
                self.assertEqual(
                    sources_pdf.extract_source_units(
                        SourceFile(Path("empty.pdf"), text)
                    ),
                    [],
                )

    def test_page_unit_preserves_relative_path(self) -> None:
        """Keep repository-relative provenance after PDF extraction."""

        path = Path("papers/example.pdf")
        units = sources_pdf.extract_source_units(
            SourceFile(path, "A page of text")
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].path, path)
        self.assertEqual(units[0].section, "Page 1")


if __name__ == "__main__":
    unittest.main()
