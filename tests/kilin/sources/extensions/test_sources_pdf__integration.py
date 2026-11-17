"""Integration tests for committed PDF source fixtures.

The two-page synthetic PDF tests text extraction and page provenance.
A scientific-paper fixture tests a historical text-spacing regression.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from kilin.repository import scan_repository
from kilin.sources.sources import (
    extract_source_units,
    read_repository_files,
)
from tests.kilin.sources._support import fixture_path

PAPER_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "pdfs"

try:
    from kilin.sources.extensions import sources_pdf
except ModuleNotFoundError as error:
    if error.name not in {"pdfminer", "pypdf"}:
        raise
    sources_pdf = None


@unittest.skipIf(sources_pdf is None, "Optional PDF readers unavailable")
class TestPdfIntegration(unittest.TestCase):
    """Verify PDF extraction with actual local parser dependencies."""

    def setUp(self) -> None:
        """Restore the PDF parser logger after each integration test."""

        logger = logging.getLogger("pdfminer")
        self.addCleanup(logger.setLevel, logger.level)

    def test_small_pdf_fixture_extracts_both_pages(self) -> None:
        """Extract real text without relying on a large scientific PDF."""

        file = sources_pdf.read_source_file(
            fixture_path("sample.pdf"), Path("papers/sample.pdf")
        )
        units = sources_pdf.extract_source_units(file)
        self.assertEqual(len(units), 2)
        self.assertEqual(
            [unit.section for unit in units], ["Page 1", "Page 2"]
        )
        self.assertIn("KíLín fixture page one.", units[0].text)
        self.assertIn("KíLín fixture page two.", units[1].text)
        self.assertEqual(
            [unit.path for unit in units],
            [Path("papers/sample.pdf"), Path("papers/sample.pdf")],
        )

    def test_scan_read_extract_pdf_fixture(self) -> None:
        """Run the complete repository-to-PDF-units workflow."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "papers").mkdir()
            shutil.copyfile(
                fixture_path("sample.pdf"), root / "papers" / "sample.pdf"
            )
            report = scan_repository(root, extensions=[".pdf"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(report["selected_files"], ["papers/sample.pdf"])
        self.assertEqual(len(files), 1)
        self.assertEqual(len(units), 2)
        self.assertEqual(
            [unit.section for unit in units], ["Page 1", "Page 2"]
        )

    def test_scientific_pdf_spacing_regression(self) -> None:
        """Check the historical DIVA spacing issue using a committed PDF."""

        relative = Path("papers/patil25deep.pdf")
        file = sources_pdf.read_source_file(
            PAPER_FIXTURES / "patil25deep.pdf", relative
        )
        self.assertIn("we introduce DIVA", " ".join(file.text.split()))
        self.assertNotIn("DIV A ", file.text)
        self.assertNotIn("(cid:", file.text)
        self.assertEqual(file.text.count(sources_pdf.PAGE_BREAK), 77)


if __name__ == "__main__":
    unittest.main()
