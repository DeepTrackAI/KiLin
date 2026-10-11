"""Integration tests for a small committed PDF source fixture.

The two-page synthetic PDF tests real text extraction and page provenance.
A separate, optional scientific-paper regression is retained for manual use.

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.sources.sources import (
    extract_source_units,
    read_repository_files,
)
from tests.qachat.sources._support import fixture_path

try:
    from qachat.sources.extensions import sources_pdf
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
        self.assertEqual([unit.section for unit in units],
                         ["Page 1", "Page 2"])
        self.assertIn("QAChat fixture page one.", units[0].text)
        self.assertIn("QAChat fixture page two.", units[1].text)
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
        self.assertEqual([unit.section for unit in units],
                         ["Page 1", "Page 2"])

    @unittest.skipUnless(
        os.environ.get("QACHAT_TEST_LARGE_PDF") == "1",
        "Set QACHAT_TEST_LARGE_PDF=1 for optional scientific PDF regression",
    )
    def test_optional_scientific_pdf_spacing_regression(self) -> None:
        """Check the historical DIVA spacing issue when opted in."""

        root = Path(__file__).resolve().parents[4]
        relative = Path("example_repos/literature_repo/papers/patil25deep.pdf")
        path = root / relative
        if not path.is_file():
            self.skipTest("Optional scientific PDF fixture is unavailable")
        file = sources_pdf.read_source_file(path, relative)
        self.assertIn("we introduce DIVA", " ".join(file.text.split()))
        self.assertNotIn("DIV A ", file.text)
        self.assertNotIn("(cid:", file.text)
        self.assertEqual(file.text.count(sources_pdf.PAGE_BREAK), 77)


if __name__ == "__main__":
    unittest.main()
