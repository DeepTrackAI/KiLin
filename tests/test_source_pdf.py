"""Tests for reading PDF source pages."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat.repository import scan_repository
from qachat.sources.extensions import sources_pdf
from qachat.sources.sources import (
    extract_source_units,
    read_repository_files,
)


def _page(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(extract_text=lambda: text)


class TestPdfSource(unittest.TestCase):

    def test_pdf_pages_become_natural_units(self) -> None:
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[
                _page("First page.\n"),
                _page(""),
                _page("Third page.\n"),
            ],
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.pdf").write_bytes(b"%PDF test")

            with patch.object(
                sources_pdf,
                "PdfReader",
                return_value=reader,
            ):
                report = scan_repository(root)
                files = read_repository_files(report)

            units = extract_source_units(files)

        self.assertEqual(report["selected_files"], ["guide.pdf"])
        self.assertEqual(
            [unit.kind for unit in units],
            ["pdf_page", "pdf_page"],
        )
        self.assertEqual(
            [unit.section for unit in units],
            ["Page 1", "Page 3"],
        )
        self.assertEqual(
            [unit.text for unit in units],
            ["First page.\n", "Third page.\n"],
        )

    def test_encrypted_pdf_is_rejected(self) -> None:
        reader = SimpleNamespace(is_encrypted=True, pages=[])
        with patch.object(
            sources_pdf,
            "PdfReader",
            return_value=reader,
        ):
            with self.assertRaisesRegex(ValueError, "Encrypted PDF"):
                sources_pdf.read_source_file(
                    Path("locked.pdf"),
                    Path("locked.pdf"),
                )

    def test_scanned_pdf_reports_that_ocr_is_needed(self) -> None:
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[_page(None), _page("")],
        )
        with patch.object(
            sources_pdf,
            "PdfReader",
            return_value=reader,
        ):
            with self.assertRaisesRegex(ValueError, "need OCR"):
                sources_pdf.read_source_file(
                    Path("scan.pdf"),
                    Path("scan.pdf"),
                )


if __name__ == "__main__":
    unittest.main()
