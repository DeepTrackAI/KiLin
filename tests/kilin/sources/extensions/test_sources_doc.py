"""Unit tests for old Word ``.doc`` reading.

The fixture ``fixtures/sample.doc`` is a small Word 97-2003 file with a
title, accented and Greek letters, a line starting with ``#``, a two-row
table, and a link. Protected and pre-1997 files are made by changing a few
bytes of a copy. The tests also cover files whose name does not match their
format, field codes, and that passages keep line ranges that can be cited.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.repository import scan_repository
from kilin.sources.sources import (
    extract_source_units,
    read_repository_files,
)

try:
    from docx import Document
    from kilin.sources.extensions import sources_doc, sources_docx
except ModuleNotFoundError as error:
    if error.name not in {"docx", "lxml", "olefile"}:
        raise
    Document = None
    sources_doc = None
    sources_docx = None

FIXTURE = Path(__file__).parent / "fixtures" / "sample.doc"
SAMPLE_TEXT = (
    "Study title\n\n"
    "First paragraph with café and naïve — “quotes”, α and β.\n\n"
    "\\# not a heading\n\n"
    "Name | Value\n\n"
    "alpha | 1\n\n"
    "See the website for more.\n\n"
    "Last line.\n"
)
# Start of the WordDocument stream: Word marker and Word 97 version.
WORD_97_START = b"\xec\xa5\xc1\x00"


@unittest.skipIf(sources_doc is None, "Optional Word readers unavailable")
class TestDocReading(unittest.TestCase):
    """Verify old Word text extraction, file sniffing, and errors."""

    def setUp(self) -> None:
        """Create a temporary folder for copied and changed Word files."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def _changed_copy(self, offset: int, value: int) -> Path:
        """Copy the fixture with one byte of its Word header changed."""

        data = bytearray(FIXTURE.read_bytes())
        start = data.index(WORD_97_START)
        data[start + offset] = value
        path = self.root / "changed.doc"
        path.write_bytes(bytes(data))
        return path

    def test_body_text_tables_and_letters(self) -> None:
        """Keep paragraphs, table rows, and non-English letters in order."""

        source = sources_doc.read_source_file(FIXTURE, Path("sample.doc"))
        self.assertEqual(source.text, SAMPLE_TEXT)

    def test_scanned_doc_is_indexed_with_citable_lines(self) -> None:
        """Select, read, and split a .doc like any other source file."""

        shutil.copy(FIXTURE, self.root / "paper.doc")
        report = scan_repository(self.root, extensions=[".doc"])
        self.assertEqual(report["selected_files"], ["paper.doc"])
        files = read_repository_files(report)
        units = extract_source_units(files)
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual(unit.path, Path("paper.doc"))
        lines = files[0].text.splitlines(keepends=True)
        self.assertEqual(
            "".join(lines[unit.start_line - 1 : unit.end_line]), unit.text
        )

    def test_sample_doc_and_docx_have_equivalent_body_text(self) -> None:
        """Compare text content from both stable, converted fixtures."""

        modern = FIXTURE.with_suffix(".docx")
        old_text = sources_doc.read_source_file(
            FIXTURE, Path("sample.doc")
        ).text
        new_text = sources_docx.read_source_file(
            modern, Path("sample.docx")
        ).text
        # Their table row separators differ, not the underlying words.
        self.assertEqual(
            " ".join(old_text.split()), " ".join(new_text.split())
        )

    def test_doc_renamed_to_docx_is_read(self) -> None:
        """Read an old Word file by its content, not by its name."""

        path = self.root / "renamed.docx"
        shutil.copy(FIXTURE, path)
        source = sources_docx.read_source_file(path, Path("renamed.docx"))
        self.assertEqual(source.text, SAMPLE_TEXT)

    def test_docx_renamed_to_doc_is_read(self) -> None:
        """Hand a .docx file with a .doc name to the .docx reader."""

        document = Document()
        document.add_heading("Methods", level=1)
        document.add_paragraph("We slept.")
        path = self.root / "renamed.doc"
        document.save(str(path))
        source = sources_doc.read_source_file(path, Path("renamed.doc"))
        self.assertEqual(source.text, "# Methods\n\nWe slept.\n")

    def test_field_codes_are_dropped(self) -> None:
        """Keep what a field shows, also for a field inside another one."""

        link = 'See \x13 HYPERLINK "https://x.org" \x14the site\x15.\r'
        nested = "\x13 IF \x13 PAGE \x141\x15 = 1 \x14yes\x15\r"
        self.assertEqual(
            sources_doc._blocks(link + nested), ["See the site.", "yes"]
        )

    def test_breaks_and_hidden_marks(self) -> None:
        """Turn breaks into blocks and drop marks that carry no text."""

        raw = "one\x0btwo\x0cthree non\x1ebreaking op\x1ftional\x01\r"
        self.assertEqual(
            sources_doc._blocks(raw),
            ["one", "two", "three non-breaking optional"],
        )

    def test_damaged_file_is_rejected_clearly(self) -> None:
        """Report a file that is not a Word document by its name."""

        path = self.root / "broken.doc"
        path.write_bytes(b"not a Word file")
        with self.assertRaisesRegex(
            ValueError, "Cannot read Word file.*broken.doc"
        ):
            sources_doc.read_source_file(path, Path("broken.doc"))

    def test_protected_file_is_named(self) -> None:
        """Say that a file is password-protected instead of misreading it."""

        # High byte of the flags; 0x01 there is the encryption flag.
        path = self._changed_copy(0x0B, 0x01)
        with self.assertRaisesRegex(
            ValueError, "password-protected: changed.doc"
        ):
            sources_doc.read_source_file(path, Path("changed.doc"))

    def test_word_95_file_asks_for_docx(self) -> None:
        """Ask the user to convert a file older than Word 97."""

        # Version byte 0x65 is Word 6.
        path = self._changed_copy(0x02, 0x65)
        with self.assertRaisesRegex(
            ValueError, r"Word 95 or older.*save it as \.docx: changed.doc"
        ):
            sources_doc.read_source_file(path, Path("changed.doc"))


class TestMissingOldWordReader(unittest.TestCase):
    """Verify the message shown when olefile is not installed."""

    def test_missing_package_names_the_fix(self) -> None:
        """Say which package to install, and keep the missing module name."""

        module = "kilin.sources.extensions.sources_doc"
        with patch.dict(sys.modules, {"olefile": None, module: None}):
            del sys.modules[module]
            with self.assertRaisesRegex(
                ModuleNotFoundError, "needs the olefile package"
            ) as caught:
                import_module(module)
        self.assertEqual(caught.exception.name, "olefile")


if __name__ == "__main__":
    unittest.main()
