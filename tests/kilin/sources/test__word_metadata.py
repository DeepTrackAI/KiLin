"""Verify best-effort metadata reading from Word OOXML and OLE files.

Use a stable converted ``.docx`` fixture and a deterministic OLE double to
verify that both file types return title and author properties. Missing or
corrupt properties must not prevent downstream source reading or indexing.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from kilin.sources._word_metadata import read_word_metadata

FIXTURE = (
    Path(__file__).parent
    / "extensions/fixtures/sample.docx"
)
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class TestWordMetadata(unittest.TestCase):
    """Keep Word properties readable without needing a language model."""

    def test_docx_properties_from_real_fixture(self) -> None:
        """Read title and author saved in a LibreOffice Word document."""

        self.assertEqual(
            read_word_metadata(FIXTURE),
            ("A Sample Study of Word Documents", "Ada Lovelace"),
        )

    def test_ole_properties_are_read_with_olefile(self) -> None:
        """Read the standard OLE summary property names for old Word."""

        metadata = SimpleNamespace(
            title="A Legacy Word Study", author="Grace Hopper"
        )
        container = Mock()
        container.__enter__ = Mock(return_value=container)
        container.__exit__ = Mock(return_value=False)
        container.get_metadata.return_value = metadata
        fake_olefile = SimpleNamespace(OleFileIO=Mock(return_value=container))
        with TemporaryDirectory() as directory:
            document = Path(directory) / "old.doc"
            document.write_bytes(_OLE_SIGNATURE + b"placeholder")
            with patch.dict(sys.modules, {"olefile": fake_olefile}):
                self.assertEqual(
                    read_word_metadata(document),
                    ("A Legacy Word Study", "Grace Hopper"),
                )
        fake_olefile.OleFileIO.assert_called_once()

    def test_missing_bad_and_non_word_files_are_nonfatal(self) -> None:
        """Return missing properties rather than raising for bad files."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in (
                ("missing.docx", None),
                ("damaged.docx", b"broken Word file"),
                ("not_word.txt", b"text"),
                ("damaged.doc", _OLE_SIGNATURE + b"broken"),
            ):
                with self.subTest(name=name):
                    file = root / name
                    if data is not None:
                        file.write_bytes(data)
                    self.assertEqual(
                        read_word_metadata(file), (None, None)
                    )


if __name__ == "__main__":
    unittest.main()
