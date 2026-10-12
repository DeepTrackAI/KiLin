"""Verify isolated temporary files and atomic text replacement.

Tests use real files in private temporary directories and simulate write and
replacement failures. The helper is deliberately independent of filesystem
permissions policy, trace logging, and conversation ownership.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat.atomic_writes import atomic_write_text


class TestAtomicTextWrites(unittest.TestCase):
    """Check successful replacement and failure rollback of one file."""

    def setUp(self) -> None:
        """Allocate a disposable output directory."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "conversation.json"

    def _temporary_files(self) -> list[Path]:
        """Find temporary files created for this destination.

        Returns
        -------
        list[pathlib.Path]
            Matching temporary paths still present in the directory.

        """

        return list(self.root.glob(f".{self.path.name}.*.tmp"))

    def test_writes_unicode_and_replaces_existing_text(self) -> None:
        """Preserve exact UTF-8 content across atomic replacements."""

        atomic_write_text(self.path, "first \u03b1\n")
        self.assertEqual(
            self.path.read_text(encoding="utf-8"), "first \u03b1\n"
        )
        atomic_write_text(self.path, "second \u03b2\n")
        self.assertEqual(
            self.path.read_text(encoding="utf-8"), "second \u03b2\n"
        )
        self.assertEqual(self._temporary_files(), [])

    def test_preexisting_tmp_symlink_is_not_followed(self) -> None:
        """Leave a predictably named symlink and its target untouched."""

        victim = self.root / "other.txt"
        victim.write_text("protected", encoding="utf-8")
        trap = self.root / "conversation.json.tmp"
        try:
            trap.symlink_to(victim)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Temporary symlinks unavailable: {error}")
        atomic_write_text(self.path, "safe")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "safe")
        self.assertEqual(victim.read_text(encoding="utf-8"), "protected")
        self.assertTrue(trap.is_symlink())
        self.assertEqual(self._temporary_files(), [])

    def test_replacement_failure_preserves_original_and_cleans_up(
        self,
    ) -> None:
        """Never replace valid data when the final rename fails."""

        atomic_write_text(self.path, "committed")
        with patch.object(
            Path, "replace", side_effect=OSError("replacement rejected")
        ):
            with self.assertRaisesRegex(OSError, "replacement rejected"):
                atomic_write_text(self.path, "aborted")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "committed")
        self.assertEqual(self._temporary_files(), [])

    def test_write_failure_preserves_original_and_cleans_up(self) -> None:
        """Remove the temporary file if the stream cannot write content."""

        atomic_write_text(self.path, "committed")
        with patch.object(
            Path, "replace", side_effect=AssertionError("must not replace")
        ):
            with self.assertRaises(UnicodeEncodeError):
                atomic_write_text(self.path, "\ud800")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "committed")
        self.assertEqual(self._temporary_files(), [])

    def test_missing_parent_is_not_implicitly_created(self) -> None:
        """Leave parent-directory creation under caller control."""

        path = self.root / "absent" / "conversation.json"
        with self.assertRaises(OSError):
            atomic_write_text(path, "text")
        self.assertFalse(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
