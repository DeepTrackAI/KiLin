"""Test private output-directory management without legacy migrations.

The core initializes outputs for new projects. Shared output helpers may
also create the directory on first use, but they must not move unrelated
folders or follow symbolic links.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.tools._shared.outputs import output_folder


class TestSharedOutputs(unittest.TestCase):
    """Verify safe output-folder initialization for shared services."""

    def test_creates_only_the_requested_outputs_directory(self) -> None:
        """A previously absent outputs directory is created on demand.

        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = output_folder(root)
            self.assertEqual(outputs, root / "outputs")
            self.assertTrue(outputs.is_dir())
            self.assertEqual(output_folder(root), outputs)

    def test_does_not_migrate_or_delete_an_unrelated_target(self) -> None:
        """Never relocate the contents of another directory.

        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            (target / "manual.md").write_text("# Manual\n")
            self.assertEqual(output_folder(root), root / "outputs")
            self.assertTrue((root / "outputs").is_dir())
            self.assertEqual(
                (target / "manual.md").read_text(), "# Manual\n"
            )
            self.assertFalse((root / "outputs/manual.md").exists())

    def test_rejects_a_linked_output_directory(self) -> None:
        """Do not write to an external directory via a folder symlink.

        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            linked = root / "outputs"
            try:
                linked.symlink_to(elsewhere, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("Symbolic links are unavailable")
            with self.assertRaisesRegex(ValueError, "cannot be a link"):
                output_folder(root)
            self.assertEqual(list(elsewhere.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
