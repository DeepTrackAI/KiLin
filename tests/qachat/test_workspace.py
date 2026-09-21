"""Tests for a project built from several local source paths."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.workspace import prepare_sources_workspace


class TestSourcesWorkspace(unittest.TestCase):
    """Only selected snapshots enter a user-chosen QAChat workspace."""

    def test_folders_and_file_share_one_workspace(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one" / "methods"
            second = root / "two" / "methods"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "scan.py").write_text("def scan(): pass\n")
            (second / "scan.m").write_text("function scan\nend\n")
            note = root / "notes.md"
            note.write_text("# Notes\n")
            workspace = root / "elsewhere" / "my-workspace"

            project = prepare_sources_workspace(
                [first, second, note], "code", workspace
            )

            self.assertEqual(project.root, workspace)
            self.assertEqual(project.source, workspace / "sources")
            self.assertTrue((workspace / "sources/methods/scan.py").is_file())
            self.assertTrue((workspace / "sources/notes.md").is_file())
            matches = list((workspace / "sources").glob("methods-*/scan.m"))
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(list((workspace / "sources").iterdir())), 3)
            self.assertTrue((workspace / "indexes").is_dir())
            self.assertTrue((workspace / "logs").is_dir())
            self.assertEqual(note.read_text(), "# Notes\n")

            reopened = prepare_sources_workspace([], "code", workspace)
            self.assertEqual(reopened.source, project.source)

    def test_new_source_is_added_without_removing_existing_snapshot(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one.py"
            second = root / "two.py"
            first.write_text("first = 1\n")
            second.write_text("second = 2\n")
            workspace = root / "workspace"

            prepare_sources_workspace([first], "code", workspace)
            prepare_sources_workspace([second], "code", workspace)

            self.assertTrue((workspace / "sources/one.py").is_file())
            self.assertTrue((workspace / "sources/two.py").is_file())

    def test_existing_unmanaged_folder_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "code.py"
            source.write_text("pass\n")
            workspace = root / "personal"
            workspace.mkdir()
            marker = workspace / "keep.txt"
            marker.write_text("mine")

            with self.assertRaisesRegex(ValueError, "empty folder"):
                prepare_sources_workspace([source], "code", workspace)

            self.assertEqual(marker.read_text(), "mine")

    def test_unsupported_file_does_not_enter_workspace(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "image.png"
            source.write_bytes(b"not an image")
            workspace = root / "workspace"

            with self.assertRaisesRegex(ValueError, "File type not selected"):
                prepare_sources_workspace([source], "code", workspace)

            self.assertFalse((workspace / "sources/image.png").exists())


if __name__ == "__main__":
    unittest.main()
