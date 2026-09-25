"""Tests for a project built from several local source paths."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.workspace import prepare_sources_workspace
from qachat.project import default_project_profile


class TestSourcesWorkspace(unittest.TestCase):
    """Only selected snapshots enter a user-chosen QAChat workspace."""

    def test_new_literature_project_accepts_typical_pdf_size(self) -> None:
        """The literature default is larger without changing code projects."""

        self.assertEqual(
            default_project_profile("code").max_file_bytes, 1024 * 1024
        )
        self.assertEqual(
            default_project_profile("literature").max_file_bytes,
            20 * 1024 * 1024,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024))

            project = prepare_sources_workspace(
                [pdf], "literature", root / "workspace"
            )

            self.assertTrue((project.source / "paper.pdf").is_file())
            self.assertEqual(project.profile.max_file_bytes, 20 * 1024 * 1024)

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

            self.assertEqual(project.root, workspace.resolve())
            self.assertEqual(project.source, (workspace / "sources").resolve())
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

    def test_new_source_is_added_without_removing_existing_snapshot(
        self,
    ) -> None:
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

    def test_checked_files_and_advanced_settings_are_used(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "research"
            (source / "package").mkdir(parents=True)
            (source / "package/keep.py").write_text("value = 1\n")
            (source / "skip.py").write_text("value = 2\n")
            workspace = root / "my-workspace"

            project = prepare_sources_workspace(
                [source],
                "code",
                workspace,
                selected_files={str(source.resolve()): ["package/keep.py"]},
                settings={"target_tokens": 90, "max_tokens": 120},
            )

            self.assertTrue(
                (workspace / "sources/research/package/keep.py").is_file()
            )
            self.assertFalse((workspace / "sources/research/skip.py").exists())
            self.assertEqual(project.profile.target_tokens, 90)
            self.assertEqual(project.profile.max_tokens, 120)

    def test_invalid_settings_do_not_replace_profile(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "code.py"
            source.write_text("pass\n")
            workspace = root / "workspace"
            prepare_sources_workspace([source], "code", workspace)
            before = (workspace / "profile.ini").read_text()

            with self.assertRaisesRegex(ValueError, "overlap_tokens"):
                prepare_sources_workspace(
                    [],
                    "code",
                    workspace,
                    settings={"target_tokens": 5, "overlap_tokens": 5},
                )

            self.assertEqual((workspace / "profile.ini").read_text(), before)

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
