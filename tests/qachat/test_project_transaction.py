"""Test atomic new-project completion and best-effort rollback.

Simulate failures during packaged resource installation, managed source
copying, mirrored snapshot construction, and the final profile commit.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.project import initialize_project, prepare_project_workspace
from kilin.workspace import prepare_sources_workspace


class TestProjectTransaction(unittest.TestCase):
    """Ensure a failed first attempt never commits a project profile."""

    def setUp(self) -> None:
        """Create independent source and workspace fixtures."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = self.root / "source.md"
        self.original.write_text("# Study\n", encoding="utf-8")

    def test_resource_install_failure_rolls_back_new_project(self) -> None:
        """Installing an incomplete template set must not commit profile."""

        workspace = self.root / "workspace"
        workspace.mkdir()
        keep = workspace / "keep.txt"
        keep.write_text("user-owned\n")
        with patch(
            "kilin.project.install_project_resources",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                initialize_project(workspace, "literature")
        self.assertFalse((workspace / "profile.ini").exists())
        self.assertEqual(keep.read_text(), "user-owned\n")
        self.assertFalse((workspace / "templates").exists())
        project = initialize_project(workspace, "literature")
        self.assertTrue((project.root / "profile.ini").is_file())
        self.assertTrue(
            (project.root / "templates/research_briefing.md").is_file()
        )

    def test_partial_resource_copy_is_rolled_back(self) -> None:
        """A failed template write does not leave half a packaged file."""

        workspace = self.root / "workspace"
        workspace.mkdir()

        def interrupted(source, destination):
            destination.write(source.read(7))
            raise OSError("resource write failed")

        with patch(
            "kilin.project_layout.shutil.copyfileobj",
            side_effect=interrupted,
        ):
            with self.assertRaisesRegex(OSError, "resource write failed"):
                initialize_project(workspace, "literature")
        self.assertFalse((workspace / "profile.ini").exists())
        self.assertFalse(
            (workspace / "templates/research_briefing.md").exists()
        )

    def test_final_profile_commit_failure_preserves_user_files(self) -> None:
        """A failed atomic profile write rolls back created resources."""

        workspace = self.root / "workspace"
        workspace.mkdir()
        keep = workspace / "original.md"
        keep.write_text("manual\n")
        with patch(
            "kilin.project.atomic_write_text",
            side_effect=OSError("cannot commit"),
        ):
            with self.assertRaisesRegex(OSError, "cannot commit"):
                initialize_project(workspace, "literature")
        self.assertFalse((workspace / "profile.ini").exists())
        self.assertEqual(keep.read_text(), "manual\n")
        self.assertFalse((workspace / "templates").exists())

    def test_profile_is_committed_after_resources_are_ready(self) -> None:
        """The final profile write observes installed templates and folders."""

        workspace = self.root / "workspace"
        workspace.mkdir()
        from kilin.project import atomic_write_text

        def inspect_commit(path, text):
            self.assertTrue(
                (workspace / "templates/research_briefing.md").is_file()
            )
            self.assertTrue((workspace / "outputs").is_dir())
            self.assertFalse((workspace / "profile.ini").exists())
            return atomic_write_text(path, text)

        with patch(
            "kilin.project.atomic_write_text", side_effect=inspect_commit
        ):
            initialize_project(workspace, "literature")

    def test_multisource_copy_failure_rolls_back_project(self) -> None:
        """New-project source copies are prepared before profile commit."""

        workspace = self.root / "workspace"
        from kilin import workspace as source_module

        with patch.object(
            source_module, "_copy_file", side_effect=OSError("copy fail")
        ):
            with self.assertRaisesRegex(OSError, "copy fail"):
                prepare_sources_workspace(
                    [self.original], "literature", workspace
                )
        self.assertFalse((workspace / "profile.ini").exists())
        self.assertFalse((workspace / "sources.json").exists())
        project = prepare_sources_workspace(
            [self.original], "literature", workspace
        )
        self.assertTrue((project.root / "profile.ini").exists())

    def test_mirror_copy_failure_does_not_commit(self) -> None:
        """Mirrored workspaces keep the source original and stay retryable."""

        origin = self.root / "repository"
        origin.mkdir()
        (origin / "code.py").write_text("number = 2\n")
        sandbox = self.root / "sandbox"
        with patch(
            "kilin.project.shutil.copy2", side_effect=OSError("copy fail")
        ):
            with self.assertRaisesRegex(OSError, "copy fail"):
                prepare_project_workspace(origin, "code", sandbox)
        workspace = sandbox / "project_repository"
        self.assertFalse((workspace / "profile.ini").exists())
        self.assertFalse((workspace / "source_snapshot.json").exists())
        self.assertTrue((origin / "code.py").exists())
        project = prepare_project_workspace(origin, "code", sandbox)
        self.assertTrue((project.root / "profile.ini").is_file())
        self.assertTrue((project.source / "code.py").is_file())

    def test_existing_project_never_restores_deleted_template(self) -> None:
        """Reopening preserves a user's deliberate template deletion."""

        root = self.root / "existing"
        root.mkdir()
        project = initialize_project(root, "literature")
        deleted = project.root / "templates/research_briefing.md"
        deleted.unlink()
        initialize_project(root, "literature")
        self.assertFalse(deleted.exists())


if __name__ == "__main__":
    unittest.main()
