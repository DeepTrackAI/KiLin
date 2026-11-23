"""Protect saved and partially created projects during forceful GUI Stop.

These Qt-free tests exercise the filesystem policy used after killing the
engine. A stopped process ID and delayed GUI events never authorize recursive
removal of workspace contents.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin._project_creation import (
    claim_workspace,
    cleanup_stopped_creation,
)


class TestStoppedCreationCleanup(unittest.TestCase):
    """Only empty, identifiable creation claims may be removed."""

    def setUp(self) -> None:
        """Keep all candidate workspaces inside a temporary directory."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()

    def make_claim(
        self, name: str, pid: int = 42, *, marker: str | None = None
    ) -> Path:
        """Create a directory with a realistic exclusive-creation marker.

        Parameters
        ----------
        name: str
            Folder name within the temporary projects parent.
        pid: int, optional
            Claimed process ID; defaults to 42.
        marker: str | None, optional
            Literal marker contents, or the standard value when omitted.

        Returns
        -------
        pathlib.Path
            Created workspace directory.

        """

        folder = self.parent / name
        folder.mkdir()
        (folder / ".kilin-creating").write_text(
            marker if marker is not None
            else f"pid={pid}\ncreated=1\n",
            encoding="ascii",
        )
        return folder

    def test_completed_project_survives_stale_project_created_event(self):
        """A published profile is safe even when opened is not processed."""

        project = self.parent / "committed"
        (project / "sources").mkdir(parents=True)
        (project / "profile.ini").write_text("[project]\n", encoding="utf-8")
        (project / "sources" / "user.txt").write_bytes(b"user content")
        removed, preserved = cleanup_stopped_creation(
            self.parent, project, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertEqual(
            (project / "sources" / "user.txt").read_bytes(),
            b"user content",
        )

    def test_even_matching_marker_cannot_delete_published_project(self):
        """Presence of profile or any other entry blocks removal."""

        project = self.make_claim("published")
        (project / "profile.ini").write_text("[project]\n", encoding="utf-8")
        (project / "notes.txt").write_text("do not remove", encoding="utf-8")
        removed, preserved = cleanup_stopped_creation(
            self.parent, None, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertEqual((project / "notes.txt").read_text(
            encoding="utf-8",
        ), "do not remove")
        self.assertTrue((project / ".kilin-creating").exists())

    def test_partial_source_files_are_retained_for_manual_recovery(self):
        """A PID marker is not permission to erase copied or edited files."""

        project = self.make_claim("copying")
        (project / "sources").mkdir()
        (project / "sources" / "partial.py").write_text(
            "value = 1\n", encoding="utf-8",
        )
        removed, preserved = cleanup_stopped_creation(
            self.parent, None, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertTrue((project / ".kilin-creating").exists())
        self.assertEqual(
            (project / "sources" / "partial.py").read_text(encoding="utf-8"),
            "value = 1\n",
        )

    def test_only_empty_marker_owned_claim_is_removed(self):
        """Use rmdir only where no project file could be lost."""

        empty = self.make_claim("empty")
        other = self.make_claim("other", 7)
        removed, preserved = cleanup_stopped_creation(
            self.parent, None, 42
        )
        self.assertEqual(removed, [str(empty)])
        self.assertEqual(preserved, [])
        self.assertFalse(empty.exists())
        self.assertTrue(other.is_dir())
        self.assertEqual(
            cleanup_stopped_creation(self.parent, None, 42), ([], [])
        )

    def test_reported_workspace_is_protected_even_if_marker_only(self):
        """Never infer that a queued opened event makes work disposable."""

        project = self.make_claim("reported")
        removed, preserved = cleanup_stopped_creation(
            self.parent, project, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertTrue(project.is_dir())

    def test_unknown_or_other_engine_pid_never_grants_cleanup(self):
        """Only a matching stopped child may release an empty claim."""

        project = self.make_claim("other-engine", 7)
        for pid in (0, -1, 41):
            with self.subTest(pid=pid):
                self.assertEqual(
                    cleanup_stopped_creation(self.parent, None, pid),
                    ([], []),
                )
                self.assertTrue(project.is_dir())

    def test_core_markers_prove_created_vs_existing_directories(self):
        """Exercise both marker variants produced by the real core API."""

        existing = self.parent / "existing"
        existing.mkdir()
        with claim_workspace(existing):
            marker = (existing / ".kilin-creating").read_text(encoding="utf-8")
            self.assertEqual(
                marker, f"pid={os.getpid()}\ncreated=0\n"
            )
            self.assertEqual(
                cleanup_stopped_creation(
                    self.parent, None, os.getpid()
                ),
                ([], [str(existing)]),
            )
        self.assertTrue(existing.is_dir())
        self.assertEqual(list(existing.iterdir()), [])

        fresh = self.parent / "fresh"
        with claim_workspace(fresh):
            marker = (fresh / ".kilin-creating").read_text(encoding="utf-8")
            self.assertEqual(
                marker, f"pid={os.getpid()}\ncreated=1\n"
            )
        self.assertFalse(fresh.exists())

    def test_preexisting_empty_folder_is_never_removed(self):
        """A marker must prove the attempt created this exact directory."""

        project = self.make_claim(
            "preexisting", marker="pid=42\ncreated=0\n"
        )
        removed, preserved = cleanup_stopped_creation(
            self.parent, None, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertTrue(project.is_dir())

    def test_old_marker_without_created_flag_is_preserved(self):
        """Do not guess ownership from an older PID-only marker."""

        project = self.make_claim("old", marker="pid=42\n")
        removed, preserved = cleanup_stopped_creation(
            self.parent, None, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertTrue(project.is_dir())
        self.assertTrue((project / ".kilin-creating").is_file())

    def test_malformed_marker_is_not_an_ownership_lease(self):
        """Reject truncated or appended marker text, without guessing."""

        invalid = ("pid=42", "pid=42\ncreated=1\nextra", "")
        for index, value in enumerate(invalid):
            project = self.make_claim(f"bad-{index}", marker=value)
            self.assertTrue(project.exists())
        self.assertEqual(
            cleanup_stopped_creation(self.parent, None, 42), ([], [])
        )
        self.assertEqual(len(list(self.parent.iterdir())), 3)

    def test_links_and_folders_outside_parent_are_not_deleted(self):
        """Follow neither linked workspaces nor linked marker files."""

        outside = self.parent.parent / (self.parent.name + "-outside")
        outside.mkdir()
        self.addCleanup(lambda: outside.rmdir())
        other = outside / "notes.txt"
        other.write_text("keep", encoding="utf-8")
        self.addCleanup(lambda: other.unlink())
        linked_folder = self.parent / "linked"
        linked_marker = self.parent / "marker-link"
        linked_marker.mkdir()
        try:
            linked_folder.symlink_to(outside, target_is_directory=True)
            (linked_marker / ".kilin-creating").symlink_to(other)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        removed, preserved = cleanup_stopped_creation(
            self.parent, outside, 42
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(outside)])
        self.assertTrue(linked_folder.is_symlink())
        self.assertTrue((linked_marker / ".kilin-creating").is_symlink())
        self.assertEqual(other.read_text(encoding="utf-8"), "keep")

    def test_new_file_added_while_stopping_is_never_deleted(self):
        """If a concurrent entry appears, rmdir must fail closed."""

        project = self.make_claim("racing")
        marker = project / ".kilin-creating"
        unlink = Path.unlink

        def add_file_before_unlink(path, *args, **kwargs):
            if path == marker:
                (project / "notes.txt").write_text("keep me", encoding="utf-8")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", add_file_before_unlink):
            removed, preserved = cleanup_stopped_creation(
                self.parent, None, 42
            )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(project)])
        self.assertEqual((project / "notes.txt").read_text(
            encoding="utf-8",
        ), "keep me")

    def test_reported_custom_workspace_does_not_authorize_deletion(self):
        """A workspace outside the default parent can only be reported."""

        project = self.make_claim("unrelated")
        outer = self.parent.parent / (self.parent.name + "-custom")
        outer.mkdir()
        self.addCleanup(outer.rmdir)
        (outer / "profile.ini").write_text("[project]\n", encoding="utf-8")
        self.addCleanup(lambda: (outer / "profile.ini").unlink())
        removed, preserved = cleanup_stopped_creation(
            self.parent, outer, 0
        )
        self.assertEqual(removed, [])
        self.assertEqual(preserved, [str(outer)])
        self.assertTrue(project.is_dir())
        self.assertTrue((outer / "profile.ini").is_file())


if __name__ == "__main__":
    unittest.main()
