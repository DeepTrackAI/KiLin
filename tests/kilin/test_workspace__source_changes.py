"""Test finding and syncing source copies whose originals changed.

These tests change originals and copies after an import and check which
copies are listed as changed, deleted, or edited, which sources are
skipped, that the check changes nothing, and that a sync updates or
removes copies while keeping the user's edits.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from kilin.workspace import (
    find_source_changes,
    prepare_sources_workspace,
    sync_project_sources,
)


class TestSourceChanges(unittest.TestCase):
    """Compare originals and copies with their import stamps."""

    def setUp(self) -> None:
        """Import a two-file folder and a direct file into a new project."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.folder = self.root / "repository"
        self.folder.mkdir()
        for name in ("a.py", "b.py"):
            self._write(self.folder / name, "x = 1\n", 1)
        self.single = self._write(self.root / "notes.md", "notes\n", 1)
        self.project = prepare_sources_workspace(
            [self.folder, self.single], "code", self.root / "workspace"
        )
        self.copies = self.project.root / "sources"

    @staticmethod
    def _write(path: Path, content: str, second: int) -> Path:
        """Write a file and give it a fixed modification time.

        Parameters
        ----------
        path: pathlib.Path
            File to write.
        content: str
            New contents.
        second: int
            Modification time in seconds since the epoch.

        Returns
        -------
        pathlib.Path
            The written file.

        """

        path.write_text(content, encoding="utf-8")
        os.utime(path, ns=(second * 10**9, second * 10**9))
        return path

    def _changes(self, changed=(), deleted=(), edited=()) -> None:
        """Check the copies listed by ``find_source_changes``.

        Parameters
        ----------
        changed, deleted, edited: Sequence[str], optional
            Expected copies in each group, relative to ``sources/`` and
            sorted. Default to none.

        """

        self.assertEqual(
            find_source_changes(self.project),
            {
                "changed": [f"sources/{path}" for path in changed],
                "deleted": [f"sources/{path}" for path in deleted],
                "edited": [f"sources/{path}" for path in edited],
            },
        )

    def test_nothing_changed(self) -> None:
        """Freshly imported copies match their originals."""

        self._changes()

    def test_changed_originals_are_listed(self) -> None:
        """Changing an original lists its copy as changed."""

        self._write(self.folder / "a.py", "x = 22\n", 2)
        self._write(self.single, "more notes\n", 2)
        self._changes(changed=["notes.md", "repository/a.py"])

    def test_same_size_with_new_time_is_changed(self) -> None:
        """A new modification time alone counts as a change."""

        self._write(self.folder / "a.py", "x = 2\n", 2)
        self._changes(changed=["repository/a.py"])

    def test_deleted_original_is_listed(self) -> None:
        """Deleting a file from the original folder lists its copy."""

        (self.folder / "b.py").unlink()
        self._changes(deleted=["repository/b.py"])

    def test_copy_edited_in_project_is_kept(self) -> None:
        """A copy edited in the project is listed as edited, not changed."""

        self._write(self.folder / "a.py", "x = 22\n", 2)
        self._write(self.copies / "repository/a.py", "x = 333\n", 3)
        (self.folder / "b.py").unlink()
        self._write(self.copies / "repository/b.py", "y = 1\n", 3)
        self._changes(edited=["repository/a.py", "repository/b.py"])

    def test_copy_edited_with_unchanged_original_is_not_listed(self) -> None:
        """Only copies whose originals changed or were deleted are listed."""

        self._write(self.copies / "repository/a.py", "x = 333\n", 3)
        self._changes()

    def test_new_file_in_original_is_not_listed(self) -> None:
        """Files added to an original folder after import are left out."""

        self._write(self.folder / "c.py", "z = 1\n", 2)
        self._changes()

    def test_missing_or_empty_originals_are_skipped(self) -> None:
        """A missing or empty original is ignored, not reported deleted."""

        self.single.unlink()
        for path in self.folder.iterdir():
            path.unlink()
        self._changes()
        self.folder.rmdir()
        self._changes()

    def test_original_of_another_kind_is_skipped(self) -> None:
        """A folder replaced by a file is ignored."""

        shutil.rmtree(self.folder)
        self._write(self.folder, "now a file\n", 2)
        self._changes()

    def test_files_without_stamps_are_skipped(self) -> None:
        """Copies imported before stamps existed cannot be compared."""

        manifest = self.project.root / "sources.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for record in payload["sources"]:
            del record["stamps"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        self._write(self.folder / "a.py", "x = 22\n", 2)
        self._changes()

    def test_nothing_is_modified(self) -> None:
        """The check changes neither copies nor the manifest."""

        manifest = self.project.root / "sources.json"
        before = manifest.read_bytes()
        self._write(self.folder / "a.py", "x = 22\n", 2)
        (self.folder / "b.py").unlink()
        find_source_changes(self.project)
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(
            (self.copies / "repository/a.py").read_text(encoding="utf-8"),
            "x = 1\n",
        )
        self.assertTrue((self.copies / "repository/b.py").is_file())

    def test_requires_an_opened_project(self) -> None:
        """Anything other than an opened project is rejected."""

        with self.assertRaises(TypeError):
            find_source_changes(self.project.root)  # type: ignore[arg-type]


    def test_sync_updates_removes_and_keeps(self) -> None:
        """Changed copies are updated, deleted ones removed, edits kept."""

        self._write(self.single, "more notes\n", 2)
        (self.folder / "a.py").unlink()
        self._write(self.folder / "b.py", "y = 22\n", 2)
        self._write(self.copies / "repository/b.py", "mine\n", 3)
        done = sync_project_sources(self.project)
        self.assertEqual(
            done,
            {
                "changed": ["sources/notes.md"],
                "deleted": ["sources/repository/a.py"],
                "edited": ["sources/repository/b.py"],
            },
        )
        self.assertEqual(
            (self.copies / "notes.md").read_text(encoding="utf-8"),
            "more notes\n",
        )
        self.assertFalse((self.copies / "repository/a.py").exists())
        self.assertEqual(
            (self.copies / "repository/b.py").read_text(encoding="utf-8"),
            "mine\n",
        )
        self.assertTrue(self.single.is_file())
        self._changes(edited=["repository/b.py"])

    def test_sync_renews_the_manifest(self) -> None:
        """Updated copies get new stamps; removed copies leave the record."""

        self._write(self.folder / "a.py", "x = 22\n", 2)
        (self.folder / "b.py").unlink()
        sync_project_sources(self.project)
        manifest = self.project.root / "sources.json"
        records = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
        folder = next(item for item in records if item["name"] == "repository")
        self.assertEqual(folder["files"], ["a.py"])
        self.assertEqual(
            folder["stamps"], {"a.py": {"size": 7, "mtime_ns": 2 * 10**9}}
        )

    def test_sync_removes_folders_left_empty(self) -> None:
        """A renamed subfolder does not leave an empty copy behind."""

        (self.folder / "lib").mkdir()
        nested = self._write(self.folder / "lib/c.py", "z = 1\n", 1)
        second = prepare_sources_workspace(
            [self.folder], "code", self.root / "second"
        )
        nested.unlink()
        nested.parent.rmdir()
        sync_project_sources(second)
        self.assertFalse((second.root / "sources/repository/lib").exists())
        self.assertTrue((second.root / "sources/repository/a.py").is_file())

    def test_sync_without_changes_writes_nothing(self) -> None:
        """Nothing to sync leaves sources.json untouched."""

        manifest = self.project.root / "sources.json"
        before = manifest.stat().st_mtime_ns
        self.assertEqual(
            sync_project_sources(self.project),
            {"changed": [], "deleted": [], "edited": []},
        )
        self.assertEqual(manifest.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
