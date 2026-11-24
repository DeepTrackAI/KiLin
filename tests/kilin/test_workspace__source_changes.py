"""Test finding and syncing source copies whose originals changed.

These tests change originals and copies after an import and check which
copies are listed as changed, deleted, or edited, including missing and
relocated original paths. Checks do not mutate snapshots, while explicit
synchronization updates or removes copies without overwriting user edits.

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
from unittest.mock import patch

from kilin.project import open_project
from kilin.workspace import (
    find_source_changes,
    list_source_versions,
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

        path.write_bytes(content.encode("utf-8"))
        os.utime(path, ns=(second * 10**9, second * 10**9))
        return path

    def _changes(
        self, added=(), changed=(), deleted=(), edited=()
    ) -> None:
        """Check the copies listed by ``find_source_changes``.

        Parameters
        ----------
        added: Sequence[str], optional
            Expected newly discovered paths. Defaults to none.
        changed: Sequence[str], optional
            Expected changed paths relative to ``sources/``. Defaults to none.
        deleted: Sequence[str], optional
            Expected deleted paths relative to ``sources/``. Defaults to none.
        edited: Sequence[str], optional
            Expected edited paths relative to ``sources/``. Defaults to none.

        """

        self.assertEqual(
            find_source_changes(self.project),
            {
                "added": [f"sources/{path}" for path in added],
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

    def test_new_file_in_original_is_listed(self) -> None:
        """New originals can be selected without reimporting the folder."""

        self._write(self.folder / "c.py", "z = 1\n", 2)
        self._changes(added=["repository/c.py"])

    def test_missing_or_empty_originals_are_deleted(self) -> None:
        """Missing direct files and files in empty folders are detected."""

        self.single.unlink()
        for path in self.folder.iterdir():
            path.unlink()
        deleted = ["notes.md", "repository/a.py", "repository/b.py"]
        self._changes(deleted=deleted)
        self.folder.rmdir()
        self._changes(deleted=deleted)

    def test_original_of_another_kind_is_deleted(self) -> None:
        """Replacing a folder with a file removes the tracked old paths."""

        shutil.rmtree(self.folder)
        self._write(self.folder, "now a file\n", 2)
        self._changes(deleted=["repository/a.py", "repository/b.py"])

    def test_direct_original_replaced_by_directory_is_deleted(self) -> None:
        """Changing a direct file to a directory invalidates its copy."""

        self.single.unlink()
        self.single.mkdir()
        self._changes(deleted=["notes.md"])

    def test_renamed_or_moved_paths_are_deletions(self) -> None:
        """Old paths are deleted without importing new paths automatically."""

        (self.folder / "a.py").rename(self.folder / "renamed.py")
        (self.folder / "sub").mkdir()
        (self.folder / "b.py").rename(self.folder / "sub/b.py")
        self._changes(
            added=["repository/renamed.py", "repository/sub/b.py"],
            deleted=["repository/a.py", "repository/b.py"],
        )
        sync_project_sources(self.project)
        self.assertFalse((self.copies / "repository/a.py").exists())
        self.assertFalse((self.copies / "repository/b.py").exists())
        self.assertFalse((self.copies / "repository/renamed.py").exists())
        self.assertFalse((self.copies / "repository/sub/b.py").exists())

    def test_original_directory_relocation_is_deletion(self) -> None:
        """Relocating the source directory is not followed implicitly."""

        self.folder.rename(self.root / "repository_moved")
        self._changes(deleted=["repository/a.py", "repository/b.py"])
        done = sync_project_sources(self.project)
        self.assertEqual(
            done["deleted"],
            ["sources/repository/a.py", "sources/repository/b.py"],
        )
        self.assertFalse((self.copies / "repository/a.py").exists())
        self.assertFalse((self.copies / "repository/b.py").exists())

    def test_symlink_original_is_not_followed(self) -> None:
        """Replacing an original with a link never causes a file copy."""

        target = self.root / "another.py"
        self._write(target, "updated\n", 3)
        (self.folder / "a.py").unlink()
        (self.folder / "a.py").symlink_to(target)
        self._changes()

    def test_linked_original_parent_is_not_followed(self) -> None:
        """A new symlinked parent cannot supply replacement content."""

        nested = self.folder / "lib"
        nested.mkdir()
        self._write(nested / "c.py", "z = 1\n", 1)
        second = prepare_sources_workspace(
            [self.folder], "code", self.root / "second"
        )
        shutil.rmtree(nested)
        external = self.root / "external"
        external.mkdir()
        self._write(external / "c.py", "external\n", 3)
        nested.symlink_to(external, target_is_directory=True)
        self.assertEqual(
            find_source_changes(second),
            {"added": [], "changed": [], "deleted": [], "edited": []},
        )
        self.assertEqual(
            sync_project_sources(second),
            {"added": [], "changed": [], "deleted": [], "edited": []},
        )
        self.assertEqual(
            (second.source / "repository/lib/c.py").read_text(),
            "z = 1\n",
        )

    def test_unreadable_original_is_not_treated_as_deleted(self) -> None:
        """Filesystem access failures abort checking rather than deleting."""

        original = self.folder / "a.py"
        original_lstat = Path.lstat

        def guarded_lstat(path: Path):
            """Simulate an inaccessible tracked file."""

            if path == original:
                raise PermissionError("source unavailable")
            return original_lstat(path)

        with patch.object(Path, "lstat", guarded_lstat):
            with self.assertRaises(PermissionError):
                find_source_changes(self.project)
        self.assertTrue((self.copies / "repository/a.py").is_file())

    def test_identical_metadata_does_not_prove_identical_content(self) -> None:
        """The documented stat heuristic intentionally misses this case."""

        target = self.copies / "repository/a.py"
        stamp = target.stat()
        self._write(target, "z = 1\n", 1)
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self._write(self.folder / "a.py", "y = 1\n", 2)
        self._changes(changed=["repository/a.py"])
        # The copy has the old stamp, despite having different bytes.
        # This is a documented limitation rather than a silent regression.

    def test_hash_avoids_recopied_identical_contents(self) -> None:
        """Reverted contents need only filesystem metadata and new stamp."""

        original = self.folder / "a.py"
        copy = self.copies / "repository/a.py"
        inode = copy.stat().st_ino
        self._write(original, "x = 22\n", 2)
        self._write(original, "x = 1\n", 3)
        self._changes(changed=["repository/a.py"])
        with patch("kilin.workspace._copy_file") as copy_file:
            done = sync_project_sources(self.project)
        copy_file.assert_not_called()
        self.assertEqual(done["changed"], ["sources/repository/a.py"])
        self.assertEqual(copy.stat().st_mtime_ns, 3 * 10**9)
        self.assertEqual(copy.stat().st_ino, inode)
        self.assertEqual(copy.read_text(encoding="utf-8"), "x = 1\n")
        self._changes()

    def test_sync_replaces_different_contents_with_same_size(self) -> None:
        """A same-sized content change still replaces an unedited copy."""

        original = self.folder / "a.py"
        self._write(original, "x = 2\n", 2)
        done = sync_project_sources(self.project)
        self.assertEqual(done["changed"], ["sources/repository/a.py"])
        self.assertEqual(
            (self.copies / "repository/a.py").read_text(
                encoding="utf-8"
            ),
            "x = 2\n",
        )
        self._changes()

    def test_removed_workspace_copy_is_not_recreated(self) -> None:
        """A deliberate user removal is not undone by synchronization."""

        (self.copies / "repository/a.py").unlink()
        self._write(self.folder / "a.py", "x = 22\n", 2)
        self._changes()
        self.assertEqual(
            sync_project_sources(self.project),
            {"added": [], "changed": [], "deleted": [], "edited": []},
        )
        self.assertFalse((self.copies / "repository/a.py").exists())

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
                "added": [],
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

    def test_sync_removes_all_missing_originals(self) -> None:
        """Sync can remove a direct file and last files in an empty folder."""

        self.single.unlink()
        for file in self.folder.iterdir():
            file.unlink()
        done = sync_project_sources(self.project)
        self.assertEqual(
            done["deleted"],
            [
                "sources/notes.md", "sources/repository/a.py",
                "sources/repository/b.py",
            ],
        )
        self.assertFalse((self.copies / "notes.md").exists())
        self.assertFalse((self.copies / "repository/a.py").exists())
        self.assertFalse((self.copies / "repository/b.py").exists())
        self._changes()
        manifest = self.project.root / "sources.json"
        for record in json.loads(manifest.read_text())["sources"]:
            self.assertEqual(record["files"], [])
            self.assertEqual(record["stamps"], {})

    def test_deleted_original_keeps_edited_copy(self) -> None:
        """A missing original does not discard a modified workspace copy."""

        target = self.copies / "repository/a.py"
        self._write(target, "mine\n", 3)
        self.folder.rename(self.root / "repository_moved")
        self._changes(
            deleted=["repository/b.py"], edited=["repository/a.py"]
        )
        done = sync_project_sources(self.project)
        self.assertEqual(done["edited"], ["sources/repository/a.py"])
        self.assertEqual(target.read_text(encoding="utf-8"), "mine\n")
        self.assertFalse((self.copies / "repository/b.py").exists())

    def test_only_edited_does_not_rewrite_manifest(self) -> None:
        """Listing an edited copy makes no manifest changes."""

        self._write(self.folder / "a.py", "new\n", 2)
        self._write(self.copies / "repository/a.py", "mine\n", 3)
        manifest = self.project.root / "sources.json"
        before = manifest.read_bytes()
        with patch("kilin.workspace._write_records") as save:
            result = sync_project_sources(self.project)
        save.assert_not_called()
        self.assertEqual(result["edited"], ["sources/repository/a.py"])
        self.assertEqual(manifest.read_bytes(), before)

    def test_sync_rejects_updated_file_size_limit(self) -> None:
        """Oversized or empty originals cannot replace indexed copies."""

        original = self.folder / "a.py"
        copy = self.copies / "repository/a.py"
        manifest = self.project.root / "sources.json"
        before = manifest.read_bytes()
        original.write_bytes(b"x" * (self.project.profile.max_file_bytes + 1))
        self._changes(changed=["repository/a.py"])
        with self.assertRaisesRegex(ValueError, "file-size limit"):
            sync_project_sources(self.project)
        with self.assertRaisesRegex(ValueError, "file-size limit"):
            sync_project_sources(self.project, decisions={
                "sources/repository/a.py": "keep_both",
            })
        self.assertEqual(copy.read_text(encoding="utf-8"), "x = 1\n")
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(list_source_versions(self.project), [])
        original.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "file-size limit"):
            sync_project_sources(self.project)
        self.assertEqual(manifest.read_bytes(), before)


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
            {"added": [], "changed": [], "deleted": [], "edited": []},
        )
        self.assertEqual(manifest.stat().st_mtime_ns, before)


    def test_nested_new_folder_is_discovered_recursively(self) -> None:
        """A new nested folder contributes only eligible nonempty files."""

        nested = self.folder / "analysis" / "runs"
        nested.mkdir(parents=True)
        self._write(nested / "model.py", "result = 4\n", 2)
        self._write(nested / "readme.md", "# Notes\n", 2)
        self._write(nested / "binary.bin", "raw", 2)
        (self.folder / "empty").mkdir()
        expected = [
            "repository/analysis/runs/model.py",
            "repository/analysis/runs/readme.md",
        ]
        self._changes(added=expected)
        self.assertFalse((self.copies / expected[0]).exists())
        decision = {f"sources/{expected[0]}": "import"}
        done = sync_project_sources(self.project, decisions=decision)
        self.assertEqual(done["added"], [f"sources/{expected[0]}"])
        self.assertEqual(
            (self.copies / expected[0]).read_text(encoding="utf-8"),
            "result = 4\n",
        )
        self._changes(added=[expected[1]])

    def test_file_by_file_decisions_preserve_unselected_sources(self) -> None:
        """An explicit decision updates only the selected original."""

        self._write(self.folder / "a.py", "a = 20\n", 2)
        self._write(self.folder / "b.py", "b = 30\n", 2)
        done = sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "replace",
        })
        self.assertEqual(done["changed"], ["sources/repository/a.py"])
        self.assertEqual(done["added"], [])
        self.assertEqual(
            (self.copies / "repository/b.py").read_text(), "x = 1\n"
        )
        self._changes(changed=["repository/b.py"])
        self.assertEqual(
            sync_project_sources(self.project, decisions={}),
            {"added": [], "changed": [], "deleted": [], "edited": []},
        )

    def test_invalid_batch_rejected_before_any_copy(self) -> None:
        """An invalid action cannot partially apply an earlier decision."""

        self._write(self.folder / "a.py", "a = 20\n", 2)
        self._write(self.folder / "b.py", "b = 30\n", 2)
        with self.assertRaises(ValueError):
            sync_project_sources(self.project, decisions={
                "sources/repository/a.py": "replace",
                "sources/repository/b.py": "delete",
            })
        self.assertEqual(
            (self.copies / "repository/a.py").read_text(), "x = 1\n"
        )
        with self.assertRaises(ValueError):
            sync_project_sources(self.project, decisions={
                "sources/repository/../notes.md": "replace",
            })
        self.assertEqual(list_source_versions(self.project), [])

    def test_keep_both_preserves_distinct_retrievable_versions(self) -> None:
        """Versioned copies have independent paths and saved ownership."""

        self._write(self.folder / "a.py", "value = 2\n", 2)
        done = sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        self.assertEqual(done["changed"], ["sources/repository/a.py"])
        self.assertEqual(
            (self.copies / "repository/a.py").read_text(), "value = 2\n"
        )
        first = list_source_versions(self.project)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["source"], "sources/repository/a.py")
        self.assertTrue(first[0]["version"].endswith(".py"))
        self.assertIn("/.kilin_versions/", "/" + first[0]["version"])
        self.assertEqual(
            (self.project.root / first[0]["version"]).read_text(),
            "x = 1\n",
        )
        self._write(self.folder / "a.py", "value = 3\n", 3)
        sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        versions = list_source_versions(self.project)
        self.assertEqual(len(versions), 2)
        self.assertNotEqual(versions[0]["version"], versions[1]["version"])
        self.assertEqual(
            {
                (self.project.root / item["version"]).read_text()
                for item in versions
            }, {"x = 1\n", "value = 2\n"}
        )
        manifest = json.loads(
            (self.project.root / "sources.json").read_text()
        )
        folder = next(
            item for item in manifest["sources"]
            if item["name"] == "repository"
        )
        self.assertEqual(len(folder["versions"]), 2)

    def test_keep_both_saves_independently_edited_copy(self) -> None:
        """An explicitly retained edit survives an original update."""

        target = self.copies / "repository/a.py"
        self._write(target, "my edit\n", 3)
        self._write(self.folder / "a.py", "upstream\n", 2)
        self._changes(edited=["repository/a.py"])
        sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        self.assertEqual(target.read_text(), "upstream\n")
        version = list_source_versions(self.project)[0]
        self.assertEqual(
            (self.project.root / version["version"]).read_text(),
            "my edit\n",
        )

    def test_metadata_only_keep_both_does_not_duplicate_content(self) -> None:
        """No version is needed when the original bytes are unchanged."""

        self._write(self.folder / "a.py", "x = 1\n", 2)
        sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        self.assertEqual(list_source_versions(self.project), [])
        self._changes()

    def test_deleted_keep_detaches_copy_from_missing_origin(self) -> None:
        """Restoring a kept original permits safe, explicit reconciliation."""

        copy = self.copies / "repository/a.py"
        (self.folder / "a.py").unlink()
        done = sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep",
        })
        self.assertEqual(done["deleted"], [])
        self.assertTrue(copy.is_file())
        self._changes()
        self._write(self.folder / "a.py", "new upstream\n", 2)
        self._changes(changed=["repository/a.py"])
        # Detached copies cannot be replaced by the automatic policy.
        self.assertEqual(sync_project_sources(self.project), {
            "added": [], "changed": [], "deleted": [], "edited": [],
        })
        self.assertEqual(copy.read_text(), "x = 1\n")
        sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        self.assertEqual(copy.read_text(), "new upstream\n")
        version = list_source_versions(self.project)[0]
        self.assertEqual(
            (self.project.root / version["version"]).read_text(),
            "x = 1\n",
        )
        self._changes()


    def test_keep_deleted_direct_file_reconciles_without_overwriting(
        self,
    ) -> None:
        """A retained direct file can be safely reconnected to its origin."""

        copy = self.copies / "notes.md"
        self.single.unlink()
        sync_project_sources(self.project, decisions={
            "sources/notes.md": "keep",
        })
        self._changes()
        self._write(self.single, "notes restored\n", 3)
        self._changes(changed=["notes.md"])
        sync_project_sources(self.project)
        self.assertEqual(copy.read_text(), "notes\n")
        sync_project_sources(self.project, decisions={
            "sources/notes.md": "replace",
        })
        self.assertEqual(copy.read_text(), "notes restored\n")
        self._changes()

    def test_delete_only_direct_file_keeps_managed_sources(self) -> None:
        """The final direct-file deletion must not remove ``sources/``."""

        directory = self.root / "last-project"
        single_project = prepare_sources_workspace(
            [self.single], "literature", directory
        )
        self.single.unlink()
        result = sync_project_sources(single_project)
        self.assertEqual(result["deleted"], ["sources/notes.md"])
        self.assertTrue(single_project.source.is_dir())
        reopened = open_project(directory)
        self.assertEqual(reopened.source, single_project.source)
        self.assertFalse((reopened.source / "notes.md").exists())
        self.assertEqual(find_source_changes(reopened)["deleted"], [])

    def test_deletion_never_prunes_folder_or_sibling(self) -> None:
        """Folder cleanup stops at its imported directory boundary."""

        directory = self.root / "folder-only"
        folder = self.root / "nested"
        folder.mkdir()
        nested = folder / "sub"
        nested.mkdir()
        file = self._write(nested / "a.py", "a = 1\n", 1)
        project = prepare_sources_workspace(
            [folder], "code", directory
        )
        file.unlink()
        result = sync_project_sources(project)
        self.assertEqual(result["deleted"], ["sources/nested/sub/a.py"])
        self.assertTrue(project.source.is_dir())
        self.assertTrue((project.source / "nested").is_dir())
        self.assertFalse((project.source / "nested/sub").exists())
        self.assertTrue(open_project(project.root).source.is_dir())

    def test_direct_file_reappearing_is_a_new_original(self) -> None:
        """Re-created direct imports can be imported individually."""

        self.single.unlink()
        sync_project_sources(self.project)
        self._write(self.single, "notes restored\n", 3)
        self._changes(added=["notes.md"])
        sync_project_sources(self.project, decisions={
            "sources/notes.md": "import",
        })
        self._changes()
        self.assertEqual((self.copies / "notes.md").read_text(),
                         "notes restored\n")

    def test_archive_failure_does_not_overwrite_original_copy(self) -> None:
        """An archive I/O error must leave the active snapshot intact."""

        self._write(self.folder / "a.py", "upstream\n", 2)
        with patch(
            "kilin.workspace._archive_source_version",
            side_effect=OSError("archive failed"),
        ):
            with self.assertRaisesRegex(OSError, "archive failed"):
                sync_project_sources(self.project, decisions={
                    "sources/repository/a.py": "keep_both",
                })
        self.assertEqual(
            (self.copies / "repository/a.py").read_text(), "x = 1\n"
        )
        self._changes(changed=["repository/a.py"])

    def test_version_survives_removal_of_active_copy(self) -> None:
        """Deleting the active source does not delete retained history."""

        from kilin.workspace import remove_workspace_sources

        self._write(self.folder / "a.py", "upstream\n", 2)
        sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        version = list_source_versions(self.project)[0]
        remove_workspace_sources(
            self.project, ["sources/repository/a.py"]
        )
        self.assertEqual(list_source_versions(self.project), [version])
        self.assertTrue((self.project.root / version["version"]).is_file())

    def test_invalid_version_manifest_path_is_rejected(self) -> None:
        """Persistence must never admit archive traversal paths."""

        self._write(self.folder / "a.py", "upstream\n", 2)
        sync_project_sources(self.project, decisions={
            "sources/repository/a.py": "keep_both",
        })
        manifest = self.project.root / "sources.json"
        payload = json.loads(manifest.read_text())
        payload["sources"][0]["versions"][0]["path"] = (
            ".kilin_versions/repository/../../private.py"
        )
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "version path"):
            list_source_versions(self.project)


if __name__ == "__main__":
    unittest.main()
