"""Test explicit imports and safe removal of managed source snapshots.

These tests exercise direct-file selection, project-owned source manifests
and their copy stamps, folder deletion, safety boundaries, and effective RAG
scan selection.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.project import initialize_project, open_project
from kilin.repository import scan_repository
from kilin.workspace import (
    add_project_sources,
    include_explicit_imports,
    prepare_sources_workspace,
    remove_workspace_sources,
)


class TestWorkspaceSourceManagement(unittest.TestCase):
    """Exercise source management using disposable project paths."""

    def setUp(self) -> None:
        """Initialize an isolated source directory and workspace path."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.destination = self.root / "workspace"

    def _file(self, name: str, content: str = "sample\n") -> Path:
        """Create a temporary original source.

        Parameters
        ----------
        name: str
            Filename inside the fixture root.
        content: str, optional
            Contents of the original file. Defaults to sample text.

        Returns
        -------
        pathlib.Path
            New original source path.

        """

        source = self.root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
        return source

    def _records(self) -> list[dict[str, object]]:
        """Return the saved managed-source provenance entries.

        Returns
        -------
        list[dict[str, object]]
            Persisted sources.json records.

        """

        return json.loads(
            (self.destination / "sources.json").read_text(encoding="utf-8")
        )["sources"]

    def test_explicit_suffix_and_protected_file_are_imported(self) -> None:
        """Explicit file imports do not change automatic scan policy."""

        png = self._file("image.xyz", "data")
        secret = self._file(".env.local", "KEY=sample\n")
        project = prepare_sources_workspace(
            [png, secret], "code", self.destination
        )
        self.assertNotIn(".xyz", project.profile.extensions)
        self.assertTrue((project.source / "image.xyz").is_file())
        self.assertTrue((project.source / ".env.local").is_file())
        scanned = scan_repository(
            project.source, extensions=project.profile.extensions
        )
        self.assertNotIn("image.xyz", scanned["selected_files"])
        self.assertNotIn(".env.local", scanned["selected_files"])
        expanded = include_explicit_imports(project, scanned)
        self.assertIn("image.xyz", expanded["selected_files"])
        self.assertIn(".env.local", expanded["selected_files"])
        for row in expanded["entries"]:
            with self.subTest(path=row["path"]):
                self.assertEqual(row["kind"], "file")
                self.assertNotIn("type_id", row)
        self.assertEqual(project.profile.type_id, "code")

    def test_remove_direct_file_updates_manifest_not_original(self) -> None:
        """Removing a source copy does not delete its external original."""

        original = self._file("note.md")
        project = prepare_sources_workspace(
            [original], "code", self.destination
        )
        removed = remove_workspace_sources(project, ["sources/note.md"])
        self.assertEqual(removed, ["sources/note.md"])
        self.assertFalse((project.source / "note.md").exists())
        self.assertTrue(original.is_file())
        self.assertEqual(self._records(), [])

    def test_import_stamps_each_copy_and_removal_drops_it(self) -> None:
        """Every tracked copy has a stamp; removed copies lose theirs."""

        source = self.root / "repository"
        (source / "sub").mkdir(parents=True)
        (source / "sub/a.py").write_text("a = 1\n", encoding="utf-8")
        (source / "sub/b.py").write_text("bb = 22\n", encoding="utf-8")
        project = prepare_sources_workspace(
            [source], "code", self.destination
        )
        stamps = self._records()[0]["stamps"]
        self.assertEqual(set(stamps), {"sub/a.py", "sub/b.py"})
        copy = project.source / "repository/sub/b.py"
        self.assertEqual(stamps["sub/b.py"], {
            "size": copy.stat().st_size,
            "mtime_ns": copy.stat().st_mtime_ns,
        })
        remove_workspace_sources(
            project, ["sources/repository/sub/a.py"]
        )
        self.assertEqual(set(self._records()[0]["stamps"]), {"sub/b.py"})

    def test_reimport_renews_only_replaced_stamps(self) -> None:
        """Replaced copies get new stamps; preserved copies keep theirs."""

        source = self.root / "repository"
        source.mkdir()
        changed = source / "changed.py"
        kept = source / "kept.py"
        changed.write_text("x = 1\n", encoding="utf-8")
        kept.write_text("y = 1\n", encoding="utf-8")
        for path in (changed, kept):
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        prepare_sources_workspace([source], "code", self.destination)
        before = self._records()[0]["stamps"]

        changed.write_text("x = 22\n", encoding="utf-8")
        os.utime(changed, ns=(2_000_000_000, 2_000_000_000))
        add_project_sources(
            self.destination, [source],
            selected_files={str(source): ["changed.py"]},
            replace_existing=True,
        )
        after = self._records()[0]["stamps"]
        self.assertEqual(
            after["changed.py"], {"size": 7, "mtime_ns": 2_000_000_000}
        )
        self.assertEqual(after["kept.py"], before["kept.py"])

    def test_manifest_without_stamps_is_still_accepted(self) -> None:
        """Older manifests open, and the next import adds stamps."""

        original = self._file("note.md")
        prepare_sources_workspace([original], "code", self.destination)
        manifest = self.destination / "sources.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        del payload["sources"][0]["stamps"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        open_project(self.destination)
        add_project_sources(self.destination, [])
        self.assertNotIn("stamps", self._records()[0])
        add_project_sources(self.destination, [original])
        self.assertEqual(set(self._records()[0]["stamps"]), {"note.md"})

    def test_invalid_stamps_are_rejected(self) -> None:
        """Stamps must be non-negative integers for tracked files only."""

        original = self._file("note.md")
        prepare_sources_workspace([original], "code", self.destination)
        manifest = self.destination / "sources.json"
        valid = json.loads(manifest.read_text(encoding="utf-8"))
        for stamps in (
            [],
            {"other.md": {"size": 1, "mtime_ns": 1}},
            {"note.md": {"size": -1, "mtime_ns": 1}},
            {"note.md": {"size": 1}},
            {"note.md": {"size": 1.5, "mtime_ns": 1}},
            {"note.md": {"size": True, "mtime_ns": 1}},
        ):
            with self.subTest(stamps=stamps):
                payload = json.loads(json.dumps(valid))
                payload["sources"][0]["stamps"] = stamps
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "stamps"):
                    add_project_sources(self.destination, [original])

    def test_nested_deletion_updates_directory_inventory(self) -> None:
        """Removing one imported file preserves other tracked files."""

        source = self.root / "repository"
        (source / "sub").mkdir(parents=True)
        (source / "sub/a.py").write_text("a = 1\n", encoding="utf-8")
        (source / "sub/b.py").write_text("b = 2\n", encoding="utf-8")
        project = prepare_sources_workspace(
            [source], "code", self.destination
        )
        remove_workspace_sources(
            project, ["sources/repository/sub/a.py"]
        )
        self.assertEqual(self._records()[0]["files"], ["sub/b.py"])
        self.assertTrue((source / "sub/a.py").is_file())
        remove_workspace_sources(project, ["sources/repository"])
        self.assertEqual(self._records(), [])

    def test_invalid_paths_rejected_before_mutation(self) -> None:
        """Traversal, the managed root, and external paths are rejected."""

        original = self._file("note.md")
        project = prepare_sources_workspace(
            [original], "code", self.destination
        )
        for path in (
            "sources", "../note.md", "sources/../profile.ini",
            "outputs/file.md", "sources\\note.md", "/sources/note.md",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    remove_workspace_sources(project, [path])
        self.assertTrue((project.source / "note.md").is_file())

    def test_symlink_target_is_rejected(self) -> None:
        """A source symlink never deletes the external target."""

        original = self._file("note.md")
        project = prepare_sources_workspace(
            [original], "code", self.destination
        )
        link = project.source / "linked.md"
        link.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            remove_workspace_sources(project, ["sources/linked.md"])
        self.assertTrue(original.is_file())

    def test_external_source_projects_cannot_delete_original_files(
        self,
    ) -> None:
        """Deleting managed copies cannot target a root-as-source project."""

        external = self.root / "external"
        project = initialize_project(external, "code")
        profile = external / "profile.ini"
        profile.write_text(
            profile.read_text(encoding="utf-8").replace(
                "source = sources", "source = ."
            ),
            encoding="utf-8",
        )
        original = external / "analysis.py"
        original.write_text("number = 1\n", encoding="utf-8")
        project = open_project(external)
        self.assertEqual(project.source, external)
        with self.assertRaisesRegex(ValueError, "managed sources"):
            remove_workspace_sources(project, ["sources/analysis.py"])
        self.assertTrue(original.is_file())

    def test_removed_explicit_file_is_no_longer_scanned(self) -> None:
        """RAG selection tracks only remaining direct import snapshots."""

        original = self._file("special.xyz")
        project = prepare_sources_workspace(
            [original], "code", self.destination
        )
        remove_workspace_sources(project, ["sources/special.xyz"])
        report = scan_repository(
            project.source, extensions=project.profile.extensions
        )
        self.assertNotIn(
            "special.xyz",
            include_explicit_imports(project, report)["selected_files"],
        )


if __name__ == "__main__":
    unittest.main()
