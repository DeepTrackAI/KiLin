"""Test explicit imports and safe removal of managed source snapshots.

These tests exercise direct-file selection, project-owned source manifests,
folder deletion, safety boundaries, and effective RAG scan selection.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.project import initialize_project
from qachat.repository import scan_repository
from qachat.workspace import (
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
            (self.destination / "sources.json").read_text()
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

    def test_remove_direct_file_updates_manifest_not_original(self) -> None:
        """Removing a source copy does not delete its source outside QAChat."""

        original = self._file("note.md")
        project = prepare_sources_workspace(
            [original], "code", self.destination
        )
        removed = remove_workspace_sources(project, ["sources/note.md"])
        self.assertEqual(removed, ["sources/note.md"])
        self.assertFalse((project.source / "note.md").exists())
        self.assertTrue(original.is_file())
        self.assertEqual(self._records(), [])

    def test_nested_deletion_updates_directory_inventory(self) -> None:
        """Removing one imported file preserves other tracked files."""

        source = self.root / "repository"
        (source / "sub").mkdir(parents=True)
        (source / "sub/a.py").write_text("a = 1\n")
        (source / "sub/b.py").write_text("b = 2\n")
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
        external.mkdir()
        original = external / "analysis.py"
        original.write_text("number = 1\n")
        project = initialize_project(external, "code")
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
