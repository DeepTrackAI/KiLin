"""Test read-only source import previews against the core copying policy.

The preview reports eligible sources, rejected files, and the same destination
names that a subsequent cumulative import will allocate. It must never write
or reserve workspace files; the importer revalidates at commit time.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.workspace import (
    add_project_sources,
    prepare_sources_workspace,
    preview_project_sources,
)
from kilin.repository_reason_codes import (
    SELECTION_REASON_EMPTY_FILE,
    SELECTION_REASON_FILE_TOO_LARGE,
    SELECTION_REASON_FILE_TYPE_NOT_SELECTED,
)


class TestSourceImportPreview(unittest.TestCase):
    """Compare reported imports against the managed workspace policy."""

    def setUp(self) -> None:
        """Create an existing workspace and a separate input directory."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        initial = self.root / "initial.txt"
        initial.write_text("original", encoding="utf-8")
        self.project = prepare_sources_workspace(
            [initial], "literature", self.root / "workspace"
        )
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()

    def _file(self, relative: str, content: str) -> Path:
        """Create a file beneath the test input directory."""

        path = self.inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_direct_file_exceptions_and_size_rules(self) -> None:
        """Explicit files bypass suffix rules but must obey size checks."""

        allowed = self._file("data.other", "usable")
        empty = self._file("empty.txt", "")
        large = self.inputs / "large.txt"
        large.write_bytes(
            b"a" * (self.project.profile.max_file_bytes + 1)
        )
        manifest = self.project.root / "sources.json"
        before = manifest.read_bytes()
        paths = [str(allowed), str(empty), str(large)]
        plan = preview_project_sources(self.project, paths)
        self.assertEqual(plan["sources"], [str(allowed)])
        self.assertEqual(plan["add"], [{
            "name": "data.other", "kind": "file", "files": 1,
            "state": "new",
        }])
        self.assertEqual(plan["files"], [
            {"path": "empty.txt", "reason_code": SELECTION_REASON_EMPTY_FILE},
            {"path": "large.txt", "reason_code":
             SELECTION_REASON_FILE_TOO_LARGE},
        ])
        self.assertEqual(manifest.read_bytes(), before)
        self.assertFalse((self.project.source / "data.other").exists())
        self.assertEqual(plan["skipped"], [])
        add_project_sources(self.project.root, plan["sources"])
        self.assertEqual(
            (self.project.source / "data.other").read_text(), "usable"
        )

    def test_missing_link_and_workspace_paths_are_rejected(self) -> None:
        """Preview rejects paths the real importer would refuse."""

        missing = self.inputs / "missing.txt"
        with self.assertRaises(FileNotFoundError):
            preview_project_sources(self.project, [missing])
        link = self.inputs / "link.txt"
        try:
            link.symlink_to(self.root / "initial.txt")
        except (OSError, NotImplementedError):
            pass
        else:
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                preview_project_sources(self.project, [link])
        with self.assertRaisesRegex(ValueError, "workspace as its own"):
            preview_project_sources(
                self.project, [self.project.source / "initial.txt"]
            )
        for invalid in ([""], [object()]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    preview_project_sources(self.project, invalid)

    def test_duplicate_names_match_import_destination(self) -> None:
        """Simultaneous equal basenames receive the actual hashed names."""

        first = self._file("alpha/note.txt", "first")
        second = self._file("beta/note.txt", "second")
        paths = [str(first), str(second), str(first)]
        plan = preview_project_sources(self.project, paths)
        renamed = f"note-{sha256(str(second).encode()).hexdigest()[:8]}.txt"
        self.assertEqual(plan["sources"], paths[:2])
        self.assertEqual(plan["add"], [
            {"name": "note.txt", "kind": "file", "files": 1,
             "state": "new"},
            {"name": renamed, "kind": "file", "files": 1,
             "state": "renamed"},
        ])
        add_project_sources(self.project.root, plan["sources"])
        self.assertEqual(
            (self.project.source / "note.txt").read_text(), "first"
        )
        self.assertEqual(
            (self.project.source / renamed).read_text(), "second"
        )

    def test_existing_source_and_skipped_folder(self) -> None:
        """Original-path identity wins over a matching destination name."""

        folder = self.inputs / "unsuitable"
        folder.mkdir()
        (folder / "code.py").write_text("x = 1", encoding="utf-8")
        existing = self.root / "initial.txt"
        plan = preview_project_sources(
            self.project, [str(folder), str(existing)]
        )
        self.assertEqual(plan["skipped"], [str(folder)])
        self.assertEqual(plan["sources"], [str(existing)])
        self.assertEqual(plan["add"][0]["state"], "existing")
        self.assertEqual(plan["files"], [{
            "path": "unsuitable/code.py",
            "reason_code": SELECTION_REASON_FILE_TYPE_NOT_SELECTED,
        }])

    def test_invalid_collisions_and_incomplete_scans(self) -> None:
        """Preflight rejects destination failures before any copying."""

        source = self._file("new/initial.txt", "another file")
        digest = sha256(str(source).encode()).hexdigest()[:8]
        occupied = self.project.source / f"initial-{digest}.txt"
        occupied.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "distinct source name"):
            preview_project_sources(self.project, [source])
        self.assertEqual(occupied.read_text(), "keep")
        with patch(
            "kilin.workspace.scan_repository",
            return_value={"complete": False},
        ):
            with self.assertRaisesRegex(ValueError, "completely scan"):
                preview_project_sources(self.project, [self.inputs])


if __name__ == "__main__":
    unittest.main()
