"""Test GUI-independent management of outputs and template files.

Use disposable project directories to exercise file operations, naming,
replacement, deletion, containment, and project inventory without models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.project import initialize_project
from kilin.project_files import (
    add_project_files,
    list_project_files,
    remove_project_files,
    replace_project_file,
)


class TestProjectFiles(unittest.TestCase):
    """Exercise user-owned files in the two editable workspace folders."""

    def setUp(self) -> None:
        """Create an isolated project and external files."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        folder = self.root / "project"
        folder.mkdir()
        self.project = initialize_project(folder, "code")
        self.original = self.root / "report.md"
        self.original.write_text("# Original\n", encoding="utf-8")

    def test_both_editable_folders_accept_files(self) -> None:
        """Outputs and templates are equally editable through the core."""

        for folder in ("outputs", "templates"):
            with self.subTest(folder=folder):
                paths = add_project_files(
                    self.project, folder, [self.original]
                )
                self.assertEqual(paths, [f"{folder}/report.md"])
                target = self.project.root / paths[0]
                self.assertEqual(target.read_text(), "# Original\n")
                self.assertEqual(
                    remove_project_files(self.project, paths), paths
                )
                self.assertFalse(target.exists())
        self.assertTrue(self.original.is_file())

    def test_name_collision_and_explicit_replacement(self) -> None:
        """New files use suffixes unless replacement is requested."""

        first = add_project_files(
            self.project, "outputs", [self.original]
        )
        second = add_project_files(
            self.project, "outputs", [self.original]
        )
        self.assertEqual(first, ["outputs/report.md"])
        self.assertEqual(second, ["outputs/report-2.md"])
        self.original.write_text("# Changed\n", encoding="utf-8")
        replaced = add_project_files(
            self.project, "outputs", [self.original], replace=True
        )
        self.assertEqual(replaced, first)
        self.assertEqual(
            (self.project.root / first[0]).read_text(), "# Changed\n"
        )
        self.assertTrue((self.project.root / second[0]).exists())

    def test_replace_existing_file_in_both_folders(self) -> None:
        """Explicit replacement keeps the selected destination name."""

        for folder in ("outputs", "templates"):
            with self.subTest(folder=folder):
                path = add_project_files(
                    self.project, folder, [self.original]
                )[0]
                replacement = self.root / "replacement.txt"
                replacement.write_text("Replaced\n", encoding="utf-8")
                self.assertEqual(
                    replace_project_file(self.project, path, replacement),
                    path,
                )
                self.assertEqual(
                    (self.project.root / path).read_text(), "Replaced\n"
                )

    def test_folder_addition_and_deletion(self) -> None:
        """Nested directories and their files may be added or deleted."""

        source = self.root / "custom"
        source.mkdir()
        (source / "nested").mkdir()
        (source / "nested/example.md").write_text("Example\n")
        paths = add_project_files(self.project, "templates", [source])
        self.assertEqual(paths, ["templates/custom"])
        self.assertIn(
            "templates/custom/nested/example.md",
            list_project_files(self.project),
        )
        self.assertEqual(remove_project_files(self.project, paths), paths)
        self.assertFalse((self.project.root / paths[0]).exists())
        self.assertTrue((source / "nested/example.md").exists())

    def test_empty_folders_remain_listed(self) -> None:
        """Core inventory includes empty outputs and templates folders."""

        listed = list_project_files(self.project)
        self.assertIn("outputs/", listed)
        self.assertIn("templates/", listed)
        self.assertIn("indexes/", listed)
        self.assertIn("profile.ini", listed)

    def test_invalid_paths_and_managed_root_are_rejected(self) -> None:
        """Deletion cannot escape, or remove, a protected project folder."""

        for path in (
            "outputs", "templates", "profile.ini", "outputs/../logs/x",
            "/outputs/report.md", "outputs\\report.md", "C:/x",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    remove_project_files(self.project, [path])
        self.assertTrue((self.project.root / "profile.ini").exists())

    def test_linked_targets_and_parents_are_rejected(self) -> None:
        """A link cannot redirect a deletion outside a project."""

        outside = self.root / "outside.txt"
        outside.write_text("safe\n")
        linked = self.project.root / "outputs/link.txt"
        linked.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "link"):
            remove_project_files(self.project, ["outputs/link.txt"])
        with self.assertRaisesRegex(ValueError, "link"):
            replace_project_file(
                self.project, "outputs/link.txt", self.original
            )
        self.assertEqual(outside.read_text(), "safe\n")

    def test_all_removal_paths_are_checked_before_deleting(self) -> None:
        """Invalid second paths do not delete a valid first file."""

        path = add_project_files(
            self.project, "outputs", [self.original]
        )[0]
        with self.assertRaises(ValueError):
            remove_project_files(
                self.project, [path, "templates/does-not-exist.md"]
            )
        self.assertTrue((self.project.root / path).is_file())

    def test_add_rejects_importing_own_managed_directory(self) -> None:
        """A folder may not recursively import itself."""

        with self.assertRaisesRegex(ValueError, "already"):
            add_project_files(
                self.project, "outputs", [self.project.root / "outputs"]
            )

    def test_bad_replacement_or_invalid_folder(self) -> None:
        """Only existing output/template files can be replaced."""

        with self.assertRaises(ValueError):
            replace_project_file(
                self.project, "templates/missing.md", self.original
            )
        with self.assertRaises(ValueError):
            add_project_files(self.project, "sources", [self.original])


if __name__ == "__main__":
    unittest.main()
