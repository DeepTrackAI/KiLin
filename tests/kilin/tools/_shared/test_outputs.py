"""Test private output-directory management without legacy migrations.

The core initializes outputs for new projects. Shared output helpers may
also create the directory on first use, but they must not move unrelated
folders or follow symbolic links.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.tools._shared.outputs import (
    output_folder,
    publish_output_file,
    save_output_file,
    staged_output_file,
)


class TestSharedOutputs(unittest.TestCase):
    """Verify safe output-folder initialization for shared services."""

    def test_creates_only_the_requested_outputs_directory(self) -> None:
        """A previously absent outputs directory is created on demand.

        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = output_folder(root)
            self.assertEqual(outputs, root / "outputs")
            self.assertTrue(outputs.is_dir())
            self.assertEqual(output_folder(root), outputs)

    def test_does_not_migrate_or_delete_an_unrelated_target(self) -> None:
        """Never relocate the contents of another directory.

        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            (target / "manual.md").write_text("# Manual\n", encoding="utf-8")
            self.assertEqual(output_folder(root), root / "outputs")
            self.assertTrue((root / "outputs").is_dir())
            self.assertEqual(
                (target / "manual.md").read_text(encoding="utf-8"),
                "# Manual\n",
            )
            self.assertFalse((root / "outputs/manual.md").exists())

    def test_rejects_a_linked_output_directory(self) -> None:
        """Do not write to an external directory via a folder symlink.

        """

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            linked = root / "outputs"
            try:
                linked.symlink_to(elsewhere, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("Symbolic links are unavailable")
            with self.assertRaisesRegex(ValueError, "cannot be a link"):
                output_folder(root)
            self.assertEqual(list(elsewhere.iterdir()), [])


class TestOutputFilePublication(unittest.TestCase):
    """Check atomic, contained creation of project-owned output files."""

    def setUp(self) -> None:
        """Create an isolated output directory for each test.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = output_folder(self.root)

    def test_saves_content_and_numbers_existing_names(self) -> None:
        """Preserve previous content while naming subsequent outputs.

        """

        first = save_output_file(self.root, "report.md", "first")
        second = save_output_file(self.root, "report.md", "second")
        third = save_output_file(self.root, "report.md", "third")

        self.assertEqual(
            (first, second, third),
            (
                Path("outputs/report.md"),
                Path("outputs/report-2.md"),
                Path("outputs/report-3.md"),
            ),
        )
        self.assertEqual(
            (self.root / first).read_text(encoding="utf-8"), "first"
        )
        self.assertEqual(
            (self.root / second).read_text(encoding="utf-8"), "second"
        )
        self.assertEqual(
            (self.root / third).read_text(encoding="utf-8"), "third"
        )
        self.assertEqual(
            sorted(path.name for path in self.folder.iterdir()),
            ["report-2.md", "report-3.md", "report.md"],
        )

    def test_rejects_unsafe_filenames_without_writing(self) -> None:
        """Reject traversal and ambiguous names on every platform.

        """

        bad_names = (
            "", ".", "..", "../escape.md", "/tmp/escape.md",
            "subfolder/report.md", r"..\escape.md", r"C:\escape.md",
            "report:stream.md", "report.md.", "report.md ",
            "NUL.txt", "CON", "COM1.csv", "COM¹.txt",
            'bad*name.md', 'report?.md', '<name>.md',
            "bad\x00name.md",
        )
        for name in bad_names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    save_output_file(self.root, name, "unsafe")

        self.assertEqual(list(self.folder.iterdir()), [])
        self.assertFalse((self.root / "escape.md").exists())

    def test_existing_and_dangling_symlinks_are_not_followed(self) -> None:
        """Skip occupied symlink names, never their outside targets.

        """

        existing = self.root / "outside.md"
        existing.write_text("protected", encoding="utf-8")
        dangling = self.root / "missing.md"
        link = self.folder / "report.md"
        missing_link = self.folder / "report-2.md"
        try:
            link.symlink_to(existing)
            missing_link.symlink_to(dangling)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")

        saved = save_output_file(self.root, "report.md", "new")
        self.assertEqual(saved, Path("outputs/report-3.md"))
        self.assertEqual(
            (self.root / saved).read_text(encoding="utf-8"), "new"
        )
        self.assertEqual(existing.read_text(encoding="utf-8"), "protected")
        self.assertFalse(dangling.exists())
        self.assertTrue(link.is_symlink())
        self.assertTrue(missing_link.is_symlink())

    def test_name_collision_after_selection_does_not_overwrite(self) -> None:
        """A file arriving during publication receives the first name.

        """

        original_link = os.link
        first = True

        def competing_link(source: Path, destination: Path, **kwargs) -> None:
            """Introduce a competing file before the first link attempt.

            """

            nonlocal first
            if first:
                first = False
                destination.write_text("other writer", encoding="utf-8")
            original_link(source, destination, **kwargs)

        with patch(
            "kilin.tools._shared.outputs.os.link",
            side_effect=competing_link,
        ):
            saved = save_output_file(self.root, "report.md", "new")

        self.assertEqual(saved, Path("outputs/report-2.md"))
        self.assertEqual(
            (self.folder / "report.md").read_text(encoding="utf-8"),
            "other writer",
        )
        self.assertEqual(
            (self.root / saved).read_text(encoding="utf-8"), "new"
        )

    def test_concurrent_writers_receive_unique_names(self) -> None:
        """The output directory never loses a concurrent writer's data.

        """

        data = tuple(f"entry {index}" for index in range(12))
        with ThreadPoolExecutor(max_workers=6) as pool:
            paths = list(
                pool.map(
                    lambda content: save_output_file(
                        self.root, "report.md", content
                    ),
                    data,
                )
            )

        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(
            {path.name for path in paths},
            {"report.md"}
            | {f"report-{index}.md" for index in range(2, 13)},
        )
        self.assertEqual(
            {(self.root / path).read_text(encoding="utf-8") for path in paths},
            set(data),
        )
        self.assertFalse(
            any(name.startswith(".kilin-output-")
                for name in os.listdir(self.folder))
        )

    def test_stage_is_scoped_and_cleaned_after_failure(self) -> None:
        """Failed writers do not leave private temporary artifacts.

        """

        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            with staged_output_file(self.root, suffix=".pdf") as stage:
                self.assertEqual(stage.suffix, ".pdf")
                self.assertTrue(stage.is_file())
                stage.write_bytes(b"partial")
                raise RuntimeError("writer failed")
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_rejects_external_or_linked_stage(self) -> None:
        """Only regular staging files from outputs may be published.

        """

        outside = self.root / "external.md"
        outside.write_text("protected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "belong"):
            publish_output_file(self.root, "report.md", outside)

        linked = self.folder / "linked.md"
        try:
            linked.symlink_to(outside)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        with self.assertRaisesRegex(ValueError, "regular"):
            publish_output_file(self.root, "report.md", linked)
        self.assertFalse((self.folder / "report.md").exists())
        self.assertEqual(outside.read_text(encoding="utf-8"), "protected")

    def test_exclusive_copy_when_hard_links_are_unavailable(self) -> None:
        """The fallback preserves existing outputs without hard links.

        """

        first = self.folder / "report.md"
        first.write_text("protected", encoding="utf-8")
        with patch(
            "kilin.tools._shared.outputs.os.link",
            side_effect=OSError("hard links unavailable"),
        ):
            saved = save_output_file(self.root, "report.md", "new")

        self.assertEqual(saved, Path("outputs/report-2.md"))
        self.assertEqual(first.read_text(encoding="utf-8"), "protected")
        self.assertEqual(
            (self.root / saved).read_text(encoding="utf-8"), "new"
        )
        self.assertEqual(len(list(self.folder.iterdir())), 2)

    def test_failed_copy_does_not_leave_partial_output(self) -> None:
        """An interrupted copy cleans up its exclusive destination.

        """

        with patch(
            "kilin.tools._shared.outputs.os.link",
            side_effect=OSError("hard links unavailable"),
        ), patch(
            "kilin.tools._shared.outputs.shutil.copyfileobj",
            side_effect=OSError("copy failed"),
        ):
            with self.assertRaisesRegex(OSError, "copy failed"):
                save_output_file(self.root, "report.md", "new")

        self.assertEqual(list(self.folder.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
