"""Unit tests for repository metadata scans and selection policies.

Tests use a disposable repository containing source files, generated
artifacts, and protected names. No repository file may be executed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import repository


class _RepositoryFixture(unittest.TestCase):
    """Provide an isolated repository with representative file types."""

    def setUp(self) -> None:
        """Create a repository with selectable and protected entries."""

        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "small project"
        files = {
            "README.md": "# A small project\n",
            "src/main.py": "raise RuntimeError('never run')\n",
            "src/helpers.py": "def add(a, b): return a + b\n",
            "src_extra/other.py": "x = 1\n",
            "tests/test_main.py": "assert True\n",
            "docs/guide.MD": "Some instructions.\n",
            "settings.toml": "name = 'example'\n",
            "empty.py": "",
            "image.png": "not selected",
            ".env": "FAKE_SECRET=not-real",
            ".env.local": "FAKE_SECRET=not-real",
            ".git/config": "generated metadata",
            ".gitignore": ".venv/",
            ".github/workflow.yml": "name: checks",
            ".venv/package.py": "installed package",
            ".vscode/settings.json": "{}",
            "models/config.json": "{}",
            "models/layer.py": "class Layer: pass",
            "cache.safetensors": "fake weights",
            "credentials.json": "{}",
        }
        for name, content in files.items():
            self._write(name, content)

    def _write(self, relative: str, content: str) -> Path:
        """Write a UTF-8 file under the temporary repository."""

        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    @staticmethod
    def _by_path(report: dict[str, object]) -> dict[str, dict]:
        """Index reported entries by repository-relative path."""

        return {entry["path"]: entry for entry in report["entries"]}


class TestRepositoryScanSelection(_RepositoryFixture):
    """Test selection, ordering, and metadata-only scan behavior."""

    def test_packaged_selection_defaults_match_runtime(self) -> None:
        """Load selection defaults from the bundled JSON resource."""

        path = Path(repository.__file__).with_name("kilin_defaults.json")
        defaults = json.loads(path.read_text(encoding="utf-8"))
        # New domain sections must not change repository selection policy.
        self.assertTrue(
            {"DEFAULT_MODELS", "REPOSITORY", "ANSWERING"}
            <= set(defaults)
        )
        defaults = defaults["REPOSITORY"]
        self.assertEqual(
            set(defaults),
            {
                "DEFAULT_EXTENSIONS",
                "SKIP_DIRECTORIES",
                "SKIP_FILES",
                "BLOCKED_SUFFIXES",
            },
        )
        self.assertEqual(
            repository.DEFAULT_EXTENSIONS,
            tuple(defaults["DEFAULT_EXTENSIONS"]),
        )
        for name in (
            "SKIP_DIRECTORIES",
            "SKIP_FILES",
            "BLOCKED_SUFFIXES",
        ):
            with self.subTest(setting=name):
                self.assertEqual(
                    getattr(repository, name), set(defaults[name])
                )

    def test_public_extension_normalizer_reuses_scanner_grammar(
        self,
    ) -> None:
        """Normalize and validate literal suffixes before file scanning."""

        self.assertEqual(
            repository.normalize_extensions(
                ["py", ".PY", ".tar.GZ", ".tar.gz"]
            ),
            (".py", ".tar.gz"),
        )
        for value in (".py*", ".py?", ".py file", ".py/file"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "wildcards"):
                    repository.normalize_extensions([value])
        with self.assertRaisesRegex(ValueError, "extensions"):
            repository.normalize_extensions(".py")
        self.assertEqual(repository.normalize_extensions([]), ())

    def test_default_selection_and_report_metadata(self) -> None:
        """Return sorted relative paths and coherent selection totals."""

        report = repository.scan_repository(self.root)
        expected = [
            ".github/workflow.yml",
            ".gitignore",
            ".vscode/settings.json",
            "README.md",
            "docs/guide.MD",
            "models/config.json",
            "models/layer.py",
            "settings.toml",
            "src/helpers.py",
            "src/main.py",
            "src_extra/other.py",
            "tests/test_main.py",
        ]
        self.assertEqual(report["repository"], str(self.root.resolve()))
        self.assertEqual(report["selected_files"], expected)
        self.assertEqual(report["extensions"], list(
            repository.DEFAULT_EXTENSIONS
        ))
        self.assertEqual(report["include_paths"], [])
        self.assertEqual(report["exclude_paths"], [])
        self.assertEqual(report["max_file_bytes"], 1_048_576)
        self.assertTrue(report["complete"])
        rows = report["entries"]
        self.assertEqual(
            [row["path"] for row in rows],
            sorted(row["path"] for row in rows),
        )
        self.assertEqual(report["visited_entries"], 27)
        self.assertEqual(
            report["selected_bytes"],
            sum((self.root / name).stat().st_size for name in expected),
        )
        indexed = self._by_path(report)
        for name in expected:
            with self.subTest(name=name):
                self.assertEqual(indexed[name]["status"], "selected")
                self.assertEqual(indexed[name]["kind"], "file")
                self.assertEqual(
                    indexed[name]["size_bytes"],
                    (self.root / name).stat().st_size,
                )
                self.assertEqual(
                    indexed[name]["reason"], "matches your selection"
                )

    def test_preview_never_reads_or_executes_selected_files(self) -> None:
        """Inspect filesystem metadata without reading or running code."""

        message = "A repository scan must inspect metadata only."
        with (
            patch.object(Path, "open", side_effect=AssertionError(message)),
            patch.object(
                builtins, "open", side_effect=AssertionError(message)
            ),
            patch.object(
                subprocess, "run", side_effect=AssertionError(message)
            ),
        ):
            report = repository.scan_repository(self.root)
        self.assertIn("src/main.py", report["selected_files"])

    def test_include_and_exclude_scope_boundaries(self) -> None:
        """Select literal paths without matching similarly named siblings."""

        cases = (
            (["src"], [], ["src/helpers.py", "src/main.py"]),
            (["src/main.py"], [], ["src/main.py"]),
            (["src"], ["src/helpers.py"], ["src/main.py"]),
            (["src\\main.py"], [], ["src/main.py"]),
            (["missing"], [], []),
            (["."], ["src"], None),
            ([], ["."], []),
        )
        for includes, excludes, expected in cases:
            with self.subTest(includes=includes, excludes=excludes):
                report = repository.scan_repository(
                    self.root,
                    include_paths=includes,
                    exclude_paths=excludes,
                )
                selected = report["selected_files"]
                if expected is None:
                    self.assertNotIn("src/main.py", selected)
                    self.assertIn("src_extra/other.py", selected)
                else:
                    self.assertEqual(selected, expected)

    def test_duplicate_paths_are_normalized_in_report(self) -> None:
        """Normalize separators and deduplicate selection scopes."""

        report = repository.scan_repository(
            self.root,
            include_paths=("src", "src/", "src\\main.py", "src"),
            exclude_paths=("src/helpers.py", "src/helpers.py"),
        )
        self.assertEqual(
            report["include_paths"], ["src", "src/main.py"]
        )
        self.assertEqual(report["exclude_paths"], ["src/helpers.py"])
        self.assertEqual(report["selected_files"], ["src/main.py"])

    def test_extension_normalization_and_compound_suffixes(self) -> None:
        """Accept literal suffixes case-insensitively, including compounds."""

        self._write("archive.tar.GZ", "payload")
        self._write("notarchive.gz.tmp", "payload")
        report = repository.scan_repository(
            self.root,
            extensions=["TXT", ".tar.gz", "txt", ".TAR.GZ"],
        )
        self.assertEqual(report["extensions"], [".txt", ".tar.gz"])
        self.assertEqual(report["selected_files"], ["archive.tar.GZ"])
        report = repository.scan_repository(
            self.root, extensions=[".MD"]
        )
        self.assertEqual(
            report["selected_files"], ["README.md", "docs/guide.MD"]
        )

    def test_empty_extensions_select_no_files(self) -> None:
        """Permit an empty extension set without selecting any file."""

        report = repository.scan_repository(self.root, extensions=[])
        self.assertTrue(report["complete"])
        self.assertEqual(report["selected_files"], [])
        self.assertEqual(report["selected_bytes"], 0)

    def test_extension_matches_special_dotfiles(self) -> None:
        """Apply literal suffix selection to permitted dotfiles."""

        report = repository.scan_repository(
            self.root, extensions=[".gitignore", ".gitattributes"]
        )
        self.assertEqual(report["selected_files"], [".gitignore"])

    def test_structured_skip_reasons_preserve_selection_policy(
        self,
    ) -> None:
        """Provide stable codes for each existing selection decision."""

        report = repository.scan_repository(self.root)
        rows = self._by_path(report)
        expected = {
            "README.md": ("selected", "matches your selection"),
            ".git": (
                "excluded_directory",
                "environment, cache, or generated folder",
            ),
            ".env": (
                "protected_file", "potentially secret or generated file"
            ),
            "cache.safetensors": (
                "blocked_file_type",
                "model weights or potentially secret file",
            ),
            "image.png": ("file_type_not_selected", "file type not selected"),
            "empty.py": ("empty_file", "empty file"),
        }
        for path, (code, explanation) in expected.items():
            with self.subTest(path=path):
                self.assertEqual(rows[path]["reason_code"], code)
                self.assertEqual(rows[path]["reason"], explanation)

        scoped = repository.scan_repository(
            self.root,
            include_paths=["src"],
            exclude_paths=["src/helpers.py"],
        )
        scoped_rows = self._by_path(scoped)
        self.assertEqual(
            scoped_rows["src/helpers.py"]["reason_code"],
            "excluded_path",
        )
        self.assertEqual(
            scoped_rows["README.md"]["reason_code"],
            "outside_scope",
        )

    def test_size_reason_code_does_not_depend_on_explanation(self) -> None:
        """Keep the skip decision independent of displayed wording."""

        self._write("large.txt", "12345")
        original = repository._skip_reason

        def changed_text(relative: str, *args: object):
            """Simulate rewording without changing the reason code."""

            result = original(relative, *args)
            if result is not None and result[0] == "file_too_large":
                return result[0], "This file exceeds the limit."
            return result

        with patch.object(
            repository, "_skip_reason", side_effect=changed_text
        ):
            report = repository.scan_repository(
                self.root, extensions=[".txt"], max_file_bytes=4
            )
        row = self._by_path(report)["large.txt"]
        self.assertEqual(row["reason_code"], "file_too_large")
        self.assertEqual(row["reason"], "This file exceeds the limit.")
        self.assertEqual(row["status"], "skipped")

    def test_size_boundary_and_empty_files(self) -> None:
        """Select files at the limit and skip empty or larger files."""

        self._write("four.txt", "1234")
        self._write("five.txt", "12345")
        report = repository.scan_repository(
            self.root, extensions=[".txt"], max_file_bytes=4
        )
        self.assertEqual(report["selected_files"], ["four.txt"])
        rows = self._by_path(report)
        self.assertEqual(rows["empty.py"]["reason"], "file type not selected")
        self.assertEqual(
            rows["five.txt"]["reason"], "larger than the file-size limit"
        )
        self.assertEqual(rows["four.txt"]["size_bytes"], 4)
        report = repository.scan_repository(
            self.root, extensions=[".py"]
        )
        self.assertEqual(
            self._by_path(report)["empty.py"]["reason"], "empty file"
        )

    def test_protected_files_remain_excluded_when_included(self) -> None:
        """Never override built-in secrets and weight-file exclusions."""

        cases = (
            (".env", "potentially secret or generated file"),
            (".env.local", "potentially secret or generated file"),
            ("credentials.json", "potentially secret or generated file"),
            ("cache.safetensors", "model weights or potentially secret file"),
        )
        for name, reason in cases:
            with self.subTest(name=name):
                report = repository.scan_repository(
                    self.root, extensions=[".env", ".local", ".json",
                                           ".safetensors"],
                    include_paths=[name],
                )
                self.assertEqual(report["selected_files"], [])
                self.assertEqual(self._by_path(report)[name]["reason"], reason)

    def test_protected_directory_is_not_traversed(self) -> None:
        """Exclude generated directories even when explicitly included."""

        protected = (".git", ".venv", "__pycache__", "weights", "logs")
        for name in protected:
            self._write(f"{name}/nested.py", "x = 1")
            with self.subTest(name=name):
                report = repository.scan_repository(
                    self.root, include_paths=[name]
                )
                self.assertEqual(report["selected_files"], [])
                self.assertEqual(
                    self._by_path(report)[name]["reason"],
                    "environment, cache, or generated folder",
                )
                self.assertNotIn(f"{name}/nested.py", self._by_path(report))

    def test_other_regular_file_types_are_excluded(self) -> None:
        """Distinguish unsupported suffixes from protected weight formats."""

        report = repository.scan_repository(self.root)
        rows = self._by_path(report)
        self.assertEqual(rows["image.png"]["reason"],
                         "file type not selected")
        self.assertEqual(rows["cache.safetensors"]["reason"],
                         "model weights or potentially secret file")

    def test_entry_limit_is_inclusive(self) -> None:
        """Allow exactly the visited count but reject a smaller limit."""

        count = repository.scan_repository(self.root)["visited_entries"]
        report = repository.scan_repository(self.root, max_entries=count)
        self.assertEqual(report["visited_entries"], count)
        with self.assertRaisesRegex(ValueError, "Scan limit reached"):
            repository.scan_repository(self.root, max_entries=count - 1)


class TestRepositoryScanValidation(_RepositoryFixture):
    """Test rejection of unsafe roots and invalid scan parameters."""

    def test_invalid_selection_settings(self) -> None:
        """Reject malformed scopes, wildcards, and unsupported types."""

        cases = (
            {"include_paths": "src"},
            {"include_paths": None},
            {"include_paths": [17]},
            {"include_paths": [""]},
            {"include_paths": [" "]},
            {"include_paths": ["../outside"]},
            {"exclude_paths": ["a/../b"]},
            {"exclude_paths": ["/absolute"]},
            {"include_paths": ["C:\\private"]},
            {"include_paths": ["src/*.py"]},
            {"include_paths": ["src/file?.py"]},
            {"include_paths": ["src/[a].py"]},
            {"include_paths": ["src:main.py"]},
            {"exclude_paths": ["x\0y"]},
        )
        for settings in cases:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    repository.scan_repository(self.root, **settings)

    def test_invalid_extension_settings(self) -> None:
        """Require a sequence of literal, valid extension strings."""

        cases = (
            ".py", None, [None], ["*"], [""], ["."],
            ["py*"], [".p y"], ["py/file"], [".py?"], [".py."],
        )
        for extensions in cases:
            with self.subTest(extensions=extensions):
                with self.assertRaises(ValueError):
                    repository.scan_repository(
                        self.root, extensions=extensions
                    )

    def test_invalid_size_and_entry_limits(self) -> None:
        """Allow only positive, exact integer resource limits."""

        for field in ("max_file_bytes", "max_entries"):
            for value in (0, -1, True, False, 1.5, "10", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        repository.scan_repository(
                            self.root, **{field: value}
                        )

    def test_invalid_repository_roots(self) -> None:
        """Reject remote identifiers, nonexistent paths, and files."""

        for value in ("", "  ", None, 5, "https://site/repo",
                      "http://site/repo", "git@example:repo"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    repository.scan_repository(value)
        with self.assertRaises(FileNotFoundError):
            repository.scan_repository(self.root / "missing")
        with self.assertRaises(NotADirectoryError):
            repository.scan_repository(self.root / "README.md")

    def test_home_directory_is_not_selectable(self) -> None:
        """Prevent accidentally scanning a user's entire home directory."""

        with patch.object(Path, "home", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "home folder"):
                repository.scan_repository(self.root)

    def test_relative_repository_paths_use_project_root(self) -> None:
        """Resolve a relative source against KíLín's project root."""

        with patch.object(
            repository, "PROJECT_ROOT", self.root.parent
        ):
            report = repository.scan_repository(self.root.name)
        self.assertEqual(report["repository"], str(self.root.resolve()))


class TestRepositoryScanFilesystem(_RepositoryFixture):
    """Test safe traversal and recoverable filesystem failures."""

    def test_directory_and_file_symlinks_are_not_followed(self) -> None:
        """Never select content through descendant symbolic links."""

        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("outside", encoding="utf-8")
        try:
            (self.root / "linked-folder").symlink_to(
                outside, target_is_directory=True
            )
            (self.root / "linked-file.py").symlink_to(
                outside / "secret.py"
            )
            (self.root / "broken.py").symlink_to(
                outside / "missing.py"
            )
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Filesystem does not support symlinks: {error}")

        report = repository.scan_repository(self.root)
        rows = self._by_path(report)
        for name in ("linked-folder", "linked-file.py", "broken.py"):
            with self.subTest(name=name):
                self.assertEqual(rows[name]["kind"], "link")
                self.assertEqual(rows[name]["status"], "skipped")
                self.assertEqual(
                    rows[name]["reason_code"], "link_not_followed"
                )
                self.assertEqual(rows[name]["reason"],
                                 "links are not followed")
        self.assertNotIn("linked-folder/secret.py", report["selected_files"])
        self.assertTrue(report["complete"])

    def test_descendant_becoming_link_is_not_traversed(self) -> None:
        """Recheck directory safety before entering a pending child."""

        original = Path.is_symlink
        child = (self.root / "src").resolve()

        def is_symlink(path: Path) -> bool:
            """Simulate a directory turning into a link during scanning."""

            return path == child or original(path)

        with patch.object(
            Path, "is_symlink", autospec=True, side_effect=is_symlink
        ):
            report = repository.scan_repository(self.root)

        row = self._by_path(report)["src"]
        self.assertEqual(row["kind"], "link")
        self.assertEqual(row["reason"], "links are not followed")
        self.assertNotIn("src/main.py", report["selected_files"])
        self.assertTrue(report["complete"])

    def test_explicit_root_symlink_is_resolved(self) -> None:
        """Resolve the chosen root before descending into its files."""

        link = Path(self.temporary_directory.name) / "project-link"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Filesystem does not support symlinks: {error}")
        report = repository.scan_repository(link)
        self.assertEqual(report["repository"], str(self.root.resolve()))
        self.assertIn("README.md", report["selected_files"])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs unavailable")
    def test_special_files_are_not_selected(self) -> None:
        """Treat filesystem special entries as nonregular files."""

        fifo = self.root / "pipe.py"
        try:
            os.mkfifo(fifo)
        except OSError as error:
            self.skipTest(f"Filesystem cannot create a FIFO: {error}")
        report = repository.scan_repository(self.root)
        row = self._by_path(report)["pipe.py"]
        self.assertEqual(row["kind"], "other")
        self.assertEqual(row["reason_code"], "unsupported_entry")
        self.assertEqual(row["reason"], "not a regular file")

    def test_root_listing_failure_propagates(self) -> None:
        """Raise an error rather than return an incomplete root inventory."""

        original = repository.os.scandir
        resolved_root = self.root.resolve()

        def scandir(path: Path):
            """Fail when listing the selected repository root."""

            if Path(path) == resolved_root:
                raise PermissionError("denied")
            return original(path)

        with patch.object(repository.os, "scandir", side_effect=scandir):
            with self.assertRaisesRegex(OSError, "Cannot list"):
                repository.scan_repository(self.root)

    def test_descendant_listing_failure_marks_scan_incomplete(self) -> None:
        """Preserve other entries while recording an unreadable subtree."""

        original = repository.os.scandir
        blocked = (self.root / "src").resolve()

        def scandir(path: Path):
            """Reject the src directory while allowing other listings."""

            if Path(path) == blocked:
                raise PermissionError("denied")
            return original(path)

        with patch.object(repository.os, "scandir", side_effect=scandir):
            report = repository.scan_repository(self.root)

        self.assertFalse(report["complete"])
        self.assertIn("README.md", report["selected_files"])
        self.assertNotIn("src/main.py", report["selected_files"])
        row = self._by_path(report)["src"]
        self.assertEqual(row["kind"], "folder")
        self.assertEqual(row["reason_code"], "inspection_failed")
        self.assertEqual(row["reason"],
                         "cannot list folder (PermissionError)")

    def test_entry_inspection_failure_marks_scan_incomplete(self) -> None:
        """Represent an unreadable entry without aborting the scan."""

        original = repository._skip_reason

        def reason(relative: str, *args: object):
            """Simulate metadata inspection failure for one entry."""

            if relative == "src/main.py":
                raise PermissionError("denied")
            return original(relative, *args)

        with patch.object(repository, "_skip_reason", side_effect=reason):
            report = repository.scan_repository(self.root)

        self.assertFalse(report["complete"])
        row = self._by_path(report)["src/main.py"]
        self.assertEqual(row["kind"], "unknown")
        self.assertEqual(row["reason_code"], "inspection_failed")
        self.assertEqual(row["reason"],
                         "cannot inspect entry (PermissionError)")
        self.assertIn("src/helpers.py", report["selected_files"])


if __name__ == "__main__":
    unittest.main()
