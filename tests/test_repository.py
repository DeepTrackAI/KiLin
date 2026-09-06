"""Tests for safe repository metadata selection."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import builtins
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat import repository


class TestRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
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
        for name, text in files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _by_path(report: dict[str, object]) -> dict[str, dict[str, object]]:
        entries = report["entries"]
        return {entry["path"]: entry for entry in entries}

    def test_scan_repository(self) -> None:
        self._check_default_preview()
        self._check_preview_never_reads_or_runs_files()
        self._check_include_and_exclude_paths()
        self._check_extensions_and_size()
        self._check_protected_entries()
        self._check_invalid_settings()
        self._check_invalid_repository()
        self._check_links_are_not_followed()
        self._check_entry_limit()

    def _check_default_preview(self) -> None:
        report = repository.scan_repository(self.root)
        self.assertTrue(report["complete"])
        self.assertEqual(
            report["selected_files"],
            [
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
            ],
        )
        rows = self._by_path(report)
        self.assertEqual(rows["image.png"]["reason"], "file type not selected")
        self.assertEqual(rows["empty.py"]["reason"], "empty file")
        self.assertEqual(
            rows[".env"]["reason"],
            "potentially secret or generated file",
        )
        self.assertEqual(
            rows[".env.local"]["reason"],
            "potentially secret or generated file",
        )
        self.assertNotIn(".git/config", rows)

    def _check_preview_never_reads_or_runs_files(self) -> None:
        message = "The scanner must only inspect metadata."
        with patch.object(Path, "open", side_effect=AssertionError(message)):
            with patch.object(
                builtins,
                "open",
                side_effect=AssertionError(message),
            ):
                with patch.object(
                    subprocess,
                    "run",
                    side_effect=AssertionError(message),
                ):
                    report = repository.scan_repository(self.root)
        self.assertIn("src/main.py", report["selected_files"])

    def _check_include_and_exclude_paths(self) -> None:
        choices = (
            (["src"], [], ["src/helpers.py", "src/main.py"]),
            (["src/main.py"], [], ["src/main.py"]),
            (["src"], ["src/helpers.py"], ["src/main.py"]),
            (["src\\main.py"], [], ["src/main.py"]),
            (["missing"], [], []),
        )
        for includes, excludes, expected in choices:
            with self.subTest(includes=includes, excludes=excludes):
                report = repository.scan_repository(
                    self.root,
                    include_paths=includes,
                    exclude_paths=excludes,
                )
                self.assertEqual(report["selected_files"], expected)

    def _check_extensions_and_size(self) -> None:
        (self.root / "four.txt").write_bytes(b"1234")
        (self.root / "five.txt").write_bytes(b"12345")
        report = repository.scan_repository(
            self.root,
            extensions=["TXT", ".txt"],
            max_file_bytes=4,
        )
        self.assertEqual(report["extensions"], [".txt"])
        self.assertEqual(report["selected_files"], ["four.txt"])
        rows = self._by_path(report)
        self.assertEqual(
            rows["five.txt"]["reason"],
            "larger than the file-size limit",
        )

    def _check_protected_entries(self) -> None:
        report = repository.scan_repository(
            self.root,
            extensions=[".json", ".safetensors", ".py"],
            include_paths=[
                "credentials.json",
                "cache.safetensors",
                ".venv",
            ],
        )
        self.assertEqual(report["selected_files"], [])

    def _check_invalid_settings(self) -> None:
        options: list[dict[str, object]] = [
            {"include_paths": "src"},
            {"exclude_paths": ["../outside"]},
            {"include_paths": ["src/*.py"]},
            {"extensions": ".py"},
            {"extensions": ["*"]},
            {"max_file_bytes": 0},
            {"max_entries": True},
        ]
        for settings in options:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    repository.scan_repository(self.root, **settings)

    def _check_invalid_repository(self) -> None:
        invalid = (
            "",
            "https://github.com/example/project",
            "git@example:project",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    repository.scan_repository(value)
        with self.assertRaises(FileNotFoundError):
            repository.scan_repository(self.root / "missing")
        with self.assertRaises(NotADirectoryError):
            repository.scan_repository(self.root / "README.md")

    def _check_links_are_not_followed(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        (outside / "outside.py").write_text("outside", encoding="utf-8")
        link = self.root / "linked-folder"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"This system cannot create a test link: {error}")
        report = repository.scan_repository(self.root)
        row = self._by_path(report)["linked-folder"]
        self.assertEqual(row["kind"], "link")
        self.assertNotIn("linked-folder/outside.py", report["selected_files"])

    def _check_entry_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Scan limit reached"):
            repository.scan_repository(self.root, max_entries=2)


if __name__ == "__main__":
    unittest.main()
