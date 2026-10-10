"""Unit tests for repository content manifests and comparison.

Construct disposable source trees and validate SHA-256 fingerprints against
independent digests. Comparison tests do not depend on filesystem access.

"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat import repository


class TestRepositoryManifestBuild(unittest.TestCase):
    """Test hashing and normalization of trusted scan reports."""

    def setUp(self) -> None:
        """Create a small repository with text and binary content."""

        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "source"
        self.root.mkdir()
        self.data = {
            "README.md": b"# README\n",
            "src/code.py": b"print('hello')\n",
            "notes.txt": b"one\ntwo\n",
            "raw.dat": b"\x00\xff\x10\x80",
        }
        for relative, content in self.data.items():
            self._write(relative, content)

    def _write(self, relative: str, content: bytes) -> Path:
        """Create one binary source file in the temporary repository."""

        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def _manifest(self, **settings: object) -> dict[str, object]:
        """Scan and fingerprint the current repository content."""

        report = repository.scan_repository(self.root, **settings)
        self.assertTrue(report["complete"])
        return repository.build_repository_manifest(report)

    def test_manifest_has_expected_schema_and_sha256_values(self) -> None:
        """Hash selected bytes correctly and omit absolute source paths."""

        manifest = self._manifest(extensions=[".py", ".md", ".txt"])
        self.assertEqual(manifest["format_version"], 1)
        expected = {
            relative: {
                "size_bytes": len(self.data[relative]),
                "sha256": sha256(self.data[relative]).hexdigest(),
            }
            for relative in ("README.md", "notes.txt", "src/code.py")
        }
        self.assertEqual(manifest["files"], expected)
        self.assertEqual(set(manifest), {"format_version", "files"})
        self.assertNotIn(str(self.root), json.dumps(manifest))
        self.assertEqual(manifest, json.loads(json.dumps(manifest)))

    def test_binary_files_are_hashed_as_raw_bytes(self) -> None:
        """Hash non-UTF-8 input without decoding the source file."""

        manifest = self._manifest(extensions=[".dat"])
        self.assertEqual(
            manifest["files"]["raw.dat"],
            {
                "size_bytes": 4,
                "sha256": sha256(b"\x00\xff\x10\x80").hexdigest(),
            },
        )

    def test_hashes_are_stable_across_repeated_scans(self) -> None:
        """Ignore unmodified timestamps when building a manifest."""

        initial = self._manifest(extensions=[".py", ".md"])
        source = self.root / "src/code.py"
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        later = self._manifest(extensions=[".py", ".md"])
        self.assertEqual(initial, later)

    def test_manifest_is_independent_of_absolute_repository_path(self) -> None:
        """Produce the same manifest from equivalent source directories."""

        mirror = Path(self.temporary_directory.name) / "copy"
        mirror.mkdir()
        for relative, content in self.data.items():
            target = mirror / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        first = self._manifest(extensions=[".dat", ".py", ".txt"])
        second = repository.build_repository_manifest(
            repository.scan_repository(
                mirror, extensions=[".dat", ".py", ".txt"]
            )
        )
        self.assertEqual(first, second)

    def test_same_size_edit_changes_hash_not_size(self) -> None:
        """Detect content changes when filesystem file sizes are unchanged."""

        before = self._manifest(extensions=[".txt"])
        self._write("notes.txt", b"six\ntwo\n")
        after = self._manifest(extensions=[".txt"])
        earlier = before["files"]["notes.txt"]
        current = after["files"]["notes.txt"]
        self.assertEqual(earlier["size_bytes"], current["size_bytes"])
        self.assertNotEqual(earlier["sha256"], current["sha256"])

    def test_large_selected_file_is_hashed_in_blocks(self) -> None:
        """Hash a file exceeding one 1-MiB input block correctly."""

        payload = (b"0123456789abcdef" * 65_537) + b"end"
        self._write("large.dat", payload)
        manifest = self._manifest(
            extensions=[".dat"], max_file_bytes=2_000_000
        )
        self.assertEqual(manifest["files"]["large.dat"], {
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        })

    def test_empty_selection_generates_empty_manifest(self) -> None:
        """Represent no selected files as a valid, versioned manifest."""

        report = repository.scan_repository(self.root, extensions=[])
        with patch.object(Path, "open", side_effect=AssertionError):
            manifest = repository.build_repository_manifest(report)
        self.assertEqual(manifest, {"format_version": 1, "files": {}})

    def test_report_source_and_selection_fields_are_validated(self) -> None:
        """Reject reports without a string root and list of selected files."""

        cases = (
            {},
            {"repository": str(self.root)},
            {"repository": 123, "selected_files": []},
            {"repository": str(self.root), "selected_files": "README.md"},
            {"repository": str(self.root), "selected_files": [None]},
            {"repository": str(self.root), "selected_files": [123]},
        )
        for report in cases:
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    repository.build_repository_manifest(report)

    def test_missing_root_and_selected_file_propagate(self) -> None:
        """Report vanished roots or files instead of silently omitting them."""

        scan = repository.scan_repository(self.root)
        (self.root / "README.md").unlink()
        with self.assertRaises(FileNotFoundError):
            repository.build_repository_manifest(scan)
        report = {
            "repository": str(self.root / "missing"),
            "selected_files": [],
        }
        with self.assertRaises(FileNotFoundError):
            repository.build_repository_manifest(report)

    def test_directories_cannot_be_hashed_as_files(self) -> None:
        """Reject a report that lists a directory as a selected file."""

        report = {
            "repository": str(self.root),
            "selected_files": ["src"],
        }
        with self.assertRaises(FileNotFoundError):
            repository.build_repository_manifest(report)

    def test_file_read_error_propagates(self) -> None:
        """Preserve a file-read failure during content hashing."""

        report = repository.scan_repository(self.root, extensions=[".md"])
        with patch.object(Path, "open", side_effect=PermissionError("no")):
            with self.assertRaises(PermissionError):
                repository.build_repository_manifest(report)


class TestRepositoryManifestComparison(unittest.TestCase):
    """Test structural validation and deterministic manifest differences."""

    @staticmethod
    def _manifest(**files: dict[str, object]) -> dict[str, object]:
        """Construct one supported manifest without touching the disk."""

        return {"format_version": 1, "files": files}

    @staticmethod
    def _record(marker: str = "a", size: int = 2) -> dict[str, object]:
        """Make one size and hexadecimal digest record."""

        return {"size_bytes": size, "sha256": marker * 64}

    def test_equal_manifests_match_without_changes(self) -> None:
        """Ignore dictionary insertion order and retained metadata fields."""

        saved = self._manifest(
            **{"b.py": self._record("b"), "a.py": self._record("a")}
        )
        current = self._manifest(
            **{"a.py": {**self._record("a"), "extra": "ignored"},
               "b.py": self._record("b")}
        )
        saved_before = deepcopy(saved)
        current_before = deepcopy(current)
        self.assertEqual(
            repository.compare_repository_manifests(saved, current),
            {"matches": True, "added": [], "removed": [],
             "modified": []},
        )
        self.assertEqual(saved, saved_before)
        self.assertEqual(current, current_before)

    def test_new_baseline_lists_all_current_files_as_added(self) -> None:
        """Treat None as missing baseline, even for an empty manifest."""

        for files, expected in (
            ({"b.py": self._record(), "a.py": self._record()},
             ["a.py", "b.py"]),
            ({}, []),
        ):
            with self.subTest(files=files):
                self.assertEqual(
                    repository.compare_repository_manifests(
                        None, self._manifest(**files)
                    ),
                    {"matches": False, "added": expected,
                     "removed": [], "modified": []},
                )

    def test_empty_saved_and_current_manifests_match(self) -> None:
        """Distinguish an existing empty baseline from a missing one."""

        empty = self._manifest()
        self.assertTrue(
            repository.compare_repository_manifests(
                empty, empty
            )["matches"]
        )

    def test_added_removed_and_modified_paths_are_sorted(self) -> None:
        """Separate and sort each kind of difference deterministically."""

        saved = self._manifest(**{
            "z-removed.py": self._record(),
            "b-removed.py": self._record(),
            "unchanged.py": self._record(),
            "a-modified.py": self._record("a"),
            "z-modified.py": self._record("a"),
        })
        current = self._manifest(**{
            "y-added.py": self._record(),
            "a-added.py": self._record(),
            "unchanged.py": self._record(),
            "a-modified.py": self._record("c"),
            "z-modified.py": self._record("a", size=3),
        })
        self.assertEqual(
            repository.compare_repository_manifests(saved, current),
            {
                "matches": False,
                "added": ["a-added.py", "y-added.py"],
                "removed": ["b-removed.py", "z-removed.py"],
                "modified": ["a-modified.py", "z-modified.py"],
            },
        )

    def test_size_and_digest_each_participate_in_comparison(self) -> None:
        """Mark a record modified when either normalized field differs."""

        initial = self._manifest(**{"main.py": self._record()})
        variants = (self._record(size=3), self._record("b"))
        for record in variants:
            with self.subTest(record=record):
                result = repository.compare_repository_manifests(
                    initial, self._manifest(**{"main.py": record})
                )
                self.assertFalse(result["matches"])
                self.assertEqual(result["modified"], ["main.py"])

    def test_invalid_versions_and_file_mappings(self) -> None:
        """Reject unsupported versions and absent or malformed mappings."""

        invalid = (
            {},
            {"format_version": 2, "files": {}},
            {"format_version": "1", "files": {}},
            {"format_version": 1},
            {"format_version": 1, "files": []},
            {"format_version": 1, "files": None},
        )
        valid = self._manifest()
        for value in invalid:
            for saved in (False, True):
                with self.subTest(value=value, saved=saved):
                    with self.assertRaises(ValueError):
                        if saved:
                            repository.compare_repository_manifests(
                                value, valid
                            )
                        else:
                            repository.compare_repository_manifests(
                                valid, value
                            )

    def test_invalid_file_record_structure(self) -> None:
        """Require string keys, dict records, exact sizes, and text hashes."""

        invalid_files = (
            {7: self._record()},
            {"a.py": []},
            {"a.py": None},
            {"a.py": {}},
            {"a.py": {"size_bytes": True, "sha256": "a" * 64}},
            {"a.py": {"size_bytes": 2.0, "sha256": "a" * 64}},
            {"a.py": {"size_bytes": 2, "sha256": None}},
            {"a.py": {"size_bytes": 2, "sha256": 3}},
        )
        for files in invalid_files:
            with self.subTest(files=files):
                with self.assertRaises(ValueError):
                    repository.compare_repository_manifests(
                        self._manifest(),
                        {"format_version": 1, "files": files},
                    )


if __name__ == "__main__":
    unittest.main()
