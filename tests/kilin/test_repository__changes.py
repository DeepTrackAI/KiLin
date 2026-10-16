"""Unit tests for repository change detection and saved manifests.

Run the complete scan, hash, and compare workflow on disposable repositories.
Saved manifests are supplied as mappings or local JSON files; no network or
external repository is used.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import repository


class TestRepositoryChangeDetection(unittest.TestCase):
    """Test change detection against an actual temporary source tree."""

    def setUp(self) -> None:
        """Create a known initial manifest and source tree."""

        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "source"
        self._write("README.md", "A project\n")
        self._write("src/main.py", "x = 1\n")
        self._write("src/helper.py", "y = 2\n")
        self._write("other.py", "z = 3\n")
        self._write("ignore.bin", "binary content")
        self._write(".env", "FAKE_SECRET=dummy")
        self.baseline = repository.build_repository_manifest(
            repository.scan_repository(self.root)
        )

    def _write(self, relative: str, content: str) -> Path:
        """Create or replace one repository file."""

        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def _changes(
        self,
        saved: dict[str, object] | str | Path | None = None,
        **settings: object,
    ) -> dict[str, object]:
        """Compare current content to a baseline or explicit manifest."""

        if saved is None:
            saved = self.baseline
        return repository.detect_repository_changes(
            self.root, saved, **settings
        )

    def test_unchanged_source_matches_baseline(self) -> None:
        """Return no differences without altering the saved mapping."""

        before = deepcopy(self.baseline)
        self.assertEqual(
            self._changes(),
            {"matches": True, "added": [], "removed": [],
             "modified": []},
        )
        self.assertEqual(self.baseline, before)
        self.assertFalse(
            repository.repository_has_changed(self.root, self.baseline)
        )

    def test_new_selected_files_are_added(self) -> None:
        """List new matching files in lexicographic order."""

        self._write("z-new.py", "new = 1\n")
        self._write("a-new.py", "new = 2\n")
        report = self._changes()
        self.assertFalse(report["matches"])
        self.assertEqual(report["added"], ["a-new.py", "z-new.py"])
        self.assertEqual(report["removed"], [])
        self.assertEqual(report["modified"], [])
        self.assertTrue(
            repository.repository_has_changed(self.root, self.baseline)
        )

    def test_deleted_files_are_removed(self) -> None:
        """Detect selected files missing from the current scan."""

        (self.root / "src/helper.py").unlink()
        (self.root / "README.md").unlink()
        report = self._changes()
        self.assertEqual(report["removed"], ["README.md", "src/helper.py"])
        self.assertEqual(report["added"], [])
        self.assertEqual(report["modified"], [])

    def test_changed_contents_are_modified_without_size_change(self) -> None:
        """Identify edits using hashes even when file size is unchanged."""

        self._write("src/main.py", "x = 9\n")
        report = self._changes()
        self.assertEqual(report["modified"], ["src/main.py"])
        self.assertEqual(report["added"], [])
        self.assertEqual(report["removed"], [])

    def test_simultaneous_added_removed_and_modified_changes(self) -> None:
        """Report all change categories in one detection pass."""

        (self.root / "src/helper.py").unlink()
        self._write("other.py", "z = 8\n")
        self._write("src/new.py", "n = 1\n")
        report = self._changes()
        self.assertEqual(report, {
            "matches": False,
            "added": ["src/new.py"],
            "removed": ["src/helper.py"],
            "modified": ["other.py"],
        })

    def test_unselected_changes_do_not_affect_manifest(self) -> None:
        """Ignore files outside the suffix and built-in safety selection."""

        self._write("ignore.bin", "new binary value")
        self._write(".env", "FAKE_SECRET=changed")
        self._write("another.jpeg", "not selected")
        report = self._changes()
        self.assertTrue(report["matches"])
        self.assertEqual(report["added"], [])
        self.assertEqual(report["modified"], [])

    def test_include_scope_is_respected(self) -> None:
        """Limit comparisons to the same explicit selection as the baseline."""

        settings = {"include_paths": ["src"]}
        source_baseline = repository.build_repository_manifest(
            repository.scan_repository(self.root, **settings)
        )
        self._write("README.md", "Changed project\n")
        self.assertTrue(self._changes(source_baseline, **settings)["matches"])
        self._write("src/main.py", "x = 9\n")
        changes = self._changes(source_baseline, **settings)
        self.assertEqual(changes["modified"], ["src/main.py"])
        self.assertEqual(changes["added"], [])

    def test_exclude_scope_is_respected(self) -> None:
        """Keep explicitly excluded modifications out of the comparison."""

        settings = {"exclude_paths": ["src"]}
        source_baseline = repository.build_repository_manifest(
            repository.scan_repository(self.root, **settings)
        )
        self._write("src/main.py", "x = 9\n")
        self.assertTrue(self._changes(source_baseline, **settings)["matches"])
        self._write("other.py", "z = 8\n")
        self.assertEqual(
            self._changes(source_baseline, **settings)["modified"],
            ["other.py"],
        )

    def test_changed_selection_policy_is_reflected_in_differences(
        self,
    ) -> None:
        """Compare selected paths without implicitly remapping scopes."""

        report = self._changes(extensions=[".py"])
        self.assertFalse(report["matches"])
        self.assertEqual(report["removed"], ["README.md"])
        self.assertEqual(report["added"], [])

    def test_missing_baseline_is_not_a_match(self) -> None:
        """Differentiate no saved manifest from an empty existing baseline."""

        report = repository.detect_repository_changes(self.root, None)
        self.assertFalse(report["matches"])
        self.assertEqual(report["added"], sorted(
            self.baseline["files"]
        ))
        self.assertEqual(report["removed"], [])
        self.assertEqual(report["modified"], [])
        self.assertTrue(repository.repository_has_changed(self.root, None))

    def test_missing_baseline_is_not_a_match_for_empty_selection(self) -> None:
        """Flag a missing baseline even when there are no selected files."""

        options = {"extensions": []}
        result = repository.detect_repository_changes(
            self.root, None, **options
        )
        self.assertEqual(
            result, {"matches": False, "added": [], "removed": [],
                     "modified": []}
        )
        self.assertTrue(
            repository.repository_has_changed(self.root, None, **options)
        )
        empty = {"format_version": 1, "files": {}}
        self.assertFalse(repository.repository_has_changed(
            self.root, empty, **options
        ))

    def test_missing_manifest_file_behaves_like_none(self) -> None:
        """Do not create a missing saved-manifest file during detection."""

        missing = Path(self.temporary_directory.name) / "missing.json"
        report = self._changes(missing)
        self.assertFalse(report["matches"])
        self.assertEqual(report["added"], sorted(self.baseline["files"]))
        self.assertFalse(missing.exists())

    def test_saved_json_file_matches_dictionary_baseline(self) -> None:
        """Read an existing JSON manifest as a complete saved baseline."""

        path = Path(self.temporary_directory.name) / "saved.json"
        path.write_text(json.dumps(self.baseline), encoding="utf-8")
        self.assertEqual(self._changes(path), self._changes(self.baseline))
        self._write("src/main.py", "x = 9\n")
        self.assertEqual(self._changes(path)["modified"], ["src/main.py"])
        self.assertTrue(repository.repository_has_changed(self.root, path))

    def test_relative_saved_manifest_path_uses_working_directory(
        self,
    ) -> None:
        """Resolve a relative saved-manifest path against the process cwd."""

        path = Path(self.temporary_directory.name) / "saved.json"
        path.write_text(json.dumps(self.baseline), encoding="utf-8")
        # A relative saved path is based on cwd, not PROJECT_ROOT.
        original = Path.cwd()
        try:
            os.chdir(self.temporary_directory.name)
            changes = self._changes("saved.json")
        finally:
            os.chdir(original)
        self.assertTrue(changes["matches"])

    def test_existing_directory_is_not_a_saved_manifest(self) -> None:
        """Reject a directory instead of treating it as an absent baseline."""

        with self.assertRaisesRegex(ValueError, "not a file"):
            self._changes(self.root)

    def test_invalid_manifest_arguments(self) -> None:
        """Reject non-path values and whitespace-only manifest paths."""

        for invalid in ("", "  ", 123, [], True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self._changes(invalid)

    def test_invalid_manifest_json_rejected(self) -> None:
        """Propagate JSON parsing errors without treating them as missing."""

        path = Path(self.temporary_directory.name) / "corrupt.json"
        path.write_text("{bad JSON", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self._changes(path)

    def test_saved_manifest_json_must_contain_an_object(self) -> None:
        """Require a top-level mapping before comparing saved records."""

        path = Path(self.temporary_directory.name) / "array.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must contain an object"):
            self._changes(path)

    def test_invalid_saved_manifest_version_propagates(self) -> None:
        """Reject an unsupported schema version during comparison."""

        invalid = {"format_version": 99, "files": {}}
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self._changes(invalid)

    def test_invalid_scan_settings_propagate(self) -> None:
        """Do not hide selection validation errors in convenience wrappers."""

        with self.assertRaises(ValueError):
            self._changes(include_paths=["../unsafe"])
        with self.assertRaises(ValueError):
            repository.repository_has_changed(
                self.root, self.baseline, max_entries=0
            )

    def test_detect_wrapper_forwards_all_selection_arguments(self) -> None:
        """Pass selection limits to scan and compare the resulting manifest."""

        report = {
            "repository": str(self.root), "complete": True,
            "selected_files": [],
        }
        manifest = {"format_version": 1, "files": {}}
        expected = {"matches": True, "added": [], "removed": [],
                    "modified": []}
        settings = {
            "extensions": [".py"],
            "include_paths": ["src"],
            "exclude_paths": ["src/helper.py"],
            "max_file_bytes": 123,
            "max_entries": 40,
        }
        with (
            patch.object(repository, "scan_repository", return_value=report)
            as scan,
            patch.object(
                repository, "build_repository_manifest", return_value=manifest
            ) as build,
            patch.object(
                repository, "_load_repository_manifest", return_value=manifest
            ) as load,
            patch.object(
                repository, "compare_repository_manifests",
                return_value=expected,
            ) as compare,
        ):
            result = repository.detect_repository_changes(
                self.root, self.baseline, **settings
            )

        self.assertIs(result, expected)
        scan.assert_called_once_with(self.root, **settings)
        build.assert_called_once_with(report)
        load.assert_called_once_with(self.baseline)
        compare.assert_called_once_with(manifest, manifest)

    def test_boolean_wrapper_negates_detection_match(self) -> None:
        """Forward options and negate only the matches field."""

        options = {"extensions": [".py"], "max_file_bytes": 10}
        for matches in (True, False):
            with self.subTest(matches=matches):
                with patch.object(
                    repository, "detect_repository_changes",
                    return_value={"matches": matches},
                ) as detect:
                    changed = repository.repository_has_changed(
                        self.root, self.baseline, **options
                    )
                self.assertEqual(changed, not matches)
                detect.assert_called_once_with(
                    self.root,
                    self.baseline,
                    extensions=[".py"],
                    include_paths=(),
                    exclude_paths=(),
                    max_file_bytes=10,
                    max_entries=10_000,
                )


    def test_incomplete_scan_prevents_change_reporting(self) -> None:
        """Never classify inaccessible source files as removed."""

        report = repository.scan_repository(self.root)
        report["complete"] = False
        with patch.object(
            repository, "scan_repository", return_value=report
        ):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self._changes()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                repository.repository_has_changed(
                    self.root, self.baseline
                )


if __name__ == "__main__":
    unittest.main()
