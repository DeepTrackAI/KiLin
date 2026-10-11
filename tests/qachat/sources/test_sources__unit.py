"""Unit tests for source records, reading, dispatch, and extraction.

These tests exercise source coordination without a repository scan, model
loading, or execution of any selected source file. Filesystem operations use
small temporary directories; extension behavior is isolated with mocks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from qachat.sources import sources
from qachat.sources.sources import (
    SourceFile,
    SourceUnit,
    extract_source_units,
    read_repository_files,
)


class TestSourceRecords(unittest.TestCase):
    """Verify source-record fields and immutable provenance."""

    def test_source_file_is_frozen_and_equatable(self) -> None:
        """Keep relative paths and text in an immutable value record."""

        record = SourceFile(Path("src/example.py"), "value = 1\n")
        self.assertEqual(
            record, SourceFile(Path("src/example.py"), "value = 1\n")
        )
        with self.assertRaises(FrozenInstanceError):
            record.text = "changed"  # type: ignore[misc]

    def test_source_unit_defaults_and_children(self) -> None:
        """Retain optional source coordinates and nested units."""

        child = SourceUnit(Path("src/a.py"), "pass\n", 2, 2, "method")
        parent = SourceUnit(
            Path("src/a.py"),
            "class A:\n    pass\n",
            1,
            2,
            "python_class",
            symbol="A",
            section="Classes",
            cell=3,
            children=(child,),
        )
        self.assertEqual(parent.children, (child,))
        self.assertEqual((parent.symbol, parent.section, parent.cell),
                         ("A", "Classes", 3))
        self.assertIsNone(child.symbol)
        self.assertEqual(child.children, ())
        with self.assertRaises(FrozenInstanceError):
            parent.kind = "other"  # type: ignore[misc]


class TestReadRepositoryFiles(unittest.TestCase):
    """Verify report validation, safe paths, reading, and callbacks."""

    def setUp(self) -> None:
        """Create a local repository with text and binary examples."""

        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "repo"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "src" / "one.rst").write_text(
            "First file.\n", encoding="utf-8"
        )
        (self.root / "two.rst").write_text(
            "Second file.\n", encoding="utf-8"
        )
        (self.root / "invalid.rst").write_bytes(b"\xff\xfe")
        self.report = {
            "repository": str(self.root),
            "selected_files": ["src/one.rst", "two.rst"],
        }

    def test_reads_in_report_order_and_preserves_paths(self) -> None:
        """Respect selected order without sorting or rewriting text."""

        files = read_repository_files(self.report)
        self.assertEqual(
            files,
            [
                SourceFile(Path("src/one.rst"), "First file.\n"),
                SourceFile(Path("two.rst"), "Second file.\n"),
            ],
        )

    def test_empty_selection_does_not_call_progress(self) -> None:
        """Return no records and avoid progress notifications."""

        callback = Mock()
        report = {**self.report, "selected_files": []}
        self.assertEqual(
            read_repository_files(report, progress_callback=callback), []
        )
        callback.assert_not_called()

    def test_progress_reports_completed_files(self) -> None:
        """Notify callers with one-based progress after each read."""

        callback = Mock()
        read_repository_files(self.report, progress_callback=callback)
        self.assertEqual(
            callback.call_args_list,
            [call(1, 2), call(2, 2)],
        )

    def test_callback_exception_propagates_and_stops(self) -> None:
        """Do not suppress callback failures or read later files."""

        callback = Mock(side_effect=RuntimeError("stop"))
        with patch.object(
            sources, "_read_text_file", wraps=sources._read_text_file
        ) as read:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                read_repository_files(
                    self.report, progress_callback=callback
                )
        self.assertEqual(read.call_count, 1)
        callback.assert_called_once_with(1, 2)

    def test_invalid_report_shape_is_rejected(self) -> None:
        """Reject missing keys and incorrect report field types."""

        invalid = (
            None,
            {},
            {"repository": str(self.root)},
            {"selected_files": []},
            {"repository": 12, "selected_files": []},
            {"repository": str(self.root), "selected_files": ()},
            {"repository": str(self.root), "selected_files": "one"},
        )
        for report in invalid:
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    read_repository_files(report)

    def test_invalid_callback_is_rejected_before_reading(self) -> None:
        """Validate callback types before accessing selected files."""

        with patch.object(sources, "_read_text_file") as reader:
            with self.assertRaisesRegex(ValueError, "progress_callback"):
                read_repository_files(
                    self.report, progress_callback="invalid"
                )
        reader.assert_not_called()

    def test_invalid_selected_path_type_is_rejected(self) -> None:
        """Require strings for every selected path."""

        for invalid in (None, 42, ["two.rst"], Path("two.rst")):
            with self.subTest(value=invalid):
                report = {**self.report, "selected_files": [invalid]}
                with self.assertRaisesRegex(ValueError, "invalid file"):
                    read_repository_files(report)

    def test_absolute_and_parent_paths_are_rejected(self) -> None:
        """Reject lexical paths that escape the repository."""

        for invalid in ("../outside.rst", "src/../../outside.rst",
                        "/etc/passwd"):
            with self.subTest(path=invalid):
                report = {**self.report, "selected_files": [invalid]}
                with self.assertRaisesRegex(ValueError, "inside"):
                    read_repository_files(report)

    def test_symlinks_are_rejected(self) -> None:
        """Reject symlinks even when their targets are inside the root."""

        link = self.root / "linked.rst"
        try:
            link.symlink_to(self.root / "two.rst")
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symlinks are unavailable: {error}")
        report = {**self.report, "selected_files": ["linked.rst"]}
        with self.assertRaisesRegex(ValueError, "regular"):
            read_repository_files(report)

    def test_symlinked_directory_cannot_escape_root(self) -> None:
        """Reject a descendant path resolving outside the repository."""

        outside = self.root.parent / "outside.rst"
        outside.write_text("outside", encoding="utf-8")
        link = self.root / "external"
        try:
            link.symlink_to(self.root.parent, target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symlinks are unavailable: {error}")
        report = {
            **self.report,
            "selected_files": ["external/outside.rst"],
        }
        with self.assertRaisesRegex(ValueError, "regular"):
            read_repository_files(report)

    def test_missing_selected_file_propagates_os_error(self) -> None:
        """Do not silently omit missing selected files."""

        report = {**self.report, "selected_files": ["missing.rst"]}
        with self.assertRaises(FileNotFoundError):
            read_repository_files(report)

    def test_directory_is_not_a_source_file(self) -> None:
        """Reject selected paths that resolve to directories."""

        report = {**self.report, "selected_files": ["src"]}
        with self.assertRaisesRegex(ValueError, "regular"):
            read_repository_files(report)

    def test_invalid_utf8_is_reported_as_value_error(self) -> None:
        """Translate decoding failures without dropping provenance."""

        report = {**self.report, "selected_files": ["invalid.rst"]}
        with self.assertRaisesRegex(ValueError, "not UTF-8.*invalid.rst"):
            read_repository_files(report)

    def test_text_read_os_error_propagates(self) -> None:
        """Preserve filesystem read failures."""

        report = {**self.report, "selected_files": ["two.rst"]}
        with patch.object(
            sources.Path, "read_text", side_effect=OSError("unreadable")
        ):
            with self.assertRaisesRegex(OSError, "unreadable"):
                read_repository_files(report)

    def test_extension_reader_receives_canonical_paths(self) -> None:
        """Call an extension reader with canonical and relative paths."""

        relative = Path("two.rst")
        reader = Mock(return_value=SourceFile(relative, "custom"))
        extension = SimpleNamespace(read_source_file=reader)
        report = {**self.report, "selected_files": ["two.rst"]}
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            self.assertEqual(
                read_repository_files(report),
                [SourceFile(relative, "custom")],
            )
        reader.assert_called_once_with(
            (self.root / "two.rst").resolve(), relative
        )

    def test_invalid_extension_reader_result_is_rejected(self) -> None:
        """Require extension readers to return SourceFile objects."""

        report = {**self.report, "selected_files": ["two.rst"]}
        for invalid in (None, "text", [], 123):
            with self.subTest(value=invalid):
                extension = SimpleNamespace(
                    read_source_file=Mock(return_value=invalid)
                )
                with patch.object(
                    sources, "_extension_module", return_value=extension
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Invalid source reader"
                    ):
                        read_repository_files(report)

    def test_extension_reader_must_preserve_file_provenance(self) -> None:
        """Reject a reader that returns a different repository path."""

        report = {**self.report, "selected_files": ["two.rst"]}
        invalid = (
            SourceFile(Path("wrong.rst"), "data"),
            SourceFile(Path("two.rst"), 17),
            SourceFile("two.rst", "data"),
        )
        callback = Mock()
        for record in invalid:
            with self.subTest(record=record):
                extension = SimpleNamespace(
                    read_source_file=Mock(return_value=record)
                )
                with patch.object(
                    sources, "_extension_module", return_value=extension
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Invalid source reader"
                    ):
                        read_repository_files(
                            report, progress_callback=callback
                        )
        callback.assert_not_called()

    def test_extension_reader_failure_propagates(self) -> None:
        """Do not convert extension exceptions into empty files."""

        extension = SimpleNamespace(
            read_source_file=Mock(side_effect=ValueError("reader failed"))
        )
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            with self.assertRaisesRegex(ValueError, "reader failed"):
                read_repository_files(self.report)


class TestExtensionDiscovery(unittest.TestCase):
    """Verify missing hooks and broken dependencies are distinct."""

    def test_extension_name_uses_lowercase_suffix(self) -> None:
        """Normalize case when locating an extension module."""

        module = ModuleType("test_extension")
        with patch.object(
            sources, "import_module", return_value=module
        ) as imp:
            self.assertIs(sources._extension_module(Path("DOC.PY")), module)
        imp.assert_called_once_with(
            "qachat.sources.extensions.sources_py"
        )

    def test_missing_suffix_does_not_attempt_import(self) -> None:
        """Avoid dispatch for files with no suffix."""

        with patch.object(sources, "import_module") as imp:
            self.assertIsNone(sources._extension_module(Path("Makefile")))
        imp.assert_not_called()

    def test_missing_extension_is_an_optional_fallback(self) -> None:
        """Suppress only the missing requested extension module."""

        name = "qachat.sources.extensions.sources_rst"
        with patch.object(
            sources, "import_module",
            side_effect=ModuleNotFoundError(name=name),
        ):
            self.assertIsNone(sources._extension_module(Path("doc.rst")))

    def test_missing_dependency_inside_extension_propagates(self) -> None:
        """Do not disguise broken extension dependencies as absent hooks."""

        with patch.object(
            sources, "import_module",
            side_effect=ModuleNotFoundError(name="missing_dependency"),
        ):
            with self.assertRaises(ModuleNotFoundError):
                sources._extension_module(Path("doc.pdf"))

    def test_non_module_import_error_propagates(self) -> None:
        """Keep other extension-import failures visible."""

        with patch.object(
            sources, "import_module", side_effect=ImportError("bad import")
        ):
            with self.assertRaisesRegex(ImportError, "bad import"):
                sources._extension_module(Path("doc.py"))


class TestExtractSourceUnits(unittest.TestCase):
    """Verify extraction dispatch, fallbacks, and progress semantics."""

    def test_fallback_retains_nonblank_file_and_lines(self) -> None:
        """Preserve exact text and inclusive one-based line coordinates."""

        file = SourceFile(Path("docs/guide.rst"), "Title\n\nBody\n")
        units = extract_source_units([file])
        self.assertEqual(
            units,
            [SourceUnit(file.path, file.text, 1, 3, "file")],
        )

    def test_fallback_handles_unknown_and_extensionless_files(self) -> None:
        """Use a whole-file unit for unknown formats and no suffix."""

        files = [
            SourceFile(Path("LICENSE"), "Terms"),
            SourceFile(Path("custom.xyz"), "Data\n"),
        ]
        units = extract_source_units(files)
        self.assertEqual([unit.kind for unit in units], ["file", "file"])
        self.assertEqual([unit.path for unit in units],
                         [file.path for file in files])

    def test_fallback_skips_blank_and_empty_sources(self) -> None:
        """Omit whitespace-only fallback units without losing progress."""

        callback = Mock()
        files = [
            SourceFile(Path("blank.rst"), " \n\t\n"),
            SourceFile(Path("empty.rst"), ""),
        ]
        self.assertEqual(
            extract_source_units(files, progress_callback=callback), []
        )
        self.assertEqual(
            callback.call_args_list,
            [call(1, 2), call(2, 2)],
        )

    def test_extraction_preserves_file_and_unit_order(self) -> None:
        """Concatenate extension results in input order."""

        files = [
            SourceFile(Path("a.xyz"), "first"),
            SourceFile(Path("b.xyz"), "second"),
        ]
        first = SourceUnit(files[0].path, "first", 1, 1, "custom")
        second = SourceUnit(files[1].path, "second", 1, 1, "custom")
        extractor = Mock(side_effect=[[first], [second]])
        extension = SimpleNamespace(extract_source_units=extractor)
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            units = extract_source_units(files)
        self.assertEqual(units, [first, second])
        self.assertEqual(extractor.call_count, 2)

    def test_extractor_must_preserve_unit_provenance(self) -> None:
        """Reject units claiming to originate from a different file."""

        file = SourceFile(Path("a.xyz"), "data")
        wrong = SourceUnit(Path("wrong.xyz"), "data", 1, 1, "custom")
        extension = SimpleNamespace(
            extract_source_units=Mock(return_value=[wrong])
        )
        callback = Mock()
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Invalid source extractor"
            ):
                extract_source_units(
                    [file], progress_callback=callback
                )
        callback.assert_not_called()

    def test_extractor_nested_units_must_preserve_provenance(self) -> None:
        """Apply the same path contract to nested child units."""

        file = SourceFile(Path("a.xyz"), "data")
        wrong_child = SourceUnit(
            Path("wrong.xyz"), "data", 1, 1, "custom"
        )
        parent = SourceUnit(
            file.path, "data", 1, 1, "custom",
            children=(wrong_child,),
        )
        extension = SimpleNamespace(
            extract_source_units=Mock(return_value=[parent])
        )
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Invalid source extractor"
            ):
                extract_source_units([file])

    def test_extension_can_return_no_units(self) -> None:
        """Treat an empty list as a valid extractor result."""

        extension = SimpleNamespace(extract_source_units=Mock(return_value=[]))
        file = SourceFile(Path("a.xyz"), "data")
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            self.assertEqual(extract_source_units([file]), [])

    def test_invalid_extractor_results_are_rejected(self) -> None:
        """Require a list of SourceUnit records, not arbitrary objects."""

        file = SourceFile(Path("a.xyz"), "data")
        invalid = (None, "text", (), ["bad"], [SourceUnit(
            file.path, "data", 1, 1, "custom"
        ), None])
        for value in invalid:
            with self.subTest(value=value):
                extension = SimpleNamespace(
                    extract_source_units=Mock(return_value=value)
                )
                with patch.object(
                    sources, "_extension_module", return_value=extension
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Invalid source extractor"
                    ):
                        extract_source_units([file])

    def test_extension_extraction_failure_propagates(self) -> None:
        """Do not silently replace failing extraction with a fallback."""

        extension = SimpleNamespace(
            extract_source_units=Mock(side_effect=ValueError("invalid data"))
        )
        with patch.object(
            sources, "_extension_module", return_value=extension
        ):
            with self.assertRaisesRegex(ValueError, "invalid data"):
                extract_source_units([SourceFile(Path("x.xyz"), "data")])

    def test_extraction_callback_order_and_failure(self) -> None:
        """Call after each file and propagate callback exceptions."""

        files = [
            SourceFile(Path("a.rst"), "first"),
            SourceFile(Path("b.rst"), "second"),
        ]
        callback = Mock()
        extract_source_units(files, progress_callback=callback)
        self.assertEqual(
            callback.call_args_list,
            [call(1, 2), call(2, 2)],
        )
        callback.reset_mock()
        callback.side_effect = RuntimeError("stop")
        with self.assertRaisesRegex(RuntimeError, "stop"):
            extract_source_units(files, progress_callback=callback)
        callback.assert_called_once_with(1, 2)

    def test_invalid_extraction_callback_is_rejected(self) -> None:
        """Validate callbacks before attempting any extraction."""

        with self.assertRaisesRegex(ValueError, "progress_callback"):
            extract_source_units(
                [SourceFile(Path("a.rst"), "data")],
                progress_callback=1,
            )

    def test_no_input_produces_no_units_or_progress(self) -> None:
        """Handle empty input without invoking callbacks."""

        callback = Mock()
        self.assertEqual(
            extract_source_units([], progress_callback=callback), []
        )
        callback.assert_not_called()

    def test_source_unit_helper_preserves_coordinates(self) -> None:
        """Preserve optional provenance in the shared unit constructor."""

        path = Path("doc.ipynb")
        child = SourceUnit(path, "line", 1, 1, "child")
        unit = sources._source_unit(
            path,
            ["one\n", "two\n"],
            1,
            2,
            kind="custom",
            symbol="name",
            section="section",
            cell=4,
            children=(child,),
        )
        self.assertEqual(unit.text, "one\ntwo\n")
        self.assertEqual((unit.start_line, unit.end_line), (1, 2))
        self.assertEqual((unit.symbol, unit.section, unit.cell),
                         ("name", "section", 4))
        self.assertEqual(unit.children, (child,))

    def test_source_unit_helper_rejects_empty_slices(self) -> None:
        """Avoid constructing units for reversed or blank slices."""

        for lines, start, end in (
            (["content\n"], 2, 1),
            ([" \n", "\t"], 1, 2),
            ([], 1, 0),
        ):
            with self.subTest(lines=lines, start=start, end=end):
                self.assertIsNone(
                    sources._source_unit(
                        Path("a.rst"), lines, start, end, kind="file"
                    )
                )


if __name__ == "__main__":
    unittest.main()
