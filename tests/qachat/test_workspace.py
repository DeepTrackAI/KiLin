"""Test multi-source workspaces, imported snapshots, and source provenance.

Exercise the public workspace API using isolated local directories and real
project profiles. Verify cumulative imports, selected copies, provenance,
configuration, and filesystem safeguards without loading models or accessing
a user's files.

Expected failures document independently reproducible workspace defects; they
are not changes to the production module. The tests distinguish present
implementation behavior from the desired snapshot-preservation contract.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from qachat import workspace
from qachat.project import (
    default_project_profile,
    open_project,
)
from qachat.project_type import project_type
from qachat.repository import SKIP_DIRECTORIES


class WorkspaceFixture(unittest.TestCase):
    """Provide disposable sources and a workspace for each test.

    """

    def setUp(self) -> None:
        """Allocate an independent temporary directory.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.destination = self.root / "workspace"

    def _file(
        self,
        relative: str,
        content: str = "value = 1\n",
    ) -> Path:
        """Write a file relative to the fixture directory.

        Parameters
        ----------
        relative: str
            Relative file path to create.
        content: str, optional
            Contents of the file. Defaults to a short Python assignment.

        Returns
        -------
        pathlib.Path
            Source file with the requested content.

        """

        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _folder(self, relative: str) -> Path:
        """Create a directory under the fixture root.

        Parameters
        ----------
        relative: str
            Relative folder path.

        Returns
        -------
        pathlib.Path
            Existing temporary directory.

        """

        folder = self.root / relative
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _prepare(
        self,
        sources: list[Path] | tuple[Path, ...],
        *,
        kind: str = "code",
        destination: Path | None = None,
        **kwargs: object,
    ) -> workspace.QAProject:
        """Prepare the real workspace with the supplied options.

        Parameters
        ----------
        sources: list[pathlib.Path] | tuple[pathlib.Path, ...]
            Original source paths to import.
        kind: str, optional
            Project kind. Defaults to ``code``.
        destination: pathlib.Path | None, optional
            Override the workspace directory. Defaults to the fixture path.
        **kwargs: object
            Additional public preparation options.

        Returns
        -------
        qachat.project.QAProject
            Opened project representing the managed copy.

        """

        return workspace.prepare_sources_workspace(
            sources,
            kind,
            destination if destination is not None else self.destination,
            **kwargs,
        )

    def _records(self) -> list[dict[str, object]]:
        """Load the persisted provenance entries.

        Returns
        -------
        list[dict[str, object]]
            Source records from ``sources.json``.

        """

        payload = json.loads(
            (self.destination / "sources.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["version"], 1)
        return payload["sources"]

    def _write_records(self, records: object, version: object = 1) -> None:
        """Replace fixture provenance with an explicit payload.

        Parameters
        ----------
        records: object
            Candidate source-record collection.
        version: object, optional
            Format version to write. Defaults to 1.

        """

        (self.destination / "sources.json").write_text(
            json.dumps({"version": version, "sources": records}) + "\n",
            encoding="utf-8",
        )


class TestWorkspaceInitialization(WorkspaceFixture):
    """Check project creation, managed ownership, and safe reopening.

    """

    def test_new_workspace_has_profile_and_managed_directories(self) -> None:
        """Create the expected project structure and provenance record."""

        original = self._file("inputs/code.py")
        project = self._prepare([original])
        self.assertEqual(project.root, self.destination.resolve())
        self.assertEqual(project.source, self.destination / "sources")
        self.assertEqual(project.profile.kind, "code")
        self.assertEqual(project.profile.source, "sources")
        self.assertTrue(project.indexes.is_dir())
        self.assertTrue(project.logs.is_dir())
        self.assertTrue((self.destination / "profile.ini").is_file())
        self.assertEqual(
            (project.source / "code.py").read_bytes(),
            original.read_bytes(),
        )
        self.assertEqual(
            self._records(),
            [{
                "original": str(original.resolve()),
                "name": "code.py",
                "files": ["code.py"],
            }],
        )

    def test_empty_list_reopens_existing_workspace_without_changes(
        self,
    ) -> None:
        """An empty import is a no-op only for an existing workspace."""

        original = self._file("file.py")
        project = self._prepare([original])
        copy = project.source / "file.py"
        copy.write_text("local = True\n", encoding="utf-8")
        original.write_text("external = True\n", encoding="utf-8")
        before = (self.destination / "sources.json").read_bytes()
        reopened = self._prepare([])
        self.assertEqual(reopened, open_project(self.destination))
        self.assertEqual(copy.read_text(), "local = True\n")
        self.assertEqual(
            (self.destination / "sources.json").read_bytes(), before
        )

    def test_initial_empty_list_is_rejected_without_creating_project(
        self,
    ) -> None:
        """A new workspace needs at least one imported source."""

        with self.assertRaisesRegex(ValueError, "Add a source"):
            self._prepare([])
        self.assertFalse(self.destination.exists())

    def test_empty_existing_directory_can_be_adopted(self) -> None:
        """An empty user-chosen folder may become a QAChat workspace."""

        self.destination.mkdir()
        original = self._file("code.py")
        project = self._prepare([original])
        self.assertEqual(project.root, self.destination)

    def test_trace_only_workspace_is_accepted(self) -> None:
        """Preparation may follow creation of a logs/traces directory."""

        (self.destination / "logs/traces").mkdir(parents=True)
        original = self._file("code.py")
        project = self._prepare([original])
        self.assertTrue((project.source / "code.py").is_file())

    def test_trace_only_workspaces_reject_extra_content(self) -> None:
        """Do not adopt directories resembling but not matching trace setup."""

        cases = (
            ("logs/traces", "other.txt"),
            ("logs", "logs/extra.txt"),
            ("logs/traces", "logs/extra.txt"),
        )
        for index, (initial, extra) in enumerate(cases):
            with self.subTest(initial=initial, extra=extra):
                destination = self.root / f"case-{index}"
                (destination / initial).mkdir(parents=True)
                extra_path = destination / extra
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                extra_path.write_text("keep", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "empty folder"):
                    self._prepare(
                        [self._file("code.py")],
                        destination=destination,
                    )
                self.assertEqual(extra_path.read_text(), "keep")

    def test_trace_preparation_rejects_symlinked_logs(self) -> None:
        """Do not adopt a workspace whose trace location redirects outside."""

        self.destination.mkdir()
        log_target = self._folder("external-logs")
        (self.destination / "logs").symlink_to(
            log_target, target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "empty folder"):
            self._prepare([self._file("code.py")])

    def test_trace_preparation_rejects_symlinked_traces(self) -> None:
        """A permitted trace setup must contain actual directories."""

        logs = self.destination / "logs"
        logs.mkdir(parents=True)
        (logs / "traces").symlink_to(
            self._folder("external-traces"), target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "empty folder"):
            self._prepare([self._file("code.py")])

    def test_existing_unmanaged_folder_is_not_overwritten(self) -> None:
        """Protect unrelated user content in a nonempty destination."""

        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("mine", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty folder"):
            self._prepare([self._file("code.py")])
        self.assertEqual(marker.read_text(), "mine")
        self.assertFalse((self.destination / "profile.ini").exists())

    def test_existing_workspace_path_must_be_directory(self) -> None:
        """Reject selecting a regular file as the project root."""

        self.destination.write_text("not a folder", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            self._prepare([self._file("code.py")])

    def test_workspace_directory_must_not_be_home_or_drive_root(
        self,
    ) -> None:
        """Reject locations whose ownership cannot safely be assumed."""

        original = self._file("code.py")
        for directory in (Path.home(), Path(self.root.anchor)):
            with self.subTest(directory=directory):
                with self.assertRaisesRegex(ValueError, "home or drive"):
                    self._prepare([original], destination=directory)

    def test_blank_workspace_directory_is_rejected(self) -> None:
        """A blank destination is not interpreted as the current directory."""

        with self.assertRaisesRegex(ValueError, "workspace folder"):
            workspace.prepare_sources_workspace(
                [self._file("code.py")], "code", " "
            )

    def test_unknown_project_kind_is_rejected(self) -> None:
        """Unknown kinds cannot initialize a workspace."""

        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            self._prepare([self._file("code.py")], kind="unknown")
        self.assertFalse(self.destination.exists())

    def test_existing_project_kind_must_match(self) -> None:
        """A code workspace cannot be reopened as literature."""

        self._prepare([self._file("code.py")])
        before = (self.destination / "profile.ini").read_bytes()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self._prepare([], kind="literature")
        self.assertEqual(
            (self.destination / "profile.ini").read_bytes(), before
        )

    def test_existing_profile_must_retain_sources_ownership(self) -> None:
        """Do not import into a profile pointing at a different folder."""

        self._prepare([self._file("code.py")])
        profile = self.destination / "profile.ini"
        text = profile.read_text(encoding="utf-8")
        profile.write_text(
            text.replace("source = sources", "source = ."),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "managed sources"):
            self._prepare([])

    def test_unexpected_initialized_kind_is_rejected(self) -> None:
        """Check the defensive kind guard after project initialization."""

        project = self._prepare([self._file("code.py")])
        mismatched = replace(
            project, profile=replace(project.profile, kind="literature")
        )
        with patch.object(
            workspace, "initialize_project", return_value=mismatched
        ):
            with self.assertRaisesRegex(ValueError, "type does not match"):
                self._prepare([])

    def test_existing_profile_errors_are_not_suppressed(self) -> None:
        """A malformed existing profile is not silently regenerated."""

        self._prepare([self._file("code.py")])
        path = self.destination / "profile.ini"
        path.write_text("[project]\nkind = code\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing profile setting"):
            self._prepare([])


class TestWorkspaceSelections(WorkspaceFixture):
    """Verify supported file types, size limits, and explicit selections.

    """

    def test_literature_import_uses_packaged_size_limit(self) -> None:
        """Import a realistic PDF under literature's configured budget."""

        code = default_project_profile("code")
        literature = default_project_profile("literature")
        self.assertEqual(
            code.max_file_bytes, project_type("code").max_file_bytes
        )
        self.assertEqual(
            literature.max_file_bytes,
            project_type("literature").max_file_bytes,
        )
        pdf = self.root / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024))
        self.assertGreaterEqual(literature.max_file_bytes, pdf.stat().st_size)
        project = self._prepare([pdf], kind="literature")
        self.assertEqual(
            (project.source / "paper.pdf").stat().st_size,
            pdf.stat().st_size,
        )
        self.assertEqual(
            project.profile.max_file_bytes, literature.max_file_bytes
        )

    def test_folder_selection_respects_extensions_and_skips(self) -> None:
        """Directory imports use repository scanning rules."""

        source = self._folder("research")
        self._file("research/code.py")
        self._file("research/readme.md", "# Readme\n")
        self._file("research/image.png", "image")
        self._file("research/.git/hidden.py")
        self._file("research/__pycache__/cached.py")
        project = self._prepare([source])
        self.assertTrue((project.source / "research/code.py").exists())
        self.assertTrue((project.source / "research/readme.md").exists())
        for relative in (
            "research/image.png",
            "research/.git/hidden.py",
            "research/__pycache__/cached.py",
        ):
            with self.subTest(relative=relative):
                self.assertFalse((project.source / relative).exists())
        self.assertEqual(
            self._records()[0]["files"], ["code.py", "readme.md"]
        )

    def test_direct_file_currently_bypasses_directory_skip_rules(
        self,
    ) -> None:
        """Direct selection uses suffix and size, not the full scanner."""

        self.assertIn("__pycache__", SKIP_DIRECTORIES)
        original = self._file("research/__pycache__/code.py")
        project = self._prepare([original])
        self.assertTrue((project.source / "code.py").is_file())

    def test_checked_files_and_advanced_settings_are_used(self) -> None:
        """Import checked files and update the active profile."""

        source = self._folder("research")
        self._file("research/package/keep.py")
        self._file("research/skip.py")
        project = self._prepare(
            [source],
            selected_files={str(source): ["package/keep.py"]},
            settings={"target_tokens": 90, "max_tokens": 120},
        )
        self.assertTrue(
            (project.source / "research/package/keep.py").is_file()
        )
        self.assertFalse((project.source / "research/skip.py").exists())
        self.assertEqual(project.profile.target_tokens, 90)
        self.assertEqual(project.profile.max_tokens, 120)
        self.assertEqual(
            self._records()[0]["files"], ["package/keep.py"]
        )

    def test_directory_selection_uses_scan_order_and_deduplicates(
        self,
    ) -> None:
        """The order of selected files follows the deterministic scan."""

        source = self._folder("research")
        self._file("research/z.py")
        self._file("research/a.py")
        self._file("research/b.py")
        self._prepare(
            [source],
            selected_files={str(source): ["z.py", "a.py", "a.py"]},
        )
        self.assertEqual(self._records()[0]["files"], ["a.py", "z.py"])

    def test_invalid_directory_selections_are_rejected(self) -> None:
        """Reject missing, empty, absolute, and deselected file paths."""

        source = self._folder("research")
        self._file("research/keep.py")
        self._file("research/deselected.png", "not supported")
        invalid = (
            [],
            ["absent.py"],
            ["deselected.png"],
            ["../elsewhere.py"],
            [str(self.root / "research/keep.py")],
        )
        for selected in invalid:
            with self.subTest(selected=selected):
                destination = self.root / f"target-{invalid.index(selected)}"
                with self.assertRaisesRegex(ValueError, "Invalid file"):
                    self._prepare(
                        [source],
                        destination=destination,
                        selected_files={str(source): selected},
                    )
                self.assertFalse((destination / "sources/research").exists())

    def test_direct_file_selection_must_name_the_file(self) -> None:
        """A direct source cannot be selected through another filename."""

        source = self._file("folder/code.py")
        for selection in ([], ["other.py"], ["code.py", "code.py"]):
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, "Invalid file"):
                    self._prepare(
                        [source],
                        selected_files={str(source): selection},
                    )

    def test_direct_file_selection_accepts_exact_name(self) -> None:
        """The exact direct filename is a valid checked selection."""

        source = self._file("folder/code.py")
        project = self._prepare(
            [source], selected_files={str(source): ["code.py"]}
        )
        self.assertEqual(
            (project.source / "code.py").read_bytes(), source.read_bytes()
        )

    def test_unsupported_file_does_not_enter_workspace(self) -> None:
        """Reject direct files outside the project's accepted suffixes."""

        original = self._file("image.png", "not an image")
        with self.assertRaisesRegex(ValueError, "File type not selected"):
            self._prepare([original])
        self.assertFalse(
            (self.destination / "sources/image.png").exists()
        )

    def test_empty_and_oversized_direct_files_are_rejected(self) -> None:
        """A direct source must be nonempty and within its profile limit."""

        empty = self._file("empty.py", "")
        with self.assertRaisesRegex(ValueError, "empty or exceeds"):
            self._prepare([empty])
        maximum = default_project_profile("code").max_file_bytes
        huge = self.root / "huge.py"
        with huge.open("wb") as stream:
            stream.truncate(maximum + 1)
        with self.assertRaisesRegex(ValueError, "empty or exceeds"):
            self._prepare([huge])

    def test_size_at_limit_and_uppercase_extension_are_accepted(
        self,
    ) -> None:
        """Apply an inclusive size limit and case-insensitive suffix rules."""

        original = self._file("CODE.PY", "abcde")
        project = self._prepare(
            [original], settings={"max_file_bytes": 5}
        )
        self.assertEqual(
            (project.source / "CODE.PY").read_text(), "abcde"
        )

    def test_settings_can_select_additional_literal_extensions(
        self,
    ) -> None:
        """Allow new file types through the profile settings."""

        original = self._file("experiment.txt", "result")
        project = self._prepare(
            [original], settings={"extensions": [".TXT", ".py"]}
        )
        self.assertIn(".txt", project.profile.extensions)
        self.assertTrue((project.source / "experiment.txt").exists())

    def test_directory_scan_must_be_complete(self) -> None:
        """Incomplete scans abort import rather than copy partial evidence."""

        source = self._folder("folder")
        self._file("folder/code.py")
        incomplete = {"complete": False, "selected_files": ["code.py"]}
        with patch.object(
            workspace, "scan_repository", return_value=incomplete
        ):
            with self.assertRaisesRegex(ValueError, "completely scan"):
                self._prepare([source])
        self.assertFalse(
            (self.destination / "sources/folder/code.py").exists()
        )

    def test_empty_or_unselected_directory_is_rejected(self) -> None:
        """A directory with no eligible files cannot be imported."""

        source = self._folder("empty")
        self._file("empty/unsupported.png")
        with self.assertRaisesRegex(ValueError, "No selected files"):
            self._prepare([source])
        self.assertFalse((self.destination / "sources/empty").exists())

    def test_scan_excludes_nested_workspace(self) -> None:
        """The destination must not be recopied during parent import."""

        source = self._folder("research")
        self._file("research/code.py")
        destination = source / "workspace"
        with patch.object(
            workspace, "scan_repository", wraps=workspace.scan_repository
        ) as scanner:
            project = self._prepare([source], destination=destination)
        self.assertEqual(project.root, destination)
        self.assertFalse((project.source / "research/workspace").exists())
        self.assertEqual(
            scanner.call_args.kwargs["exclude_paths"], ("workspace",)
        )

    def test_scan_without_nested_workspace_uses_no_exclusion(
        self,
    ) -> None:
        """No exclusion is needed when the workspace is elsewhere."""

        source = self._folder("research")
        self._file("research/code.py")
        with patch.object(
            workspace, "scan_repository", wraps=workspace.scan_repository
        ) as scanner:
            self._prepare([source])
        self.assertEqual(scanner.call_args.kwargs["exclude_paths"], ())

    def test_repository_scan_limits_follow_active_profile(self) -> None:
        """Folder selection uses effective profile size and suffix options."""

        source = self._folder("research")
        self._file("research/code.py")
        with patch.object(
            workspace, "scan_repository", wraps=workspace.scan_repository
        ) as scanner:
            project = self._prepare(
                [source], settings={"max_file_bytes": 2048}
            )
        self.assertEqual(
            scanner.call_args.kwargs["max_file_bytes"],
            project.profile.max_file_bytes,
        )
        self.assertEqual(
            scanner.call_args.kwargs["extensions"],
            project.profile.extensions,
        )


class TestSnapshotBehavior(WorkspaceFixture):
    """Characterize imported files as project-owned local snapshots.

    """

    def test_original_changes_do_not_affect_snapshot_without_import(
        self,
    ) -> None:
        """External modifications have no implicit workspace effect."""

        original = self._file("code.py", "old = 1\n")
        project = self._prepare([original])
        original.write_text("new = 2\n", encoding="utf-8")
        self.assertEqual(
            (project.source / "code.py").read_text(), "old = 1\n"
        )
        self._prepare([])
        self.assertEqual(
            (project.source / "code.py").read_text(), "old = 1\n"
        )

    def test_deleted_original_does_not_delete_snapshot(self) -> None:
        """Losing the original does not remove its imported contents."""

        original = self._file("code.py")
        project = self._prepare([original])
        original.unlink()
        self._prepare([])
        self.assertTrue((project.source / "code.py").is_file())
        self.assertEqual(self._records()[0]["original"], str(original))

    def test_existing_snapshot_can_be_edited_independently(self) -> None:
        """Editing a workspace copy never writes to the external source."""

        original = self._file("code.py", "original = True\n")
        project = self._prepare([original])
        (project.source / "code.py").write_text(
            "local = True\n", encoding="utf-8"
        )
        self.assertEqual(original.read_text(), "original = True\n")
        self.assertEqual(
            (project.source / "code.py").read_text(), "local = True\n"
        )

    def test_reimport_currently_replaces_existing_snapshot(self) -> None:
        """Document the current explicit re-import copy semantics."""

        original = self._file("code.py", "old = 1\n")
        project = self._prepare([original])
        original.write_text("new = 2\n", encoding="utf-8")
        self._prepare([original])
        self.assertEqual(
            (project.source / "code.py").read_text(), "new = 2\n"
        )
        self.assertEqual(len(self._records()), 1)

    def test_reimport_does_not_remove_deselected_directory_files(
        self,
    ) -> None:
        """Removing a selection retains previously imported files."""

        folder = self._folder("research")
        self._file("research/a.py")
        self._file("research/b.py")
        self._prepare([folder])
        (folder / "a.py").unlink()
        self._prepare([folder])
        self.assertTrue(
            (self.destination / "sources/research/a.py").is_file()
        )
        self.assertTrue(
            (self.destination / "sources/research/b.py").is_file()
        )

    def test_new_source_is_added_without_removing_existing_snapshot(
        self,
    ) -> None:
        """Independent imports add entries rather than replacing sources."""

        first = self._file("one.py", "first = 1\n")
        second = self._file("two.py", "second = 2\n")
        self._prepare([first])
        self._prepare([second])
        self.assertTrue((self.destination / "sources/one.py").is_file())
        self.assertTrue((self.destination / "sources/two.py").is_file())
        self.assertEqual(len(self._records()), 2)

    def test_folder_and_file_share_one_workspace(self) -> None:
        """Mix directory and single-file imports with name collisions."""

        first = self._folder("one/methods")
        second = self._folder("two/methods")
        self._file("one/methods/scan.py", "def scan(): pass\n")
        self._file("two/methods/scan.m", "function scan\nend\n")
        note = self._file("notes.md", "# Notes\n")
        project = self._prepare([first, second, note])
        self.assertTrue((project.source / "methods/scan.py").is_file())
        self.assertTrue((project.source / "notes.md").is_file())
        matches = list((project.source).glob("methods-*/scan.m"))
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(list(project.source.iterdir())), 3)
        self.assertEqual(note.read_text(), "# Notes\n")
        self.assertEqual(len(self._records()), 3)
        self._prepare([])
        self.assertEqual(len(self._records()), 3)

    def test_identical_file_basenames_have_distinct_sources(self) -> None:
        """Different originals are not conflated by a shared filename."""

        first = self._file("one/model.py", "origin = 1\n")
        second = self._file("two/model.py", "origin = 2\n")
        project = self._prepare([first, second])
        digest = sha256(str(second).encode()).hexdigest()[:8]
        renamed = f"model-{digest}.py"
        self.assertEqual(
            (project.source / "model.py").read_text(), "origin = 1\n"
        )
        self.assertEqual(
            (project.source / renamed).read_text(), "origin = 2\n"
        )
        self.assertEqual(
            [record["name"] for record in self._records()],
            ["model.py", renamed],
        )

    def test_reimport_reuses_persisted_collision_name(self) -> None:
        """An existing original retains its name after other imports."""

        first = self._file("one/model.py")
        second = self._file("two/model.py", "two = 2\n")
        self._prepare([first, second])
        existing = self._records()[1]["name"]
        second.write_text("updated = 3\n", encoding="utf-8")
        self._prepare([second])
        self.assertEqual(self._records()[1]["name"], existing)
        self.assertEqual(
            (self.destination / "sources" / str(existing)).read_text(),
            "updated = 3\n",
        )

    def test_duplicate_inputs_in_one_call_are_imported_once(self) -> None:
        """Repeated canonical source paths produce one record and copy."""

        source = self._file("code.py")
        project = self._prepare([source, source, Path(str(source))])
        self.assertEqual(len(self._records()), 1)
        self.assertEqual(len(list(project.source.iterdir())), 1)

    def test_tuple_of_sources_is_supported(self) -> None:
        """The documented list and tuple input forms are accepted."""

        source = self._file("code.py")
        self._prepare((source,))
        self.assertEqual(len(self._records()), 1)

    def test_imports_are_cumulative_across_multiple_sessions(self) -> None:
        """Reopening preserves existing records and adds new originals."""

        first = self._file("first.py")
        second = self._file("second.py")
        self._prepare([first])
        original = self._records()[0]
        self._prepare([])
        self._prepare([second])
        records = self._records()
        self.assertEqual(records[0], original)
        self.assertEqual(records[1]["original"], str(second))

    @unittest.expectedFailure
    def test_reimport_cannot_silently_overwrite_local_changes(self) -> None:
        """Future snapshot policy should protect locally edited copies."""

        original = self._file("code.py", "original = 1\n")
        project = self._prepare([original])
        snapshot = project.source / "code.py"
        snapshot.write_text("local = 10\n", encoding="utf-8")
        original.write_text("original = 2\n", encoding="utf-8")
        self._prepare([original])
        self.assertEqual(snapshot.read_text(), "local = 10\n")

    @unittest.expectedFailure
    def test_reselection_manifest_lists_all_retained_snapshots(self) -> None:
        """Provenance must not forget earlier snapshots still on disk."""

        folder = self._folder("research")
        self._file("research/a.py")
        self._file("research/b.py")
        self._prepare(
            [folder], selected_files={str(folder): ["a.py"]}
        )
        self._prepare(
            [folder], selected_files={str(folder): ["b.py"]}
        )
        self.assertTrue(
            (self.destination / "sources/research/a.py").exists()
        )
        self.assertEqual(
            sorted(self._records()[0]["files"]), ["a.py", "b.py"]
        )

    @unittest.expectedFailure
    def test_unmatched_selection_mapping_must_not_import_all_files(
        self,
    ) -> None:
        """A mistyped selection key must not silently broaden an import."""

        folder = self._folder("research")
        self._file("research/selected.py")
        self._file("research/private.py")
        wrong_key = str(folder) + "-mistyped"
        with self.assertRaisesRegex(ValueError, "selection"):
            self._prepare(
                [folder],
                selected_files={wrong_key: ["selected.py"]},
            )


class TestWorkspaceValidation(WorkspaceFixture):
    """Exercise input validation and profile changes.

    """

    def test_invalid_source_container_is_rejected(self) -> None:
        """A scalar path is not a list of input sources."""

        with self.assertRaisesRegex(ValueError, "source files"):
            workspace.prepare_sources_workspace(
                str(self._file("code.py")), "code", self.destination
            )

    def test_missing_source_is_not_imported(self) -> None:
        """A selected original must exist at import time."""

        with self.assertRaises(FileNotFoundError):
            self._prepare([self.root / "missing.py"])
        self.assertFalse(self.destination.exists())

    def test_non_file_non_directory_source_is_rejected(self) -> None:
        """Special filesystem nodes cannot be imported as regular files."""

        original = self._file("code.py")
        with patch.object(Path, "is_file", return_value=False), patch.object(
            Path, "is_dir", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "regular file"):
                self._prepare([original])

    def test_source_root_must_not_be_workspace_itself(self) -> None:
        """Reject adding the destination as one of its own sources."""

        self._prepare([self._file("code.py")])
        with self.assertRaisesRegex(ValueError, "own source"):
            self._prepare([self.destination])

    def test_project_managed_directories_are_not_importable(self) -> None:
        """Exclude sources, indexes, and logs as new source origins."""

        self._prepare([self._file("code.py")])
        for folder in ("sources", "indexes", "logs"):
            with self.subTest(folder=folder):
                path = self.destination / folder
                with self.assertRaisesRegex(ValueError, "own source"):
                    self._prepare([path])

    def test_workspace_outputs_can_be_imported_explicitly(self) -> None:
        """User-approved generated output files may become evidence."""

        self._prepare([self._file("code.py")])
        generated = self.destination / "outputs/result.py"
        generated.parent.mkdir()
        generated.write_text("result = 2\n", encoding="utf-8")
        project = self._prepare([generated])
        self.assertEqual(
            (project.source / "result.py").read_text(),
            "result = 2\n",
        )

    def test_workspace_templates_can_be_imported_explicitly(self) -> None:
        """A template beneath the workspace can be an explicit source."""

        self._prepare([self._file("code.py")])
        template = self.destination / "templates/report.md"
        template.parent.mkdir()
        template.write_text("# Report\n", encoding="utf-8")
        project = self._prepare([template])
        self.assertTrue((project.source / "report.md").exists())

    def test_workspace_output_root_itself_is_not_allowed(self) -> None:
        """Only descendants of outputs/templates, not root folders, count."""

        self._prepare([self._file("code.py")])
        outputs = self.destination / "outputs"
        outputs.mkdir()
        with self.assertRaisesRegex(ValueError, "own source"):
            self._prepare([outputs])

    def test_workspace_source_subdirectory_is_not_allowed(self) -> None:
        """The copied tree cannot recursively import itself."""

        project = self._prepare([self._file("code.py")])
        with self.assertRaisesRegex(ValueError, "own source"):
            self._prepare([project.source / "code.py"])

    def test_invalid_settings_do_not_replace_existing_profile(self) -> None:
        """Validate a proposed profile change before writing it."""

        original = self._file("code.py")
        self._prepare([original])
        before = (self.destination / "profile.ini").read_bytes()
        with self.assertRaisesRegex(ValueError, "overlap_tokens"):
            self._prepare(
                [], settings={"target_tokens": 5, "overlap_tokens": 5}
            )
        self.assertEqual(
            (self.destination / "profile.ini").read_bytes(), before
        )

    def test_unknown_settings_do_not_change_profile(self) -> None:
        """Unknown advanced keys fail through project configuration."""

        self._prepare([self._file("code.py")])
        before = (self.destination / "profile.ini").read_bytes()
        with self.assertRaisesRegex(ValueError, "Unknown project setting"):
            self._prepare([], settings={"unknown_setting": True})
        self.assertEqual(
            (self.destination / "profile.ini").read_bytes(), before
        )

    def test_settings_can_be_applied_when_reopening(self) -> None:
        """An empty import can update project profile options."""

        self._prepare([self._file("code.py")])
        project = self._prepare(
            [], settings={"target_tokens": 125, "max_tokens": 155}
        )
        self.assertEqual(project.profile.target_tokens, 125)
        self.assertEqual(project.profile.max_tokens, 155)

    def test_existing_unsupported_source_root_symlink_is_rejected(
        self,
    ) -> None:
        """Sources may not redirect through a symbolic link."""

        self._prepare([self._file("code.py")])
        source_root = self.destination / "sources"
        original = self.root / "moved-sources"
        source_root.rename(original)
        source_root.symlink_to(original, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be a link"):
            self._prepare([])

    def test_top_level_source_symlink_is_rejected(self) -> None:
        """Reject a user-supplied symlink rather than following it."""

        original = self._file("code.py")
        alias = self.root / "alias.py"
        alias.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self._prepare([alias])
        self.assertFalse(self.destination.exists())

    def test_folder_symlinks_are_not_followed(self) -> None:
        """The repository scanner must skip symlinked descendants."""

        source = self._folder("research")
        external = self._file("external/code.py")
        self._file("research/visible.py")
        (source / "alias.py").symlink_to(external)
        project = self._prepare([source])
        self.assertTrue(
            (project.source / "research/visible.py").is_file()
        )
        self.assertFalse(
            (project.source / "research/alias.py").exists()
        )

    def test_existing_destination_symlink_is_rejected(self) -> None:
        """Do not replace a direct snapshot that has become a symlink."""

        original = self._file("code.py")
        project = self._prepare([original])
        target = self._file("outside/marker.txt", "protected")
        copy = project.source / "code.py"
        copy.unlink()
        copy.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self._prepare([original])
        self.assertEqual(target.read_text(), "protected")

    def test_existing_destination_parent_symlink_is_rejected(
        self,
    ) -> None:
        """Prevent directory copies through a symlinked snapshot folder."""

        folder = self._folder("research")
        self._file("research/code.py")
        project = self._prepare([folder])
        snapshot = project.source / "research"
        outside = self._folder("outside")
        (snapshot / "code.py").unlink()
        snapshot.rmdir()
        snapshot.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self._prepare([folder])
        self.assertFalse((outside / "code.py").exists())

    def test_source_copy_error_is_propagated(self) -> None:
        """Copy failures must be reported instead of recording success."""

        original = self._file("code.py")
        with patch.object(
            workspace.shutil, "copy2", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self._prepare([original])
        self.assertFalse((self.destination / "sources.json").exists())

    def test_a_failed_later_source_preserves_earlier_copy(self) -> None:
        """Nontransactional imports retain earlier committed source entries."""

        first = self._file("one.py")
        second = self._file("other.png", "not supported")
        with self.assertRaisesRegex(ValueError, "File type not selected"):
            self._prepare([first, second])
        self.assertTrue((self.destination / "sources/one.py").exists())
        self.assertEqual(len(self._records()), 1)

    @unittest.expectedFailure
    def test_temporary_copy_symlink_must_not_modify_external_file(
        self,
    ) -> None:
        """A stale .tmp symlink must not redirect the copy outside QAChat."""

        original = self._file("code.py", "source = 1\n")
        self._prepare([original])
        outside = self._file("outside/marker.txt", "PROTECTED\n")
        temporary = self.destination / "sources/code.py.tmp"
        temporary.symlink_to(outside)
        self._prepare([original])
        self.assertEqual(outside.read_text(), "PROTECTED\n")
        self.assertFalse(
            (self.destination / "sources/code.py").is_symlink()
        )


class TestWorkspaceProvenance(WorkspaceFixture):
    """Validate source-origin metadata and deterministic naming.

    """

    def test_missing_manifest_is_treated_as_no_previous_records(self) -> None:
        """An absent provenance file does not crash subsequent imports."""

        original = self._file("one.py")
        self._prepare([original])
        (self.destination / "sources.json").unlink()
        second = self._file("two.py")
        self._prepare([second])
        self.assertEqual(len(self._records()), 1)
        self.assertEqual(self._records()[0]["original"], str(second))

    def test_invalid_manifest_structure_is_rejected(self) -> None:
        """Reject invalid JSON, versions, and source-list representations."""

        self._prepare([self._file("code.py")])
        invalid = (
            "{broken",
            "[]",
            "{}",
            '{"version": 2, "sources": []}',
            '{"version": 1, "sources": null}',
            '{"version": 1, "sources": {}}',
        )
        for raw in invalid:
            with self.subTest(payload=raw):
                (self.destination / "sources.json").write_text(
                    raw, encoding="utf-8"
                )
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    self._prepare([])

    def test_invalid_source_record_entries_are_rejected(self) -> None:
        """Disallow records without string source or name attributes."""

        self._prepare([self._file("code.py")])
        invalid = (
            [None],
            [4],
            [{}],
            [{"original": 5, "name": "item.py"}],
            [{"original": "/tmp/item.py", "name": None}],
        )
        for record in invalid:
            with self.subTest(record=record):
                self._write_records(record)
                with self.assertRaisesRegex(ValueError, "source entry"):
                    self._prepare([])

    def test_invalid_destination_names_are_rejected(self) -> None:
        """An origin record cannot redirect imports by path traversal."""

        original = self._file("code.py")
        self._prepare([original])
        for name in ("", ".", "..", "../outside", "sub/file.py"):
            with self.subTest(name=name):
                self._write_records(
                    [{"original": str(original), "name": name}]
                )
                with self.assertRaisesRegex(ValueError, "source name"):
                    self._prepare([])

    def test_existing_untracked_filename_is_disambiguated(self) -> None:
        """Do not overwrite an unrelated file already inside sources."""

        self._prepare([self._file("initial.py")])
        original = self._file("external/existing.py")
        untracked = self.destination / "sources/existing.py"
        untracked.write_text("keep = True\n", encoding="utf-8")
        self._prepare([original])
        digest = sha256(str(original).encode()).hexdigest()[:8]
        name = f"existing-{digest}.py"
        self.assertEqual(untracked.read_text(), "keep = True\n")
        self.assertTrue((untracked.parent / name).exists())

    def test_hashed_collision_is_rejected(self) -> None:
        """Do not allocate names already occupied or reserved."""

        original = self._file("external/existing.py")
        digest = sha256(str(original).encode()).hexdigest()[:8]
        records = {"existing.py", f"existing-{digest}.py"}
        with self.assertRaisesRegex(ValueError, "distinct source name"):
            workspace._unique_name(original, records, self.root)

    def test_records_are_written_atomically_with_temporary_sibling(
        self,
    ) -> None:
        """A successful import leaves a complete JSON file and no temp."""

        self._prepare([self._file("code.py")])
        self.assertFalse((self.destination / "sources.json.tmp").exists())
        raw = (self.destination / "sources.json").read_text()
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(json.loads(raw)["version"], 1)

    @unittest.expectedFailure
    def test_temporary_manifest_symlink_must_not_modify_external_file(
        self,
    ) -> None:
        """A predictable metadata temp file must not redirect writes."""

        first = self._file("one.py")
        second = self._file("two.py")
        self._prepare([first])
        protected = self._file("outside/marker.txt", "PROTECTED\n")
        (self.destination / "sources.json.tmp").symlink_to(protected)
        self._prepare([second])
        self.assertEqual(protected.read_text(), "PROTECTED\n")

    @unittest.expectedFailure
    def test_duplicate_stored_names_must_be_rejected(self) -> None:
        """Two originals cannot legitimately own the same copied filename."""

        first = self._file("one/code.py", "first = 1\n")
        second = self._file("two/code.py", "second = 2\n")
        self._prepare([first])
        self._write_records([
            {"original": str(first), "name": "code.py"},
            {"original": str(second), "name": "code.py"},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate|conflict"):
            self._prepare([second])


class TestWorkspaceTracing(WorkspaceFixture):
    """Check the event contract without a persistent trace backend.

    """

    def test_trace_records_preparation_and_copy_events(self) -> None:
        """Emit a start, source copy, and completion with useful metadata."""

        first = self._file("one.py")
        second = self._file("two.py")
        trace = MagicMock()
        project = self._prepare([first, second], trace=trace)
        events = [call.args[0] for call in trace.emit.call_args_list]
        self.assertEqual(events, [
            "workspace.prepare.started",
            "workspace.source.copied",
            "workspace.source.copied",
            "workspace.prepare.completed",
        ])
        first_data = trace.emit.call_args_list[0].kwargs["data"]
        self.assertEqual(first_data["source_paths"], [str(first), str(second)])
        self.assertEqual(first_data["kind"], "code")
        self.assertEqual(
            trace.emit.call_args_list[1].kwargs["data"]["copied_files"],
            ["one.py"],
        )
        completion = trace.emit.call_args_list[-1].kwargs
        self.assertEqual(completion["data"]["source_count"], 2)
        self.assertEqual(
            completion["data"]["project_directory"], str(project.root)
        )
        self.assertGreaterEqual(completion["duration_ms"], 0)

    def test_no_sources_emits_only_start_and_completion(self) -> None:
        """A reopened workspace has no source-copy events."""

        self._prepare([self._file("code.py")])
        trace = MagicMock()
        self._prepare([], trace=trace)
        self.assertEqual(
            [call.args[0] for call in trace.emit.call_args_list],
            ["workspace.prepare.started", "workspace.prepare.completed"],
        )

    def test_configure_project_receives_caller_trace(self) -> None:
        """Forward the provided trace sink to configuration updates."""

        original = self._file("code.py")
        self._prepare([original])
        trace = MagicMock()
        with patch.object(
            workspace, "configure_project", wraps=workspace.configure_project
        ) as configure:
            self._prepare(
                [], settings={"retrieval_count": 7}, trace=trace
            )
        self.assertIs(configure.call_args.kwargs["trace"], trace)
        trace.settings_changed.assert_called_once()

    def test_trace_events_are_not_emitted_for_failed_copies(self) -> None:
        """Failed imports should not emit copied or completed events."""

        trace = MagicMock()
        with self.assertRaisesRegex(ValueError, "File type"):
            self._prepare([self._file("image.png")], trace=trace)
        self.assertEqual(
            [call.args[0] for call in trace.emit.call_args_list],
            ["workspace.prepare.started"],
        )


if __name__ == "__main__":
    unittest.main()
