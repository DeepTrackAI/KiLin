"""Verify exclusive project creation and explicit existing-project updates.

Use real files and independent processes to ensure collision rejection does
not merely raise an error after modifying another project's state.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import kilin
from kilin.atomic_writes import atomic_create_text
from kilin._project_creation import claim_workspace


class TestExclusiveProjectCreation(unittest.TestCase):
    """Protect occupied destinations independently of every interface."""

    def setUp(self):
        """Prepare sources outside every candidate project."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source.py"
        self.source.write_text("value = 1\n", encoding="utf-8")
        self.destination = self.root / "project"

    def snapshot(self, root):
        """Record tree contents and modification times without following links.

        Parameters
        ----------
        root: pathlib.Path
            Existing tree to compare.

        Returns
        -------
        dict
            Relative path to mode, modification time, and bytes or link target.

        """
        result = {}
        for path in [root, *root.rglob("*")]:
            stat = path.lstat()
            content = (
                os.readlink(path) if path.is_symlink()
                else path.read_bytes() if path.is_file() else None
            )
            result[path.relative_to(root).as_posix()] = (
                stat.st_mode, stat.st_mtime_ns, content
            )
        return result

    def create(self, **options):
        """Create an explicit managed fixture with optional profile settings.

        Parameters
        ----------
        **options: object
            Additional creation options.

        Returns
        -------
        kilin.QAProject
            Fresh managed project.

        """
        return kilin.prepare_sources_workspace(
            [self.source], "code", self.destination, **options
        )

    def test_explicit_collision_preserves_every_entry_and_emits_no_trace(self):
        """Reject without touching profile, index, conversation, or trace."""
        project = self.create()
        for relative in (
            "indexes/settings.json", "logs/conversations/saved.json",
            "logs/traces/saved.json", "outputs/result.txt",
            "templates/custom.md",
        ):
            path = project.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("user content: " + relative, encoding="utf-8")
        before = self.snapshot(project.root)
        trace = Mock()
        with self.assertRaises(kilin.ProjectAlreadyExistsError):
            self.create(settings={"temperature": 0.7}, trace=trace)
        trace.emit.assert_not_called()
        self.assertEqual(self.snapshot(project.root), before)
        self.assertEqual(kilin.open_project(project.root).profile.temperature,
                         project.profile.temperature)

    def test_initialize_cannot_bypass_existing_project_guard(self):
        """Enforce ownership through the lower-level public initializer."""
        project = self.create()
        before = self.snapshot(project.root)
        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                with self.assertRaises(kilin.ProjectAlreadyExistsError):
                    kilin.initialize_project(project.root, type_id)
                self.assertEqual(self.snapshot(project.root), before)

    def test_partial_and_unrelated_destinations_are_protected(self):
        """Logs-only, abandoned claims, and unrelated files are not adopted."""
        for relative in (
            "notes.txt", "logs/traces/partial.json", ".kilin-creating",
            "sources/partial.py",
        ):
            with self.subTest(relative=relative):
                root = self.root / relative.replace("/", "-")
                path = root / relative
                path.parent.mkdir(parents=True)
                path.write_text("preserve", encoding="utf-8")
                before = self.snapshot(root)
                for create in (
                    lambda: kilin.initialize_project(root, "code"),
                    lambda: kilin.prepare_sources_workspace([], "code", root),
                ):
                    with self.assertRaises(kilin.WorkspaceOccupiedError):
                        create()
                    self.assertEqual(self.snapshot(root), before)

    def test_invalid_profile_entry_is_still_an_existing_project(self):
        """Do not silently regenerate a malformed or directory profile."""
        self.destination.mkdir()
        profile = self.destination / "profile.ini"
        profile.write_text("invalid", encoding="utf-8")
        with self.assertRaises(kilin.ProjectAlreadyExistsError):
            self.create()
        self.assertEqual(profile.read_text(encoding="utf-8"), "invalid")
        profile.unlink()
        profile.mkdir()
        with self.assertRaises(kilin.ProjectAlreadyExistsError):
            self.create()
        self.assertTrue(profile.is_dir())

    def test_default_collisions_allocate_independent_projects(self):
        """Allocate independent roots for the same sources or other types."""
        parent = self.root / "projects"
        first = kilin.prepare_sources_workspace(
            [self.source], "code", workspace_root=parent
        )
        before = self.snapshot(first.root)
        self.source.write_text("value = 2\n", encoding="utf-8")
        second = kilin.prepare_sources_workspace(
            [self.source], "code", workspace_root=parent,
            settings={"temperature": 0.7},
        )
        third = kilin.prepare_sources_workspace(
            [self.source], "literature", workspace_root=parent
        )
        self.assertEqual(second.root.name, first.root.name + "-2")
        self.assertEqual(third.root.name, first.root.name + "-3")
        self.assertEqual(self.snapshot(first.root), before)
        self.assertEqual(first.profile.temperature, 0.0)
        self.assertEqual(second.profile.temperature, 0.7)
        self.assertEqual(third.profile.type_id, "literature")

    def test_default_skips_empty_files_links_and_partial_directories(self):
        """Any occupied default name belongs to its current owner."""
        parent = self.root / "projects"
        base = kilin.default_workspace_path([self.source], parent)
        base.mkdir(parents=True)
        (parent / (base.name + "-2")).write_text("keep", encoding="utf-8")
        linked = parent / (base.name + "-3")
        try:
            linked.symlink_to(self.root / "missing", target_is_directory=True)
        except OSError:
            self.skipTest("Symbolic links are unavailable")
        partial = parent / (base.name + "-4")
        partial.mkdir()
        (partial / ".kilin-creating").write_text("old", encoding="utf-8")
        before = self.snapshot(parent)
        project = kilin.prepare_sources_workspace(
            [self.source], "code", workspace_root=parent
        )
        self.assertEqual(project.root.name, base.name + "-5")
        after = self.snapshot(parent)
        for key, value in before.items():
            if key != ".":
                self.assertEqual(after[key], value)

    def test_default_first_slot_is_a_file_or_link(self):
        """Skip occupied preferred names without inspecting their targets."""
        target = self.root / "outside"
        target.mkdir()
        (target / "keep.txt").write_text("keep", encoding="utf-8")
        for label in ("file", "link", "broken", "linked-sources"):
            with self.subTest(entry=label):
                parent = self.root / label
                parent.mkdir()
                base = kilin.default_workspace_path([self.source], parent)
                if label == "file":
                    base.write_text("keep", encoding="utf-8")
                elif label == "linked-sources":
                    base.mkdir()
                    (base / "sources").symlink_to(target)
                else:
                    destination = target if label == "link" else target / "x"
                    base.symlink_to(destination, target_is_directory=True)
                before = self.snapshot(self.root)
                project = kilin.prepare_sources_workspace(
                    [self.source], "code", workspace_root=parent
                )
                self.assertEqual(project.root, parent / (base.name + "-2"))
                self.assertTrue((project.source / self.source.name).is_file())
                after = self.snapshot(self.root)
                for key, value in before.items():
                    if key not in {".", label}:
                        self.assertEqual(after[key], value)

    def test_mirror_source_cannot_alias_workspace_state(self):
        """Never replace the creation marker or core folders with a mirror."""
        for name in (
            "indexes", "logs", "outputs", "templates", "profile.ini",
            "sources.json", "source_snapshot.json", ".kilin-creating",
        ):
            with self.subTest(name=name):
                source = self.root / "origins" / name
                source.mkdir(parents=True)
                original = source / "data.py"
                original.write_text("value = 1\n", encoding="utf-8")
                parent = self.root / "projects"
                with self.assertRaisesRegex(ValueError, "managed workspace"):
                    kilin.prepare_project_workspace(source, "code", parent)
                self.assertFalse(parent.exists())
                self.assertEqual(original.read_text(), "value = 1\n")
        with patch("kilin.project.scan_repository") as scan:
            with self.assertRaisesRegex(ValueError, "managed workspace"):
                kilin.prepare_project_workspace(
                    Path(self.root.anchor), "code", self.root / "projects"
                )
        scan.assert_not_called()

    def test_source_can_already_contain_an_independent_project(self):
        """Copy an existing project as a source without mutating its owner."""
        original = self.create()
        before = self.snapshot(original.root)
        other = kilin.prepare_sources_workspace(
            [original.root], "code", workspace_root=self.root / "copies"
        )
        self.assertNotEqual(other.root, original.root)
        self.assertEqual(self.snapshot(original.root), before)
        self.assertTrue((other.root / "profile.ini").is_file())

    def test_explicit_workspace_links_and_nested_projects_are_rejected(self):
        """Canonical path checks prevent aliasing another project."""
        project = self.create()
        before = self.snapshot(project.root)
        alias = self.root / "alias"
        try:
            alias.symlink_to(project.root, target_is_directory=True)
        except OSError:
            self.skipTest("Symbolic links are unavailable")
        for path in (alias, project.root / "new-child"):
            with self.subTest(path=path):
                with self.assertRaises(kilin.WorkspaceOccupiedError):
                    kilin.prepare_sources_workspace([], "code", path)
        self.assertEqual(self.snapshot(project.root), before)

    def test_empty_project_import_is_explicit_and_preserves_settings(self):
        """Separate creation, read-only opening, and explicit importing."""
        project = kilin.prepare_sources_workspace(
            [], "code", self.destination, settings={"temperature": 0.7}
        )
        profile = (project.root / "profile.ini").read_bytes()
        with self.assertRaises(kilin.ProjectAlreadyExistsError):
            self.create()
        imported = kilin.add_project_sources(project.root, [self.source])
        self.assertEqual(imported.root, project.root)
        self.assertEqual((project.root / "profile.ini").read_bytes(), profile)
        self.assertTrue((project.source / self.source.name).is_file())
        self.assertEqual(imported.profile.temperature, 0.7)

    def test_import_never_creates_a_missing_project(self):
        """A deliberate update requires a saved, valid project."""
        with self.assertRaises(FileNotFoundError):
            kilin.add_project_sources(self.destination, [self.source])
        self.assertFalse(self.destination.exists())

    def test_open_is_read_only_even_when_managed_folders_are_missing(self):
        """Inspection must not recreate directories or resources."""
        project = kilin.prepare_sources_workspace([], "code", self.destination)
        for name in ("indexes", "logs", "outputs", "templates"):
            (project.root / name).rmdir()
        before = self.snapshot(project.root)
        opened = kilin.open_project(project.root)
        self.assertEqual(opened.root, project.root)
        self.assertEqual(self.snapshot(project.root), before)

    def test_failed_creation_restores_empty_destination(self):
        """Rollback keeps a pre-existing empty root but removes owned files."""
        self.destination.mkdir()
        inode = self.destination.stat().st_ino
        with patch("kilin.workspace._copy_file", side_effect=OSError("fail")):
            with self.assertRaisesRegex(OSError, "fail"):
                self.create()
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(self.destination.stat().st_ino, inode)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertTrue((self.create().root / "profile.ini").is_file())

    def test_failed_creation_preserves_unrelated_concurrent_file(self):
        """Rollback cannot recursively remove unjournaled contents."""
        def fail(root, _layout):
            (root / "unrelated.txt").write_text("preserve", encoding="utf-8")
            raise OSError("interrupted")

        with patch(
            "kilin.project.install_project_resources", side_effect=fail
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.create()
        self.assertEqual(
            (self.destination / "unrelated.txt").read_text(encoding="utf-8"),
            "preserve",
        )
        self.assertFalse((self.destination / "profile.ini").exists())
        self.assertFalse((self.destination / ".kilin-creating").exists())

    def test_profile_publication_never_replaces_unexpected_profile(self):
        """Fail closed at publication and preserve the conflicting file."""
        def competing_commit(path, text):
            path.write_text("another owner", encoding="utf-8")
            return atomic_create_text(path, text)

        with patch(
            "kilin.project.atomic_create_text", side_effect=competing_commit
        ):
            with self.assertRaises(kilin.ProjectAlreadyExistsError):
                self.create()
        self.assertEqual(
            (self.destination / "profile.ini").read_text(encoding="utf-8"),
            "another owner",
        )

    def test_atomic_publish_preserves_links_and_cleans_temporary_files(self):
        """A linked destination is never followed or replaced."""
        self.destination.mkdir()
        path = self.destination / "profile.ini"
        try:
            path.symlink_to(self.source)
        except OSError:
            self.skipTest("Symbolic links are unavailable")
        with self.assertRaises(FileExistsError):
            atomic_create_text(path, "replacement")
        self.assertTrue(path.is_symlink())
        self.assertEqual(
            self.source.read_text(encoding="utf-8"), "value = 1\n"
        )
        self.assertEqual(list(self.destination.iterdir()), [path])

    def test_claim_rejects_another_attempt_without_releasing_owner(self):
        """A losing creator cannot remove the winning creator's marker."""
        with claim_workspace(self.destination) as root:
            with self.assertRaises(kilin.WorkspaceOccupiedError):
                self.create()
            self.assertTrue((root / ".kilin-creating").is_file())
        self.assertFalse(self.destination.exists())

    def test_mirror_creation_and_synchronization_are_explicit(self):
        """Creation is independent; updating only occurs on an exact target."""
        source = self.root / "repository"
        source.mkdir()
        (source / "main.py").write_text("value = 1\n", encoding="utf-8")
        first = kilin.prepare_project_workspace(
            source, "code", self.root / "p"
        )
        before = self.snapshot(first.root)
        (source / "main.py").write_text("value = 2\n", encoding="utf-8")
        second = kilin.prepare_project_workspace(
            source, "code", self.root / "p"
        )
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(self.snapshot(first.root), before)
        result = kilin.synchronize_project_workspace(source, first.root)
        self.assertEqual(result.root, first.root)
        self.assertEqual(
            (first.source / "main.py").read_text(encoding="utf-8"),
            "value = 2\n",
        )

    def run_creators(self, *, default, existing_empty=False):
        """Start independent creators together and return their outcomes.

        Parameters
        ----------
        default: bool
            Exercise default allocation rather than an explicit destination.
        existing_empty: bool, optional
            Precreate the explicit destination. Defaults to False.

        Returns
        -------
        list[dict]
            Child-process creation results.

        """
        if existing_empty:
            self.destination.mkdir()
        script = """
import json
from pathlib import Path
import sys
import kilin
source, destination, automatic = sys.argv[1:]
print('ready', flush=True)
sys.stdin.readline()
try:
    options = ({'workspace_root': destination} if automatic == '1'
               else {'workspace_directory': destination})
    project = kilin.prepare_sources_workspace([source], 'code', **options)
    result = {'status': 'created', 'root': str(project.root)}
except kilin.WorkspaceOccupiedError:
    result = {'status': 'occupied'}
print(json.dumps(result), flush=True)
"""
        workers = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(self.source),
                 str(self.destination), "1" if default else "0"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            for _ in range(4)
        ]
        try:
            # No timing assumption: every process waits for the same command.
            for worker in workers:
                self.assertEqual(worker.stdout.readline().strip(), "ready")
            for worker in workers:
                worker.stdin.write("go\n")
                worker.stdin.flush()
            results = []
            for worker in workers:
                output, errors = worker.communicate(timeout=30)
                self.assertEqual(worker.returncode, 0, errors)
                results.append(json.loads(output))
            return results
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate()

    def test_concurrent_explicit_creation_has_exactly_one_owner(self):
        """Competing processes cannot both initialize a nonexistent root."""
        results = self.run_creators(default=False)
        self.assertEqual(sum(r["status"] == "created" for r in results), 1)
        self.assertTrue((self.destination / "profile.ini").is_file())
        self.assertFalse((self.destination / ".kilin-creating").exists())

    def test_concurrent_creation_of_existing_empty_folder_has_one_owner(self):
        """The marker also protects directories not made by this process."""
        results = self.run_creators(default=False, existing_empty=True)
        self.assertEqual(sum(r["status"] == "created" for r in results), 1)

    def test_concurrent_default_creation_allocates_distinct_destinations(self):
        """Independent default creations all succeed without sharing roots."""
        results = self.run_creators(default=True)
        self.assertTrue(all(r["status"] == "created" for r in results))
        self.assertEqual(len({r["root"] for r in results}), 4)
        for result in results:
            self.assertTrue((Path(result["root"]) / "profile.ini").is_file())


if __name__ == "__main__":
    unittest.main()
