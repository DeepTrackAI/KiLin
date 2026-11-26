"""Test project-wide trace numbering and shared trace publication."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin._trace_project import (
    TRACE_FORMAT_VERSION,
    ProjectTraceSequence,
    project_trace_directory,
    write_project_trace,
)
from kilin.conversation_ownership import ConversationFileLock


class TestProjectTrace(unittest.TestCase):
    """Exercise allocation, concurrent publication and failure boundaries."""

    def setUp(self):
        """Create an isolated managed project directory."""

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / "logs" / "traces"
        self.directory.mkdir(parents=True)
        self.counter = ProjectTraceSequence(self.directory)

    def test_reopening_and_separate_projects(self):
        """Keep one sequence per project, including across instances."""

        self.assertEqual(self.counter.next(), 1)
        self.assertEqual(ProjectTraceSequence(self.directory).next(), 2)
        other = self.root / "other" / "logs" / "traces"
        other.mkdir(parents=True)
        self.assertEqual(ProjectTraceSequence(other).next(), 1)
        self.assertEqual(self.counter.next(), 3)

    def test_threaded_allocation(self):
        """Serialize short leases across competing threads."""

        with ThreadPoolExecutor(max_workers=4) as executor:
            numbers = list(executor.map(
                lambda _: ProjectTraceSequence(self.directory).next(),
                range(32),
            ))
        self.assertEqual(sorted(numbers), list(range(1, 33)))

    def test_process_allocation(self):
        """Allocate unique numbers across independent Python processes."""

        code = (
            "from pathlib import Path; import sys, json; "
            "from kilin._trace_project import ProjectTraceSequence; "
            "s = ProjectTraceSequence(Path(sys.argv[1])); "
            "print(json.dumps([s.next() for _ in range(8)]))"
        )
        processes = []
        try:
            for _ in range(3):
                processes.append(subprocess.Popen(
                    [sys.executable, "-c", code, str(self.directory)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ))
            numbers = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=45)
                self.assertEqual(process.returncode, 0, stderr)
                numbers.extend(json.loads(stdout))
            self.assertEqual(sorted(numbers), list(range(1, 25)))
            self.assertEqual(self.counter.next(), 25)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_corrupted_counter_is_not_repaired(self):
        """Do not recycle numbers after a malformed counter is found."""

        self.counter.path.write_text('{"last_sequence": -1}', 'utf-8')
        before = self.counter.path.read_bytes()
        with self.assertRaises(OSError):
            self.counter.next()
        self.assertEqual(self.counter.path.read_bytes(), before)

    def test_missing_counter_with_archived_trace(self):
        """Archived events prevent sequence reuse after counter loss."""

        archive = self.directory / "archived"
        archive.mkdir()
        trace = archive / "conversation_old.json"
        trace.write_text(
            json.dumps({"sequence_scope": "project"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(OSError, "counter is missing"):
            self.counter.next()
        self.assertFalse(self.counter.path.exists())

    def test_missing_counter_recognizes_custom_trace_names(self):
        """Do not restart numbering for any supported managed JSON name."""

        names = (
            "conversation.json", "custom.json", "conversation_old.JSON",
            "archived/custom.JSON", "../exports/custom.json",
        )
        for number, name in enumerate(names):
            with self.subTest(name=name):
                directory = self.root / str(number) / "logs" / "traces"
                directory.mkdir(parents=True)
                counter = ProjectTraceSequence(directory)
                trace = directory / name
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write_text(
                    json.dumps({"sequence_scope": "project"}),
                    encoding="utf-8",
                )
                before = trace.read_bytes()
                for operation in (counter.next, counter.mark_incomplete):
                    with self.assertRaisesRegex(OSError, "counter is missing"):
                        operation()
                    self.assertFalse(counter.path.exists())
                self.assertEqual(trace.read_bytes(), before)

    def test_unrelated_json_does_not_block_initial_numbering(self):
        """Ignore unrelated regular logs without requiring object JSON."""

        contents = {
            "application.json": b'{"status": "ready"}',
            "items.JSON": b"[]",
            "unfinished.json": b"{",
            "binary.json": b"\xff",
        }
        for name, content in contents.items():
            (self.directory / name).write_bytes(content)
        (self.directory / "directory.json").mkdir()
        self.assertEqual(self.counter.next(), 1)
        for name, content in contents.items():
            self.assertEqual((self.directory / name).read_bytes(), content)

    def test_invalid_canonical_trace_blocks_initial_numbering(self):
        """Keep malformed recognized traces from certifying fresh history."""

        for name in ("project.json", "conversation_old.json"):
            trace = self.directory / name
            for content in (b"[]", b"{", b"\xff"):
                with self.subTest(name=name, content=content):
                    trace.write_bytes(content)
                    for operation in (
                        self.counter.next, self.counter.mark_incomplete,
                    ):
                        with self.assertRaises(OSError):
                            operation()
                        self.assertFalse(self.counter.path.exists())
                    self.assertEqual(trace.read_bytes(), content)
            trace.unlink()

    def test_unreadable_custom_json_blocks_initial_numbering(self):
        """Inaccessible potential history must not permit number reuse."""

        trace = self.directory / "custom.json"
        trace.write_text("{}", encoding="utf-8")
        with patch.object(
            Path, "read_text", side_effect=PermissionError("unreadable")
        ):
            for operation in (
                self.counter.next, self.counter.mark_incomplete,
            ):
                with self.assertRaises(PermissionError):
                    operation()
                self.assertFalse(self.counter.path.exists())
        self.assertEqual(trace.read_text(encoding="utf-8"), "{}")

    def test_linked_custom_json_blocks_initial_numbering(self):
        """Do not follow or ignore a linked potential trace on recovery."""

        target = self.root / "outside.json"
        target.write_text(
            json.dumps({"sequence_scope": "project"}), encoding="utf-8"
        )
        link = self.directory / "custom.JSON"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic links are unavailable.")
        before = target.read_bytes()
        for operation in (self.counter.next, self.counter.mark_incomplete):
            with self.assertRaisesRegex(OSError, "Linked trace file"):
                operation()
            self.assertFalse(self.counter.path.exists())
        self.assertEqual(target.read_bytes(), before)
        self.assertTrue(link.is_symlink())

    def test_unreadable_history_directories_block_initial_numbering(self):
        """Do not mistake failed traversal for the absence of prior traces."""

        real_scandir = os.scandir
        for number, denied_name in enumerate(("", "traces/archived")):
            with self.subTest(denied_name=denied_name):
                logs = self.root / str(number) / "logs"
                directory = logs / "traces"
                archive = directory / "archived"
                archive.mkdir(parents=True)
                trace = archive / "conversation_old.json"
                trace.write_text(
                    json.dumps({"sequence_scope": "project"}),
                    encoding="utf-8",
                )
                before = trace.read_bytes()
                counter = ProjectTraceSequence(directory)
                denied = logs / denied_name

                def scandir(path):
                    """Fail only the selected history-directory traversal."""

                    if Path(path) == denied:
                        raise PermissionError("History is unreadable.")
                    return real_scandir(path)

                with patch.object(os, "scandir", side_effect=scandir):
                    for operation in (counter.next, counter.mark_incomplete):
                        with self.assertRaises(PermissionError):
                            operation()
                        self.assertFalse(counter.path.exists())
                self.assertEqual(trace.read_bytes(), before)

    def test_linked_history_directory_blocks_initial_numbering(self):
        """Reject a linked archive without following its target directory."""

        target = self.root / "outside"
        target.mkdir()
        trace = target / "conversation_old.json"
        trace.write_text(
            json.dumps({"sequence_scope": "project"}), encoding="utf-8"
        )
        link = self.directory / "archived"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic links are unavailable.")
        before = trace.read_bytes()
        for operation in (self.counter.next, self.counter.mark_incomplete):
            with self.assertRaisesRegex(OSError, "Linked trace directory"):
                operation()
            self.assertFalse(self.counter.path.exists())
        self.assertEqual(trace.read_bytes(), before)
        self.assertTrue(link.is_symlink())

    def test_first_allocation_creates_missing_logs_tree(self):
        """Allow a fresh workspace whose logs have not yet been created."""

        logs = self.root / "new" / "logs"
        counter = ProjectTraceSequence(logs / "traces")
        self.assertFalse(logs.exists())
        self.assertEqual(counter.next(), 1)
        self.assertTrue(counter.path.is_file())

    def test_existing_counter_does_not_scan_history(self):
        """Established allocation and health updates need no history scan."""

        self.assertEqual(self.counter.next(), 1)
        with patch.object(
            os, "scandir", side_effect=AssertionError("Unexpected traversal")
        ):
            self.assertEqual(self.counter.next(), 2)
            self.counter.mark_incomplete()
        state = json.loads(self.counter.path.read_text(encoding="utf-8"))
        self.assertEqual(state["last_sequence"], 2)
        self.assertTrue(state["incomplete"])

    def test_recording_health_is_sticky(self):
        """A later successful allocation cannot clear a recording failure."""

        self.assertEqual(self.counter.next(), 1)
        self.counter.mark_incomplete()
        self.assertEqual(ProjectTraceSequence(self.directory).next(), 2)
        state = json.loads(self.counter.path.read_text(encoding="utf-8"))
        self.assertTrue(state["incomplete"])
        self.assertEqual(state["last_sequence"], 2)

    def test_independent_conversation_ownership(self):
        """An open conversation must not block project numbering."""

        lock = ConversationFileLock(self.directory / "conversation_a.json")
        lock.acquire()
        try:
            self.assertEqual(self.counter.next(), 1)
        finally:
            lock.release()

    def test_project_publication_preserves_other_runs(self):
        """Replace only a writer's own sessions during publication."""

        path = self.directory / "project.json"
        metadata = {
            "trace_format_version": TRACE_FORMAT_VERSION,
            "trace_scope": "project",
            "sequence_scope": "project",
        }
        write_project_trace(path, "a", {"01_setup": [1]}, metadata)
        write_project_trace(path, "b", {"01_setup": [2]}, metadata)
        write_project_trace(path, "a", {"01_setup": [1, 3]}, metadata)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["sessions"], {
            "01_setup_a": [1, 3], "01_setup_b": [2],
        })
        self.assertNotIn("conversation", payload)

    def test_invalid_project_trace_is_not_overwritten(self):
        """Do not replace malformed shared diagnostics."""

        path = self.directory / "project.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaises(OSError):
            write_project_trace(path, "a", {}, {})
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")

    def test_path_resolution(self):
        """Recognize a managed workspace and reject escaping paths."""

        path = self.directory / "conversation_a.json"
        self.assertIsNone(project_trace_directory(path))
        (self.root / "profile.ini").touch()
        self.assertEqual(project_trace_directory(path), self.directory)
        self.assertEqual(
            project_trace_directory(path, self.root), self.directory
        )
        with self.assertRaises(ValueError):
            project_trace_directory(self.root / "outside.json", self.root)


if __name__ == "__main__":
    unittest.main()
