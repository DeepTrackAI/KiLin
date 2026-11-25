"""Test project-wide trace numbering and shared trace publication."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

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
