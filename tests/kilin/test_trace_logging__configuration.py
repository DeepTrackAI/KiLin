"""Exercise compact project configuration recording and failure boundaries."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch
import warnings

from kilin import _trace_project
from kilin._trace_configuration import (
    CONFIGURATION_EVENT,
    CONFIGURATION_REFERENCE,
    configuration_bytes,
    configuration_reference,
)
from kilin.trace_logging import NullTraceLogger, TraceConfig, TraceLogger


def _configuration(root, value=1):
    """Produce a JSON-ready observation with every effective setting group."""
    return {
        "profile": {"temperature": 0.0, "thinking": False},
        "effort_levels": {"mlx": [{"name": "Low", "model": "model"}]},
        "router_models": {"mlx": "router"},
        "search_depth": [5, 8, 10, 15, 20],
        "source_cards": {"enabled": False},
        "behavior": {"project": str(root), "prompt": "KíLín"},
        "tools": {"example": {"limit": value}},
    }


def _observations(payload):
    """Read configuration observations from the actual stored hierarchy."""
    nodes = [node for session in payload["sessions"].values()
             for node in session.get("operations", {}).values()
             if node["event"]["event_type"] == CONFIGURATION_EVENT]
    return sorted(nodes, key=lambda node: node["event"]["sequence"])


def _event_metadata(value):
    """Collect event metadata from the controlled diagnostic hierarchy."""
    if isinstance(value, dict):
        if "sequence" in value and "event_type" in value:
            yield value
        for child in value.values():
            yield from _event_metadata(child)
    elif isinstance(value, list):
        for child in value:
            yield from _event_metadata(child)


class TestConfigurationRecording(unittest.TestCase):
    """Verify event identity, equivalence and publication across writers."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "logs" / "traces"
        self.directory.mkdir(parents=True)
        (self.root / "profile.ini").touch()
        self.path = self.directory / "project.json"
        self.config = _configuration(self.root)

    def _logger(self, policy="raise", name="project.json", scope="project"):
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=self.directory / name,
                        failure_policy=policy),
            default_directory=self.directory, project_directory=self.root,
            scope=scope,
        )
        self.addCleanup(logger.close)
        return logger

    def _emit(self, logger, configuration=None):
        logger.emit(CONFIGURATION_EVENT, component="gui_engine",
                    data=self.config if configuration is None
                    else configuration)

    def _nodes(self):
        return _observations(json.loads(self.path.read_text("utf-8")))

    def _record_change_and_restore(self):
        """Publish known A, A, B, B, A, A inputs across two runs."""
        first = self._logger()
        with first.span("gui.project.open", component="gui_engine"):
            self._emit(first)
        first.close()
        second = self._logger()
        with second.span("gui.project.open", component="gui_engine"):
            self._emit(second)
        changed = _configuration(self.root, 2)
        self._emit(second, changed)
        self._emit(second, changed)
        self._emit(second)
        self._emit(second)
        second.close()
        return {first.trace_id: (1,), second.trace_id: (1, 2, 2, 1, 1)}

    def test_change_and_restore_across_logger_runs(self):
        submitted = self._record_change_and_restore()
        changed = _configuration(self.root, 2)
        rows = self._nodes()
        self.assertEqual(len(rows), 6)
        for full, reference, expected in (
            (0, 1, self.config), (2, 3, changed), (4, 5, self.config),
        ):
            self.assertEqual(rows[full]["details"], expected)
            self.assertEqual(rows[reference]["details"],
                             configuration_reference(
                                 rows[full]["event"]["sequence"],
                                 configuration_bytes(expected),
                             ))
        self.assertNotEqual(rows[0]["event"]["trace_id"],
                            rows[1]["event"]["trace_id"])
        self.assertIn("span_id", rows[1]["event"])
        self._assert_ordered_observations(submitted)

    def test_json_order_and_type_sensitive_comparison(self):
        logger = self._logger()
        self._emit(logger)
        reordered = dict(reversed(list(self.config.items())))
        reordered["profile"] = {"thinking": False, "temperature": 0.0}
        self._emit(logger, reordered)
        self.assertIn(CONFIGURATION_REFERENCE, self._nodes()[-1]["details"])
        for value in (1.0, True, 1):
            changed = _configuration(self.root, value)
            self._emit(logger, changed)
            self.assertEqual(self._nodes()[-1]["details"], changed)
        reversed_list = deepcopy(self.config)
        reversed_list["search_depth"].reverse()
        self._emit(logger, reversed_list)
        self.assertEqual(self._nodes()[-1]["details"], reversed_list)

    def test_all_setting_groups_and_historical_path_are_significant(self):
        logger = self._logger()
        for key in self.config:
            with self.subTest(key=key):
                self._emit(logger)
                changed = deepcopy(self.config)
                if key == "behavior":
                    changed[key]["project"] += "-renamed"
                elif isinstance(changed[key], list):
                    changed[key].append(99)
                else:
                    changed[key]["extra"] = "changed"
                self._emit(logger, changed)
                self.assertEqual(self._nodes()[-1]["details"], changed)

    def test_separate_workspaces_and_conversation_traces(self):
        logger = self._logger()
        self._emit(logger)
        conversation = self._logger(name="conversation_a.json",
                                    scope="conversation")
        self._emit(conversation)
        self._emit(conversation)
        conversation.close()
        stored = _observations(json.loads(
            conversation.readable_path.read_text("utf-8")
        ))
        self.assertEqual([node["details"] for node in stored],
                         [self.config, self.config])
        other = self.root / "other"
        directory = other / "logs" / "traces"
        directory.mkdir(parents=True)
        (other / "profile.ini").touch()
        with_logger = TraceLogger(
            TraceConfig(enabled=True, output_path=directory / "project.json"),
            default_directory=directory, project_directory=other,
            scope="project",
        )
        try:
            self._emit(with_logger)
            stored = _observations(json.loads(
                with_logger.readable_path.read_text("utf-8")
            ))
            self.assertEqual(stored[0]["details"], self.config)
        finally:
            with_logger.close()

    def test_one_project_read_and_write_per_observation(self):
        logger = self._logger()
        self._emit(logger)
        with (
            patch.object(_trace_project, "_read_object",
                         wraps=_trace_project._read_object) as read,
            patch.object(_trace_project, "atomic_write_text",
                         wraps=_trace_project.atomic_write_text) as write,
        ):
            self._emit(logger)
        self.assertEqual(sum(call.args[0] == self.path
                             for call in read.call_args_list), 1)
        self.assertEqual(sum(call.args[0] == self.path
                             for call in write.call_args_list), 1)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(write.call_count, 2)

    def _record_threaded(self):
        """Publish known inputs concurrently through two independent sinks."""
        first, second = self._logger(), self._logger()
        barrier = Barrier(2)

        def run(logger, values):
            for value in values:
                barrier.wait(timeout=10)
                self._emit(logger, _configuration(self.root, value))

        expected = {first.trace_id: (1, 2, 1, 2, 1),
                    second.trace_id: (1, 1, 2, 1, 2)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(run, logger, expected[logger.trace_id])
                     for logger in (first, second)]
            for task in tasks:
                task.result(timeout=30)
        first.close()
        second.close()
        return expected

    def test_threaded_publication(self):
        self._assert_ordered_observations(self._record_threaded())

    def _assert_recording_identity(self):
        """Check the actual persisted metadata without a replay reader."""
        payload = json.loads(self.path.read_text("utf-8"))
        counter = json.loads((self.directory / ".sequence.json")
                             .read_text("utf-8"))
        self.assertEqual(payload["trace_scope"], "project")
        self.assertEqual(payload["sequence_scope"], "project")
        self.assertEqual(payload["trace_format_version"],
                         counter["format_version"])
        self.assertIs(payload["trace_incomplete"], False)
        self.assertIs(counter["incomplete"], False)
        events = sorted(_event_metadata(payload["sessions"]),
                        key=lambda event: event["sequence"])
        self.assertEqual([event["sequence"] for event in events],
                         list(range(1, counter["last_sequence"] + 1)))
        ids = [event["event_id"] for event in events]
        self.assertTrue(all(isinstance(value, str) and value for value in ids))
        self.assertEqual(len(set(ids)), len(events))

    def _assert_ordered_observations(self, expected):
        """Check raw full/reference decisions against each writer's inputs."""
        rows = self._nodes()
        self.assertEqual(len(rows), sum(map(len, expected.values())))
        positions = dict.fromkeys(expected, 0)
        resolved = {}
        latest = None
        for row in rows:
            event = row["event"]
            trace_id = event["trace_id"]
            self.assertIn(trace_id, expected)
            position = positions[trace_id]
            self.assertLess(position, len(expected[trace_id]))
            configuration = _configuration(self.root,
                                           expected[trace_id][position])
            positions[trace_id] += 1
            sequence = event["sequence"]
            encoded = configuration_bytes(configuration)
            if latest is not None and latest[1] == encoded:
                self.assertEqual(row["details"], configuration_reference(
                    latest[0], encoded
                ))
            else:
                self.assertEqual(row["details"], configuration)
                latest = (sequence, encoded)
            resolved[sequence] = configuration
        self.assertEqual(positions, {key: len(values)
                                     for key, values in expected.items()})
        self._assert_recording_identity()
        return resolved

    def _record_processes(self):
        """Publish known inputs from separate Python processes."""
        code = """
import json, sys
from pathlib import Path
from kilin.trace_logging import TraceConfig, TraceLogger
root = Path(sys.argv[1])
path = root / 'logs' / 'traces' / 'project.json'
logger = TraceLogger(TraceConfig(enabled=True, output_path=path,
                                failure_policy='raise'),
                     default_directory=path.parent, project_directory=root,
                     scope='project')
try:
    data = json.loads(sys.argv[2])
    for value in json.loads(sys.argv[3]):
        data['tools']['example']['limit'] = value
        logger.emit('gui.project.configuration.effective',
                    component='gui_engine', data=data)
finally:
    logger.close()
print(logger.trace_id)
"""
        processes = []
        expected = {}
        try:
            for values in ([1, 2, 2, 1], [1, 1, 2, 1], [2, 1, 1, 2]):
                process = subprocess.Popen(
                    [sys.executable, "-c", code, str(self.root),
                     json.dumps(self.config), json.dumps(values)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append((process, values))
            for process, values in processes:
                stdout, stderr = process.communicate(timeout=45)
                self.assertEqual(process.returncode, 0, stderr)
                trace_id = stdout.strip()
                self.assertTrue(trace_id)
                self.assertNotIn(trace_id, expected)
                expected[trace_id] = values
        finally:
            for process, _ in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        return expected

    def test_independent_process_publication(self):
        self._assert_ordered_observations(self._record_processes())

    def test_failed_publication_preserves_snapshot_and_recording_health(self):
        for policy in ("warn", "raise"):
            with self.subTest(policy=policy):
                logger = self._logger(policy)
                self._emit(logger)
                before = self.path.read_bytes()
                counter = self.directory / ".sequence.json"
                high = json.loads(counter.read_text("utf-8"))["last_sequence"]
                original = _trace_project.atomic_write_text

                def fail(path, content):
                    if path == self.path:
                        raise OSError("simulated publication failure")
                    original(path, content)

                with patch.object(_trace_project, "atomic_write_text",
                                  side_effect=fail):
                    if policy == "raise":
                        with self.assertRaisesRegex(RuntimeError,
                                                    "publication failure"):
                            self._emit(logger)
                    else:
                        with warnings.catch_warnings(record=True) as seen:
                            warnings.simplefilter("always", RuntimeWarning)
                            self._emit(logger)
                        self.assertTrue(any("publication failure" in
                                            str(w.message) for w in seen))
                self.assertEqual(self.path.read_bytes(), before)
                state = json.loads(counter.read_text("utf-8"))
                self.assertTrue(state["incomplete"])
                self.assertEqual(state["last_sequence"], high + 1)
                logger.close()
                # A new writer can acquire the lease after the failure.
                next_logger = self._logger()
                self._emit(next_logger)
                next_logger.close()
                self.assertIn(CONFIGURATION_REFERENCE,
                              self._nodes()[-1]["details"])

    def test_disabled_trace_performs_no_configuration_io(self):
        with (
            patch.object(_trace_project, "_read_object") as read,
            patch.object(_trace_project, "atomic_write_text") as write,
        ):
            self._emit(NullTraceLogger())
        read.assert_not_called()
        write.assert_not_called()
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
