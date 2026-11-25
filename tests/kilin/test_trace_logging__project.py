"""Exercise project-numbered diagnostics through the existing trace logger."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import trace_logging
from kilin.trace_logging import (
    TRACE_EVENT_ANSWER_ACTIVITY,
    TRACE_EVENT_PROGRESS_UPDATED,
    TraceConfig,
    TraceLogger,
)


def _event_numbers(value):
    """Extract numbered event records from a hierarchical trace."""

    if isinstance(value, dict):
        if "sequence" in value and "event_type" in value:
            yield value["sequence"]
        for child in value.values():
            yield from _event_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _event_numbers(child)


class TestProjectTraceLogger(unittest.TestCase):
    """Verify numbering and ownership of actual diagnostic events."""

    def setUp(self):
        """Prepare a project directory without creating a conversation."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "logs" / "traces"
        self.directory.mkdir(parents=True)
        (self.root / "profile.ini").touch()

    def _logger(self, name, scope="conversation"):
        """Create a trace using the ordinary trace configuration."""

        logger = TraceLogger(
            TraceConfig(
                enabled=True,
                output_path=self.directory / name,
                failure_policy="raise",
            ),
            default_directory=self.directory,
            project_directory=self.root,
            scope=scope,
        )
        self.addCleanup(logger.close)
        return logger

    def _numbers(self, name):
        """Read event numbers from the saved hierarchical sessions."""

        payload = json.loads(
            (self.directory / name).read_text(encoding="utf-8")
        )
        return list(_event_numbers(payload["sessions"]))

    def test_interleaved_project_and_conversation_traces(self):
        """Do not duplicate events or restart the sequence on reopening."""

        project = self._logger("project.json", scope="project")
        first = self._logger("conversation_a.json")
        other = self._logger("project.json", scope="project")
        second = self._logger("conversation_b.json")
        for logger in (project, first, other, second):
            logger.emit("test.sample", component="tests")
            logger.close()
        previous = self._numbers("project.json")
        reopened = self._logger("conversation_a.json")
        reopened.emit("test.reopened", component="tests")
        reopened.close()
        numbers = previous + self._numbers("conversation_a.json")
        numbers += self._numbers("conversation_b.json")
        self.assertEqual(sorted(numbers), list(range(1, len(numbers) + 1)))
        self.assertEqual(len(previous), 6)
        payload = json.loads(
            (self.directory / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["trace_scope"], "project")
        self.assertNotIn("conversation", payload)

    def test_span_start_and_completion_have_separate_numbers(self):
        """Keep both lifecycle events, plus independent nested events."""

        logger = self._logger("conversation_a.json")
        with logger.span("test.operation", component="tests"):
            logger.emit("test.nested.completed", component="tests")
        logger.close()
        numbers = self._numbers("conversation_a.json")
        self.assertEqual(sorted(numbers), list(range(1, 6)))

    def test_flush_does_not_rewrite_published_trace(self):
        """Preserve committed bytes for both trace ownership scopes."""

        for scope, name in (
            ("conversation", "conversation_a.json"),
            ("project", "project.json"),
        ):
            with self.subTest(scope=scope):
                logger = self._logger(name, scope=scope)
                logger.emit("test.ready", component="tests")
                before = logger.readable_path.read_bytes()
                with patch.object(
                    logger, "_write_readable_trace",
                    wraps=logger._write_readable_trace,
                ) as write:
                    logger.flush()
                    write.assert_not_called()
                self.assertEqual(logger.readable_path.read_bytes(), before)

    def test_flush_publishes_buffered_events_only_once(self):
        """Retain throttled diagnostics without a later redundant write."""

        for scope, name in (
            ("conversation", "conversation_b.json"),
            ("project", "project.json"),
        ):
            with self.subTest(scope=scope):
                logger = self._logger(name, scope=scope)
                before = self._numbers(name)
                with patch.object(
                    trace_logging, "perf_counter",
                    return_value=logger._last_readable_write,
                ):
                    logger.emit(
                        TRACE_EVENT_PROGRESS_UPDATED, component="tests",
                        data={"stage": "embedding", "percent": 25.0},
                    )
                    logger.emit(
                        TRACE_EVENT_ANSWER_ACTIVITY, component="tests",
                        data={"label": "Starting"},
                    )
                self.assertEqual(self._numbers(name), before)
                logger.flush()
                self.assertEqual(len(self._numbers(name)), len(before) + 2)
                after = logger.readable_path.read_bytes()
                with patch.object(
                    logger, "_write_readable_trace",
                    wraps=logger._write_readable_trace,
                ) as write:
                    logger.flush()
                    write.assert_not_called()
                self.assertEqual(logger.readable_path.read_bytes(), after)

    def test_checkpoint_publishes_pending_events(self):
        """A successful checkpoint leaves no additional flush work."""

        logger = self._logger("conversation_c.json")
        with patch.object(
            trace_logging, "perf_counter",
            return_value=logger._last_readable_write,
        ):
            logger.emit(
                TRACE_EVENT_PROGRESS_UPDATED, component="tests",
                data={"stage": "embedding", "percent": 50.0},
            )
        logger.update_conversation({"name": "Saved"})
        self.assertEqual(len(self._numbers("conversation_c.json")), 2)
        before = logger.readable_path.read_bytes()
        with patch.object(
            logger, "_write_readable_trace",
            wraps=logger._write_readable_trace,
        ) as write:
            logger.flush()
            write.assert_not_called()
        self.assertEqual(logger.readable_path.read_bytes(), before)

    def test_archive_does_not_duplicate_project_numbers(self):
        """Keep original event ownership when resetting a trace."""

        name = "conversation_a.json"
        logger = self._logger(name)
        logger.emit("test.before", component="tests")
        logger.flush()
        archive = "conversation_archived.json"
        (self.directory / name).rename(self.directory / archive)
        logger.reset_conversation_state(archived=True)
        logger.emit("test.after", component="tests")
        logger.close()
        numbers = self._numbers(archive) + self._numbers(name)
        self.assertEqual(sorted(numbers), list(range(1, len(numbers) + 1)))


if __name__ == "__main__":
    unittest.main()
