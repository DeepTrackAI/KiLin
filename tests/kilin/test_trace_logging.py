"""Unit tests for structured traces and persisted conversation records.

Tests use temporary local files and deterministic event inputs. They verify
observable JSON output, full diagnostic tracing, and I/O failures
without loading models, accessing the network, or using a GUI.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import warnings

import kilin
from kilin.conversation_format import CONVERSATION_SESSION_KEY
from kilin import trace_logging
from kilin.trace_logging import (
    FAILURE_POLICY_RAISE,
    FAILURE_POLICY_WARN,
    TRACE_EVENT_EMBEDDING_TOKENS_PREPARED,
    TRACE_EVENT_LLM_CALL_INPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
    TRACE_EVENT_PROGRESS_UPDATED,
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
    TraceTextStream,
    create_trace_logger,
    trace_text_preview,
)


def _payload(path: Path) -> dict[str, object]:
    """Load a trace file and assert its top-level JSON object."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise AssertionError("The saved trace must be a JSON object.")
    return result


def _operations(
    payload: dict[str, object], category: str
) -> dict[str, object]:
    """Return the named operations from one trace session."""
    for session in payload["sessions"].values():
        if session["category"] == category:
            return session.get("operations", {})
    return {}


def _turns(payload: dict[str, object]) -> dict[str, object]:
    """Return serialized conversation turns without changing the record."""
    return payload["sessions"][CONVERSATION_SESSION_KEY]["turns"]


class TestTraceConfig(unittest.TestCase):
    """Keep immutable tracing configuration and validation explicit."""

    def test_default_flags_and_immutable_configuration(self) -> None:
        """Keep tracing opt-in at the low-level configuration boundary."""
        config = TraceConfig()
        self.assertFalse(config.enabled)
        self.assertIsNone(config.output_path)
        self.assertFalse(config.include_embedding_vectors)
        self.assertEqual(config.failure_policy, FAILURE_POLICY_WARN)
        self.assertFalse(hasattr(config, "mode"))
        with self.assertRaises(FrozenInstanceError):
            config.enabled = True

    def test_packaged_defaults_do_not_select_trace_modes(self) -> None:
        """Keep tracing detail out of package-wide settings."""
        config_path = Path(trace_logging.__file__).with_name(
            "kilin_defaults.json"
        )
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertNotIn("TRACE", settings)
        self.assertNotIn("TraceMode", trace_logging.__all__)
        self.assertNotIn("TRACE_MODE_STANDARD", trace_logging.__all__)
        self.assertNotIn("TRACE_MODE_FORENSIC", trace_logging.__all__)
        for name in (
            "TraceMode",
            "TRACE_MODE_STANDARD",
            "TRACE_MODE_FORENSIC",
            "TRACE_MODE_CHOICES",
            "DEFAULT_TRACE_MODE",
        ):
            self.assertFalse(hasattr(kilin, name))
        for mode in ("standard", "forensic"):
            with self.subTest(mode=mode):
                with self.assertRaises(TypeError):
                    TraceConfig(enabled=True, mode=mode)

    def test_accepts_supported_policies_and_paths(self) -> None:
        """Allow independent write policies and optional vector recording."""
        for policy in (FAILURE_POLICY_WARN, FAILURE_POLICY_RAISE):
            with self.subTest(policy=policy):
                config = TraceConfig(
                    enabled=True,
                    output_path=Path("trace.json"),
                    include_embedding_vectors=True,
                    failure_policy=policy,
                )
                self.assertEqual(config.failure_policy, policy)
                self.assertTrue(config.include_embedding_vectors)
        self.assertEqual(
            TraceConfig(output_path="trace.json").output_path,
            "trace.json",
        )

    def test_rejects_invalid_configuration_before_writing(self) -> None:
        """Reject unsupported values and integer substitutes for booleans."""
        invalid = (
            {"enabled": 1},
            {"enabled": "yes"},
            {"include_embedding_vectors": 1},
            {"failure_policy": "ignore"},
            {"output_path": 5},
        )
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    TraceConfig(**settings)


    def test_concrete_logger_requires_enabled_configuration(self) -> None:
        """Reject disabled configuration before opening any trace file."""
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "enabled"):
                TraceLogger(
                    TraceConfig(enabled=False),
                    default_directory=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


class TestTraceLoggerEvents(unittest.TestCase):
    """Verify structured events against their actual on-disk projection."""

    def setUp(self) -> None:
        """Allocate an isolated location for every logging test."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "nested" / "conversation.json"

    def _logger(self, **settings: object) -> TraceLogger:
        """Create a trace at the test's exact conversation path."""
        config = TraceConfig(
            enabled=True, output_path=self.path, **settings
        )
        logger = TraceLogger(config, default_directory=self.root)
        self.addCleanup(logger.close)
        return logger

    def test_startup_and_close_have_one_ordered_lifecycle(self) -> None:
        """Persist startup metadata, events, and exactly one close event."""
        logger = self._logger()
        self.assertEqual(logger.output_path, self.path)
        self.assertEqual(logger.readable_path, self.path)
        self.assertTrue(self.path.is_file())
        logger.emit("custom.ready", component=" custom ", data={"ok": 1})
        logger.close()
        logger.close()
        logger.emit("ignored.later", component="test")
        payload = _payload(self.path)
        self.assertEqual(payload["event_count"], 3)
        trace_events = _operations(payload, "trace")
        self.assertEqual(len(trace_events), 2)
        self.assertEqual(
            [key.split("_", 1)[1] for key in trace_events],
            ["trace_started", "trace_closed"],
        )
        runtime = _operations(payload, "runtime")
        self.assertEqual(runtime["01_custom_ready"]["details"], {"ok": 1})
        self.assertEqual(
            [item["event"]["sequence"] for item in trace_events.values()],
            [1, 3],
        )
        self.assertEqual(runtime["01_custom_ready"]["event"]["sequence"], 2)
        self.assertEqual(
            trace_events["01_trace_started"]["details"]["mode"],
            "forensic",
        )

    def test_event_validation_precedes_mutation(self) -> None:
        """Reject blank names without consuming sequence numbers."""
        logger = self._logger()
        for event_type, component in (
            ("", "test"), ("   ", "test"), (12, "test"),
            ("custom.event", ""), ("custom.event", 5),
        ):
            with self.subTest(event_type=event_type, component=component):
                with self.assertRaises(ValueError):
                    logger.emit(event_type, component=component)
        logger.emit(" custom.valid ", component=" test ")
        payload = _payload(self.path)
        self.assertEqual(payload["event_count"], 2)
        self.assertIn("01_custom_valid", _operations(payload, "runtime"))

    def test_enabled_trace_records_all_diagnostic_event_types(self) -> None:
        """Retain model and token events without mode-based filtering."""
        logger = self._logger()
        for event_type in (
            TRACE_EVENT_EMBEDDING_TOKENS_PREPARED,
            TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
        ):
            logger.emit(event_type, component="model", data={"ids": [1]})
        logger.emit(
            TRACE_EVENT_LLM_CALL_INPUT,
            component="chat",
            data={"input_count": 2},
        )
        logger.emit(
            TRACE_EVENT_LLM_CALL_OUTPUT,
            component="chat",
            data={"output_count": 1},
        )
        payload = _payload(self.path)
        self.assertEqual(payload["event_count"], 5)
        runtime = _operations(payload, "runtime")
        self.assertEqual(len(runtime), 3)
        embedding = _operations(payload, "indexing")
        self.assertIn("01_embedding_tokens_prepared", embedding)
        self.assertIn("01_llm_call_output_tokens", runtime)
        self.assertIn("02_llm_call_input", runtime)
        self.assertIn("03_llm_call_output", runtime)
        self.assertTrue(logger.include_content)
        self.assertTrue(logger.include_token_ids)

    def test_full_content_is_independent_of_embedding_vectors(
        self,
    ) -> None:
        """Keep full diagnostics enabled while vectors remain opt-in."""
        logger = self._logger()
        self.assertTrue(logger.include_content)
        self.assertTrue(logger.include_token_ids)
        self.assertFalse(logger.include_embedding_vectors)
        with TemporaryDirectory() as directory:
            separate = TraceLogger(
                TraceConfig(
                    enabled=True,
                    include_embedding_vectors=True,
                ),
                default_directory=directory,
            )
            self.assertTrue(separate.include_embedding_vectors)
            self.assertTrue(separate.include_content)
            self.assertTrue(separate.include_token_ids)
            separate.close()

    def test_settings_record_only_changed_effective_values(self) -> None:
        """Include added and removed keys, not unchanged settings."""
        logger = self._logger()
        self.assertFalse(logger.settings_changed(
            "settings.changed", component="rag",
            before={"a": 1}, after={"a": 1},
        ))
        self.assertTrue(logger.settings_changed(
            "settings.changed", component="rag",
            before={"a": 1, "removed": True},
            after={"a": 2, "added": "yes"},
            data={"scope": "turn"},
        ))
        payload = _payload(self.path)
        changes = _operations(payload, "runtime")[
            "01_settings_changed"
        ]["details"]
        self.assertEqual(changes["scope"], "turn")
        self.assertEqual(changes["changes"], {
            "a": {"before": 1, "after": 2},
            "added": {"before": None, "after": "yes"},
            "removed": {"before": True, "after": None},
        })

    def test_progress_is_grouped_separately_from_execution(self) -> None:
        """Save progress snapshots under their source or indexing session."""
        logger = self._logger()
        logger.emit(
            TRACE_EVENT_PROGRESS_UPDATED,
            component="rag",
            data={"stage": "extracting", "percent": 25.0},
        )
        logger.emit("custom.done", component="rag")
        payload = _payload(self.path)
        self.assertEqual(payload["progress_checkpoint_count"], 1)
        self.assertEqual(payload["execution_event_count"], 2)
        source = payload["sessions"]["03_sources"]
        entry = source["progress_checkpoints"]["01_25_percent"]
        self.assertEqual(entry["stage"], "extracting")
        self.assertEqual(entry["percent"], 25.0)
        self.assertNotIn(
            "01_progress_updated", _operations(payload, "runtime")
        )

    def test_progress_write_throttling_flushes_at_next_event(self) -> None:
        """Keep a throttled checkpoint until the next persisted operation."""
        logger = self._logger()
        with patch.object(logger, "_write_readable_trace") as write:
            with patch.object(
                trace_logging, "perf_counter", return_value=0.1
            ):
                logger.emit(
                    TRACE_EVENT_PROGRESS_UPDATED,
                    component="rag",
                    data={"stage": "embedding", "percent": 50.0},
                )
            write.assert_not_called()
        logger.emit("index.done", component="rag")
        payload = _payload(self.path)
        self.assertEqual(payload["progress_checkpoint_count"], 1)
        self.assertEqual(
            payload["sessions"]["04_indexing"]["progress_checkpoints"][
                "01_50_percent"
            ]["percent"],
            50.0,
        )


    def test_concurrent_emits_keep_unique_ordered_sequences(self) -> None:
        """Serialize parallel writers inside a single logger instance."""
        logger = self._logger()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(
                lambda number: logger.emit(
                    "parallel.event", component="test", data={"id": number}
                ),
                range(16),
            ))
        operations = _operations(_payload(self.path), "runtime")
        self.assertEqual(len(operations), 16)
        self.assertEqual(
            [item["event"]["sequence"] for item in operations.values()],
            list(range(2, 18)),
        )
        self.assertEqual(
            {item["details"]["id"] for item in operations.values()},
            set(range(16)),
        )

    def test_close_flushes_throttled_progress(self) -> None:
        """Flush an otherwise deferred progress record at logger close."""
        logger = self._logger()
        with patch.object(trace_logging, "perf_counter", return_value=0.1):
            logger.emit(
                TRACE_EVENT_PROGRESS_UPDATED,
                component="rag",
                data={"stage": "indexing", "percent": 75.0},
            )
        logger.close()
        payload = _payload(self.path)
        self.assertEqual(payload["progress_checkpoint_count"], 1)
        self.assertEqual(
            payload["sessions"]["04_indexing"]["progress_checkpoints"][
                "01_75_percent"
            ]["percent"],
            75.0,
        )


    def test_event_categories_and_turn_stages_are_distinct(self) -> None:
        """Route source/index/project events independently of turn stages."""
        logger = self._logger()
        for name in ("source.read", "index.saved", "project.created"):
            logger.emit(name, component="test")
        for name in (
            "embedding.query", TRACE_EVENT_LLM_CALL_INPUT,
            TRACE_EVENT_LLM_CALL_OUTPUT,
        ):
            logger.emit(name, component="test", turn_id="turn-a")
        logger.emit(
            TRACE_EVENT_PROGRESS_UPDATED,
            component="test",
            data={"stage": "startup", "percent": 10.0},
        )
        logger.close()
        payload = _payload(self.path)
        self.assertIn("01_source_read", _operations(payload, "sources"))
        self.assertIn("01_index_saved", _operations(payload, "indexing"))
        self.assertIn("01_project_created", _operations(payload, "project"))
        self.assertIn(
            "01_10_percent",
            payload["sessions"]["02_project"]["progress_checkpoints"],
        )
        stages = _turns(payload)["turn_0001"]["stages"]
        self.assertIn("02_embedding", stages)
        self.assertIn("04_llm_input", stages)
        self.assertIn("05_generation", stages)

    def test_duplicate_operation_duration_keeps_other_diagnostics(
        self,
    ) -> None:
        """Do not repeat the canonical elapsed time in event details."""
        logger = self._logger()
        logger.emit(
            "runtime.finished", component="runtime", duration_ms=7.25,
            data={
                "operation_duration_ms": 7.25,
                "other_duration_ms": 9.0,
                "value": "retained",
            },
        )
        entry = _operations(_payload(self.path), "runtime")[
            "01_runtime_finished"
        ]
        self.assertEqual(entry["timing"]["operation_duration_ms"], 7.25)
        self.assertEqual(entry["details"], {
            "other_duration_ms": 9.0,
            "value": "retained",
        })


class TestTraceSpans(unittest.TestCase):
    """Exercise operation correlation and exception propagation."""

    def setUp(self) -> None:
        """Provide a local trace for every context-manager test."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "conversation.json"
        self.logger = TraceLogger(
            TraceConfig(enabled=True, output_path=self.path),
            default_directory=temporary.name,
        )
        self.addCleanup(self.logger.close)

    def test_nested_spans_merge_lifecycle_and_restore_parent(self) -> None:
        """Correlate nested operations without duplicating their endings."""
        with self.logger.span("outer", component="test") as outer:
            with self.logger.span("inner", component="test") as inner:
                self.logger.emit("inside", component="test")
            self.logger.emit("after.inner", component="test")
        self.logger.emit("outside", component="test")
        operations = _operations(_payload(self.path), "runtime")
        self.assertEqual(len(operations), 5)
        self.assertEqual(operations["01_outer"]["event"]["span_id"], outer)
        self.assertEqual(operations["02_inner"]["event"]["span_id"], inner)
        self.assertEqual(
            operations["02_inner"]["event"]["parent_span_id"], outer
        )
        self.assertEqual(operations["03_inside"]["event"]["span_id"], inner)
        self.assertEqual(
            operations["04_after_inner"]["event"]["span_id"], outer
        )
        self.assertNotIn("span_id", operations["05_outside"]["event"])
        for name in ("01_outer", "02_inner"):
            self.assertEqual(operations[name]["event"]["status"], "completed")
            self.assertGreaterEqual(
                operations[name]["timing"]["operation_duration_ms"], 0
            )

    def test_failed_span_records_traceback_and_restores_context(self) -> None:
        """Re-raise body errors while retaining failure type and traceback."""
        with self.assertRaisesRegex(KeyError, "missing"):
            with self.logger.span("failing", component="test"):
                raise KeyError("missing")
        self.logger.emit("after.failure", component="test")
        operations = _operations(_payload(self.path), "runtime")
        failure = operations["01_failing"]
        self.assertEqual(failure["event"]["status"], "failed")
        self.assertEqual(failure["result"]["error_type"], "KeyError")
        self.assertIn("missing", failure["result"]["message"])
        self.assertIn("KeyError", failure["result"]["traceback"])
        self.assertNotIn("span_id", operations["02_after_failure"]["event"])

    def test_explicit_parent_and_exception_event(self) -> None:
        """Preserve caller-provided parent IDs and standalone failures."""
        with self.logger.span(
            "task", component="test", parent_span_id="external"
        ):
            pass
        try:
            raise RuntimeError("broken")
        except RuntimeError as error:
            self.logger.exception("runtime.failed", error, component="test")
        operations = _operations(_payload(self.path), "runtime")
        self.assertEqual(
            operations["01_task"]["event"]["parent_span_id"],
            "external",
        )
        standalone = operations["02_runtime_failed"]
        self.assertEqual(standalone["event"]["status"], "failed")
        self.assertEqual(standalone["details"]["error_type"], "RuntimeError")
        self.assertIn("broken", standalone["details"]["traceback"])


class TestTraceArtifacts(unittest.TestCase):
    """Check inline content, conversion, and controlled legacy resolution."""

    def setUp(self) -> None:
        """Initialize full diagnostic tracing in a disposable directory."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "trace.json"
        self.logger = TraceLogger(
            TraceConfig(
                enabled=True, output_path=self.path,
            ),
            default_directory=self.root,
        )
        self.addCleanup(self.logger.close)

    def test_inline_artifacts_render_under_named_sections(self) -> None:
        """Render LLM input, output, and retrieval evidence exactly once."""
        trace = self.logger
        trace.emit(
            TRACE_EVENT_LLM_CALL_INPUT,
            component="chat",
            data={
                "messages_artifact": trace.artifact(
                    "messages", [{"role": "user", "content": "Hello"}]
                ),
                "rendered_prompt_artifact": trace.artifact(
                    "rendered", "<start>Hello", text=True
                ),
                "input_token_ids_artifact": trace.artifact("ids", [3, 4]),
            },
        )
        trace.emit(
            TRACE_EVENT_LLM_CALL_OUTPUT,
            component="chat",
            data={"answer_artifact": trace.artifact(
                "answer", "Result", text=True
            )},
        )
        trace.emit(
            "retrieval.context.expanded",
            component="rag",
            data={"chunks_artifact": trace.artifact(
                "chunks", [{"text": "source"}]
            )},
        )
        operations = _operations(_payload(self.path), "runtime")
        inputs = operations["01_llm_call_input"]
        self.assertNotIn("details", inputs)
        self.assertEqual(
            inputs["exact_content"]["03_input_token_ids"]["token_count"],
            2,
        )
        self.assertEqual(
            inputs["exact_content"]["01_structured_messages"]["value"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(
            inputs["exact_content"]["02_rendered_prompt"]["value"],
            "<start>Hello",
        )
        self.assertEqual(
            operations["02_llm_call_output"]["exact_content"][
                "01_generated_answer"
            ]["value"],
            "Result",
        )
        self.assertEqual(
            operations["03_retrieval_context_expanded"]["exact_content"][
                "01_retrieved_chunks"
            ]["value"],
            [{"text": "source"}],
        )

    def test_artifact_sanitizes_name_and_converts_python_values(self) -> None:
        """Keep dataclasses, paths, and sets JSON-compatible inline."""
        @dataclass(frozen=True)
        class Sample:
            """Represent a dataclass payload without external dependencies."""
            path: Path
            values: set[int]

        reference = self.logger.artifact(
            "a/b ?", Sample(Path("source.py"), {3, 1})
        )
        self.assertEqual(reference, {
            "name": "a-b", "format": "json",
            "inline_content": {"path": "source.py", "values": [1, 3]},
        })
        with self.assertRaises(ValueError):
            self.logger.artifact("...", 12)

    def test_legacy_artifact_stays_inside_output_directory(self) -> None:
        """Reject traversal and accept readable old local artifact files."""
        (self.root / "payload.json").write_text(
            '[1, 2]', encoding="utf-8"
        )
        outside = self.root.parent / "outside.json"
        with patch.object(
            self.logger, "_artifact_content",
            wraps=self.logger._artifact_content,
        ) as content:
            self.assertEqual(content({
                "path": "payload.json", "format": "json"
            }), [1, 2])
            self.assertIsNone(content({
                "path": "../outside.json", "format": "json"
            }))
            self.assertIsNone(content({
                "path": str(outside), "format": "json"
            }))
            self.assertIsNone(content({"path": "missing.json"}))
            self.assertEqual(content({
                "inline_content": "preferred", "path": "missing.json"
            }), "preferred")


    def test_legacy_symlink_artifact_cannot_escape_trace_directory(
        self,
    ) -> None:
        """Reject an in-directory link pointing outside the trace tree."""
        with TemporaryDirectory() as other_directory:
            outside = Path(other_directory) / "secret.json"
            outside.write_text('["private"]', encoding="utf-8")
            link = self.root / "linked.json"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symbolic links are unavailable")
            self.assertIsNone(self.logger._artifact_content({
                "path": "linked.json", "format": "json"
            }))


    def test_artifact_is_unavailable_after_logger_closes(self) -> None:
        """Avoid producing untracked content after the trace is closed."""
        self.assertIsNotNone(self.logger.artifact("before", "content"))
        self.logger.close()
        self.assertIsNone(self.logger.artifact("after", "content"))

    def test_legacy_text_and_invalid_json_artifacts(self) -> None:
        """Read legacy text files while ignoring malformed JSON assets."""
        (self.root / "output.txt").write_text(
            "legacy text", encoding="utf-8"
        )
        (self.root / "broken.json").write_text(
            "not json", encoding="utf-8"
        )
        self.assertEqual(
            self.logger._artifact_content({"path": "output.txt"}),
            "legacy text",
        )
        self.assertIsNone(self.logger._artifact_content({
            "path": "broken.json", "format": "json"
        }))
        self.assertIsNone(self.logger._artifact_content({
            "path": 5, "format": "json"
        }))

    def test_unknown_event_values_are_serialized_as_strings(self) -> None:
        """Keep otherwise unsupported diagnostics JSON-representable."""
        class Diagnostic:
            """Represent an object with a controlled string conversion."""

            def __str__(self) -> str:
                """Return the marker used by this serialization test."""
                return "custom diagnostic"

        self.logger.emit(
            "runtime.details", component="runtime",
            data={"diagnostic": Diagnostic()},
        )
        entry = _operations(_payload(self.path), "runtime")[
            "01_runtime_details"
        ]
        self.assertEqual(
            entry["details"]["diagnostic"], "custom diagnostic"
        )


class TestTracePersistence(unittest.TestCase):
    """Protect resumable state and readable history across trace reuse."""

    def setUp(self) -> None:
        """Reserve a file for sequential logger instances."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "conversation.json"

    def _open(self) -> TraceLogger:
        """Open or reopen the exact shared conversation trace."""
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=self.path),
            default_directory=self.root,
        )
        self.addCleanup(logger.close)
        return logger

    def test_conversation_update_merges_state_and_uses_same_file(self) -> None:
        """Write resumable conversation data into the grouped trace file."""
        logger = self._open()
        logger.update_conversation({"turns": [{"question": "first"}]})
        logger.update_conversation({"checkpoint_id": "checkpoint_1"})
        session = _payload(self.path)["sessions"][CONVERSATION_SESSION_KEY]
        self.assertEqual(
            session["resume_state"], {
                "turns": [{"question": "first"}],
                "checkpoint_id": "checkpoint_1",
            },
        )
        self.assertEqual(session["operations"], {})
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

    def test_reopen_appends_new_readable_turns_and_retains_state(self) -> None:
        """Keep earlier turns when their IDs repeat in another logger run."""
        first = self._open()
        first.emit(
            "turn.started", component="rag", turn_id="turn-reused",
            data={"question": "First"},
        )
        first.update_conversation({"checkpoint_id": "first"})
        first.close()
        second = self._open()
        second.emit(
            "turn.started", component="rag", turn_id="turn-reused",
            data={"question": "Second"},
        )
        second.update_conversation({"checkpoint_id": "second"})
        second.close()
        payload = _payload(self.path)
        turns = _turns(payload)
        self.assertEqual(list(turns), ["turn_0001", "turn_0002"])
        self.assertEqual(turns["turn_0001"]["title"], "Turn 1: First")
        self.assertEqual(turns["turn_0002"]["title"], "Turn 2: Second")
        self.assertEqual(
            payload["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["checkpoint_id"],
            "second",
        )
        self.assertEqual(payload["event_count"], 3)

    def test_legacy_standard_trace_remains_readable_and_complete(
        self,
    ) -> None:
        """Retain legacy mode metadata without filtering later events."""
        first = self._open()
        first.emit("custom.earlier", component="test")
        first.close()
        prior = _payload(self.path)
        legacy_start = _operations(prior, "trace")["01_trace_started"]
        legacy_start["details"]["mode"] = "standard"
        self.path.write_text(json.dumps(prior), encoding="utf-8")

        later = self._open()
        later.emit(
            TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
            component="chat",
            data={"ids": [123, 456]},
        )
        later.close()
        updated = _payload(self.path)
        self.assertEqual(
            _operations(updated, "trace")["01_trace_started"]["details"]
            ["mode"],
            "standard",
        )
        self.assertIn("custom_earlier", self.path.read_text(encoding="utf-8"))
        self.assertIn(
            "llm_call_output_tokens", self.path.read_text(encoding="utf-8")
        )
        self.assertIn("123", self.path.read_text(encoding="utf-8"))

    def test_reopen_advances_from_highest_number_with_gaps(self) -> None:
        """Use the highest saved turn and operation number, not their count."""
        first = self._open()
        first.emit("turn.started", component="rag", turn_id="old")
        first.emit("conversation.ready", component="rag")
        first.close()
        payload = _payload(self.path)
        conversation = payload["sessions"][CONVERSATION_SESSION_KEY]
        conversation["turns"]["turn_0042"] = {
            "title": "Legacy", "stages": {}, "turn_id": "legacy"
        }
        conversation["operations"]["35_saved"] = {"event": {}}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        second = self._open()
        second.emit("turn.started", component="rag", turn_id="new")
        second.emit("conversation.updated", component="rag")
        second.close()
        conversation = _payload(self.path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]
        self.assertIn("turn_0043", conversation["turns"])
        self.assertIn("36_conversation_updated", conversation["operations"])
        self.assertIn("turn_0042", conversation["turns"])
        self.assertIn("35_saved", conversation["operations"])

    def test_reset_clears_only_conversation_events_and_state(self) -> None:
        """Retain project context but discard saved and current turns."""
        first = self._open()
        first.emit("project.ready", component="project")
        first.emit("turn.started", component="rag", turn_id="old")
        first.update_conversation({"checkpoint_id": "old"})
        first.close()
        second = self._open()
        second.emit("turn.started", component="rag", turn_id="discard")
        second.reset_conversation_state()
        second.emit("conversation.reset", component="rag")
        second.emit("turn.started", component="rag", turn_id="fresh")
        second.update_conversation({"checkpoint_id": "fresh"})
        payload = _payload(self.path)
        self.assertEqual(list(_turns(payload)), ["turn_0001"])
        self.assertEqual(
            _turns(payload)["turn_0001"]["turn_id"], "fresh"
        )
        self.assertEqual(
            payload["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["checkpoint_id"], "fresh"
        )
        self.assertTrue(any(
            item["category"] == "project"
            for item in payload["sessions"].values()
        ))

    def test_invalid_existing_trace_is_not_overwritten(self) -> None:
        """Reject unreadable or conflicting conversation formats safely."""
        for text in (
            "not json", "[]", '{"conversation": {"turns": []}}',
            '{"sessions": {"05_conversations": {'
            '"category": "other", "turns": {},'
            '"operations": {}, "resume_state": {}}}}',
        ):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                before = self.path.read_bytes()
                with self.assertRaises(ValueError):
                    self._open()
                self.assertEqual(self.path.read_bytes(), before)

    @unittest.skipUnless(os.name == "posix", "Requires POSIX file modes")
    def test_output_is_private_on_posix(self) -> None:
        """Restrict the trace and its newly created directory permissions."""
        path = self.root / "new" / "conversation.json"
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=path),
            default_directory=self.root,
        )
        self.addCleanup(logger.close)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class TestTraceWriteFailures(unittest.TestCase):
    """Exercise the documented distinction between event and state writes."""

    def setUp(self) -> None:
        """Create a logger before injecting a filesystem failure."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "trace.json"

    def _logger(self, policy: str) -> TraceLogger:
        """Return a logger whose future writes can be mocked."""
        logger = TraceLogger(
            TraceConfig(
                enabled=True, output_path=self.path,
                failure_policy=policy,
            ),
            default_directory=self.root,
        )
        # Release its lock before the folder is removed (Windows cannot
        # delete an open file).
        self.addCleanup(logger.close)
        return logger

    def test_warn_policy_disables_later_event_writes(self) -> None:
        """Warn once without replacing the last valid trace JSON."""
        logger = self._logger(FAILURE_POLICY_WARN)
        before = self.path.read_bytes()
        with patch.object(
            logger, "_write_readable_trace", side_effect=OSError("disk full")
        ) as write:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                logger.emit("failing.event", component="test")
                logger.emit("ignored.event", component="test")
                logger.close()
        self.assertEqual(write.call_count, 1)
        self.assertEqual(len(recorded), 1)
        self.assertIs(recorded[0].category, RuntimeWarning)
        self.assertIn("disk full", str(recorded[0].message))
        self.assertEqual(self.path.read_bytes(), before)

    def test_raise_policy_reports_write_failure_and_stops(self) -> None:
        """Propagate the first event-write error while disabling later ones."""
        logger = self._logger(FAILURE_POLICY_RAISE)
        with patch.object(
            logger, "_write_readable_trace", side_effect=OSError("full")
        ) as write:
            with self.assertRaisesRegex(RuntimeError, "Could not write"):
                logger.emit("failing.event", component="test")
            logger.emit("ignored.event", component="test")
        write.assert_called_once()

    def test_failed_state_write_cannot_be_committed_by_later_event(
        self,
    ) -> None:
        """Roll back attempted state even when logging subsequently works."""
        logger = self._logger(FAILURE_POLICY_WARN)
        logger.update_conversation({
            "checkpoint_id": "committed",
            "turns": [{"question": "Earlier"}],
        })
        before = _payload(self.path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["resume_state"]
        with patch.object(
            logger, "_write_readable_trace", side_effect=OSError("full")
        ):
            with self.assertRaisesRegex(OSError, "full"):
                logger.update_conversation({
                    "checkpoint_id": "uncommitted",
                    "turns": [{"question": "Unsaved"}],
                })
        self.assertEqual(logger._conversation, before)
        logger.emit("after.failed.checkpoint", component="test")
        actual = _payload(self.path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["resume_state"]
        self.assertEqual(actual, before)
        self.assertFalse(logger._failed)
        logger.update_conversation({"checkpoint_id": "next"})
        actual = _payload(self.path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["resume_state"]
        self.assertEqual(actual["checkpoint_id"], "next")
        self.assertEqual(actual["turns"], before["turns"])

    def test_failed_rename_preserves_saved_resume_state(self) -> None:
        """Leave an earlier checkpoint intact when atomic rename fails."""
        logger = self._logger(FAILURE_POLICY_WARN)
        logger.update_conversation({"checkpoint_id": "committed"})
        with patch.object(
            Path, "replace", side_effect=OSError("rename refused")
        ):
            with self.assertRaisesRegex(OSError, "rename refused"):
                logger.update_conversation({"checkpoint_id": "unsaved"})
        logger.emit("after.rename.failure", component="test")
        self.assertEqual(
            _payload(self.path)["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["checkpoint_id"],
            "committed",
        )

    def test_conversation_update_propagates_io_error(self) -> None:
        """Keep direct resumable-state writes outside event failure policy."""
        logger = self._logger(FAILURE_POLICY_WARN)
        with patch.object(
            logger, "_write_readable_trace", side_effect=OSError("full")
        ):
            with self.assertRaisesRegex(OSError, "full"):
                logger.update_conversation({"checkpoint_id": "next"})


class TestTraceFactoryAndNull(unittest.TestCase):
    """Verify disabled tracing, ownership, and path selection."""

    def test_factory_returns_unowned_null_for_disabled_options(self) -> None:
        """Avoid creating any output for null or disabled configurations."""
        with TemporaryDirectory() as directory:
            for candidate in (None, TraceConfig(enabled=False)):
                with self.subTest(candidate=candidate):
                    sink, owned = create_trace_logger(
                        candidate, default_directory=directory
                    )
                    self.assertIsInstance(sink, NullTraceLogger)
                    self.assertFalse(owned)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_factory_respects_caller_owned_sink_and_configurations(
        self,
    ) -> None:
        """Never take ownership of a sink supplied by the caller."""
        with TemporaryDirectory() as directory:
            config = TraceConfig(enabled=True)
            created, owned = create_trace_logger(
                config, default_directory=directory
            )
            self.assertTrue(owned)
            self.assertTrue(created.output_path.is_file())
            self.assertEqual(created.output_path.suffix, ".json")
            shared, owned = create_trace_logger(
                created, default_directory=directory
            )
            self.assertIs(shared, created)
            self.assertFalse(owned)
            created.close()
            with self.assertRaises(ValueError):
                create_trace_logger(123, default_directory=directory)

    def test_null_trace_preserves_calling_protocol_without_output(
        self,
    ) -> None:
        """Allow disabled tracing in the same orchestration code paths."""
        sink = NullTraceLogger()
        self.assertFalse(sink.enabled)
        self.assertFalse(sink.include_content)
        self.assertFalse(sink.include_token_ids)
        self.assertFalse(sink.include_embedding_vectors)
        self.assertIsNone(sink.output_path)
        self.assertTrue(sink.new_id("query").startswith("query_"))
        with sink.span("work", component="test") as span_id:
            self.assertTrue(span_id.startswith("span_"))
            sink.emit("work.progress", component="test")
            sink.exception("failed", RuntimeError("x"), component="test")
            sink.update_conversation({"key": "value"})
            sink.reset_conversation_state()
            self.assertIsNone(sink.artifact("content", "secret"))
            self.assertFalse(sink.settings_changed(
                "settings", component="test", before={}, after={"a": 1}
            ))
        sink.close()
        sink.close()


    def test_explicit_directory_generates_a_conversation_filename(
        self,
    ) -> None:
        """Create a timestamped filename below an explicitly chosen folder."""
        with TemporaryDirectory() as directory:
            selected = Path(directory) / "custom"
            sink, owned = create_trace_logger(
                TraceConfig(enabled=True, output_path=selected),
                default_directory=directory,
            )
            self.assertTrue(owned)
            self.assertEqual(sink.output_path.parent, selected)
            self.assertTrue(sink.output_path.name.startswith("conversation_"))
            self.assertEqual(sink.output_path.suffix, ".json")
            sink.close()


class TestTraceTextStream(unittest.TestCase):
    """Exercise line buffering and dynamic routing of diagnostic output."""

    def test_buffers_fragments_and_ignores_blank_lines(self) -> None:
        """Emit completed lines immediately and fragments only on flush."""
        sink = SimpleNamespace(emit=Mock())
        stream = TraceTextStream(sink, stream_name="stderr")
        self.assertEqual(stream.write("first"), 5)
        sink.emit.assert_not_called()
        self.assertEqual(stream.write(" line\n\n  \nsecond"), 16)
        self.assertEqual(sink.emit.call_count, 1)
        stream.flush()
        stream.flush()
        self.assertEqual(sink.emit.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["data"]["message"]
                for call in sink.emit.call_args_list
            ],
            ["first line", "second"],
        )
        self.assertTrue(all(
            call.kwargs["data"]["stream"] == "stderr"
            for call in sink.emit.call_args_list
        ))

    def test_provider_routes_later_lines_to_new_sink(self) -> None:
        """Resolve the active trace at emission rather than construction."""
        first, second = Mock(), Mock()
        current = [first]
        stream = TraceTextStream(lambda: current[0], stream_name="stdout")
        stream.write("one\n")
        current[0] = second
        stream.write("two\n")
        first.emit.assert_called_once()
        second.emit.assert_called_once()
        self.assertEqual(
            second.emit.call_args.kwargs["data"]["message"], "two"
        )

    def test_preview_keeps_short_text_and_ends_of_long_text(self) -> None:
        """Return a concise diagnostic preview without rewriting content."""
        self.assertEqual(trace_text_preview("  one   two  "), "one two")
        self.assertEqual(
            trace_text_preview("one two three four five six seven eight"),
            "one two three … six seven eight",
        )


class TestTraceExclusiveOwnership(unittest.TestCase):
    """Protect real trace JSON files against independent active writers."""

    def setUp(self) -> None:
        """Reserve one shared output path for each trace test."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "conversation_shared.json"

    def _open(self, *, policy: str = FAILURE_POLICY_WARN) -> TraceLogger:
        """Create a new concrete logger using the same file path."""

        return TraceLogger(
            TraceConfig(
                enabled=True,
                output_path=self.path,
                failure_policy=policy,
            ),
            default_directory=self.root,
        )

    def test_competing_logger_does_not_overwrite_existing_events(
        self,
    ) -> None:
        """Refuse simultaneous independent loggers without losing events."""

        from kilin.conversation_ownership import ConversationLockedError

        first = self._open()
        self.addCleanup(first.close)
        first.emit("owner.event", component="test")
        before = self.path.read_bytes()
        with self.assertRaises(ConversationLockedError):
            self._open()
        self.assertEqual(self.path.read_bytes(), before)
        first.close()
        second = self._open()
        self.addCleanup(second.close)
        second.emit("later.event", component="test")
        self.assertIn(b"owner_event", self.path.read_bytes())
        self.assertIn(b"later_event", self.path.read_bytes())

    def test_invalid_trace_releases_lease_before_raising(self) -> None:
        """Reject corrupted JSON twice without misreporting contention."""

        self.path.write_text("not json", encoding="utf-8")
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "trace is invalid"):
                self._open()
        self.path.unlink()
        logger = self._open()
        logger.close()

    def test_strict_close_failure_still_releases_lease(self) -> None:
        """Release OS ownership when the trace.closed write raises."""

        logger = self._open(policy=FAILURE_POLICY_RAISE)
        with patch.object(
            logger, "_write_readable_trace", side_effect=OSError("full")
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not write"):
                logger.close()
        self.assertTrue(logger._closed)
        self.assertFalse(logger._file_lock.holds(self.path))
        another = self._open()
        another.close()

    def test_detach_releases_without_rewriting_existing_json(self) -> None:
        """Allow archival detachment without losing the last saved trace."""

        logger = self._open()
        logger.emit("last.saved", component="test")
        before = self.path.read_bytes()
        logger.close(emit_event=False)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            logger.update_conversation({"turns": []})
        new_logger = self._open()
        new_logger.close()
