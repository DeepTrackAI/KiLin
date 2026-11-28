"""Validate compact references through real headless workspace operations."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from kilin import rag
from kilin._trace_configuration import (
    CONFIGURATION_EVENT,
    CONFIGURATION_REFERENCE,
    configuration_bytes,
    configuration_reference,
)
from kilin.gui.engine import ChatEngine
from kilin.gui.preferences import DEFAULTS_ENVIRONMENT
from kilin.system_testing import Experiment, InvalidRecordingError
from kilin.system_testing.traces import read_recording
from tests.kilin import test_trace_logging__configuration as writer_tests


def _chat_model(model_id, **kwargs):
    """Replace only the loaded chat model, not the engine or RAG session."""
    model = Mock()
    model.model_id = model_id
    model.backend = kwargs["backend"]
    model.temperature = kwargs["temperature"]
    model.thinking = kwargs["thinking"]
    model.max_new_tokens = kwargs["max_new_tokens"]
    model.messages = []

    def configure(values):
        for key, value in values.items():
            setattr(model, key, value)

    model.configure_generation.side_effect = configure
    return model


def _embedding_model(model_id, **kwargs):
    """Supply deterministic model identity without downloading weights."""
    model = Mock()
    model.model_id = model_id
    model.backend = kwargs["backend"]
    model.model_source = model_id
    model._retrieval_model_identity = "hf:fixture"
    model.count_tokens.side_effect = lambda text: len(text.split())
    model.embed_documents.side_effect = (
        lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    )
    return model


def _snapshot(root):
    """Retain paths and bytes to detect any modification during inspection."""
    return {path.relative_to(root).as_posix():
            path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")}


def _nodes(payload):
    """Find raw configuration observations in chronological order."""
    return sorted([
        node for session in payload["sessions"].values()
        for node in session.get("operations", {}).values()
        if node["event"]["event_type"] == CONFIGURATION_EVENT
    ], key=lambda node: node["event"]["sequence"])


class TestConfigurationInspection(unittest.TestCase):
    """Test producer/reader equivalence and rejection of altered history."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "original"
        self.target = self.root / "replay"
        self.events = []
        for patcher in (
            patch.dict(os.environ, {
                DEFAULTS_ENVIRONMENT: str(self.root / "preferences.json"),
            }),
            patch.object(rag, "choose_backend", return_value="mlx"),
            patch.object(rag, "ChatSession", side_effect=_chat_model),
            patch.object(rag, "EmbeddingSession",
                         side_effect=_embedding_model),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.engine = ChatEngine(self.root, self.events.append,
                                 trace_enabled=True)
        self.addCleanup(self.engine.close)
        self.engine.open_project([], "writing", str(self.workspace))
        self.assertEqual(self.events[-1]["event"], "opened", self.events)
        self.engine.close()
        self._resume()
        self.engine.close()
        self._resume()
        self.engine.close()
        self.experiment = Experiment(self.workspace)
        self.path = self.workspace / "logs" / "traces" / "project.json"
        self.assertTrue(self.experiment.validate().valid)

    def _resume(self):
        self.engine.resume_project(str(self.workspace))
        self.assertEqual(self.events[-1]["event"], "opened", self.events)

    def _raw(self):
        return json.loads(self.path.read_text("utf-8"))

    def _write(self, payload):
        self.path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")

    def test_compact_and_legacy_full_observations_are_equivalent(self):
        payload = self._raw()
        nodes = _nodes(payload)
        self.assertEqual(len(nodes), 3)
        full = nodes[0]["details"]
        expected = configuration_reference(
            nodes[0]["event"]["sequence"], configuration_bytes(full)
        )
        self.assertEqual([node["details"] for node in nodes[1:]],
                         [expected, expected])
        before = _snapshot(self.workspace)
        report = self.experiment.validate()
        self.assertTrue(report.valid, report.issues)
        events = [event for event in report.events
                  if event.event_type == CONFIGURATION_EVENT]
        self.assertEqual([event.details for event in events], [full] * 3)
        self.assertEqual([event.metadata for event in events],
                         [node["event"] for node in nodes])
        self.assertIsNot(events[0].details, events[1].details)
        self.assertIsNot(events[0].details["behavior"],
                         events[1].details["behavior"])
        self.assertEqual(before, _snapshot(self.workspace))
        for node in nodes[1:]:
            node["details"] = deepcopy(full)
        self._write(payload)
        legacy_before = _snapshot(self.workspace)
        legacy = self.experiment.validate()
        self.assertTrue(legacy.valid, legacy.issues)
        self.assertEqual(report, legacy)
        self.assertEqual(legacy_before, _snapshot(self.workspace))

    def test_workspace_replay_preserves_openings(self):
        before = _snapshot(self.workspace)
        original = self.experiment.validate()
        replayed = self.experiment.replay(self.target)
        self.assertTrue(replayed.validation.valid, replayed.validation.issues)
        self.assertEqual(replayed.operation_count, len(original.operations))
        self.assertEqual([op.kind for op in replayed.validation.operations],
                         [op.kind for op in original.operations])
        self.assertEqual(replayed.differences, ())
        comparison = self.experiment.compare(self.target)
        self.assertTrue(comparison.complete, comparison.data["alignment"])
        self.assertEqual(before, _snapshot(self.workspace))

    def test_replay_with_real_source_and_index_reuse(self):
        source = self.root / "source.txt"
        source.write_text("Retained source evidence.\n", encoding="utf-8")
        original = self.root / "indexed"
        self.engine.open_project([str(source)], "writing", str(original))
        self.assertEqual(self.events[-1]["event"], "opened", self.events)
        self.engine.close()
        self.engine.resume_project(str(original))
        self.assertEqual(self.events[-1]["event"], "opened", self.events)
        self.engine.close()
        experiment = Experiment(original)
        before = _snapshot(original)
        report = experiment.validate()
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(
            len(report.operations[0].matching("index.rebuilt")), 1
        )
        self.assertEqual(len(report.operations[2].matching("index.reused")), 1)
        replayed = experiment.replay(self.target)
        self.assertTrue(replayed.validation.valid, replayed.validation.issues)
        self.assertEqual(replayed.operation_count, len(report.operations))
        self.assertEqual(before, _snapshot(original))

    def test_rename_remains_full_and_replays_with_fixed_directory(self):
        self._resume()
        renamed = self.workspace.with_name("renamed")
        self.engine.rename_project(renamed.name)
        self.assertFalse(self.workspace.exists())
        self.workspace = renamed
        self.path = renamed / "logs" / "traces" / "project.json"
        self.engine.close()
        self._resume()
        self.engine.close()
        rows = _nodes(self._raw())
        self.assertEqual(rows[-2]["details"]["behavior"]["project"],
                         str(renamed))
        self.assertEqual(rows[-1]["details"], configuration_reference(
            rows[-2]["event"]["sequence"],
            configuration_bytes(rows[-2]["details"]),
        ))
        experiment = Experiment(renamed)
        before = _snapshot(renamed)
        report = experiment.validate()
        self.assertTrue(report.valid, report.issues)
        replayed = experiment.replay(self.target)
        self.assertTrue(replayed.validation.valid, replayed.validation.issues)
        self.assertEqual(replayed.operation_count, len(report.operations))
        self.assertEqual(before, _snapshot(renamed))

    def test_corrupted_references_block_validation_and_replay(self):
        original = self._raw()
        for case in ("missing", "forward", "self", "reference", "digest",
                     "mixed", "bool", "wrong_type", "wrong_event",
                     "superseded", "changed_target",
                     "duplicate_target"):
            with self.subTest(case=case):
                payload = deepcopy(original)
                first, second, third = _nodes(payload)
                reference = third["details"][CONFIGURATION_REFERENCE]
                if case == "missing":
                    reference["sequence"] = 0
                elif case == "forward":
                    reference["sequence"] = third["event"]["sequence"] + 1
                elif case == "self":
                    reference["sequence"] = third["event"]["sequence"]
                elif case == "reference":
                    reference["sequence"] = second["event"]["sequence"]
                elif case == "digest":
                    reference["sha256"] = "0" * 64
                elif case == "mixed":
                    third["details"]["profile"] = {}
                elif case == "bool":
                    reference["sequence"] = True
                elif case == "wrong_type":
                    third["details"][CONFIGURATION_REFERENCE] = "invalid"
                elif case == "wrong_event":
                    reference["sequence"] = first["event"]["sequence"] - 1
                elif case == "superseded":
                    second["details"] = deepcopy(first["details"])
                elif case == "changed_target":
                    first["details"]["profile"]["temperature"] = 0.99
                elif case == "duplicate_target":
                    # Even a duplicate of a different event type makes
                    # the complete observation an ambiguous target.
                    for session in payload["sessions"].values():
                        for node in session.get("operations", {}).values():
                            event = node["event"]
                            if event["event_type"] != CONFIGURATION_EVENT:
                                event["sequence"] = first["event"]["sequence"]
                                break
                        else:
                            continue
                        break
                self._write(payload)
                before = _snapshot(self.workspace)
                report = self.experiment.validate()
                self.assertFalse(report.valid)
                self.assertIn("configuration.reference",
                              {issue.code for issue in report.issues})
                with self.assertRaises(InvalidRecordingError):
                    self.experiment.timeline()
                with self.assertRaises(InvalidRecordingError):
                    self.experiment.replay(self.target)
                self.assertFalse(self.target.exists())
                self.assertEqual(before, _snapshot(self.workspace))

    def test_missing_target_does_not_use_current_settings(self):
        payload = self._raw()
        first, _, _ = _nodes(payload)
        first["details"] = {CONFIGURATION_REFERENCE: {
            "sequence": 1, "sha256": "0" * 64,
        }}
        self._write(payload)
        recording = read_recording(self.workspace)
        self.assertEqual(sum(issue.code == "configuration.reference"
                             for issue in recording.issues), 3)
        self.assertFalse(self.experiment.validate().valid)


class TestRecordedConfigurationResolution(unittest.TestCase):
    """Resolve producer histories without relying on GUI workspace validity."""

    def setUp(self):
        fixture = writer_tests.TestConfigurationRecording()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.fixture = fixture

    def _assert_resolved(self, submitted):
        expected = self.fixture._assert_ordered_observations(submitted)
        before = _snapshot(self.fixture.root)
        recording = read_recording(self.fixture.root)
        self.assertFalse(recording.issues, recording.issues)
        observations = [event for event in recording.events
                        if event.event_type == CONFIGURATION_EVENT]
        self.assertEqual(len(observations), len(expected))
        self.assertEqual({event.sequence: event.details
                          for event in observations}, expected)
        self.assertEqual([event.sequence for event in recording.events],
                         list(range(1, recording.last_sequence + 1)))
        self.assertEqual(len({event.metadata["event_id"]
                              for event in recording.events}),
                         len(recording.events))
        self.assertEqual(before, _snapshot(self.fixture.root))

    def test_resolve_changed_and_restored_configurations(self):
        self._assert_resolved(self.fixture._record_change_and_restore())

    def test_resolve_concurrent_thread_observations(self):
        self._assert_resolved(self.fixture._record_threaded())

    def test_resolve_concurrent_process_observations(self):
        self._assert_resolved(self.fixture._record_processes())


if __name__ == "__main__":
    unittest.main()
