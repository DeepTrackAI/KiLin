"""Protect the conversation/trace contracts without downloading models."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from qachat.conversation_format import (
    CONVERSATION_SESSION_KEY,
    conversation_state,
)
from qachat.rag import RAGSession, _conversation_trace
from qachat.trace_logging import (
    NullTraceLogger,
    TRACE_MODE_FORENSIC,
    TraceConfig,
    TraceLogger,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _group(path: Path) -> dict:
    return _read(path)["sessions"][CONVERSATION_SESSION_KEY]


def _logger(path: Path) -> TraceLogger:
    return TraceLogger(
        TraceConfig(enabled=True, output_path=path, mode=TRACE_MODE_FORENSIC),
        default_directory=path.parent,
    )


def _session(root: Path, trace) -> RAGSession:
    """Use real lifecycle code with model sessions replaced by test doubles."""

    session = RAGSession.__new__(RAGSession)
    session.project = SimpleNamespace(
        root=root,
        source=root / "sources",
        logs=root / "logs",
        profile=SimpleNamespace(
            temperature=0.0, thinking=False, max_new_tokens=256
        ),
    )
    session.conversation_path = root / "logs" / "conversation_active.json"
    session.conversation_path.parent.mkdir(parents=True, exist_ok=True)
    session.trace = trace
    session._owns_trace = False
    session._closed = False
    session._checkpoint_id = "old-checkpoint"
    session.turns = []
    session.chat_session = MagicMock()
    session.embedding_session = MagicMock()
    session.chat_session.model_id = "test/chat"
    session.chat_session.model_source = None
    session.embedding_session.model_id = "test/embedding"
    session.embedding_session.model_source = None
    session.indexes = SimpleNamespace(fingerprint="unchanged-index")
    session.index_status = "reused"
    session.backend = "mlx"
    session._parameters = MagicMock(return_value={})
    session._system_prompt = MagicMock(return_value="Test system")
    return session


class TestConversationLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "logs" / "conversation_active.json"

    def test_unspecified_trace_path_binds_to_conversation(self) -> None:
        config = TraceConfig(enabled=True)
        trace, owned = _conversation_trace(config, self.path)
        self.assertTrue(owned)
        self.assertEqual(trace.output_path, self.path)
        self.assertIsNone(config.output_path)
        trace.close()

    def test_conflicting_config_is_rejected_before_file_creation(self):
        other = self.root / "conversation_other.json"
        with self.assertRaisesRegex(ValueError, "conversation path"):
            _conversation_trace(
                TraceConfig(enabled=True, output_path=other), self.path
            )
        self.assertFalse(self.path.exists())
        self.assertFalse(other.exists())

    def test_switch_rejects_mismatched_sink_without_mutating_session(self):
        trace = _logger(self.path)
        session = _session(self.root, trace)
        other = self.root / "logs" / "conversation_other.json"
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "conversation path"):
            session.switch_conversation(other, trace=trace)
        self.assertIs(session.trace, trace)
        self.assertEqual(session.conversation_path, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        session.chat_session.reset.assert_not_called()

    def test_reset_preserves_archive_and_clears_active_state(self):
        trace = _logger(self.path)
        session = _session(self.root, trace)
        trace.emit("project.prepared", component="project")
        trace.emit(
            "turn.started",
            component="rag",
            turn_id="old",
            data={"question": "Old question"},
        )
        trace.update_conversation(
            {"turns": [], "checkpoint_id": "old-checkpoint"}
        )
        before = self.path.read_bytes()
        session.turns = [object()]
        models = session.chat_session, session.embedding_session
        archived = session.reset()

        self.assertIsNotNone(archived)
        self.assertEqual(archived.read_bytes(), before)
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)
        self.assertEqual(conversation_state(_read(self.path)), {})
        self.assertEqual(_group(self.path).get("turns", {}), {})
        self.assertNotIn("Old question", self.path.read_text())
        self.assertIn("project_prepared", self.path.read_text())
        session.chat_session.reset.assert_called_once_with()
        for model in models:
            model.close.assert_not_called()

        reopened = _logger(self.path)
        session.switch_conversation(self.path, trace=reopened)
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)
        reopened.emit("turn.started", component="rag", turn_id="new")
        session._save_conversation(trace_turn_id="new")
        self.assertEqual(list(_group(self.path)["turns"]), ["turn_0001"])
        state = conversation_state(_read(self.path))
        self.assertIsNone(state["checkpoint"]["parent_checkpoint_id"])
        self.assertEqual(archived.read_bytes(), before)

    def test_reset_without_trace_does_not_create_replacement(self):
        session = _session(self.root, NullTraceLogger())
        session.conversation_path.write_text('{"conversation": {}}')
        session.turns = [object()]
        archived = session.reset()
        self.assertTrue(archived.is_file())
        self.assertFalse(self.path.exists())
        self.assertEqual(session._load_conversation(), ([], [], None))
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)

    def test_reset_without_active_file_is_valid(self):
        session = _session(self.root, NullTraceLogger())
        self.assertIsNone(session.reset())
        self.assertFalse(self.path.exists())


class TestTraceReopening(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "conversation_trace.json"

    def test_sequential_reopens_preserve_turns_and_operations(self):
        for number in range(1, 4):
            trace = _logger(self.path)
            trace.emit("conversation.restored", component="rag")
            trace.emit(
                "turn.started",
                component="rag",
                turn_id=f"id-{number}",
                data={"question": f"Question {number}"},
            )
            trace.update_conversation({"checkpoint_id": f"cp-{number}"})
            trace.close()
        group = _group(self.path)
        self.assertEqual(
            list(group["turns"]),
            ["turn_0001", "turn_0002", "turn_0003"],
        )
        self.assertEqual(len(group["operations"]), 3)
        for number, turn in enumerate(group["turns"].values(), 1):
            self.assertEqual(turn["turn_id"], f"id-{number}")
            self.assertEqual(
                turn["title"], f"Turn {number}: Question {number}"
            )
        self.assertEqual(
            conversation_state(_read(self.path))["checkpoint_id"], "cp-3"
        )

    def test_interleaved_events_keep_their_own_turn_numbers(self):
        trace = _logger(self.path)
        trace.emit("turn.started", component="rag", turn_id="saved")
        trace.close()
        trace = _logger(self.path)
        trace.emit("answer.started", component="chat", turn_id="first")
        trace.emit("answer.started", component="chat", turn_id="second")
        trace.emit(
            "turn.started",
            component="rag",
            turn_id="first",
            data={"question": "Delayed first"},
        )
        trace.emit(
            "turn.started",
            component="rag",
            turn_id="second",
            data={"question": "Delayed second"},
        )
        trace.close()
        turns = _group(self.path)["turns"]
        self.assertEqual(turns["turn_0002"]["title"], "Turn 2: Delayed first")
        self.assertEqual(turns["turn_0003"]["title"], "Turn 3: Delayed second")

    def test_reopening_uses_highest_number_not_saved_record_count(self):
        trace = _logger(self.path)
        trace.emit("turn.started", component="rag", turn_id="saved")
        trace.close()
        payload = _read(self.path)
        turns = payload["sessions"][CONVERSATION_SESSION_KEY]["turns"]
        turns["turn_0009"] = turns.pop("turn_0001")
        self.path.write_text(json.dumps(payload))

        trace = _logger(self.path)
        trace.emit("turn.started", component="rag", turn_id="next")
        trace.close()
        turns = _group(self.path)["turns"]
        self.assertEqual(list(turns), ["turn_0009", "turn_0010"])
        self.assertEqual(turns["turn_0010"]["turn_id"], "next")

    def test_reset_clears_saved_and_current_conversation_operations(self):
        trace = _logger(self.path)
        trace.emit("conversation.restored", component="rag")
        trace.emit("turn.started", component="rag", turn_id="saved")
        trace.close()
        trace = _logger(self.path)
        trace.emit("conversation.renamed", component="rag")
        trace.emit("turn.started", component="rag", turn_id="current")
        trace.reset_conversation_state()
        trace.emit("conversation.reset", component="rag")
        trace.emit("turn.started", component="rag", turn_id="new")
        trace.close()
        group = _group(self.path)
        self.assertEqual(list(group["turns"]), ["turn_0001"])
        self.assertEqual(list(group["operations"]), ["01_conversation_reset"])
        self.assertEqual(group["turns"]["turn_0001"]["turn_id"], "new")


if __name__ == "__main__":
    unittest.main()
