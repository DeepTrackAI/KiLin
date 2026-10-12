"""Test conversation lifecycle integration with RAG and structured traces.

Use actual JSON persistence and trace writers but replace model sessions with
test doubles. Check resets, switches, trace restarts, archived conversations,
and interaction with ConversationStore without downloading model weights.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from qachat.conversation_format import (
    CONVERSATION_SESSION_KEY,
    conversation_state,
    new_conversation_session,
)
from qachat.rag import RAGSession, _conversation_trace
from qachat.conversations import ConversationStore
from qachat.trace_logging import (
    NullTraceLogger,
    TRACE_MODE_FORENSIC,
    TraceConfig,
    TraceLogger,
)


def _read(path: Path) -> dict:
    """Read a saved conversation JSON document.

    Parameters
    ----------
    path: pathlib.Path
        Local JSON file to inspect.

    Returns
    -------
    dict
        Parsed JSON document.

    """

    return json.loads(path.read_text(encoding="utf-8"))


def _group(path: Path) -> dict:
    """Read the canonical grouped conversation session.

    Parameters
    ----------
    path: pathlib.Path
        Local trace JSON document.

    Returns
    -------
    dict
        Stored conversation session.

    """

    return _read(path)["sessions"][CONVERSATION_SESSION_KEY]


def _logger(path: Path) -> TraceLogger:
    """Create a real forensic trace logger bound to one local file.

    Parameters
    ----------
    path: pathlib.Path
        Target conversation JSON path.

    Returns
    -------
    TraceLogger
        Open trace logger for the fixture.

    """

    return TraceLogger(
        TraceConfig(enabled=True, output_path=path, mode=TRACE_MODE_FORENSIC),
        default_directory=path.parent,
    )


def _session(root: Path, trace) -> RAGSession:
    """Make a lightweight RAG session without starting inference models.

    Parameters
    ----------
    root: pathlib.Path
        Temporary project directory.
    trace: TraceSink
        Real or disabled trace writer for the fixture.

    Returns
    -------
    RAGSession
        Session with real persistence methods and mocked model sessions.

    """

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
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY:
                         new_conversation_session()},
        }))
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



class TestCanonicalTracePersistence(unittest.TestCase):
    """Keep the trace writer strict and compatible with canonical state."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "conversation.json"

    def test_first_event_creates_complete_canonical_session(self):
        trace = _logger(self.path)
        payload = _read(self.path)
        session = payload["sessions"][CONVERSATION_SESSION_KEY]
        self.assertEqual(session["title"], "Conversation turns")
        self.assertEqual(session["category"], "conversations")
        self.assertEqual(session["turns"], {})
        self.assertEqual(session["operations"], {})
        self.assertIs(conversation_state(payload), session["resume_state"])
        trace.close()

    def test_turn_only_trace_has_both_readable_groups(self):
        trace = _logger(self.path)
        trace.emit("turn.started", component="rag", turn_id="first")
        trace.close()
        session = _group(self.path)
        self.assertEqual(list(session["turns"]), ["turn_0001"])
        self.assertEqual(session["operations"], {})
        self.assertEqual(conversation_state(_read(self.path)), {})

    def test_conversation_operation_only_trace_has_both_groups(self):
        trace = _logger(self.path)
        trace.emit("conversation.renamed", component="rag")
        trace.close()
        session = _group(self.path)
        self.assertEqual(session["turns"], {})
        self.assertEqual(list(session["operations"]),
                         ["01_conversation_renamed"])
        self.assertEqual(conversation_state(_read(self.path)), {})

    def test_reject_obsolete_and_malformed_existing_files(self):
        canonical = {
            "sessions": {
                CONVERSATION_SESSION_KEY: new_conversation_session(),
            }
        }
        bad_records = (
            {"conversation": {"turns": []}},
            {"trace": {"sessions": canonical["sessions"]}},
            {"sessions": [], "trace": {"sessions": {}}},
            {"sessions": {}},
            {"sessions": {CONVERSATION_SESSION_KEY: {
                "category": "conversations", "conversation": {},
            }}},
            {"sessions": {CONVERSATION_SESSION_KEY: {
                "category": "runtime", "important": "untouched",
            }}},
            {"sessions": {"custom": new_conversation_session()}},
            {"sessions": {CONVERSATION_SESSION_KEY:
                          new_conversation_session({
                "chat_messages": [{"role": "user"}],
            })}},
        )
        for number, record in enumerate(bad_records):
            with self.subTest(number=number):
                path = self.root / f"invalid_{number}.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "invalid"):
                    _logger(path)
                self.assertEqual(path.read_bytes(), before)

    def test_reopen_keeps_canonical_resume_state_and_trace_operations(self):
        trace = _logger(self.path)
        trace.update_conversation({"checkpoint_id": "before"})
        trace.emit("conversation.restored", component="rag")
        trace.close()
        previous = _group(self.path)
        trace = _logger(self.path)
        trace.update_conversation({"checkpoint_id": "after"})
        trace.emit("turn.started", component="rag", turn_id="new")
        trace.close()
        current = _group(self.path)
        self.assertEqual(current["resume_state"]["checkpoint_id"], "after")
        self.assertEqual(list(current["turns"]), ["turn_0001"])
        self.assertEqual(len(current["operations"]),
                         len(previous["operations"]))
        self.assertEqual(conversation_state(_read(self.path)),
                         current["resume_state"])


class TestConversationStoreIntegration(unittest.TestCase):
    """Exercise storage, trace persistence, and RAG lifecycle together."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.store = ConversationStore(self.root)

    def test_new_store_record_can_receive_traced_rag_checkpoint(self):
        item = self.store.ensure()
        path = self.store.state_path(item)
        trace = _logger(path)
        self.addCleanup(trace.close)
        session = _session(self.root, trace)
        session.conversation_path = path
        session._checkpoint_id = None
        session._save_conversation()
        state = conversation_state(_read(path))
        self.assertEqual(state["conversation_id"], item.conversation_id)
        self.assertEqual(state["turns"], [])
        self.assertEqual(state["system_prompt"], "Test system")
        self.assertTrue(state["checkpoint_id"])
        self.assertEqual(state["checkpoint"]["index_fingerprint"],
                         "unchanged-index")
        self.assertTrue(self.store.is_empty(item.conversation_id))

    def test_branch_and_switch_reuse_models_and_saved_checkpoint(self):
        parent = self.store.ensure()
        parent_path = self.store.state_path(parent)
        old_trace = _logger(parent_path)
        self.addCleanup(old_trace.close)
        session = _session(self.root, old_trace)
        session.conversation_path = parent_path
        session._checkpoint_id = None
        session._save_conversation()
        checkpoint = session._checkpoint_id
        self.assertTrue(checkpoint)
        old_trace.emit("turn.started", component="rag", turn_id="old")
        old_trace.close()

        branch = self.store.create(branch_from=parent.conversation_id)
        self.assertEqual(branch.origin_checkpoint_id, checkpoint)
        path = self.store.state_path(branch)
        self.assertEqual(_group(path)["operations"], {})
        self.assertEqual(_group(path)["turns"], {})
        new_trace = _logger(path)
        self.addCleanup(new_trace.close)
        result = session.switch_conversation(path, trace=new_trace)
        self.assertEqual(result, [])
        self.assertEqual(session._checkpoint_id, checkpoint)
        self.assertEqual(session.conversation_path, path)
        session.chat_session.reset.assert_called_once_with()
        session.chat_session.restore_history.assert_called_once()
        session.chat_session.configure_generation.assert_called_once_with(
            {"temperature": 0.0, "thinking": False, "max_new_tokens": 256}
        )
        session.chat_session.close.assert_not_called()
        session.embedding_session.close.assert_not_called()

    def test_store_rename_and_trace_metadata_can_be_kept_in_sync(self):
        item = self.store.ensure()
        path = self.store.state_path(item)
        trace = _logger(path)
        self.addCleanup(trace.close)
        renamed = self.store.rename(item.conversation_id, "Renamed")
        trace.update_conversation(asdict(renamed))
        trace.emit("conversation.renamed", component="gui_engine")
        trace.close()
        store = ConversationStore(self.root)
        self.assertEqual(store.ensure().name, "Renamed")
        self.assertEqual(conversation_state(_read(path))["name"], "Renamed")

    def test_archive_trace_record_is_not_rediscovered(self):
        item = self.store.ensure()
        path = self.store.state_path(item)
        trace = _logger(path)
        trace.emit("turn.started", component="rag", turn_id="turn")
        trace.close()
        replacement, archived = self.store.archive(item.conversation_id)
        self.assertTrue(archived.is_file())
        self.assertNotEqual(replacement.conversation_id,
                            item.conversation_id)
        reopened = ConversationStore(self.root)
        self.assertNotIn(item.conversation_id,
                         [x.conversation_id for x in reopened.conversations()])
        self.assertNotEqual(reopened.ensure().conversation_id,
                            item.conversation_id)
        self.assertFalse(archived.is_symlink())

    def test_untraced_rag_checkpoint_keeps_canonical_metadata(self):
        item = self.store.ensure()
        path = self.store.state_path(item)
        payload = _read(path)
        group = payload["sessions"][CONVERSATION_SESSION_KEY]
        group["resume_state"]["custom_note"] = {"origin": "session"}
        group["operations"] = {"01_action": {"title": "Retained"}}
        path.write_text(json.dumps(payload), encoding="utf-8")

        session = _session(self.root, NullTraceLogger())
        session.conversation_path = path
        session._checkpoint_id = None
        session._save_conversation()

        saved = _read(path)["sessions"][CONVERSATION_SESSION_KEY]
        self.assertEqual(saved["resume_state"]["custom_note"],
                         {"origin": "session"})
        self.assertEqual(saved["resume_state"]["conversation_id"],
                         item.conversation_id)
        self.assertTrue(saved["resume_state"]["checkpoint_id"])
        self.assertEqual(saved["operations"],
                         {"01_action": {"title": "Retained"}})
        self.assertEqual(ConversationStore(self.root).ensure().name,
                         item.name)

    def test_disabled_trace_checkpoint_remains_discoverable(self):
        item = self.store.ensure()
        session = _session(self.root, NullTraceLogger())
        session.conversation_path = self.store.state_path(item)
        session._checkpoint_id = None
        session._save_conversation()
        self.assertEqual(ConversationStore(self.root).ensure().conversation_id,
                         item.conversation_id)
        self.assertEqual(self.store.is_empty(item.conversation_id), True)


class TestConversationTraceSelection(unittest.TestCase):
    """Check trace binding and continuity without starting model backends."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "logs" / "conversation_trace.json"

    def test_missing_trace_configuration_uses_forensic_conversation(self):
        trace, owned = _conversation_trace(None, self.path)
        self.addCleanup(trace.close)
        self.assertTrue(owned)
        self.assertTrue(trace.enabled)
        self.assertEqual(trace.output_path, self.path)
        self.assertTrue(self.path.is_file())

    def test_explicit_matching_output_path_is_allowed(self):
        config = TraceConfig(enabled=True, output_path=self.path)
        trace, owned = _conversation_trace(config, self.path)
        self.addCleanup(trace.close)
        self.assertTrue(owned)
        self.assertEqual(trace.output_path, self.path)

    def test_existing_caller_owned_trace_is_reused(self):
        trace = _logger(self.path)
        self.addCleanup(trace.close)
        result, owned = _conversation_trace(trace, self.path)
        self.assertIs(result, trace)
        self.assertFalse(owned)

    def test_existing_mismatched_trace_is_rejected(self):
        other = self.root / "logs" / "conversation_other.json"
        trace = _logger(other)
        self.addCleanup(trace.close)
        with self.assertRaisesRegex(ValueError, "conversation path"):
            _conversation_trace(trace, self.path)
        self.assertFalse(self.path.exists())

    def test_disabled_trace_does_not_require_conversation_path(self):
        trace, owned = _conversation_trace(
            TraceConfig(enabled=False, output_path=self.path), self.path
        )
        self.assertFalse(trace.enabled)
        self.assertFalse(owned)
        self.assertFalse(self.path.exists())

    def test_reset_without_trace_does_not_load_models(self):
        session = _session(self.root, NullTraceLogger())
        session._checkpoint_id = "previous"
        session.turns = [object()]
        self.assertIsNone(session.reset())
        self.assertIsNone(session._checkpoint_id)
        self.assertEqual(session.turns, [])
        session.chat_session.reset.assert_called_once_with()
        session.embedding_session.close.assert_not_called()

    def test_switch_rejects_closed_session_without_changing_trace(self):
        trace = _logger(self.path)
        self.addCleanup(trace.close)
        session = _session(self.root, trace)
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.switch_conversation(self.path, trace=trace)
        self.assertIs(session.trace, trace)
        session.chat_session.reset.assert_not_called()

    def test_reset_closed_session_does_not_touch_disk(self):
        session = _session(self.root, NullTraceLogger())
        session._closed = True
        session.conversation_path.write_text("legacy", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reset()
        self.assertEqual(session.conversation_path.read_text(), "legacy")

    def test_grouped_state_survives_trace_restart(self):
        trace = _logger(self.path)
        trace.update_conversation({"turns": [], "checkpoint_id": "saved"})
        trace.close()
        new_trace = _logger(self.path)
        self.addCleanup(new_trace.close)
        new_trace.emit("conversation.restored", component="rag")
        self.assertEqual(
            conversation_state(_read(self.path))["checkpoint_id"], "saved"
        )

    def test_sequential_trace_reopens_preserve_grouped_metadata(self):
        trace = _logger(self.path)
        trace.update_conversation({"name": "Session", "turns": []})
        trace.close()
        reopened = _logger(self.path)
        self.addCleanup(reopened.close)
        reopened.update_conversation({"checkpoint_id": "updated"})
        state = conversation_state(_read(self.path))
        self.assertEqual(state["name"], "Session")
        self.assertEqual(state["checkpoint_id"], "updated")

if __name__ == "__main__":
    unittest.main()
