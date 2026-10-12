"""Verify GUI conversation switching respects exclusive file ownership.

Tests use a real conversation store and trace writer with a lightweight
session substitute. They do not import Qt, load models, or start the GUI.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from qachat.conversation_ownership import (
    ConversationFileLock,
    ConversationLockedError,
)
from qachat.conversations import ConversationStore
from qachat.gui.engine import ChatEngine
from qachat.trace_logging import TraceLogger


class TestGUIConversationOwnership(unittest.TestCase):
    """Preserve the active GUI trace when the requested file is owned."""

    def setUp(self) -> None:
        """Prepare two store records and a real active trace."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events: list[dict[str, object]] = []
        self.engine = ChatEngine(
            self.root, self.events.append, trace_enabled=True
        )
        self.store = ConversationStore(self.root)
        self.previous = self.store.create()
        self.target = self.store.create()
        self.engine.conversation_store = self.store
        self.engine.active_conversation = self.previous
        self.engine._start_project_trace(self.root, self.previous)
        self.previous_trace = self.engine.project_trace
        self.session = MagicMock()
        self.session.project = SimpleNamespace(
            root=self.root, profile=SimpleNamespace(retrieval_count=5)
        )
        self.session.switch_conversation.return_value = []
        self.engine.session = self.session
        self.addCleanup(self.engine._close_project_trace)

    def test_locked_target_preserves_previous_live_trace(self) -> None:
        """Leave the active session untouched if the target is occupied."""

        target_path = self.store.state_path(self.target)
        old_path = self.store.state_path(self.previous)
        with ConversationFileLock(target_path):
            with self.assertRaises(ConversationLockedError):
                self.engine._switch_conversation(self.target)
            self.assertIs(
                self.engine.project_trace, self.previous_trace
            )
            self.assertIs(
                self.engine.active_conversation, self.previous
            )
            self.assertTrue(self.previous_trace._file_lock.holds(old_path))
            self.session.switch_conversation.assert_not_called()

    def test_failed_selection_restores_store_active_identifier(
        self,
    ) -> None:
        """Do not leave the GUI store pointing at a locked target."""

        self.store.select(self.previous.conversation_id)
        target_path = self.store.state_path(self.target)
        with ConversationFileLock(target_path), patch.object(
            self.engine, "_report_error"
        ) as reported:
            self.engine.select_conversation(self.target.conversation_id)
        reported.assert_called_once()
        self.assertEqual(self.store.active(), self.previous)
        self.assertIs(
            self.engine.active_conversation, self.previous
        )
        self.assertIs(self.engine.project_trace, self.previous_trace)

    def test_failed_new_chat_restores_previous_store_selection(
        self,
    ) -> None:
        """Keep the former active identity if new trace acquisition fails."""

        self.store.select(self.previous.conversation_id)
        with patch.object(
            self.engine,
            "_switch_conversation",
            side_effect=ConversationLockedError("busy"),
        ), patch.object(self.engine, "_report_error") as reported:
            self.engine.create_conversation()
        reported.assert_called_once()
        self.assertEqual(self.store.active(), self.previous)
        self.assertIs(self.engine.active_conversation, self.previous)

    def test_successful_switch_releases_old_trace_lock(self) -> None:
        """Acquire the target before releasing the former conversation."""

        new_path = self.store.state_path(self.target)
        old_path = self.store.state_path(self.previous)
        with patch.object(
            self.engine, "_use_project_answer_settings"
        ), patch.object(
            self.engine, "_send_conversations"
        ):
            self.engine._switch_conversation(self.target)
        self.assertIs(
            self.engine.active_conversation, self.target
        )
        self.assertIsInstance(self.engine.project_trace, TraceLogger)
        self.assertTrue(self.engine.project_trace._file_lock.holds(new_path))
        self.assertFalse(self.previous_trace._file_lock.holds(old_path))
        with ConversationFileLock(old_path) as owner:
            self.assertTrue(owner.holds(old_path))

    def test_detaching_archived_trace_releases_without_rewrite(self) -> None:
        """Retain archived bytes and release ownership after deletion."""

        old_path = self.store.state_path(self.previous)
        old_bytes = old_path.read_bytes()
        _, archived = self.store.archive(
            self.previous.conversation_id,
            ownership=self.previous_trace._file_lock,
        )
        self.engine._detach_project_trace()
        self.assertEqual(archived.read_bytes(), old_bytes)
        self.assertFalse(self.previous_trace._file_lock.holds(old_path))
        with ConversationFileLock(old_path):
            pass

    def test_rename_commits_through_trace_only(self) -> None:
        """A later diagnostic event must not revert a renamed title."""

        import json

        from qachat.conversation_format import conversation_state
        from qachat import conversations

        self.store.select(self.previous.conversation_id)
        self.session._active_conversation_lock.return_value = (
            self.previous_trace._file_lock
        )
        path = self.store.state_path(self.previous)
        with patch.object(
            conversations,
            "_write_json",
            side_effect=AssertionError("unexpected second writer"),
        ), patch.object(self.engine, "_send_conversations"):
            self.engine.rename_conversation(
                self.previous.conversation_id, "Updated title"
            )
        self.previous_trace.emit("later.event", component="test")
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(conversation_state(saved)["name"], "Updated title")
        self.assertEqual(self.store.active().name, "Updated title")
        self.assertEqual(self.engine.active_conversation.name, "Updated title")

    def test_active_trace_supports_metadata_before_rag_session(
        self,
    ) -> None:
        """Borrow the GUI trace lease when the RAG model is not loaded."""

        import json

        from qachat.conversation_format import conversation_state

        self.store.select(self.previous.conversation_id)
        self.engine.session = None
        with patch.object(self.engine, "_send_conversations"):
            self.engine.rename_conversation(
                self.previous.conversation_id, "Ready title"
            )
        path = self.store.state_path(self.previous)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(conversation_state(saved)["name"], "Ready title")
        self.assertEqual(self.engine.active_conversation.name, "Ready title")

    def test_failed_rename_does_not_change_registry_or_saved_title(
        self,
    ) -> None:
        """A failed trace update must not create a stale store snapshot."""

        import json

        from qachat.conversation_format import conversation_state

        self.store.select(self.previous.conversation_id)
        self.session._active_conversation_lock.return_value = (
            self.previous_trace._file_lock
        )
        path = self.store.state_path(self.previous)
        previous_name = self.previous.name
        with patch.object(
            self.previous_trace,
            "_write_readable_trace",
            side_effect=OSError("out of space"),
        ), patch.object(self.engine, "_report_error") as report:
            self.engine.rename_conversation(
                self.previous.conversation_id, "Unsaved title"
            )
        report.assert_called_once()
        self.previous_trace.emit("later.event", component="test")
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(conversation_state(saved)["name"], previous_name)
        self.assertEqual(self.store.active(), self.previous)
        self.assertEqual(self.engine.active_conversation, self.previous)


if __name__ == "__main__":
    unittest.main()
