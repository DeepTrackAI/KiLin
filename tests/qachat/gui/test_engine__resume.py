"""Verify exact-conversation restoration by the GUI engine.

Use real conversation files and a real ConversationStore, but replace project
opening and model activation so no model or Qt application is needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat.conversations import ConversationStore
from qachat.gui import engine as gui_engine


class TestEngineResume(unittest.TestCase):
    """Check conversation selection before any model or trace is started."""

    def setUp(self) -> None:
        """Create two independent, persisted conversations."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        store = ConversationStore(self.root)
        with patch(
            "qachat.conversations._utc_timestamp",
            side_effect=[
                "2026-10-14T10:00:00.000Z",
                "2026-10-14T10:01:00.000Z",
            ],
        ):
            self.older = store.ensure()
            self.newer = store.create()
        self.paths = {
            item.conversation_id: store.state_path(item)
            for item in (self.older, self.newer)
        }
        self.events: list[dict[str, object]] = []
        self.engine = gui_engine.ChatEngine(self.root, self.events.append)
        self.project = SimpleNamespace(root=self.root)
        opener = patch.object(
            gui_engine, "open_project", return_value=self.project
        )
        self.open_project = opener.start()
        self.addCleanup(opener.stop)
        activation = patch.object(self.engine, "_activate_project")
        self.activate = activation.start()
        self.addCleanup(activation.stop)
        tracing = patch.object(self.engine, "_start_project_trace")
        self.start_trace = tracing.start()
        self.addCleanup(tracing.stop)

    def test_explicit_id_restores_an_older_conversation(self) -> None:
        """Select the requested chat rather than the latest saved one."""

        self.engine.resume_project(
            str(self.root), self.older.conversation_id
        )
        self.assertEqual(
            self.engine.active_conversation.conversation_id,
            self.older.conversation_id,
        )
        self.assertEqual(
            self.engine.conversation_store.active().conversation_id,
            self.older.conversation_id,
        )
        self.start_trace.assert_called_once_with(
            self.root, self.engine.active_conversation
        )
        self.activate.assert_called_once_with(
            self.project, self.engine.active_conversation
        )
        self.assertEqual(self.events, [])

    def test_ordinary_resume_preserves_latest_selection(self) -> None:
        """Keep the existing behavior when no ID is supplied."""

        self.engine.resume_project(str(self.root))
        self.assertEqual(
            self.engine.active_conversation.conversation_id,
            self.newer.conversation_id,
        )
        self.activate.assert_called_once()
        self.assertEqual(self.events, [])

    def test_unknown_id_reports_error_without_switching_chats(self) -> None:
        """Do not activate another chat when the requested ID is missing."""

        before = {
            identifier: path.read_bytes()
            for identifier, path in self.paths.items()
        }
        self.engine.resume_project(str(self.root), "conversation_missing")
        self.assertIsNone(self.engine.active_conversation)
        self.assertIsNone(self.engine.conversation_store)
        self.activate.assert_not_called()
        self.start_trace.assert_not_called()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["event"], "error")
        self.assertIn(
            "Could not resume the project", self.events[0]["message"]
        )
        self.assertIn("Conversation not found", self.events[0]["message"])
        self.assertEqual(
            before,
            {
                identifier: path.read_bytes()
                for identifier, path in self.paths.items()
            },
        )

    def test_malformed_explicit_id_does_not_fall_back(self) -> None:
        """Reject an empty or non-string ID instead of opening a chat."""

        for value in ("", 123):
            with self.subTest(value=value):
                self.events.clear()
                self.engine.resume_project(str(self.root), value)
                self.assertIsNone(self.engine.active_conversation)
                self.activate.assert_not_called()
                self.start_trace.assert_not_called()
                self.assertEqual(len(self.events), 1)
                self.assertEqual(self.events[0]["event"], "error")
                self.assertIn(
                    "Conversation not found", self.events[0]["message"]
                )


if __name__ == "__main__":
    unittest.main()
