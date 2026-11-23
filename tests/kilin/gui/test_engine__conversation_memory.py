"""Verify project-scoped conversation-memory commands in the GUI engine.

The tests use real project conversations and feedback, but no model weights.
They ensure that memory controls cannot mutate another workspace or rebuild
ordinary document indexes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from kilin import rag
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.index import MemoryIndex
from kilin.conversation_memory.persistence import (
    memory_index_path,
    write_memory_index,
)
from kilin.conversations import ConversationStore
from kilin.gui.engine import ChatEngine
from kilin.workspace import prepare_sources_workspace
from tests.kilin._rag_support import _turn


class TestEngineConversationMemory(unittest.TestCase):
    """Exercise feedback, index status and explicit synchronization."""

    def setUp(self) -> None:
        """Prepare a complete writing project with one saved turn."""

        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        source = self.root / "note.txt"
        source.write_text("Relevant material.\n", encoding="utf-8")
        self.project = prepare_sources_workspace(
            [source], "writing", self.root / "workspace"
        )
        self.store = ConversationStore(self.project.root)
        self.conversation = self.store.create()
        self.turn = _turn(
            "What should the introduction emphasize?",
            "Emphasize the foldon hypothesis.",
            parameters={"intent": "writing"},
        )
        path = self.store.state_path(self.conversation)
        saved = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(saved)["turns"] = [
            rag._turn_to_data(self.turn)
        ]
        path.write_text(json.dumps(saved), encoding="utf-8")
        self.events: list[dict[str, object]] = []
        self.engine = ChatEngine(self.root, self.events.append)
        self.session = MagicMock()
        self.session.project = self.project
        self.session.sync_conversation_memory.return_value = MemoryIndex(
            "test-model", (), {}, changed=True, embedded_count=2
        )
        self.engine.session = self.session
        self.project_id = str(self.project.root)

    def request(self, **changes):
        """Construct one revision-checked feedback write."""

        request = {
            "project": self.project_id,
            "conversation_id": self.conversation.conversation_id,
            "turn_uuid": self.turn.conversation_turn_uuid,
            "role": "assistant",
            "saved": True,
            "reaction": "liked",
            "revision": FeedbackStore(self.project.root).load().revision,
            "request_id": 1,
        }
        request.update(changes)
        return request

    def test_read_and_update_feedback_without_changing_checkpoint(self):
        """Preserve saved conversation bytes and return authoritative state."""

        path = self.store.state_path(self.conversation)
        before = path.read_bytes()
        self.engine.conversation_feedback(self.project_id)
        result = self.events[-1]
        self.assertEqual(result["event"], "conversation_feedback")
        self.assertEqual(result["entries"], {})
        self.assertEqual(result["project"], self.project_id)
        request = self.request(revision=result["revision"])
        self.engine.save_message_feedback(request)
        saved = next(event for event in self.events if event["event"]
                     == "message_feedback")
        self.assertEqual(saved["request_id"], 1)
        self.assertEqual(saved["revision"],
                         FeedbackStore(self.project.root).load().revision)
        self.assertEqual(saved["entries"][self.turn.conversation_turn_uuid]
                         ["assistant"], {"saved": True, "reaction": "liked"})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.events[-1]["event"], "memory_status")
        self.session.refresh_sources.assert_not_called()

    def test_stale_write_rejected_without_overwriting_feedback(self):
        """An outdated revision cannot replace the last saved reaction."""

        request = self.request()
        self.engine.save_message_feedback(request)
        self.engine.save_message_feedback({
            **request, "reaction": "disliked", "request_id": 2,
        })
        self.assertEqual(self.events[-1]["event"], "message_feedback")
        self.assertIn("error", self.events[-1])
        feedback = FeedbackStore(self.project.root).load()
        self.assertEqual(feedback.get(self.turn.conversation_turn_uuid,
                                      "assistant").reaction, "liked")

    def test_foreign_project_and_wrong_identity_are_rejected(self):
        """Never cross project boundaries or rate an unknown turn."""

        for changes in (
            {"project": str(self.root / "foreign")},
            {"conversation_id": "not-a-conversation"},
            {"turn_uuid": "a" * 32},
            {"role": "system"},
            {"request_id": 0},
            {"extra": "unaccepted"},
        ):
            self.engine.save_message_feedback(self.request(**changes))
            self.assertEqual(self.events[-1]["event"], "message_feedback")
            self.assertIn("error", self.events[-1])
        self.assertFalse((self.project.root / "memory" / "feedback.json")
                         .exists())
        self.engine.conversation_feedback(str(self.root / "foreign"))
        self.assertEqual(self.events[-1]["event"],
                         "conversation_feedback")
        self.assertIn("error", self.events[-1])

    def test_answer_in_progress_prevents_feedback_and_sync(self):
        """Explicit mutations do not overlap generation."""

        self.engine._answering = True
        self.engine.save_message_feedback(self.request())
        self.assertIn("error", self.events[-1])
        self.engine.sync_memory_index(self.project_id)
        self.assertEqual(self.events[-1]["event"], "memory_sync_failed")
        self.session.sync_conversation_memory.assert_not_called()

    def test_index_status_is_read_only_and_reports_feedback_staleness(self):
        """Show the last derived index, not a claim of current freshness."""

        snapshot = FeedbackStore(self.project.root).load()
        write_memory_index(memory_index_path(self.project.root), {
            "model_identity": "test-model",
            "manifest": {}, "policy_digest": "test-policy",
            "feedback_revision": snapshot.revision,
            "records": [
                {"role": "user"}, {"role": "assistant"},
            ],
            "vectors": {"fingerprint": [1.0, 0.0]},
        })
        self.engine.memory_status(self.project_id)
        result = self.events[-1]
        self.assertEqual((result["records"], result["questions"],
                          result["answers"], result["vectors"]),
                         (2, 1, 1, 1))
        self.assertTrue(result["enabled"])
        self.assertFalse(result["feedback_stale"])
        self.engine.save_message_feedback(self.request())
        self.assertTrue(self.events[-1]["feedback_stale"])
        self.session.sync_conversation_memory.assert_not_called()
        self.session.refresh_sources.assert_not_called()

    def test_manual_sync_reports_result_without_source_refresh(self):
        """Use the existing PR4 synchronizer and report its result."""

        self.engine.sync_memory_index(self.project_id)
        self.assertEqual([event["event"] for event in self.events], [
            "memory_sync_started", "memory_synced", "memory_status",
        ])
        self.assertEqual(self.events[1]["embedded_count"], 2)
        self.session.sync_conversation_memory.assert_called_once_with()
        self.session.refresh_sources.assert_not_called()

    def test_archived_conversation_is_not_a_feedback_target(self):
        """Archived turns remain immutable from project feedback commands."""

        request = self.request()
        self.store.archive(self.conversation.conversation_id)
        self.engine.save_message_feedback(request)
        self.assertIn("error", self.events[-1])
        self.assertFalse((self.project.root / "memory" / "feedback.json")
                         .exists())
