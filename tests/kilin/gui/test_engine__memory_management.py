"""Test project-scoped inventory and exclusion commands without models.

The core owns all file and revision validation; the GUI only sends requests.

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
from kilin.conversations import ConversationStore
from kilin.gui.engine import ChatEngine
from kilin.workspace import prepare_sources_workspace
from tests.kilin._rag_support import _turn


class TestEngineMemoryManagement(unittest.TestCase):
    def setUp(self):
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        source = self.root / "note.md"
        source.write_text("Local notes.", encoding="utf-8")
        project = prepare_sources_workspace(
            [source], "writing", self.root / "workspace"
        )
        self.project_id = str(project.root)
        self.store = ConversationStore(project.root)
        self.conversation = self.store.create()
        self.turn = _turn(
            "Did we discuss foldon experiments?", "No decision yet.",
            parameters={"intent": "writing"},
        )
        path = self.store.state_path(self.conversation)
        data = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(data)["turns"] = [rag._turn_to_data(self.turn)]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.checkpoint = path.read_bytes()
        self.events = []
        self.engine = ChatEngine(self.root, self.events.append)
        self.session = MagicMock()
        self.session.project = project
        self.engine.session = self.session
        self.feedback = FeedbackStore(project.root)

    def _request(self, **changes):
        result = {
            "project": self.project_id,
            "conversation_id": self.conversation.conversation_id,
            "turn_uuid": self.turn.conversation_turn_uuid,
            "role": "user", "excluded": True,
            "revision": self.feedback.load().revision,
            "request_id": 1,
        }
        result.update(changes)
        return result

    def test_list_includes_unindexed_messages_without_embedding(self):
        self.engine.memory_messages(self.project_id)
        event = self.events[-1]
        self.assertEqual(event["event"], "memory_messages")
        self.assertEqual(event["total"], 2)
        self.assertEqual(event["offset"], 0)
        self.assertEqual({m["status"] for m in event["messages"]},
                         {"eligible", "policy"})
        self.session.sync_conversation_memory.assert_not_called()
        self.engine.memory_messages(self.project_id, offset=1)
        self.assertEqual(len(self.events[-1]["messages"]), 1)

    def test_memory_summaries_read_current_trace_without_saving(self):
        """Correlate one read-only summary request to the active project."""

        self.engine.conversation_store = self.store
        request = {
            "request_id": 41,
            "references": [{
                "turn_uuid": self.turn.conversation_turn_uuid,
                "role": "user",
                "origins": [{
                    "conversation_id": self.conversation.conversation_id,
                    "turn_number": 1,
                }],
            }],
        }
        self.engine.memory_summaries(self.project_id, request)
        result = self.events[-1]
        self.assertEqual(result["event"], "memory_summaries")
        self.assertEqual(result["request_id"], 41)
        self.assertEqual(result["summaries"][0]["text"], self.turn.question)
        self.assertEqual(self.store.state_path(
            self.conversation
        ).read_bytes(), self.checkpoint)
        self.session.sync_conversation_memory.assert_not_called()
        for project, changed in (
            (str(self.root / "foreign"), request),
            (self.project_id, {**request, "request_id": 0}),
            (self.project_id, {**request, "references": "invalid"}),
        ):
            self.engine.memory_summaries(project, changed)
            self.assertIn("error", self.events[-1])

    def test_exclude_preserves_reactions_and_checkpoint(self):
        self.feedback.update(
            self.conversation.conversation_id,
            self.turn.conversation_turn_uuid, "user",
            saved=True, reaction="liked",
        )
        self.engine.set_memory_exclusion(self._request())
        event = next(e for e in self.events
                     if e["event"] == "memory_exclusion")
        self.assertEqual(event["request_id"], 1)
        self.assertTrue(event["entries"][self.turn.conversation_turn_uuid]
                        ["user"]["excluded"])
        state = self.feedback.load().get(
            self.turn.conversation_turn_uuid, "user"
        )
        self.assertTrue(state.saved)
        self.assertEqual(state.reaction, "liked")
        self.assertTrue(state.excluded)
        self.assertEqual(self.store.state_path(
            self.conversation
        ).read_bytes(), self.checkpoint)
        self.engine.memory_messages(self.project_id)
        self.assertIn("excluded", [
            item["status"] for item in self.events[-1]["messages"]
        ])
        self.engine.set_memory_exclusion(self._request(excluded=False))
        self.assertFalse(self.feedback.load().get(
            self.turn.conversation_turn_uuid, "user"
        ).excluded)
        self.session.refresh_sources.assert_not_called()
        self.session.sync_conversation_memory.assert_not_called()

    def test_stale_write_and_foreign_project_cannot_mutate(self):
        first = self._request()
        self.engine.set_memory_exclusion(first)
        before = self.feedback.path.read_bytes()
        for changes in (
            {"excluded": False},
            {"project": str(self.root / "foreign")},
            {"conversation_id": "missing"},
            {"turn_uuid": "not a uuid"},
            {"role": "system"},
            {"excluded": "false"},
            {"request_id": 0},
            {"unexpected": "field"},
        ):
            with self.subTest(changes=changes):
                self.engine.set_memory_exclusion({**first, **changes})
                self.assertEqual(self.events[-1]["event"],
                                 "memory_exclusion")
                self.assertIn("error", self.events[-1])
                self.assertEqual(self.feedback.path.read_bytes(), before)
        self.engine.memory_messages(str(self.root / "foreign"))
        self.assertIn("error", self.events[-1])

    def test_generation_and_archival_block_mutations(self):
        self.engine._answering = True
        self.engine.set_memory_exclusion(self._request())
        self.assertIn("error", self.events[-1])
        self.engine._answering = False
        self.store.archive(self.conversation.conversation_id)
        self.engine.set_memory_exclusion(self._request())
        self.assertIn("error", self.events[-1])
        self.assertFalse(self.feedback.path.exists())


if __name__ == "__main__":
    unittest.main()
