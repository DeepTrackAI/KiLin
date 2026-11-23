"""Test verified historical lookups through the real answering state machine.

No language model or project-source retrieval is needed by these fixtures.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import rag
from kilin.answering import RouteDecision
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversations import ConversationStore
from kilin.project_type import project_type
from tests.kilin._rag_support import _answer_patches, _session, _turn
from tests.kilin.test__conversation_memory_index import FakeEmbedding


class TestRAGHistoricalRecall(unittest.TestCase):
    def setUp(self):
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        (self.root / "indexes").mkdir()
        self.store = ConversationStore(self.root)
        previous = self.store.create()
        self.previous = previous
        self.turn = _turn(
            "Could we use optical tweezers for these experiments?",
            "We might investigate it, without reaching a conclusion.",
            parameters={"intent": "writing"},
        )
        path = self.store.state_path(previous)
        data = json.loads(path.read_text())
        conversation_state(data)["turns"] = [
            rag._turn_to_data(self.turn)
        ]
        path.write_text(json.dumps(data), encoding="utf-8")
        current = self.store.create()
        self.session = _session(self.root)
        self.session.conversation_path = self.store.state_path(current)
        self.session.project.profile.type_id = "writing"
        self.embedding = FakeEmbedding()
        self.session.embedding_session.embed_query.side_effect = (
            self.embedding.embed_query
        )
        self.session.embedding_session.embed_documents.side_effect = (
            self.embedding.embed_documents
        )
        self.config = project_type("writing")
        self.intent = self.config.intents["research"]

    def _ask(self, question):
        with patch.object(
            self.session, "_behavior_config", return_value=self.config
        ), _answer_patches(
            self.session, self.intent, RouteDecision("research", True)
        ) as mocks:
            self.session.ask(question)
        return mocks, self.session.chat_session.reply.call_args.args[0]

    def test_lookup_uses_saved_question_not_source_search(self):
        mocks, prompt = self._ask("Did we discuss optical tweezers?")
        mocks.retrieve.assert_not_called()
        self.assertIn("CONVERSATION MEMORY", prompt)
        self.assertIn("optical tweezers", prompt)
        self.assertIn("HISTORICAL LOOKUP", prompt)
        self.assertIn("researcher raised a topic", prompt)
        self.assertEqual(self.embedding.queries, [])
        turn = self.session.turns[-1]
        self.assertTrue(turn.parameters["historical_lookup"])
        self.assertEqual(turn.parameters["retrieval_mode"], "none")
        self.assertEqual(turn.parameters["memory_references"][0]["role"],
                         "user")

    def test_disliked_answer_does_not_create_false_conclusion(self):
        FeedbackStore(self.root).update(
            self.previous.conversation_id,
            self.turn.conversation_turn_uuid, "assistant",
            saved=True, reaction="disliked",
        )
        mocks, prompt = self._ask("Did we discuss optical tweezers?")
        mocks.retrieve.assert_not_called()
        self.assertIn("optical tweezers", prompt)
        self.assertNotIn("without reaching a conclusion", prompt)
        self.assertEqual(
            [item["role"] for item in self.session.turns[-1]
             .parameters["memory_references"]],
            ["user"],
        )

    def test_visible_saved_turn_still_counts_as_prior_discussion(self):
        # Ordinary retrieval avoids repeating live chat history, but an
        # explicit historical check must not suppress its matching record.
        self.session.turns = [self.turn]
        mocks, prompt = self._ask("Did we discuss optical tweezers?")
        mocks.retrieve.assert_not_called()
        self.assertIn("CONVERSATION MEMORY", prompt)
        self.assertIn("Could we use optical tweezers", prompt)
        self.assertNotIn("No matching eligible saved messages", prompt)
        self.assertEqual(
            self.session.turns[-1].parameters["memory_references"][0]
            ["turn_uuid"], self.turn.conversation_turn_uuid,
        )

    def test_no_match_means_unverified_not_never_discussed(self):
        mocks, prompt = self._ask("Did we discuss laser cooling?")
        mocks.retrieve.assert_not_called()
        self.assertIn("No matching eligible saved messages", prompt)
        self.assertIn("cannot verify", prompt)
        self.assertNotIn(
            "memory_references", self.session.turns[-1].parameters
        )
        self.assertEqual(self.embedding.queries, [])

    def test_normal_scientific_question_retains_source_search(self):
        mocks, prompt = self._ask(
            "What is the evidence for optical tweezers?"
        )
        mocks.retrieve.assert_called_once()
        self.assertNotIn("HISTORICAL LOOKUP", prompt)
        self.assertEqual(self.session.turns[-1].parameters[
            "retrieval_mode"], "search")


if __name__ == "__main__":
    unittest.main()
