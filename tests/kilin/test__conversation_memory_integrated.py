"""Verify historical recall, relevance, and memory index lifecycle.

The ranking fixtures are deterministic: they exercise relevance and
provenance independently of a particular local embedding model.

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
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.index import (
    MemoryIndex, MemoryOrigin, MemoryRecord, memory_fingerprint,
    synchronize_memory,
)
from kilin.conversation_memory.models import new_turn_uuid
from kilin.conversation_memory.recall import recall_topic
from kilin.conversation_memory.retrieval import search_memory
from kilin.conversation_memory.persistence import memory_index_path
from kilin.conversations import ConversationStore
from kilin.project_type import project_type
from tests.kilin._rag_support import _turn
from tests.kilin.test__conversation_memory_index import FakeEmbedding


class TestConversationMemoryRecall(unittest.TestCase):
    def setUp(self):
        self.embedding = FakeEmbedding()
        self.records = tuple(self._record(text, role) for text, role in (
            ("Did we consider optical tweezers for embryo experiments?",
             "user"),
            ("We could use a force clamp instrument.", "assistant"),
            ("Brainstorm a holographic microscopy project.", "user"),
            ("Consider high-throughput calibration.", "assistant"),
        ))
        self.index = MemoryIndex("model-A", self.records, {
            record.fingerprint: tuple(FakeEmbedding._vector(record.text))
            for record in self.records
        })

    @staticmethod
    def _record(text, role):
        return MemoryRecord(
            new_turn_uuid(), role, text, "2026-11-15T10:00:00+00:00",
            memory_fingerprint(role, text), 0,
            (MemoryOrigin("conversation_1", 1),),
        )

    def test_only_explicit_first_person_history_questions_match(self):
        fixtures = {
            "Did we discuss optical tweezers?": "optical tweezers",
            "Have we talked about embryo tracking?": "embryo tracking",
            "Do you remember if we discussed optical tweezers?":
                "optical tweezers",
            "What did we conclude about sample drift?": "sample drift",
            "Explain optical tweezers": None,
            "What does the article discuss about optical tweezers?": None,
            "Can we discuss optical tweezers?": None,
            "Did we?": None,
        }
        for question, expected in fixtures.items():
            with self.subTest(question=question):
                self.assertEqual(recall_topic(question), expected)

    def test_history_lookup_uses_saved_message_not_embedding(self):
        hits = search_memory(
            "Did we discuss optical tweezers?", self.index,
            self.embedding, recall_topic="optical tweezers",
            max_hits=4, max_tokens=1500,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].record.role, "user")
        self.assertIn("optical tweezers", hits[0].text)
        self.assertEqual(self.embedding.queries, [])

    def test_unrelated_history_is_not_proof_of_prior_discussion(self):
        hits = search_memory(
            "Did we discuss free energy landscapes?", self.index,
            self.embedding, recall_topic="free energy landscapes",
            max_hits=4, max_tokens=1500,
        )
        self.assertEqual(hits, ())
        self.assertEqual(self.embedding.queries, [])
        prompt = rag._answer_prompt(
            "Did we discuss free energy landscapes?", (),
            memory_recall_checked=True,
        )
        self.assertIn("cannot verify", prompt)
        self.assertIn("No matching eligible", prompt)
        self.assertNotIn("never discussed", prompt.split(
            "HISTORICAL LOOKUP:")[0])

    def test_user_question_is_not_an_accepted_conclusion(self):
        hits = search_memory(
            "Did we discuss optical tweezers?", self.index,
            self.embedding, recall_topic="optical tweezers",
            max_hits=4, max_tokens=1500,
        )
        prompt = rag._answer_prompt(
            "Did we discuss optical tweezers?", (),
            memory_hits=hits, memory_recall_checked=True,
        )
        self.assertIn("researcher raised a topic", prompt)
        self.assertIn("not an accepted decision", prompt)
        self.assertNotIn(
            "PROJECT SOURCE EVIDENCE (numbered", prompt
        )

    def test_excluded_current_turn_is_not_historical_recall(self):
        hits = search_memory(
            "Did we discuss optical tweezers?", self.index,
            self.embedding, recall_topic="optical tweezers",
            excluded_uuids=frozenset((self.records[0].turn_uuid,)),
            max_hits=4, max_tokens=1500,
        )
        self.assertEqual(hits, ())

    def test_lexical_match_beats_coarse_semantic_distractors(self):
        class CoarseEmbedding:
            def embed_query(self, text):
                return [1.0, 0.0]

        data = (
            self._record("Optical tweezers force measurements", "user"),
            self._record("A microscopy calibration brainstorm", "user"),
            self._record("A cell tracking discussion", "user"),
        )
        index = MemoryIndex("coarse", data, {
            item.fingerprint: (1.0, 0.0) for item in data
        })
        hits = search_memory(
            "optical tweezers", index, CoarseEmbedding(),
            max_hits=4, max_tokens=1200,
        )
        self.assertEqual([hit.record for hit in hits], [data[0]])


class TestMemoryLifecycle(unittest.TestCase):
    def setUp(self):
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        (self.root / "indexes").mkdir()
        self.store = ConversationStore(self.root)
        self.first = self.store.create()
        self.turn = _turn(
            "Could optical tweezers measure folding?",
            "It may be worth testing.",
            parameters={"intent": "writing"},
        )
        self._write(self.first, [self.turn])
        self.embedding = FakeEmbedding()
        self.config = project_type("writing")

    def _write(self, info, turns):
        path = self.store.state_path(info)
        saved = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(saved)["turns"] = [
            rag._turn_to_data(item) for item in turns
        ]
        path.write_text(json.dumps(saved), encoding="utf-8")

    def _sync(self):
        return synchronize_memory(
            self.root, self.config, self.embedding,
            model_identity="model-A",
        )

    def test_exclusion_survives_save_like_and_index_rebuild(self):
        uuid = self.turn.conversation_turn_uuid
        feedback = FeedbackStore(self.root)
        current = feedback.update(
            self.first.conversation_id, uuid, "assistant",
            saved=True, reaction="liked",
        )
        self.assertEqual(len(self._sync().records), 2)
        after = feedback.update(
            self.first.conversation_id, uuid, "assistant",
            saved=None, reaction=None, excluded=True,
            expected_revision=current.revision,
        )
        self.assertTrue(after.get(uuid, "assistant").excluded)
        self.assertTrue(after.get(uuid, "assistant").saved)
        indexed = self._sync()
        self.assertEqual([r.role for r in indexed.records], ["user"])
        feedback.update(
            self.first.conversation_id, uuid, "assistant",
            saved=True, reaction="liked",
        )
        self.assertEqual([r.role for r in self._sync().records], ["user"])
        restored = feedback.update(
            self.first.conversation_id, uuid, "assistant",
            saved=None, reaction=None, excluded=False,
        )
        self.assertFalse(restored.get(uuid, "assistant").excluded)
        self.assertEqual(len(self._sync().records), 2)

    def test_branch_exclusion_is_shared_by_turn_identity(self):
        branch = self.store.create(
            branch_from=self.first.conversation_id, turn_count=1
        )
        feedback = FeedbackStore(self.root)
        feedback.update(
            branch.conversation_id, self.turn.conversation_turn_uuid,
            "user", saved=None, reaction=None, excluded=True,
        )
        self.assertEqual(self._sync().records, ())
        self.store.archive(branch.conversation_id)
        self.assertEqual(self._sync().records, ())

    def test_diagnostic_only_changes_are_read_once_per_process(self):
        self._sync()
        path = self.store.state_path(self.first)
        data = json.loads(path.read_text())
        data["diagnostics"] = {"not_a_turn": "progress"}
        path.write_text(json.dumps(data), encoding="utf-8")
        import kilin.conversation_memory.index as index_module
        with patch.object(
            index_module, "_read_records",
            wraps=index_module._read_records,
        ) as reading:
            first = self._sync()
            second = self._sync()
        self.assertFalse(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(reading.call_count, 1)
        self.assertEqual(self.embedding.embedded.count(
            "Could optical tweezers measure folding?"
        ), 1)

    def test_checkpoint_edit_invalidates_diagnostic_stat_cache(self):
        self._sync()
        path = self.store.state_path(self.first)
        data = json.loads(path.read_text())
        data["diagnostics"] = {"progress": 1}
        path.write_text(json.dumps(data), encoding="utf-8")
        self._sync()
        data = json.loads(path.read_text())
        conversation_state(data)["turns"][0]["question"] = (
            "Could optical tweezers measure protein kinetics?"
        )
        path.write_text(json.dumps(data), encoding="utf-8")
        index = self._sync()
        self.assertTrue(index.changed)
        self.assertIn("protein kinetics", index.records[0].text)
        self.assertTrue(memory_index_path(self.root).exists())


if __name__ == "__main__":
    unittest.main()
