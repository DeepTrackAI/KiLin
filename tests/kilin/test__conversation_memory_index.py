"""Test independent, incremental conversation-memory indexing and origins.

The tests use real project-owned checkpoints and feedback without models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin import rag
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.index import (
    memory_fingerprint, synchronize_memory,
)
from kilin.conversation_memory.persistence import memory_index_path
from kilin.conversations import ConversationStore
from kilin.project_type import project_type
from tests.kilin._rag_support import _turn


class FakeEmbedding:
    def __init__(self):
        self.embedded = []
        self.queries = []

    def embed_documents(self, texts, *, purpose="documents"):
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text):
        lower = text.lower()
        if "foldon" in lower:
            return [1.0, 0.0, 0.0]
        if "review" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class TestConversationMemoryIndex(unittest.TestCase):
    def setUp(self):
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        (self.root / "indexes").mkdir()
        self.store = ConversationStore(self.root)
        self.main = self.store.create()
        self.embedding = FakeEmbedding()
        original = project_type("code")
        self.config = replace(
            original,
            conversation_memory=replace(
                original.conversation_memory, enabled=True,
            ),
        )
        self.turns = [
            _turn("What is a foldon?", "A foldon is a candidate unit.",
                  parameters={"intent": "code_explanation"}),
            _turn("Give a file overview", "This is a file overview.",
                  parameters={"intent": "project_inventory"}),
            _turn("Good morning", "Hello!",
                  parameters={"intent": "small_talk"}),
        ]
        self._write(self.main, self.turns)

    def _write(self, info, turns):
        path = self.store.state_path(info)
        data = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(data)["turns"] = [
            rag._turn_to_data(turn) for turn in turns
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

    def _sync(self, model="model-A"):
        return synchronize_memory(
            self.root, self.config, self.embedding,
            model_identity=model,
        )

    def test_defaults_intents_questions_and_selected_answers(self):
        index = self._sync()
        self.assertTrue(index.changed)
        self.assertEqual(index.embedded_count, 1)
        self.assertEqual(
            {(entry.role, entry.text) for entry in index.records},
            {("user", "What is a foldon?")},
        )
        self.assertEqual(len(self.embedding.embedded), 1)
        self.assertTrue(memory_index_path(self.root).is_file())
        second = self._sync()
        self.assertFalse(second.changed)
        self.assertEqual(second.embedded_count, 0)
        self.assertEqual(len(self.embedding.embedded), 1)

    def test_explicit_intent_and_hard_never(self):
        feedback = FeedbackStore(self.root)
        feedback.update(
            self.main.conversation_id,
            self.turns[1].conversation_turn_uuid,
            "user", saved=True, reaction="neutral",
        )
        feedback.update(
            self.main.conversation_id,
            self.turns[2].conversation_turn_uuid,
            "user", saved=True, reaction="liked",
        )
        index = self._sync()
        self.assertEqual(
            {(entry.role, entry.text) for entry in index.records},
            {("user", "What is a foldon?"),
             ("user", "Give a file overview")},
        )
        self.assertNotIn("Good morning", self.embedding.embedded)

    def test_feedback_roles_dislikes_and_reversal(self):
        feedback = FeedbackStore(self.root)
        uuid = self.turns[0].conversation_turn_uuid
        feedback.update(
            self.main.conversation_id, uuid,
            "assistant", saved=True, reaction="liked",
        )
        first = self._sync()
        answers = [entry for entry in first.records
                   if entry.role == "assistant"]
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0].priority, 2)
        self.assertEqual(first.embedded_count, 2)
        feedback.update(
            self.main.conversation_id, uuid,
            "assistant", saved=True, reaction="disliked",
        )
        removed = self._sync()
        self.assertFalse(any(
            entry.role == "assistant" for entry in removed.records
        ))
        self.assertTrue(any(
            entry.role == "user" for entry in removed.records
        ))
        feedback.update(
            self.main.conversation_id, uuid,
            "assistant", saved=True, reaction="neutral",
        )
        restored = self._sync()
        self.assertEqual(len(restored.records), 2)
        # An excluded vector can be rebuilt, but never changes its identity.
        self.assertEqual(
            restored.records[0].fingerprint,
            memory_fingerprint(
                restored.records[0].role, restored.records[0].text
            ),
        )

    def test_branch_reuses_identity_and_duplicate_vectors(self):
        other = self.store.create(branch_from=self.main.conversation_id)
        shared = self.turns[0]
        duplicate = _turn(
            "What is a foldon?", "A different answer to that question.",
            parameters={"intent": "code_explanation"},
        )
        self._write(other, [*self.turns, duplicate])
        index = self._sync()
        question_entries = [
            entry for entry in index.records
            if entry.text == "What is a foldon?"
        ]
        self.assertEqual(len(question_entries), 2)
        self.assertEqual(
            question_entries[0].fingerprint,
            question_entries[1].fingerprint,
        )
        shared_entry = next(
            entry for entry in index.records
            if entry.turn_uuid == shared.conversation_turn_uuid
        )
        self.assertEqual(len(shared_entry.origins), 2)
        self.assertEqual(index.embedded_count, 1)
        self.assertEqual(len(index.vectors), 1)
        self.assertIn(
            duplicate.conversation_turn_uuid,
            [entry.turn_uuid for entry in question_entries],
        )

    def test_archives_and_deleted_files_are_not_memory_origins(self):
        other = self.store.create(branch_from=self.main.conversation_id)
        self._sync()
        archive_path = self.store.archive(self.main.conversation_id)[1]
        self.assertTrue(archive_path.is_file())
        with_branch = self._sync()
        self.assertEqual(len(with_branch.records), 1)
        self.assertEqual(
            with_branch.records[0].origins[0].conversation_id,
            other.conversation_id,
        )
        self.store.archive(other.conversation_id)
        removed = self._sync()
        self.assertEqual(removed.records, ())
        self.assertEqual(removed.vectors, {})

    def test_trace_only_edits_do_not_rewrite_index(self):
        self._sync()
        path = memory_index_path(self.root)
        saved_bytes = path.read_bytes()
        conversation_path = self.store.state_path(self.main)
        data = json.loads(conversation_path.read_text(encoding="utf-8"))
        data["diagnostic_only"] = {"progress": "retrieving"}
        conversation_path.write_text(json.dumps(data), encoding="utf-8")
        repeated = self._sync()
        self.assertFalse(repeated.changed)
        self.assertEqual(repeated.embedded_count, 0)
        self.assertEqual(path.read_bytes(), saved_bytes)
        self.assertEqual(len(self.embedding.embedded), 1)

    def test_content_policy_change_does_not_reuse_excluded_answers(self):
        feedback = FeedbackStore(self.root)
        feedback.update(
            self.main.conversation_id,
            self.turns[0].conversation_turn_uuid,
            "assistant", saved=False, reaction="liked",
        )
        old = self._sync()
        self.assertEqual(len(old.records), 2)
        self.config = replace(
            self.config,
            conversation_memory=replace(
                self.config.conversation_memory, answers="none",
            ),
        )
        changed = self._sync()
        self.assertEqual(len(changed.records), 1)
        self.assertTrue(changed.changed)
        self.assertEqual(changed.embedded_count, 0)

    def test_index_symlink_is_rejected_without_deleting_target(self):
        outside = self.root / "target.json"
        outside.write_text("private")
        memory_index_path(self.root).symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "cannot be a link"):
            self._sync()
        self.assertEqual(outside.read_text(), "private")

    def test_embedding_dimension_mismatch_preserves_previous_index(self):
        self._sync()
        previous = memory_index_path(self.root).read_bytes()
        self.turns.append(_turn(
            "What is a candidate mechanism?", "A new question.",
            parameters={"intent": "code_explanation"},
        ))
        self._write(self.main, self.turns)

        def wrong_dimension(texts, *, purpose="documents"):
            return [[1.0, 0.0] for _ in texts]

        self.embedding.embed_documents = wrong_dimension
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self._sync()
        self.assertEqual(memory_index_path(self.root).read_bytes(),
                         previous)

    def test_model_change_and_corrupt_index_rebuild(self):
        self._sync()
        current_count = len(self.embedding.embedded)
        changed = self._sync(model="model-B")
        self.assertEqual(changed.embedded_count, 1)
        self.assertEqual(len(self.embedding.embedded), current_count + 1)
        memory_index_path(self.root).write_text("corrupt")
        rebuilt = self._sync(model="model-B")
        self.assertTrue(rebuilt.changed)
        self.assertEqual(rebuilt.embedded_count, 1)
        self.assertEqual(len(rebuilt.records), 1)

    def test_disabled_has_no_io_or_embedding_operations(self):
        self.config = replace(
            self.config,
            conversation_memory=replace(
                self.config.conversation_memory, enabled=False,
            ),
        )
        result = self._sync()
        self.assertFalse(result.changed)
        self.assertFalse(memory_index_path(self.root).exists())
        self.assertEqual(self.embedding.embedded, [])

    def test_answer_none_forbids_explicit_like(self):
        feedback = FeedbackStore(self.root)
        feedback.update(
            self.main.conversation_id,
            self.turns[0].conversation_turn_uuid,
            "assistant", saved=True, reaction="liked",
        )
        self.config = replace(
            self.config,
            conversation_memory=replace(
                self.config.conversation_memory, answers="none",
            ),
        )
        result = self._sync()
        self.assertEqual([entry.role for entry in result.records], ["user"])
