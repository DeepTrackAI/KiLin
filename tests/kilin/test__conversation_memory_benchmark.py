"""Deterministic retrieval-quality fixtures for project conversation memory.

The fixture embeddings are deliberately synthetic. These assertions exercise
ordering, hard exclusions, and history verification reproducibly without
claiming an empirical recall rate for a user's local embedding model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.conversation_memory.index import (
    MemoryIndex, MemoryOrigin, MemoryRecord, memory_fingerprint,
)
from kilin.conversation_memory.retrieval import search_memory


class _QueryEmbedding:
    """Return the same fixed query direction for every fixture."""

    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [1.0, 0.0]


class TestConversationMemoryBenchmark(unittest.TestCase):
    def setUp(self):
        self.embedding = _QueryEmbedding()

    @staticmethod
    def _record(number, text, *, role="user", priority=0, turn=None):
        identifier = (
            f"00000000-0000-4000-8000-{turn or number:012x}"
        ).replace("-", "")
        return MemoryRecord(
            identifier, role, text, "2026-11-15T12:00:00+00:00",
            memory_fingerprint(role, text), priority,
            (MemoryOrigin("conversation_1", number),),
        )

    def _search(self, question, items, *, topic=None):
        records = tuple(item[0] for item in items)
        index = MemoryIndex("synthetic", records, {
            item.fingerprint: vector for item, vector in items
        })
        return search_memory(
            question, index, self.embedding, max_hits=5,
            max_tokens=2200, recall_topic=topic,
        )

    def test_keyword_anchor_rejects_equal_cosine_distractors(self):
        target = self._record(1, "Optical tweezers calibration")
        irrelevant = self._record(2, "Brainstorm art exhibition")
        another = self._record(3, "Cell segmentation pipeline")
        results = self._search("optical tweezers", (
            (target, (1.0, 0.0)),
            (irrelevant, (1.0, 0.0)),
            (another, (1.0, 0.0)),
        ))
        self.assertEqual([item.record for item in results], [target])

    def test_semantic_retrieval_remains_when_lexical_anchor_absent(self):
        relevant = self._record(1, "Particle tracking methods")
        irrelevant = self._record(2, "Grant formatting rules")
        results = self._search("analyse cell migration", (
            (relevant, (1.0, 0.0)),
            (irrelevant, (0.0, 1.0)),
        ))
        self.assertEqual([item.record for item in results], [relevant])

    def test_like_cannot_revive_irrelevant_memory(self):
        liked = self._record(
            1, "Financial reimbursement forms", priority=2
        )
        results = self._search("optical tweezers", (
            (liked, (0.0, 1.0)),
        ))
        self.assertEqual(results, ())

    def test_historical_question_includes_eligible_paired_answer(self):
        question = self._record(
            1, "Did we consider optical tweezers?", turn=7
        )
        response = self._record(
            1, "We did not establish a conclusion.",
            role="assistant", turn=7,
        )
        unrelated = self._record(2, "A grant proposal for microscopy")
        results = self._search(
            "Did we discuss optical tweezers?", (
                (question, (0.0, 1.0)),
                (response, (0.0, 1.0)),
                (unrelated, (1.0, 0.0)),
            ), topic="optical tweezers",
        )
        self.assertEqual(
            [item.record for item in results], [question, response]
        )
        self.assertEqual(self.embedding.calls, 0)

    def test_partial_topic_overlap_does_not_verify_full_history(self):
        weak = self._record(1, "Optical tweezers were mentioned.")
        results = self._search(
            "Did we discuss optical tweezers force calibration?",
            ((weak, (1.0, 0.0)),),
            topic="optical tweezers force calibration",
        )
        self.assertEqual(results, ())
        self.assertEqual(self.embedding.calls, 0)


if __name__ == "__main__":
    unittest.main()
