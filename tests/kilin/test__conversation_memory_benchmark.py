"""Deterministic retrieval-quality fixtures for project conversation memory.

The fixture embeddings are deliberately synthetic. These assertions exercise
ordering, hard exclusions, and history verification reproducibly without
claiming an empirical recall rate for a user's local embedding model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import unittest

from kilin.conversation_memory.index import (
    MemoryIndex, MemoryOrigin, MemoryRecord, memory_fingerprint,
)
from kilin.conversation_memory.retrieval import search_memory
from kilin.conversation_memory.retrieval import render_memory_hits
from kilin.retrievals.document_reading import estimate_tokens


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

    def test_four_matching_questions_return_two_complete_turns(self):
        """Do not fill the four-hit budget with questions alone."""

        items = []
        for number in range(1, 5):
            question = self._record(
                number, f"Did we plan optical tweezers experiment {number}?"
            )
            answer = self._record(
                number, f"The proposed option was plan {number}.",
                role="assistant",
            )
            items.extend(((question, (0.0, 1.0)),
                          (answer, (0.0, 1.0))))
        records = tuple(item[0] for item in items)
        index = MemoryIndex("synthetic", records, {
            record.fingerprint: vector for record, vector in items
        })
        hits = search_memory(
            "What did we decide about optical tweezers?", index,
            self.embedding, max_hits=4, max_tokens=2200,
            recall_topic="optical tweezers",
        )
        self.assertEqual([hit.record.role for hit in hits], [
            "user", "assistant", "user", "assistant",
        ])
        self.assertEqual(hits[0].record.turn_uuid,
                         hits[1].record.turn_uuid)
        self.assertEqual(hits[2].record.turn_uuid,
                         hits[3].record.turn_uuid)
        self.assertEqual(self.embedding.calls, 0)

    def test_unavailable_replies_do_not_appear_as_decisions(self):
        """Represent excluded and never-indexed replies as absent."""

        question = self._record(
            1, "Did we consider optical tweezers?"
        )
        # The reply exists in the source conversation but was omitted
        # from the eligible memory index by the feedback/index policy.
        unrelated = self._record(
            2, "A different microscope discussion", role="assistant"
        )
        hits = self._search(
            "What did we decide about optical tweezers?", (
                (question, (1.0, 0.0)),
                (unrelated, (1.0, 0.0)),
            ), topic="optical tweezers",
        )
        self.assertEqual([hit.record for hit in hits], [question])

    def test_historical_pairs_fit_the_hit_and_token_limits(self):
        """Truncate lengthy content without losing the paired reply."""

        question = self._record(
            1, "Did we consider optical tweezers? " + "context " * 350
        )
        answer = self._record(
            1, "We proposed calibration first. " + "details " * 350,
            role="assistant",
        )
        items = ((question, (1.0, 0.0)), (answer, (1.0, 0.0)))
        records = tuple(item[0] for item in items)
        index = MemoryIndex("synthetic", records, {
            record.fingerprint: vector for record, vector in items
        })
        hits = search_memory(
            "What did we decide about optical tweezers?", index,
            self.embedding, max_hits=2, max_tokens=160,
            recall_topic="optical tweezers",
        )
        self.assertEqual([hit.record for hit in hits], [question, answer])
        self.assertLessEqual(
            estimate_tokens(render_memory_hits(hits)), 160,
        )
        self.assertIn("calibration", hits[1].text)
        self.assertTrue(hits[0].text.endswith("..."))
        self.assertTrue(hits[1].text.endswith("..."))
        self.assertEqual(search_memory(
            "Did we discuss optical tweezers?", index,
            self.embedding, max_hits=2, max_tokens=10,
            recall_topic="optical tweezers",
        ), ())

    def test_one_hit_policy_cannot_infer_a_missing_reply(self):
        """When configured for one hit, present only the question."""

        question = self._record(1, "Optical tweezers test proposal")
        answer = self._record(
            1, "The team did not approve it.", role="assistant"
        )
        index = MemoryIndex("synthetic", (question, answer), {})
        hits = search_memory(
            "Did we discuss optical tweezers?", index, self.embedding,
            max_hits=1, max_tokens=700, recall_topic="optical tweezers",
        )
        self.assertEqual([hit.record for hit in hits], [question])

    def test_tight_token_budget_still_verifies_matching_question(self):
        """A pair too large for metadata budget must not mean no match."""

        question = self._record(1, "Optical tweezers test proposal")
        answer = self._record(
            1, "We suggested a controlled calibration.", role="assistant"
        )
        long_origin = MemoryOrigin("conversation_" + "a" * 120, 1)
        question = replace(question, origins=(long_origin,))
        answer = replace(answer, origins=(long_origin,))
        index = MemoryIndex("synthetic", (question, answer), {})
        hits = search_memory(
            "Did we discuss optical tweezers?", index, self.embedding,
            max_hits=2, max_tokens=128,
            recall_topic="optical tweezers",
        )
        self.assertEqual([hit.record for hit in hits], [question])
        self.assertLessEqual(
            estimate_tokens(render_memory_hits(hits)), 128,
        )

    def test_relevant_answer_without_matching_question_is_standalone(self):
        """Do not attach a reply to a different turn's question."""

        question = self._record(1, "Did we discuss optical tweezers?")
        answer = self._record(
            2, "Optical tweezers could help.", role="assistant"
        )
        hits = self._search(
            "Did we discuss optical tweezers?", (
                (question, (1.0, 0.0)), (answer, (1.0, 0.0)),
            ), topic="optical tweezers",
        )
        self.assertEqual([hit.record for hit in hits], [question, answer])
        self.assertNotEqual(
            hits[0].record.turn_uuid, hits[1].record.turn_uuid,
        )


if __name__ == "__main__":
    unittest.main()
