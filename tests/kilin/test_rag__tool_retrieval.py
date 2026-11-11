"""Test the opt-in bounded retrieval service with deterministic indexes.

The shared service must return original chunks, preserve the selected
source ranking, enforce independent tool budgets, and avoid extra query
embeddings or model calls when cards are absent.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock

from kilin.retrievals.evidence_budget import EvidenceLimits
from kilin.retrievals.requests import RetrievalRequest
from kilin.retrievals.source_cards.index import (
    SourceCardVector,
    SourceCardVectorIndex,
)
from tests.kilin.test_rag__source_first import _session


class TestToolRetrieval(unittest.TestCase):

    def test_card_discovers_source_and_preserves_original_chunk(self):
        session = _session()
        request = RetrievalRequest(
            "memory", EvidenceLimits(2, 1, 2, 1000),
            card_kinds=("scientific_paper",),
        )
        result = session._retrieve_for_tool(request, "turn")
        self.assertEqual(result.sources[0], Path("c.pdf"))
        self.assertEqual(result.passages[0], session.chunks[4])
        self.assertEqual(result.passages[0].identifier, "c1")
        self.assertEqual(result.coverage["mode"], "bounded")
        session.embedding_session.embed_query.assert_called_once_with(
            "memory", trace_turn_id="turn"
        )

    def test_missing_cards_still_use_semantic_and_lexical(self):
        session = _session(cards=False)
        result = session._retrieve_for_tool(RetrievalRequest(
            "memory", EvidenceLimits(2, 1, 2, 1000),
        ))
        self.assertEqual(len(result.passages), 2)
        self.assertNotIn(Path("c.pdf"), result.sources)
        session.embedding_session.embed_query.assert_called_once()

    def test_exhaustive_coverage_and_final_limits(self):
        session = _session()
        request = RetrievalRequest(
            "memory", EvidenceLimits(1, 1, 1, 120, "exhaustive"),
        )
        result = session._retrieve_for_tool(request)
        self.assertLessEqual(len(result.passages), 1)
        self.assertLessEqual(result.coverage["estimated_evidence_tokens"],
                             120)
        self.assertFalse(result.coverage["complete"])
        self.assertIn("sources", result.coverage["limits_hit"])
        self.assertIn("a.pdf", result.coverage["missing_sources"])

    def test_explicit_scope_limits_final_evidence(self):
        session = _session()
        request = RetrievalRequest(
            "memory", EvidenceLimits(3, 1, 3, 1000),
            scope=(Path("a.pdf"), Path("b.pdf")),
        )
        result = session._retrieve_for_tool(request)
        self.assertTrue(result.passages)
        self.assertTrue(all(
            chunk.path in request.scope for chunk in result.passages
        ))

    def test_unknown_kind_rejected_without_query(self):
        session = _session()
        request = RetrievalRequest(
            "memory", EvidenceLimits(2, 1, 2, 1000),
            card_kinds=("unregistered_kind",),
        )
        with self.assertRaisesRegex(ValueError, "Unknown source-card kind"):
            session._retrieve_for_tool(request)
        session.embedding_session.embed_query.assert_not_called()

    def test_nonmatching_card_kind_is_not_used(self):
        session = _session()
        original = session.source_card_vectors.records[0]
        session.source_card_vectors = SourceCardVectorIndex(
            "embed:test", (SourceCardVector(
                original.path, "another_kind", original.card_sha256,
                original.text_sha256, original.embeddings,
            ),),
        )
        result = session._retrieve_for_tool(RetrievalRequest(
            "memory", EvidenceLimits(2, 1, 2, 1000),
            card_kinds=("scientific_paper",),
        ))
        self.assertNotIn(Path("c.pdf"), result.sources)

    def test_invalid_request_never_runs_search(self):
        session = _session()
        with self.assertRaises(TypeError):
            session._retrieve_for_tool("memory")
        session.embedding_session.embed_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
