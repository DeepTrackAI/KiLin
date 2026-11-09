"""Exercise source-first hard budgets and saved coverage reporting.

These tests use in-memory retrieval fixtures and a mocked answering session.
They verify the actual RAG turn path without downloading model weights.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from kilin import rag
from kilin.answering import RouteDecision
from kilin.retrievals.evidence_budget import (
    EvidenceLimits, evidence_tokens,
)
from tests.kilin import _rag_support as fixtures
from tests.kilin._rag_support import (
    _answer_patches, _evidence, _intent, _turn,
)
from tests.kilin.test_rag__source_first import _session
from tests.kilin.retrievals._support import chunk


class TestSourceFirstBudget(unittest.TestCase):

    def test_initial_retrieval_ignores_low_project_retrieval_count(self):
        session = _session()
        budget = EvidenceLimits(3, 2, 5, 5000)
        details = {}
        evidence = session._retrieve(
            "memory", retrieval_count=1,
            source_first=(3, True), source_limits=budget,
            source_report=details,
        )
        self.assertEqual(len({item.path for item in evidence}), 3)
        self.assertLessEqual(len(evidence), 5)
        self.assertEqual(
            len(details["selected_sources"]), 3
        )
        self.assertLessEqual(evidence_tokens(evidence), 5000)
        session.embedding_session.embed_query.assert_called_once()

    def test_oversized_chunks_do_not_bypass_expansion_budget(self):
        session = _session()
        budget = EvidenceLimits(3, 1, 3, 100)
        # The original c.pdf passage can be selected by its card but must
        # be excluded when its full provenance-bearing block cannot fit.
        session.chunks[-1] = replace(
            session.chunks[-1], text="Y" * 2000
        )
        session.indexes.semantic = replace(
            session.indexes.semantic,
            chunks=tuple(session.chunks),
        )
        from kilin.retrievals import build_lexical_index
        session.indexes.lexical = build_lexical_index(session.chunks)
        evidence = session._retrieve(
            "memory", source_first=(3, True), source_limits=budget,
        )
        self.assertLessEqual(len(evidence), 3)
        self.assertLessEqual(evidence_tokens(evidence), 100)
        self.assertNotIn("Y" * 2000, [item.text for item in evidence])
        self.assertEqual(
            [item.citation for item in evidence],
            list(range(1, len(evidence) + 1)),
        )

    def test_exhaustive_notice_reports_missing_indexed_source(self):
        session = fixtures._session()
        session.chunks = [
            chunk("a", "first", path="a.pdf"),
            chunk("b", "second", path="b.pdf"),
            chunk("c", "third", path="c.pdf"),
        ]
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=2, passages_per_source=1,
            max_passages=2, evidence_tokens=1000,
        )
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.return_value = (
                _evidence(1, path="a.pdf"),
                _evidence(2, path="b.pdf"),
            )
            answer = session.ask("Compare all papers", use_router=False)
        self.assertIn("Coverage incomplete", answer)
        self.assertIn("2 of 3 requested sources", answer)
        data = session.turns[-1].parameters["coverage"]
        self.assertEqual(data["mode"], "exhaustive")
        self.assertEqual(data["missing_sources"], ["c.pdf"])
        self.assertFalse(data["complete"])
        self.assertEqual(session.turns[-1].answer, answer)
        self.assertEqual(session.chat_session.reply.call_count, 1)
        self.assertEqual(mocks.retrieve.call_count, 1)
        self.assertEqual(
            mocks.retrieve.call_args.kwargs["source_limits"].max_passages,
            2,
        )

    def test_all_sources_overrides_inferred_single_document_focus(self):
        session = fixtures._session()
        session.chunks = [
            chunk("a", "first", path="a.pdf"),
            chunk("b", "second", path="b.pdf"),
        ]
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=1, max_passages=1, evidence_tokens=1000,
        )
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.return_value = (_evidence(1, path="a.pdf"),)
            with patch.object(
                rag, "document_scope",
                return_value=((Path("a.pdf"),), "named"),
            ):
                answer = session.ask(
                    "Summarize all papers", use_router=False
                )
        self.assertEqual(
            mocks.retrieve.call_args.kwargs["scope"], ()
        )
        self.assertIn("1 of 2 requested sources", answer)

    def test_bounded_answer_does_not_claim_exhaustive_coverage(self):
        session = fixtures._session()
        session.chunks = [
            chunk("a", "first", path="a.pdf"),
            chunk("b", "second", path="b.pdf"),
        ]
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=1, max_passages=1, evidence_tokens=1000,
        )
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.return_value = (_evidence(1, path="a.pdf"),)
            answer = session.ask("What is the topic?", use_router=False)
        self.assertNotIn("Coverage incomplete", answer)
        self.assertEqual(
            session.turns[-1].parameters["coverage"]["mode"],
            "bounded",
        )

    def test_unindexed_sources_are_reported_separately(self):
        session = fixtures._session()
        session.chunks = [chunk("a", "first", path="a.pdf")]
        session._current_source_manifest = {
            "files": {
                "a.pdf": {"size_bytes": 10},
                "scanned.pdf": {"size_bytes": 1000},
            },
        }
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=4, max_passages=4, evidence_tokens=1000,
            coverage="exhaustive",
        )
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.return_value = (_evidence(1, path="a.pdf"),)
            answer = session.ask("Discuss the findings", use_router=False)
        self.assertIn("Some sources had no searchable text", answer)
        data = session.turns[-1].parameters["coverage"]
        self.assertEqual(data["requested_sources"], 2)
        self.assertEqual(data["unindexed_sources"], ["scanned.pdf"])

    def test_source_count_limit_is_reported_from_full_ranking(self):
        session = _session()
        original_retrieve = session._retrieve
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=1, max_passages=2, evidence_tokens=1000,
        )
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.side_effect = original_retrieve
            session.ask("Compare all papers", use_router=False)
        coverage = session.turns[-1].parameters["coverage"]
        self.assertIn("sources", coverage["limits_hit"])
        self.assertEqual(coverage["requested_sources"], 3)
        self.assertEqual(coverage["represented_sources"], 1)
        session.embedding_session.embed_query.assert_called_once()

    def test_named_document_comparison_uses_new_per_source_setting(self):
        session = fixtures._session()
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=8, passages_per_source=2,
        )
        with _answer_patches(session, intent) as mocks:
            session.ask("Compare this paper", use_router=False)
        self.assertEqual(
            mocks.retrieve.call_args.kwargs["source_first"],
            (8, True),
        )

    def test_two_papers_without_all_does_not_claim_every_source(self):
        session = fixtures._session()
        session.chunks = [
            chunk("a", "first", path="a.pdf"),
            chunk("b", "second", path="b.pdf"),
            chunk("c", "third", path="c.pdf"),
        ]
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=2, max_passages=2, evidence_tokens=1000,
        )
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.return_value = (_evidence(1, path="a.pdf"),)
            answer = session.ask(
                "Compare two papers", use_router=False
            )
        self.assertNotIn("Coverage incomplete", answer)
        self.assertEqual(
            session.turns[-1].parameters["coverage"]["mode"],
            "bounded",
        )

    def test_related_turn_evidence_cannot_overrun_final_cap(self):
        session = fixtures._session()
        session.chunks = [chunk("a", "first", path="a.pdf")]
        session.turns = [
            _turn(
                answer="Earlier finding [1]",
                evidence=(_evidence(1, path="b.pdf"),),
            ),
        ]
        intent = replace(
            _intent(related=True), retrieval_strategy="source_first",
            max_sources=1, max_passages=1, evidence_tokens=1000,
        )
        decision = RouteDecision(
            intent.intent_id, False, related_turn_ids=(1,),
        )
        with _answer_patches(session, intent, decision) as mocks:
            mocks.retrieve.return_value = (_evidence(1, path="a.pdf"),)
            session.ask("How does this relate?", use_router=False)
        evidence = session.turns[-1].evidence
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].path, Path("a.pdf"))
        self.assertEqual(evidence[0].citation, 1)

    def test_exhaustive_notice_is_streamed_once_after_the_reply(self):
        session = fixtures._session()
        session.chunks = [
            chunk("a", "first", path="a.pdf"),
            chunk("b", "second", path="b.pdf"),
        ]
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_sources=1, max_passages=1, evidence_tokens=1000,
        )
        output = []
        with _answer_patches(session, intent) as mocks:
            mocks.retrieve.return_value = (_evidence(1, path="a.pdf"),)
            answer = session.ask(
                "Check every paper", use_router=False,
                token_callback=output.append,
            )
        self.assertTrue(answer.endswith(output[-1].strip()))
        self.assertEqual(sum("Coverage incomplete" in x for x in output), 1)


if __name__ == "__main__":
    unittest.main()
