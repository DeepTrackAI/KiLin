"""Exercise source-first retrieval in RAG without loading real models.

Deterministic search fixtures verify the fourth ranking channel, scoping,
and original passage evidence while keeping legacy strategies independent.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from kilin import rag
from kilin.retrievals import build_lexical_index
from kilin.retrievals.code_graph import CodeGraph
from kilin.retrievals.source_cards.index import (
    SourceCardVector,
    SourceCardVectorIndex,
)
from tests.kilin import _rag_support as fixtures
from tests.kilin._rag_support import _answer_patches, _intent
from tests.kilin.retrievals._support import chunk


def _session(*, cards: bool = True) -> rag.RAGSession:
    """Return a session with a card-ranked but passage-distant source.

    Parameters
    ----------
    cards: bool, optional
        Whether the distant source has a matching stored card.

    Returns
    -------
    RAGSession
        Deterministic in-memory index and one mocked query embedding.

    """

    session = fixtures._session()
    session.chunks = [
        chunk("a1", "memory experiments", path="a.pdf"),
        chunk("a2", "memory tests", path="a.pdf"),
        chunk("b1", "memory control", path="b.pdf"),
        chunk("b2", "memory assessment", path="b.pdf"),
        chunk("c1", "unrelated glucose", path="c.pdf"),
    ]
    session.indexes = SimpleNamespace(
        semantic=rag.SemanticIndex(
            tuple(session.chunks),
            (
                (1.0, 0.0), (0.9, 0.1),
                (0.8, 0.2), (0.7, 0.3), (0.0, 1.0),
            ),
        ),
        lexical=build_lexical_index(session.chunks),
        code_graph=CodeGraph((), ()),
    )
    session.embedding_session = Mock()
    session.embedding_session.embed_query.return_value = [1.0, 0.0]
    session.source_card_vectors = SourceCardVectorIndex(
        "embed:test",
        (
            SourceCardVector(
                Path("c.pdf"), "scientific_paper", "a" * 64,
                "b" * 64, ((1.0, 0.0),),
            ),
        ) if cards else (),
    )
    return session


class TestSourceFirstRAG(unittest.TestCase):

    def test_card_discovers_source_outside_top_passages(self):
        session = _session()
        evidence = session._retrieve(
            "memory", 2, source_first=(2, False), trace_turn_id="turn",
        )
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0].path, Path("c.pdf"))
        self.assertEqual(evidence[0].text, "unrelated glucose")
        self.assertEqual(evidence[0].citation, 1)
        self.assertTrue(all(item.path.suffix == ".pdf" for item in evidence))
        session.embedding_session.embed_query.assert_called_once_with(
            "memory", trace_turn_id="turn"
        )

    def test_all_source_channels_are_ranked_without_extra_query(self):
        session = _session()
        with patch.object(
            rag, "lexical_search", wraps=rag.lexical_search
        ) as lexical:
            session._retrieve("memory", 2, source_first=(2, False))
        self.assertEqual(
            lexical.call_args.kwargs["count"], len(session.chunks)
        )
        session.embedding_session.embed_query.assert_called_once()

    def test_registered_card_kind_does_not_change_generic_ranking(self):
        session = _session()
        record = session.source_card_vectors.records[0]
        session.source_card_vectors = SourceCardVectorIndex(
            "embed:test", (
                SourceCardVector(
                    record.path, "another_kind", record.card_sha256,
                    record.text_sha256, record.embeddings,
                ),
            )
        )
        evidence = session._retrieve(
            "memory", 2, source_first=(2, False)
        )
        self.assertEqual(evidence[0].path, Path("c.pdf"))

    def test_without_cards_ordinary_source_rankings_still_work(self):
        session = _session(cards=False)
        evidence = session._retrieve(
            "memory", 2, source_first=(2, False)
        )
        self.assertEqual(len(evidence), 2)
        self.assertNotIn(Path("c.pdf"), [item.path for item in evidence])
        session.embedding_session.embed_query.assert_called_once_with(
            "memory"
        )

    def test_multiple_named_sources_restrict_evidence(self):
        session = _session()
        evidence = session._retrieve(
            "memory", 3, source_first=(3, True),
            scope=(Path("a.pdf"), Path("c.pdf")),
        )
        self.assertEqual(
            {item.path for item in evidence},
            {Path("a.pdf"), Path("c.pdf")},
        )
        self.assertEqual(len(evidence), 3)

    def test_single_named_source_is_primary(self):
        session = _session()
        evidence = session._retrieve(
            "memory", 3, source_first=(3, False),
            scope=(Path("b.pdf"),),
        )
        self.assertEqual({item.path for item in evidence}, {Path("b.pdf")})

    def test_single_named_source_can_be_compared(self):
        session = _session()
        evidence = session._retrieve(
            "memory", 3, source_first=(3, True),
            scope=(Path("b.pdf"),),
        )
        self.assertEqual(evidence[0].path, Path("b.pdf"))
        self.assertGreater(len({item.path for item in evidence}), 1)

    def test_source_count_is_bounded(self):
        session = _session()
        evidence = session._retrieve(
            "memory", 1, source_first=(12, False)
        )
        self.assertEqual(len(evidence), 1)

    def test_hybrid_and_papers_are_still_separate(self):
        session = _session()
        hybrid = session._retrieve("memory", 2)
        papers = session._retrieve("memory", 2, papers=(2, 0))
        self.assertEqual([item.path for item in hybrid], [Path("a.pdf")] * 2)
        self.assertTrue(papers)
        self.assertNotEqual(hybrid[0].path, Path("c.pdf"))

    def test_routing_dispatches_explicit_source_first_only(self):
        session = fixtures._session()
        intent = replace(
            _intent(), retrieval_strategy="source_first",
            max_papers=4, passages_per_paper=1,
        )
        with _answer_patches(session, intent) as mocks:
            session.ask("What differs?", use_router=False)
        self.assertEqual(
            mocks.retrieve.call_args.kwargs["source_first"], (4, True)
        )
        self.assertIsNone(mocks.retrieve.call_args.kwargs["papers"])
        session = fixtures._session()
        with _answer_patches(session, _intent()) as mocks:
            session.ask("What differs?", use_router=False)
        self.assertNotIn("source_first", mocks.retrieve.call_args.kwargs)

    def test_source_rank_trace_records_four_channel_provenance(self):
        session = _session()
        session.trace = MagicMock()
        session.trace.include_content = False
        session._retrieve("memory", 2, source_first=(2, False))
        events = [
            item for item in session.trace.emit.call_args_list
            if item.args[0] == "retrieval.sources.ranked"
        ]
        self.assertEqual(len(events), 1)
        data = events[0].kwargs["data"]
        self.assertEqual(data["selected_count"], 2)
        discovered = next(
            item for item in data["sources"]
            if item["path"] == Path("c.pdf")
        )
        self.assertIn(("cards", 1), discovered["channel_ranks"])
        self.assertTrue(discovered["chosen"])

    def test_reject_conflicting_or_invalid_strategy_requests(self):
        session = _session()
        with self.assertRaises(ValueError):
            session._retrieve(
                "memory", papers=(2, 0), source_first=(2, False)
            )
        for argument in ((0, False), (2, "no"), (True, False)):
            with self.subTest(argument=argument):
                with self.assertRaises(ValueError):
                    session._retrieve("memory", source_first=argument)


if __name__ == "__main__":
    unittest.main()
