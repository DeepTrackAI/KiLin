"""Verify paper ranking, passage selection, and paper retrieval in RAG.

Rankings and vectors are small, hand-built examples; the RAG session uses
real search over in-memory indexes with a deterministic query embedding.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin import rag
from kilin.chunking import Chunk
from kilin.retrievals import build_lexical_index
from kilin.retrievals.code_graph import CodeEntity, CodeGraph
from kilin.retrievals.strategies import retrieval_papers as rp
from kilin.retrievals.strategies.retrieval_hybrid import HybridResult
from tests.kilin import _rag_support as fixtures


def _passage(identifier: str, path: str, text: str = "text") -> Chunk:
    """Return one PDF passage of a paper."""
    return Chunk(identifier, Path(path), text, 1, 2, "pdf_page", None,
                 "Page 1")


def _ranked(*items: tuple[str, str]) -> list[HybridResult]:
    """Return a fused ranking of (identifier, path) pairs, best first."""
    return [
        HybridResult(1.0 / (60 + rank), _passage(identifier, path), ("x",))
        for rank, (identifier, path) in enumerate(items, start=1)
    ]


class TestCardSimilarities(unittest.TestCase):

    def test_best_card_vector_counts_and_bad_vectors_are_skipped(self):
        scores = rp.card_similarities((1.0, 0.0), {
            Path("a.pdf"): ((0.0, 1.0), (2.0, 0.0)),
            Path("b.pdf"): ((1.0, 1.0),),
            Path("c.pdf"): ((1.0, 0.0, 0.0), (0.0, 0.0)),
            Path("d.pdf"): (),
        })
        self.assertAlmostEqual(scores[Path("a.pdf")], 1.0)
        self.assertAlmostEqual(scores[Path("b.pdf")], 2 ** -0.5)
        self.assertNotIn(Path("c.pdf"), scores)
        self.assertNotIn(Path("d.pdf"), scores)
        self.assertEqual(rp.card_similarities((0.0, 0.0), {
            Path("a.pdf"): ((1.0, 0.0),)}), {})


class TestRankPapers(unittest.TestCase):

    def test_cards_and_passages_are_fused(self):
        ranked = _ranked(("a1", "a.pdf"), ("a2", "a.pdf"), ("b1", "b.pdf"))
        papers = rp.rank_papers(ranked, {Path("b.pdf"): 0.9,
                                         Path("c.pdf"): 0.8,
                                         Path("a.pdf"): 0.1})
        # b: card 1 + passage 2 beats a: card 3 + passage 1.
        self.assertEqual([paper.path.name for paper in papers],
                         ["b.pdf", "a.pdf", "c.pdf"])
        self.assertEqual((papers[0].card_rank, papers[0].passage_rank),
                         (1, 2))
        self.assertIsNone(papers[2].passage_rank)

    def test_passages_alone_rank_papers_without_cards(self):
        ranked = _ranked(("b1", "b.pdf"), ("a1", "a.pdf"))
        self.assertEqual(
            [paper.path.name for paper in rp.rank_papers(ranked, {})],
            ["b.pdf", "a.pdf"],
        )
        self.assertEqual(rp.rank_papers([], {}), [])


class TestRelevantPapers(unittest.TestCase):

    def setUp(self):
        self.ranked = _ranked(("a1", "a.pdf"), ("b1", "b.pdf"),
                              ("a2", "a.pdf"), ("c1", "c.pdf"))
        self.papers = rp.rank_papers(self.ranked, {Path("c.pdf"): 0.9})

    def _names(self, papers):
        return [paper.path.name for paper in papers]

    def test_papers_need_a_passage_among_the_best(self):
        self.assertEqual(
            self._names(rp.relevant_papers(self.papers, self.ranked,
                                           depth=2)),
            ["a.pdf", "b.pdf"],
        )

    def test_minimum_and_required_papers(self):
        self.assertEqual(
            self._names(rp.relevant_papers(self.papers, self.ranked,
                                           depth=1, minimum=2)),
            ["a.pdf", "c.pdf"],
        )
        self.assertEqual(
            self._names(rp.relevant_papers(
                self.papers, self.ranked, depth=1,
                required=[Path("b.pdf")],
            )),
            ["b.pdf", "a.pdf"],
        )

    def test_questions_about_every_paper(self):
        for question in ("Compare the datasets used in all five project "
                         "papers.", "Review the five articles.",
                         "Compare each study.", "compare all of them",
                         "What is common across the papers?"):
            with self.subTest(question=question):
                self.assertTrue(rp.asks_for_all_papers(question))
        for question in ("How does BRAPH 2 differ from the earlier BRAPH?",
                         "How does it compare with the 2017 paper?",
                         "Compare a.pdf and b.pdf."):
            with self.subTest(question=question):
                self.assertFalse(rp.asks_for_all_papers(question))


class TestSelectPaperPassages(unittest.TestCase):

    def setUp(self):
        self.ranked = _ranked(
            ("a1", "a.pdf"), ("a2", "a.pdf"), ("a3", "a.pdf"),
            ("b1", "b.pdf"), ("c1", "c.pdf"), ("b2", "b.pdf"),
        )

    def _ids(self, results):
        return [result.chunk.identifier for result in results]

    def test_best_passages_of_the_chosen_papers(self):
        chosen = [Path("a.pdf"), Path("b.pdf")]
        self.assertEqual(
            self._ids(rp.select_paper_passages(self.ranked, chosen, count=4)),
            ["a1", "a2", "a3", "b1"],
        )

    def test_best_paper_is_always_represented(self):
        chosen = [Path("c.pdf"), Path("a.pdf")]
        self.assertEqual(
            self._ids(rp.select_paper_passages(self.ranked, chosen, count=3)),
            ["a1", "a2", "c1"],
        )

    def test_each_paper_gets_its_quota(self):
        chosen = [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")]
        self.assertEqual(
            self._ids(rp.select_paper_passages(
                self.ranked, chosen, count=3, per_paper=1
            )),
            ["a1", "b1", "c1"],
        )
        self.assertEqual(
            self._ids(rp.select_paper_passages(
                self.ranked, chosen, count=5, per_paper=1
            )),
            # Each paper's best passage, then the best remaining ones.
            ["a1", "a2", "a3", "b1", "c1"],
        )

    def test_unknown_papers_give_nothing(self):
        self.assertEqual(
            rp.select_paper_passages(self.ranked, [Path("z.pdf")], count=3),
            [],
        )


def _paper_session() -> rag.RAGSession:
    """Create a session with three papers and paper cards."""
    session = fixtures._session()
    session.chunks = [
        _passage("a1", "a.pdf", "sleep and memory consolidation"),
        _passage("a2", "a.pdf", "memory tests in older adults"),
        _passage("a3", "a.pdf", "memory capacity results"),
        _passage("b1", "b.pdf", "plant stress spectra"),
        _passage("c1", "c.pdf", "genomic risk and connectivity"),
    ]
    session.indexes = SimpleNamespace(
        semantic=rag.SemanticIndex(
            tuple(session.chunks),
            ((1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.0, 1.0), (0.1, 1.0)),
        ),
        lexical=build_lexical_index(session.chunks),
        code_graph=CodeGraph((), ()),
    )
    session.embedding_session = Mock()
    session.embedding_session.embed_query.return_value = [1.0, 0.0]
    session.paper_card_vectors = {
        Path("a.pdf"): ((1.0, 0.0),),
        Path("b.pdf"): ((0.0, 1.0),),
        Path("c.pdf"): ((0.5, 0.5),),
    }
    return session


class TestPaperRetrieval(unittest.TestCase):

    def _paths(self, evidence):
        return [item.path.name for item in evidence]

    def test_quota_covers_every_paper_asked_for_with_one_embedding(self):
        session = _paper_session()
        evidence = session._retrieve(
            "memory in all three papers", 2, trace_turn_id="t1",
            papers=(3, 1),
        )
        self.assertEqual(set(self._paths(evidence)),
                         {"a.pdf", "b.pdf", "c.pdf"})
        session.embedding_session.embed_query.assert_called_once_with(
            "memory in all three papers", trace_turn_id="t1"
        )

    def test_comparisons_leave_out_unrelated_papers(self):
        session = _paper_session()
        evidence = session._retrieve("memory", 2, papers=(8, 1))
        # Only a.pdf has a best passage; a comparison keeps two papers.
        self.assertEqual(len(set(self._paths(evidence))), 2)
        self.assertIn("a.pdf", self._paths(evidence))

    def test_one_meant_paper_is_compared_with_relevant_papers(self):
        session = _paper_session()
        evidence = session._retrieve(
            "memory", 2, papers=(8, 1), scope=(Path("b.pdf"),)
        )
        paths = self._paths(evidence)
        self.assertIn("b.pdf", paths)
        self.assertIn("a.pdf", paths)

    def test_hybrid_retrieval_is_unchanged_without_papers(self):
        session = _paper_session()
        evidence = session._retrieve("memory", 2)
        self.assertEqual(self._paths(evidence), ["a.pdf", "a.pdf"])

    def test_max_papers_limits_the_chosen_papers(self):
        session = _paper_session()
        evidence = session._retrieve("memory", 3, papers=(1, 0))
        self.assertEqual(set(self._paths(evidence)), {"a.pdf"})

    def test_named_papers_limit_and_one_named_paper_focuses(self):
        session = _paper_session()
        evidence = session._retrieve(
            "memory", 2, papers=(3, 1),
            scope=(Path("b.pdf"), Path("c.pdf")),
        )
        self.assertEqual(set(self._paths(evidence)), {"b.pdf", "c.pdf"})
        evidence = session._retrieve(
            "memory", 2, papers=(3, 1), scope=(Path("b.pdf"),)
        )
        self.assertEqual(self._paths(evidence)[0], "b.pdf")

    def test_named_paper_is_searched_not_added_in_full(self):
        session = _paper_session()
        session.indexes.code_graph = CodeGraph(
            (CodeEntity("a", "source_file", Path("a.pdf")),), ()
        )
        session.project.profile = SimpleNamespace(
            **{**vars(fixtures._profile()), "type_id": "literature"}
        )
        evidence = session._retrieve("memory in a.pdf", 1,
                                     scope=(Path("a.pdf"),))
        self.assertEqual(self._paths(evidence), ["a.pdf"])
        session.project.profile = fixtures._profile()
        evidence = session._retrieve("memory in a.pdf", 1,
                                     scope=(Path("a.pdf"),))
        # Without a paper index, a named file is still added in full.
        self.assertEqual(self._paths(evidence), ["a.pdf"] * 3)

    def test_without_cards_papers_are_ranked_by_passages(self):
        session = _paper_session()
        session.paper_card_vectors = {}
        evidence = session._retrieve("memory", 2, papers=(2, 1))
        self.assertEqual(len(set(self._paths(evidence))), 2)
        session.embedding_session.embed_query.assert_called_once_with(
            "memory"
        )


if __name__ == "__main__":
    unittest.main()
