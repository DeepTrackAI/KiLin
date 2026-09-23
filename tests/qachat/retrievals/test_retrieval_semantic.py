"""Tests for the semantic retrieval index and search backends."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from qachat.chunking import Chunk
from qachat.retrievals.strategies import retrieval_semantic as rs


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        identifier=identifier,
        path=Path(f"{identifier}.py"),
        text=text,
        start_line=1,
        end_line=1,
        kind="file",
        symbol=None,
        section=None,
    )


class _FakeEmbeddingSession:
    """Deterministic stand-in for EmbeddingSession.embed_documents/query.

    Each chunk's vector is derived from its own text length so that
    tests can assert on exact, reproducible rankings without loading a
    real model.
    """

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.clear_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, query_text: str) -> list[float]:
        return self._vector(query_text)

    def _clear_backend_cache(self) -> None:
        self.clear_calls += 1

    @staticmethod
    def _vector(text: str) -> list[float]:
        # A 2D vector so cosine similarity is easy to reason about by
        # hand: longer texts point further along the second axis.
        return [1.0, float(len(text))]


class TestBuildSemanticIndex(unittest.TestCase):

    def test_batching_matches_single_batch(self) -> None:
        chunks = [_chunk(str(i), "x" * i) for i in range(1, 6)]

        session_one = _FakeEmbeddingSession()
        index_one_batch = rs.build_semantic_index(chunks, session_one)

        session_many = _FakeEmbeddingSession()
        index_many_batches = rs.build_semantic_index(
            chunks, session_many, batch_size=2
        )

        self.assertEqual(index_one_batch.chunks, index_many_batches.chunks)
        self.assertEqual(
            index_one_batch.embeddings, index_many_batches.embeddings
        )
        # One call to embed_documents for the whole corpus...
        self.assertEqual(session_one.batches, [5])
        # ...versus three batches of at most two chunks each.
        self.assertEqual(session_many.batches, [2, 2, 1])

    def test_clears_backend_cache_each_batch(self) -> None:
        chunks = [_chunk(str(i), "x" * i) for i in range(1, 6)]
        session = _FakeEmbeddingSession()

        rs.build_semantic_index(chunks, session, batch_size=2)

        # Three batches (2, 2, 1) -> three cache clears.
        self.assertEqual(session.clear_calls, 3)

    def test_progress_callback_reports_running_total(self) -> None:
        chunks = [_chunk(str(i), "x" * i) for i in range(1, 6)]
        session = _FakeEmbeddingSession()
        seen: list[tuple[int, int]] = []

        rs.build_semantic_index(
            chunks,
            session,
            batch_size=2,
            progress_callback=lambda done, total: seen.append(
                (done, total)
            ),
        )

        self.assertEqual(seen, [(2, 5), (4, 5), (5, 5)])

    def test_rejects_empty_chunks(self) -> None:
        with self.assertRaises(ValueError):
            rs.build_semantic_index([], _FakeEmbeddingSession())


class TestSemanticSearchBackends(unittest.TestCase):

    def setUp(self) -> None:
        self.chunks = [_chunk(str(i), "x" * i) for i in range(1, 21)]
        self.session = _FakeEmbeddingSession()
        self.index = rs.build_semantic_index(self.chunks, self.session)

    def test_numpy_backend_matches_loop_backend(self) -> None:
        loop_ranked = rs.semantic_search(
            self.index, "query", self.session, count=5, backend="loop"
        )
        numpy_ranked = rs.semantic_search(
            self.index, "query", self.session, count=5, backend="numpy"
        )

        self.assertEqual(
            [result.chunk.identifier for result in loop_ranked],
            [result.chunk.identifier for result in numpy_ranked],
        )
        for loop_result, numpy_result in zip(loop_ranked, numpy_ranked):
            self.assertAlmostEqual(
                loop_result.score, numpy_result.score, places=5
            )

    def test_numpy_backend_accepts_precomputed_matrix(self) -> None:
        matrix = rs.semantic_matrix(self.index)
        ranked = rs.semantic_search(
            self.index,
            "query",
            self.session,
            count=3,
            backend="numpy",
            matrix=matrix,
        )
        self.assertEqual(len(ranked), 3)

    def test_semantic_matrix_shape(self) -> None:
        matrix = rs.semantic_matrix(self.index)
        self.assertEqual(
            matrix.shape,
            (len(self.index.chunks), len(self.index.embeddings[0])),
        )

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            rs.semantic_search(
                self.index, "query", self.session, backend="fast"
            )


if __name__ == "__main__":
    unittest.main()
