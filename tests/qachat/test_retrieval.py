"""Tests for semantic indexing and ranking."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.retrieval import (
    SemanticIndex,
    _cosine_similarity,
    build_semantic_index,
    semantic_search,
)


class FakeEmbeddingSession:

    def __init__(
        self: FakeEmbeddingSession,
        document_vectors: list[list[float]],
        query_vector: list[float],
    ) -> None:
        self.document_vectors = document_vectors
        self.query_vector = query_vector
        self.documents: list[str] = []
        self.query_text = ""

    def embed_documents(
        self: FakeEmbeddingSession,
        texts: list[str],
    ) -> list[list[float]]:
        self.documents = texts
        return self.document_vectors

    def embed_query(
        self: FakeEmbeddingSession, query_text: str
    ) -> list[float]:
        self.query_text = query_text
        return self.query_vector


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        identifier=identifier,
        path=Path(f"{identifier}.py"),
        text=text,
        start_line=1,
        end_line=1,
        kind="python_function",
        symbol=identifier,
        section=None,
    )


class TestRetrieval(unittest.TestCase):

    def test_build_and_search_semantic_index(self) -> None:
        chunks = [
            _chunk("scan", "scan repository files"),
            _chunk("chat", "reply to a user"),
        ]
        session = FakeEmbeddingSession(
            document_vectors=[[1.0, 0.0], [0.0, 1.0]],
            query_vector=[0.9, 0.1],
        )

        index = build_semantic_index(chunks, session)
        results = semantic_search(
            index,
            "formatted repository question",
            session,
            count=2,
        )

        self.assertEqual(
            session.documents,
            ["scan repository files", "reply to a user"],
        )
        self.assertEqual(session.query_text, "formatted repository question")
        self.assertEqual(index.chunks, tuple(chunks))
        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["scan", "chat"],
        )
        self.assertGreater(results[0].score, results[1].score)

    def test_invalid_index_inputs_are_rejected(self) -> None:
        chunk = _chunk("scan", "scan repository files")
        invalid_sessions = (
            FakeEmbeddingSession([], [1.0, 0.0]),
            FakeEmbeddingSession([[1.0], [1.0]], [1.0]),
        )
        with self.assertRaises(ValueError):
            build_semantic_index([], invalid_sessions[0])

        for session in invalid_sessions:
            with self.subTest(vectors=session.document_vectors):
                with self.assertRaises(ValueError):
                    build_semantic_index([chunk], session)

        valid_session = FakeEmbeddingSession([[1.0, 0.0]], [1.0, 0.0])
        index = build_semantic_index([chunk], valid_session)
        for count in (0, True):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    semantic_search(
                        index, "question", valid_session, count=count
                    )

        empty = SemanticIndex(chunks=(), embeddings=())
        with self.assertRaises(ValueError):
            semantic_search(empty, "question", valid_session)

    def test_cosine_similarity_rejects_invalid_vectors(self) -> None:
        with self.assertRaises(ValueError):
            _cosine_similarity([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            _cosine_similarity([0.0, 0.0], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
