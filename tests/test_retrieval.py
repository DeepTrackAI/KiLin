"""Tests for independent semantic, lexical, and metadata retrieval."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.retrieval import (
    SemanticIndex,
    build_lexical_index,
    build_semantic_index,
    cosine_similarity,
    lexical_search,
    metadata_search,
    semantic_search,
)


class FakeEmbedder:
    def __init__(
        self,
        document_vectors: list[list[float]],
        query_vector: list[float],
    ) -> None:
        self.document_vectors = document_vectors
        self.query_vector = query_vector
        self.batches: list[list[str]] = []
        self._position = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        start = self._position
        self._position += len(texts)
        return self.document_vectors[start : self._position]

    def embed_query(self, question: str) -> list[float]:
        del question
        return self.query_vector


def _chunk(
    identifier: str,
    text: str,
    *,
    path: str | None = None,
    symbol: str | None = None,
    section: str | None = None,
    kind: str = "text",
) -> Chunk:
    return Chunk(
        identifier=identifier,
        path=Path(path or f"{identifier}.txt"),
        text=text,
        start_line=1,
        end_line=1,
        kind=kind,
        symbol=symbol,
        section=section,
    )


class TestRetrieval(unittest.TestCase):

    def test_build_semantic_index_stores_vectors_without_model(self) -> None:
        chunks = [_chunk("a", "apple"), _chunk("b", "banana")]
        embedder = FakeEmbedder(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.0, 1.0],
        )

        index = build_semantic_index(chunks, embedder, batch_size=1)

        self.assertEqual(embedder.batches, [["apple"], ["banana"]])
        self.assertEqual(index.chunks, tuple(chunks))
        self.assertEqual(index.embeddings, ((1.0, 0.0), (0.0, 1.0)))
        self.assertFalse(hasattr(index, "embedder"))

        invalid_settings = (
            ([], FakeEmbedder([], [1.0]), 1),
            (chunks, FakeEmbedder([], [1.0]), 0),
            (chunks, FakeEmbedder([], [1.0]), True),
        )
        for selected_chunks, selected_embedder, batch_size in invalid_settings:
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    build_semantic_index(
                        selected_chunks,
                        selected_embedder,
                        batch_size=batch_size,
                    )

    def test_build_semantic_index_rejects_invalid_vectors(self) -> None:
        chunks = [_chunk("a", "apple"), _chunk("b", "banana")]
        invalid_vectors = (
            [[1.0, 0.0]],
            [[1.0], [1.0, 2.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[float("nan"), 0.0], [1.0, 0.0]],
            [[float("inf"), 0.0], [1.0, 0.0]],
            [[True, 0.0], [1.0, 0.0]],
        )
        for vectors in invalid_vectors:
            with self.subTest(vectors=vectors):
                embedder = FakeEmbedder(vectors, [1.0])
                with self.assertRaises(ValueError):
                    build_semantic_index(chunks, embedder)

        with self.assertRaises(ValueError):
            SemanticIndex(
                chunks=tuple(chunks),
                embeddings=((1.0,),),
            )

    def test_semantic_search_uses_an_explicit_query_embedder(self) -> None:
        chunks = [_chunk("a", "apple"), _chunk("b", "banana")]
        index = build_semantic_index(
            chunks,
            FakeEmbedder(
                [[1.0, 0.0], [0.0, 1.0]],
                [1.0, 0.0],
            ),
        )
        query_embedder = FakeEmbedder([], [0.0, 1.0])

        results = semantic_search(
            index,
            "Tell me about bananas",
            query_embedder,
            count=1,
        )

        self.assertEqual(results[0].chunk.identifier, "b")
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertEqual(results[0].method, "semantic")

        invalid_queries = (
            [],
            [1.0],
            [0.0, 0.0],
            [float("nan"), 0.0],
        )
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    semantic_search(
                        index,
                        "question",
                        FakeEmbedder([], query),
                    )

        with self.assertRaises(ValueError):
            semantic_search(index, "", query_embedder)
        with self.assertRaises(ValueError):
            semantic_search(index, "apple", query_embedder, count=True)

    def test_cosine_similarity_checks_vector_values(self) -> None:
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [1.0, 0.0]),
            1.0,
        )
        self.assertEqual(
            cosine_similarity([0.0, 0.0], [1.0, 0.0]),
            0.0,
        )

        invalid_pairs = (
            ([], []),
            ([1.0], [1.0, 0.0]),
            ([float("nan")], [1.0]),
            ([float("inf")], [1.0]),
            ([True], [1.0]),
        )
        for left, right in invalid_pairs:
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    cosine_similarity(left, right)

    def test_lexical_index_is_reusable(self) -> None:
        chunks = [
            _chunk("a", "class MessagePassingNetwork:"),
            _chunk("b", "def select_cluster_connect(): pass"),
            _chunk("c", "unrelated unrelated text"),
        ]
        index = build_lexical_index(chunks)

        self.assertEqual(index.chunks, tuple(chunks))
        self.assertEqual(index.document_lengths, (5, 6, 3))
        self.assertEqual(index.document_frequency["message"], 1)
        self.assertEqual(index.document_frequency["unrelated"], 1)

        results = lexical_search(
            index,
            "message passing network",
            count=3,
        )
        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["a"],
        )
        self.assertEqual(results[0].method, "lexical")

        results = lexical_search(
            index,
            "select_cluster_connect",
            count=1,
        )
        self.assertEqual(results[0].chunk.identifier, "b")
        self.assertEqual(lexical_search(index, "missing", count=2), [])
        self.assertEqual(
            lexical_search(build_lexical_index([]), "missing"),
            [],
        )
        with self.assertRaises(ValueError):
            lexical_search(index, "missing", count=0)

    def test_metadata_search_uses_only_source_metadata(self) -> None:
        chunks = [
            _chunk(
                "symbol",
                "unrelated",
                path="src/service.py",
                symbol="createElement",
            ),
            _chunk(
                "section",
                "unrelated",
                path="docs/guide.md",
                section="Create element",
            ),
            _chunk(
                "filename",
                "unrelated",
                path="src/create_element.py",
            ),
        ]

        results = metadata_search(
            chunks,
            "How does create element work?",
            count=3,
        )

        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["symbol", "filename", "section"],
        )
        self.assertTrue(all(result.method == "metadata" for result in results))
        self.assertEqual(
            metadata_search(chunks, "missing", count=3),
            [],
        )


if __name__ == "__main__":
    unittest.main()
