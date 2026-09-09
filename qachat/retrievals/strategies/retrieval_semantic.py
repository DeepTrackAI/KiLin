"""Build and search a simple in-memory semantic retrieval index.

Classes
-------
SemanticIndex
    Repository chunks paired with precomputed embedding vectors.
SemanticResult
    One ranked semantic-search result.

Functions
---------
build_semantic_index
    Embed repository chunks once and store the vectors beside them.
semantic_search
    Embed one query, compare it with indexed vectors, and rank the matches.

"""

from __future__ import annotations

from dataclasses import dataclass
import math

from qachat.chunking import Chunk
from qachat.embeddings import EmbeddingSession


__all__ = [
    "SemanticIndex",
    "SemanticResult",
    "build_semantic_index",
    "semantic_search",
]


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    """Chunks paired with vectors prepared for semantic search.

    Attributes
    ----------
    chunks: tuple[Chunk, ...]
        Searchable repository chunks.
    embeddings: tuple[tuple[float, ...], ...]
        One embedding vector for each chunk in the same order.

    """

    chunks: tuple[Chunk, ...]
    embeddings: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class SemanticResult:
    """One chunk returned by semantic ranking.

    Attributes
    ----------
    score: float
        Cosine similarity between the query and chunk vectors.
    chunk: Chunk
        Matching repository chunk.

    """

    score: float
    chunk: Chunk


def build_semantic_index(
    chunks: list[Chunk],
    embedding_session: EmbeddingSession,
) -> SemanticIndex:
    """Embed chunks once and return their searchable semantic index.

    Parameters
    ----------
    chunks: list[Chunk]
        Repository chunks to index.
    embedding_session: EmbeddingSession
        Loaded embedding model used to make document vectors.

    Returns
    -------
    SemanticIndex
        Chunks and their precomputed vectors.

    Raises
    ------
    ValueError
        If no chunks are supplied or vector dimensions disagree.

    """

    if not chunks:
        raise ValueError("At least one chunk is required for an index.")

    texts = [chunk.text for chunk in chunks]
    vectors = embedding_session.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match the chunk count.")

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not dimensions or 0 in dimensions:
        raise ValueError("Indexed vectors must share one non-zero dimension.")

    return SemanticIndex(
        chunks=tuple(chunks),
        embeddings=tuple(tuple(vector) for vector in vectors),
    )


def semantic_search(
    index: SemanticIndex,
    query_text: str,
    embedding_session: EmbeddingSession,
    *,
    count: int = 5,
) -> list[SemanticResult]:
    """Rank indexed chunks by cosine similarity to one query.

    Parameters
    ----------
    index: SemanticIndex
        Precomputed repository semantic index.
    query_text: str
        Query text already formatted for the selected embedding model.
    embedding_session: EmbeddingSession
        Loaded embedding model used to make the query vector.
    count: int, optional
        Maximum number of ranked results to return.

    Returns
    -------
    list[SemanticResult]
        Highest-scoring chunks first.

    Raises
    ------
    ValueError
        If the index is empty, count is invalid, or dimensions disagree.

    """

    if not index.chunks or not index.embeddings:
        raise ValueError("The semantic index is empty.")

    if len(index.chunks) != len(index.embeddings):
        raise ValueError("The semantic index is inconsistent.")

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")

    query_vector = embedding_session.embed_query(query_text)
    expected_dimension = len(index.embeddings[0])
    if len(query_vector) != expected_dimension:
        raise ValueError("Query and index vector dimensions do not match.")

    ranked = [
        SemanticResult(
            score=_cosine_similarity(query_vector, vector),
            chunk=chunk,
        )
        for chunk, vector in zip(
            index.chunks,
            index.embeddings,
            strict=True,
        )
    ]
    ranked.sort(key=lambda result: result.score, reverse=True)
    return ranked[: min(count, len(ranked))]


def _cosine_similarity(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
) -> float:
    """Return cosine similarity between two equal-length vectors."""

    if len(first) != len(second) or not first:
        raise ValueError("Cosine similarity needs equal non-empty vectors.")

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(first, second, strict=True)
    )
    first_length = math.sqrt(sum(value * value for value in first))
    second_length = math.sqrt(sum(value * value for value in second))

    if first_length == 0.0 or second_length == 0.0:
        raise ValueError("Cosine similarity cannot use a zero vector.")

    return dot_product / (first_length * second_length)
