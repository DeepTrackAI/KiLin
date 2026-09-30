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
semantic_matrix
    Stack an index's embeddings into one array for repeated numpy search.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from qachat.chunking import Chunk
from qachat.embeddings import EmbeddingSession

__all__ = [
    "SemanticIndex",
    "SemanticResult",
    "build_semantic_index",
    "semantic_search",
    "semantic_matrix",
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
    *,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SemanticIndex:
    """Embed chunks once and return their searchable semantic index.

    Parameters
    ----------
    chunks: list[Chunk]
        Repository chunks to index.
    embedding_session: EmbeddingSession
        Loaded embedding model used to make document vectors.
    batch_size: int | None, optional
        Number of chunks embedded per model call. ``None`` preserves the
        previous all-at-once behavior.
    progress_callback: collections.abc.Callable[[int, int], None] | None,
        optional
        Called after each embedding batch with ``current`` and ``total``.

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

    if batch_size is not None and (
        type(batch_size) is not int or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive whole number or None.")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None.")

    texts = [chunk.text for chunk in chunks]
    total = len(texts)
    effective_batch_size = total if batch_size is None else batch_size
    vectors: list[list[float]] = []
    for start in range(0, total, effective_batch_size):
        batch = texts[start : start + effective_batch_size]
        vectors.extend(embedding_session.embed_documents(batch))
        embedding_session._clear_backend_cache()
        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)

    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match the chunk count.")

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not dimensions or 0 in dimensions:
        raise ValueError("Indexed vectors must share one non-zero dimension.")

    return SemanticIndex(
        chunks=tuple(chunks),
        embeddings=tuple(tuple(vector) for vector in vectors),
    )


def semantic_matrix(index: SemanticIndex) -> np.ndarray:
    """Stack one index's embeddings into a single numpy array.

    Pass the result to ``semantic_search(..., backend="numpy")`` as
    ``matrix`` to search the same index many times without rebuilding
    this array on every call -- building it is itself an O(n) pass over
    every embedding, so a caller running many queries against one index
    (a benchmark corpus, a "literature"-kind project with a large
    document collection) should build it once, not once per query.
    """

    return np.asarray(index.embeddings, dtype=np.float32)


def semantic_search(
    index: SemanticIndex,
    query_text: str,
    embedding_session: EmbeddingSession,
    *,
    count: int = 5,
    backend: str = "loop",
    matrix: np.ndarray | None = None,
    trace_turn_id: str | None = None,
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
    backend: str, optional
        ``"loop"`` (the default) ranks with a plain Python loop over
        every chunk -- fine at repository scale, a few thousand chunks.
        ``"numpy"`` computes the identical cosine similarity with numpy
        instead, worthwhile once an index holds tens of thousands of
        chunks (a large "literature"-kind project, say). The two
        backends agree on which chunks rank where in practice, but are
        not guaranteed bit-for-bit identical: ``"numpy"`` computes in
        float32 versus the loop's float64, and its top-k selection is
        not guaranteed stable on exact score ties (the loop's is).
    matrix: numpy.ndarray | None, optional
        Only used when ``backend="numpy"``. Pass the result of
        ``semantic_matrix(index)`` to reuse it across many queries
        against the same index; left as ``None``, it is rebuilt for
        this one call.

    Returns
    -------
    list[SemanticResult]
        Highest-scoring chunks first.

    Raises
    ------
    ValueError
        If the index is empty, count is invalid, backend is unknown, or
        dimensions disagree.

    """

    if not index.chunks or not index.embeddings:
        raise ValueError("The semantic index is empty.")

    if len(index.chunks) != len(index.embeddings):
        raise ValueError("The semantic index is inconsistent.")

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")

    if backend not in ("loop", "numpy"):
        raise ValueError('backend must be "loop" or "numpy".')

    if trace_turn_id is None:
        query_vector = embedding_session.embed_query(query_text)
    else:
        query_vector = embedding_session.embed_query(
            query_text,
            trace_turn_id=trace_turn_id,
        )
    expected_dimension = len(index.embeddings[0])
    if len(query_vector) != expected_dimension:
        raise ValueError("Query and index vector dimensions do not match.")

    if backend == "numpy":
        return _semantic_search_numpy(
            index, query_vector, count=count, matrix=matrix
        )

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


def _semantic_search_numpy(
    index: SemanticIndex,
    query_vector: list[float],
    *,
    count: int,
    matrix: np.ndarray | None,
) -> list[SemanticResult]:
    """Rank one index's chunks against a query vector with numpy."""

    if matrix is None:
        matrix = semantic_matrix(index)

    query_array = np.asarray(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query_array)
    doc_norms = np.linalg.norm(matrix, axis=1)
    denominator = query_norm * doc_norms
    if query_norm == 0.0 or np.any(doc_norms == 0.0):
        raise ValueError("Cosine similarity cannot use a zero vector.")
    scores = (matrix @ query_array) / denominator

    top_count = min(count, len(index.chunks))
    top_indices = np.argpartition(-scores, top_count - 1)[:top_count]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    return [
        SemanticResult(score=float(scores[i]), chunk=index.chunks[i])
        for i in top_indices
    ]


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
