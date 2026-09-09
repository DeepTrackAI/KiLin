"""Build and search a transparent in-memory lexical index.

Classes
-------
LexicalIndex
    Repository chunks paired with sparse TF-IDF vectors.
LexicalResult
    One ranked lexical-search result.

Functions
---------
lexical_terms
    Turn text and software identifiers into one-word and two-word terms.
build_lexical_index
    Calculate TF-IDF weights once for repository chunks.
lexical_search
    Rank indexed chunks by cosine similarity to a text query.

"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from qachat.chunking import Chunk

_IDENTIFIER_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_WORD = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class LexicalIndex:
    """Chunks paired with sparse TF-IDF vectors.

    Attributes
    ----------
    chunks: tuple[Chunk, ...]
        Searchable repository chunks.
    vectors: tuple[dict[str, float], ...]
        Normalized TF-IDF term weights for each chunk.
    inverse_document_frequency: dict[str, float]
        Weight of each indexed term across all chunks.

    """

    chunks: tuple[Chunk, ...]
    vectors: tuple[dict[str, float], ...]
    inverse_document_frequency: dict[str, float]


@dataclass(frozen=True, slots=True)
class LexicalResult:
    """One chunk returned by lexical ranking.

    Attributes
    ----------
    score: float
        Cosine similarity between the query and chunk TF-IDF vectors.
    chunk: Chunk
        Matching repository chunk.

    """

    score: float
    chunk: Chunk


def lexical_terms(text: str) -> tuple[str, ...]:
    """Return lowercase one-word and two-word terms from text.

    Software identifiers are also divided at underscores and capital-letter
    boundaries. For example, ``scanRepository`` contributes ``scan`` and
    ``repository`` as searchable words.

    Parameters
    ----------
    text: str
        Text to divide into lexical terms.

    Returns
    -------
    tuple[str, ...]
        Unigrams followed by adjacent bigrams.

    Raises
    ------
    ValueError
        If text is not a string.

    """

    if not isinstance(text, str):
        raise ValueError("text must be a string.")

    words: list[str] = []
    for match in _WORD.finditer(text):
        identifier = match.group()
        original = identifier.lower()
        if len(original) > 1:
            words.append(original)

        expanded = _IDENTIFIER_BOUNDARY.sub(r"\1 \2", identifier)
        parts = [
            part.lower()
            for part in expanded.replace("_", " ").split()
            if len(part) > 1
        ]
        if parts != [original]:
            words.extend(parts)

    bigrams = [
        f"{first} {second}"
        for first, second in zip(words, words[1:], strict=False)
    ]
    return tuple([*words, *bigrams])


def build_lexical_index(
    chunks: list[Chunk],
    *,
    texts: list[str] | None = None,
) -> LexicalIndex:
    """Calculate TF-IDF vectors for repository chunks.

    Parameters
    ----------
    chunks: list[Chunk]
        Repository chunks to index.
    texts: list[str] | None, optional
        Retrieval text for each chunk. Exact chunk text is used when omitted.

    Returns
    -------
    LexicalIndex
        Chunks, normalized vectors, and reusable term weights.

    Raises
    ------
    ValueError
        If no chunks or no searchable terms are supplied.

    """

    if not chunks:
        raise ValueError("At least one chunk is required for an index.")

    if texts is None:
        texts = [chunk.text for chunk in chunks]
    if len(texts) != len(chunks):
        raise ValueError("Retrieval text count must match the chunks.")

    term_counts = [Counter(lexical_terms(text)) for text in texts]
    document_frequency = Counter(
        term for counts in term_counts for term in counts
    )
    if not document_frequency:
        raise ValueError("The chunks contain no searchable lexical terms.")

    document_count = len(chunks)
    inverse_document_frequency = {
        term: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }
    vectors = tuple(
        _tfidf_vector(counts, inverse_document_frequency)
        for counts in term_counts
    )
    return LexicalIndex(
        chunks=tuple(chunks),
        vectors=vectors,
        inverse_document_frequency=inverse_document_frequency,
    )


def lexical_search(
    index: LexicalIndex,
    query_text: str,
    *,
    count: int = 5,
) -> list[LexicalResult]:
    """Rank indexed chunks by TF-IDF cosine similarity to one query.

    Parameters
    ----------
    index: LexicalIndex
        Precomputed repository lexical index.
    query_text: str
        User query in plain text.
    count: int, optional
        Maximum number of ranked results to return.

    Returns
    -------
    list[LexicalResult]
        Highest-scoring chunks first.

    Raises
    ------
    ValueError
        If the index, query, or count is invalid.

    """

    if not index.chunks or not index.vectors:
        raise ValueError("The lexical index is empty.")

    if len(index.chunks) != len(index.vectors):
        raise ValueError("The lexical index is inconsistent.")

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be non-empty text.")

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")

    query_counts = Counter(lexical_terms(query_text))
    query_vector = _tfidf_vector(
        query_counts,
        index.inverse_document_frequency,
    )
    ranked = [
        LexicalResult(
            score=_sparse_cosine_similarity(query_vector, vector),
            chunk=chunk,
        )
        for chunk, vector in zip(
            index.chunks,
            index.vectors,
            strict=True,
        )
    ]
    ranked.sort(
        key=lambda result: (
            -result.score,
            result.chunk.identifier,
        )
    )
    return ranked[: min(count, len(ranked))]


def _tfidf_vector(
    counts: Counter[str],
    inverse_document_frequency: dict[str, float],
) -> dict[str, float]:
    """Return one normalized sparse TF-IDF vector."""

    weights = {
        term: (1.0 + math.log(frequency)) * inverse_document_frequency[term]
        for term, frequency in counts.items()
        if term in inverse_document_frequency
    }
    length = math.sqrt(sum(weight * weight for weight in weights.values()))
    if length == 0.0:
        return {}
    return {term: weight / length for term, weight in weights.items()}


def _sparse_cosine_similarity(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    """Return cosine similarity between normalized sparse vectors."""

    if len(first) > len(second):
        first, second = second, first
    return sum(
        weight * second.get(term, 0.0) for term, weight in first.items()
    )
