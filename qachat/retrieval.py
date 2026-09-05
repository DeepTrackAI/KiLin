"""Build and search independent repository retrieval indexes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log, sqrt
import re
from types import MappingProxyType
from typing import Literal, Protocol

from qachat.chunking import Chunk

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
IDENTIFIER_PART = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+"
)
METADATA_WEIGHTS = (
    ("symbol", 5.0),
    ("section", 4.0),
    ("filename", 4.0),
    ("path", 2.0),
    ("kind", 1.0),
)

RetrievalMethod = Literal["semantic", "lexical", "metadata"]
Vector = list[float] | tuple[float, ...]


class Embedder(Protocol):
    """Small interface required by semantic retrieval."""

    def embed_documents(
        self: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        """Return one document vector for each text.

        Parameters
        ----------
        texts: list[str]
            Repository text values.

        Returns
        -------
        list[list[float]]
            Document vectors.

        """

        ...

    def embed_query(self: Embedder, question: str) -> list[float]:
        """Return one vector for a repository question.

        Parameters
        ----------
        question: str
            Repository question.

        Returns
        -------
        list[float]
            Question vector.

        """

        ...


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    """Repository chunks paired with stored embedding vectors.

    Attributes
    ----------
    chunks: tuple[Chunk, ...]
        Indexed repository chunks.
    embeddings: tuple[tuple[float, ...], ...]
        One embedding per chunk.

    """

    chunks: tuple[Chunk, ...]
    embeddings: tuple[tuple[float, ...], ...]

    def __post_init__(self: SemanticIndex) -> None:
        """Check data loaded from memory or a future saved index."""

        _check_semantic_index(self.chunks, self.embeddings)


@dataclass(frozen=True, slots=True)
class LexicalIndex:
    """Repository chunks with reusable lexical corpus statistics.

    Attributes
    ----------
    chunks: tuple[Chunk, ...]
        Indexed repository chunks.
    term_frequencies: tuple[collections.abc.Mapping[str, int], ...]
        Term counts for each chunk.
    document_frequency: collections.abc.Mapping[str, int]
        Number of chunks containing each term.
    document_lengths: tuple[int, ...]
        Searchable term count for each chunk.
    average_length: float
        Average searchable term count.

    """

    chunks: tuple[Chunk, ...]
    term_frequencies: tuple[Mapping[str, int], ...]
    document_frequency: Mapping[str, int]
    document_lengths: tuple[int, ...]
    average_length: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One matching chunk and a method-specific score.

    Attributes
    ----------
    chunk: Chunk
        Matching source chunk.
    score: float
        Relevance score within one retrieval method.
    method: {"semantic", "lexical", "metadata"}
        Retrieval method that produced the result.

    """

    chunk: Chunk
    score: float
    method: RetrievalMethod


def build_semantic_index(
    chunks: list[Chunk],
    embedder: Embedder,
    *,
    batch_size: int = 32,
) -> SemanticIndex:
    """Embed repository chunks and store only the searchable data.

    Parameters
    ----------
    chunks: list[Chunk]
        Repository chunks to index.
    embedder: Embedder
        Loaded embedding session.
    batch_size: int, optional
        Number of chunks embedded in one model call.

    Returns
    -------
    SemanticIndex
        In-memory semantic index without a live model reference.

    Raises
    ------
    ValueError
        If chunks, the batch size, or returned vectors are invalid.

    """

    if not chunks:
        raise ValueError("Cannot build an index without chunks.")

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive whole number.")

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = embedder.embed_documents(
            [chunk.text for chunk in batch]
        )
        if not isinstance(batch_vectors, list):
            raise ValueError("The embedder must return a list of vectors.")
        if len(batch_vectors) != len(batch):
            raise ValueError(
                "The embedder returned the wrong number of vectors."
            )
        vectors.extend(batch_vectors)

    embeddings = _prepare_embeddings(vectors)
    return SemanticIndex(chunks=tuple(chunks), embeddings=embeddings)


def semantic_search(
    index: SemanticIndex,
    question: str,
    embedder: Embedder,
    *,
    count: int = 5,
) -> list[RetrievalResult]:
    """Return chunks with the greatest cosine similarity.

    Parameters
    ----------
    index: SemanticIndex
        Stored repository chunks and document vectors.
    question: str
        Meaning-based repository question.
    embedder: Embedder
        Loaded session used only to embed the question.
    count: int, optional
        Maximum number of results.

    Returns
    -------
    list[RetrievalResult]
        Best semantic matches in descending score order.

    Raises
    ------
    ValueError
        If the question, result count, or query vector is invalid.

    """

    _check_search(question, count)
    dimension = len(index.embeddings[0])
    query = _check_vector(
        embedder.embed_query(question),
        name="Query embedding",
        dimension=dimension,
        allow_zero=False,
    )
    results = [
        RetrievalResult(
            chunk=chunk,
            score=cosine_similarity(vector, query),
            method="semantic",
        )
        for chunk, vector in zip(
            index.chunks,
            index.embeddings,
            strict=True,
        )
    ]
    return sorted(
        results,
        key=lambda result: (-result.score, result.chunk.identifier),
    )[:count]


def build_lexical_index(chunks: list[Chunk]) -> LexicalIndex:
    """Prepare lexical statistics once for repeated searches.

    Parameters
    ----------
    chunks: list[Chunk]
        Repository chunks to index.

    Returns
    -------
    LexicalIndex
        Chunks and reusable BM25-style corpus statistics.

    """

    term_frequencies: list[Mapping[str, int]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []

    for chunk in chunks:
        terms = _lexical_terms(chunk.text)
        frequencies = Counter(terms)
        term_frequencies.append(MappingProxyType(dict(frequencies)))
        document_frequency.update(frequencies.keys())
        document_lengths.append(len(terms))

    average_length = 0.0
    if document_lengths:
        average_length = max(
            sum(document_lengths) / len(document_lengths),
            1.0,
        )

    return LexicalIndex(
        chunks=tuple(chunks),
        term_frequencies=tuple(term_frequencies),
        document_frequency=MappingProxyType(dict(document_frequency)),
        document_lengths=tuple(document_lengths),
        average_length=average_length,
    )


def lexical_search(
    index: LexicalIndex,
    question: str,
    *,
    count: int = 5,
) -> list[RetrievalResult]:
    """Rank chunks with exact terms and code-aware identifier splitting.

    Parameters
    ----------
    index: LexicalIndex
        Prepared lexical repository index.
    question: str
        Exact words or code identifiers to find.
    count: int, optional
        Maximum number of results.

    Returns
    -------
    list[RetrievalResult]
        Best lexical matches in descending score order.

    Raises
    ------
    ValueError
        If the question or result count is invalid.

    """

    _check_search(question, count)
    query_terms = _lexical_terms(question)
    if not index.chunks or not query_terms:
        return []

    query_frequency = Counter(query_terms)
    results: list[RetrievalResult] = []
    for chunk, frequencies, length in zip(
        index.chunks,
        index.term_frequencies,
        index.document_lengths,
        strict=True,
    ):
        score = _lexical_score(
            frequencies,
            length,
            query_frequency,
            index.document_frequency,
            len(index.chunks),
            index.average_length,
        )
        if score > 0.0:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    method="lexical",
                )
            )

    return sorted(
        results,
        key=lambda result: (-result.score, result.chunk.identifier),
    )[:count]


def metadata_search(
    chunks: list[Chunk],
    question: str,
    *,
    count: int = 5,
) -> list[RetrievalResult]:
    """Rank chunks by matches in paths, symbols, sections, and kinds.

    Parameters
    ----------
    chunks: list[Chunk]
        Repository chunks to search.
    question: str
        Source name or metadata description to find.
    count: int, optional
        Maximum number of results.

    Returns
    -------
    list[RetrievalResult]
        Best metadata matches in descending score order.

    Raises
    ------
    ValueError
        If the question or result count is invalid.

    """

    _check_search(question, count)
    query_terms = set(_lexical_terms(question))
    results: list[RetrievalResult] = []
    for chunk in chunks:
        values = {
            "symbol": chunk.symbol or "",
            "section": chunk.section or "",
            "filename": chunk.path.name,
            "path": chunk.path.parent.as_posix(),
            "kind": chunk.kind,
        }
        score = 0.0
        for field, weight in METADATA_WEIGHTS:
            field_terms = set(_lexical_terms(values[field]))
            score += weight * len(query_terms.intersection(field_terms))

        if score > 0.0:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    method="metadata",
                )
            )

    return sorted(
        results,
        key=lambda result: (-result.score, result.chunk.identifier),
    )[:count]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Compare two finite vectors, treating a zero vector as unrelated.

    Parameters
    ----------
    left: list[float] | tuple[float, ...]
        First vector.
    right: list[float] | tuple[float, ...]
        Second vector.

    Returns
    -------
    float
        Cosine similarity from -1 to 1.

    Raises
    ------
    ValueError
        If a vector is empty, non-finite, or has a different dimension.

    """

    checked_left = _check_vector(
        left,
        name="Left vector",
        allow_zero=True,
    )
    checked_right = _check_vector(
        right,
        name="Right vector",
        dimension=len(checked_left),
        allow_zero=True,
    )
    left_length = _vector_length(checked_left)
    right_length = _vector_length(checked_right)
    if left_length == 0.0 or right_length == 0.0:
        return 0.0

    product = sum(
        first * second
        for first, second in zip(
            checked_left,
            checked_right,
            strict=True,
        )
    )
    return product / (left_length * right_length)


def _prepare_embeddings(
    vectors: list[list[float]],
) -> tuple[tuple[float, ...], ...]:
    """Check document vectors and return immutable values.

    Parameters
    ----------
    vectors: list[list[float]]
        Raw vectors returned by an embedding model.

    Returns
    -------
    tuple[tuple[float, ...], ...]
        Checked document vectors.

    Raises
    ------
    ValueError
        If a vector is empty, inconsistent, non-finite, or zero.

    """

    if not vectors:
        raise ValueError("The embedder returned no document vectors.")

    checked: list[tuple[float, ...]] = []
    dimension: int | None = None
    for position, vector in enumerate(vectors, start=1):
        values = _check_vector(
            vector,
            name=f"Document embedding {position}",
            dimension=dimension,
            allow_zero=False,
        )
        if dimension is None:
            dimension = len(values)
        checked.append(values)

    return tuple(checked)


def _check_semantic_index(
    chunks: tuple[Chunk, ...],
    embeddings: tuple[tuple[float, ...], ...],
) -> None:
    """Check stored semantic data without requiring a live model.

    Parameters
    ----------
    chunks: tuple[Chunk, ...]
        Stored chunks.
    embeddings: tuple[tuple[float, ...], ...]
        Stored document vectors.

    Raises
    ------
    ValueError
        If stored chunks and vectors are missing or inconsistent.

    """

    if not chunks:
        raise ValueError("A semantic index must contain at least one chunk.")
    if len(chunks) != len(embeddings):
        raise ValueError("Each semantic chunk must have one embedding.")

    dimension: int | None = None
    for position, vector in enumerate(embeddings, start=1):
        values = _check_vector(
            vector,
            name=f"Stored embedding {position}",
            dimension=dimension,
            allow_zero=False,
        )
        if dimension is None:
            dimension = len(values)


def _check_vector(
    vector: Vector,
    *,
    name: str,
    dimension: int | None = None,
    allow_zero: bool,
) -> tuple[float, ...]:
    """Return one checked vector.

    Parameters
    ----------
    vector: list[float] | tuple[float, ...]
        Vector to check.
    name: str
        Name used in error messages.
    dimension: int | None, optional
        Required vector size.
    allow_zero: bool
        Whether an all-zero vector is useful in this context.

    Returns
    -------
    tuple[float, ...]
        Finite floating-point values.

    Raises
    ------
    ValueError
        If the vector is empty, malformed, inconsistent, non-finite, or zero.

    """

    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError(f"{name} must be a non-empty vector.")
    if dimension is not None and len(vector) != dimension:
        raise ValueError(f"{name} has a different embedding dimension.")

    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain only numbers.")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers.")
        values.append(number)

    checked = tuple(values)
    if not allow_zero and _vector_length(checked) == 0.0:
        raise ValueError(f"{name} must not be a zero vector.")
    return checked


def _vector_length(vector: tuple[float, ...]) -> float:
    """Return the Euclidean length of a checked vector.

    Parameters
    ----------
    vector: tuple[float, ...]
        Checked vector.

    Returns
    -------
    float
        Vector length.

    """

    return sqrt(sum(value * value for value in vector))


def _check_search(question: str, count: int) -> None:
    """Check settings shared by all search methods.

    Parameters
    ----------
    question: str
        Search question.
    count: int
        Maximum number of results.

    Raises
    ------
    ValueError
        If either value is invalid.

    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a non-empty search question.")

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")


def _lexical_terms(text: str) -> list[str]:
    """Keep complete identifiers and add their snake/camel-case parts.

    Parameters
    ----------
    text: str
        Text to divide into searchable terms.

    Returns
    -------
    list[str]
        Lowercase terms.

    """

    terms: list[str] = []
    for match in IDENTIFIER.finditer(text):
        raw = match.group(0)
        normalized = raw.lower()
        terms.append(normalized)

        has_parts = "_" in raw or any(
            character.isupper() for character in raw[1:]
        )
        if not has_parts:
            continue

        for part in IDENTIFIER_PART.findall(raw.replace("_", " ")):
            normalized_part = part.lower()
            if normalized_part != normalized:
                terms.append(normalized_part)

    return terms


def _lexical_score(
    frequencies: Mapping[str, int],
    document_length: int,
    query_frequency: Counter[str],
    document_frequency: Mapping[str, int],
    document_count: int,
    average_length: float,
) -> float:
    """Calculate one small BM25-style lexical score.

    Parameters
    ----------
    frequencies: collections.abc.Mapping[str, int]
        Current document term counts.
    document_length: int
        Current document term count.
    query_frequency: collections.Counter[str]
        Query term counts.
    document_frequency: collections.abc.Mapping[str, int]
        Number of documents containing each term.
    document_count: int
        Total number of documents.
    average_length: float
        Average document term count.

    Returns
    -------
    float
        Lexical relevance score.

    """

    score = 0.0
    term_saturation = 1.2
    length_weight = 0.75
    for term, query_count in query_frequency.items():
        term_count = frequencies.get(term, 0)
        if term_count == 0:
            continue

        containing_documents = document_frequency[term]
        inverse_frequency = log(
            1.0
            + (document_count - containing_documents + 0.5)
            / (containing_documents + 0.5)
        )
        denominator = term_count + term_saturation * (
            1.0
            - length_weight
            + length_weight * document_length / average_length
        )
        score += (
            inverse_frequency
            * term_count
            * (term_saturation + 1.0)
            / denominator
            * (1.0 + log(query_count))
        )

    return score
