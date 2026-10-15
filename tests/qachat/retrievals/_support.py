"""Deterministic, offline fixtures shared by retrieval tests.

These fixtures deliberately use simple in-memory documents and vectors. A
separate integration suite exercises the actual embedding model backends.

"""

from __future__ import annotations

from pathlib import Path

from qachat.chunking import Chunk
from qachat.retrievals.code_graph import CodeEntity, CodeGraph, CodeRelation
from qachat.retrievals.strategies.retrieval_hybrid import HybridResult
from qachat.sources.sources import SourceFile, SourceUnit


def chunk(
    identifier: str,
    text: str = "example text",
    *,
    path: str = "src/example.py",
    start: int = 1,
    end: int | None = None,
    symbol: str | None = None,
    kind: str = "python_function",
) -> Chunk:
    """Construct a source chunk with explicit, reproducible provenance.

    Parameters
    ----------
    identifier: str
        Stable identifier used to distinguish test chunks.
    text: str, optional
        Exact source content. Defaults to ``'example text'``.
    path: str, optional
        Repository-relative source path.
    start: int, optional
        First line; defaults to ``1``.
    end: int | None, optional
        Last line; defaults to the first line.
    symbol: str | None, optional
        Source symbol, if present.
    kind: str, optional
        Structural source-unit category.

    Returns
    -------
    Chunk
        New chunk carrying the requested source fields.

    """

    return Chunk(
        identifier=identifier,
        path=Path(path),
        text=text,
        start_line=start,
        end_line=start if end is None else end,
        kind=kind,
        symbol=symbol,
        section=None,
    )


def source_file(path: str, text: str) -> SourceFile:
    """Construct a minimal extracted source-file record.

    Parameters
    ----------
    path: str
        Repository-relative source path.
    text: str
        Source text.

    Returns
    -------
    SourceFile
        Source file with the required metadata.

    """

    return SourceFile(path=Path(path), text=text)


def source_unit(
    path: str,
    text: str,
    *,
    symbol: str | None = None,
    kind: str = "python_function",
    children: tuple[SourceUnit, ...] = (),
) -> SourceUnit:
    """Construct a source unit for code-graph tests.

    Parameters
    ----------
    path: str
        Repository-relative path.
    text: str
        Exact unit content.
    symbol: str | None, optional
        Source symbol.
    kind: str, optional
        Source unit kind.
    children: tuple[SourceUnit, ...], optional
        Nested source units.

    Returns
    -------
    SourceUnit
        Source unit with source lines and optional hierarchy.

    """

    return SourceUnit(
        path=Path(path),
        text=text,
        start_line=1,
        end_line=max(1, len(text.splitlines())),
        kind=kind,
        symbol=symbol,
        children=children,
    )


def graph(
    *names: str,
    relations: tuple[tuple[str, str, str], ...] = (),
    path: str = "src/example.py",
    kind: str = "python_function",
) -> CodeGraph:
    """Create a code graph with named entities and directed edges.

    Parameters
    ----------
    *names: str
        Fully qualified entity names.
    relations: tuple[tuple[str, str, str], ...], optional
        Source, relation, and target triples.
    path: str, optional
        Path shared by the graph entities.
    kind: str, optional
        Kind shared by the graph entities.

    Returns
    -------
    CodeGraph
        Minimal graph suitable for retrieval tests.

    """

    return CodeGraph(
        entities=tuple(
            CodeEntity(name, kind, Path(path)) for name in names
        ),
        relations=tuple(CodeRelation(*edge) for edge in relations),
    )


def ranked(value: Chunk, score: float = 1.0) -> HybridResult:
    """Make a ranked chunk suitable for routing and document reading.

    Parameters
    ----------
    value: Chunk
        Retrieved source chunk.
    score: float, optional
        Fusion score used for dominance.

    Returns
    -------
    HybridResult
        Ranked result from the semantic channel.

    """

    return HybridResult(score, value, ("semantic",))


class FakeEmbeddings:
    """Provide controlled, deterministic vector generation and bookkeeping.

    Parameters
    ----------
    mapping: dict[str, list[float]] | None, optional
        Exact vectors for document text and queries.
    dimension: int, optional
        Dimension used for unknown strings.

    Attributes
    ----------
    calls: list[list[str]]
        Text batches submitted for embedding.
    queries: list[tuple[str, str | None]]
        Query text and tracing identifiers.
    cleared: int
        Number of backend-cache clear requests.

    """

    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        *,
        dimension: int = 2,
    ) -> None:
        """Initialize synthetic vectors without loading a model.

        """
        self.mapping = mapping or {}
        self.dimension = dimension
        self.calls: list[list[str]] = []
        self.queries: list[tuple[str, str | None]] = []
        self.cleared = 0

    def _vector(self, value: str) -> list[float]:
        """Return a known vector or a reproducible fallback vector.

        Parameters
        ----------
        value: str
            Query or document text.

        Returns
        -------
        list[float]
            Reproducible numeric vector.

        """
        if value in self.mapping:
            return list(self.mapping[value])
        return [1.0, float(len(value))] + [0.0] * (self.dimension - 2)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch and record its exact text values.

        Parameters
        ----------
        texts: list[str]
            Batch of original text.

        Returns
        -------
        list[list[float]]
            Per-document embeddings.

        """
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(
        self,
        query_text: str,
        *,
        trace_turn_id: str | None = None,
    ) -> list[float]:
        """Embed one query while retaining the tracing argument.

        Parameters
        ----------
        query_text: str
            Query as supplied by the caller.
        trace_turn_id: str | None, optional
            Optional trace turn ID.

        Returns
        -------
        list[float]
            Query embedding.

        """
        self.queries.append((query_text, trace_turn_id))
        return self._vector(query_text)

    def _clear_backend_cache(self) -> None:
        """Record a cache-release request without using hardware.

        """
        self.cleared += 1
