"""Combine independent retrieval rankings with Reciprocal Rank Fusion.

Classes
-------
HybridResult
    One chunk with its combined score and contributing channels.

Functions
---------
hybrid_search
    Fuse semantic, lexical, and code-aware ranked results.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from qachat.chunking import Chunk


__all__ = [
    "HybridResult",
    "hybrid_search",
]


class RankedResult(Protocol):
    """Required shape of one result from a retrieval channel."""

    chunk: Chunk


@dataclass(frozen=True, slots=True)
class HybridResult:
    """One chunk ranked across retrieval channels.

    Attributes
    ----------
    score: float
        Reciprocal Rank Fusion score.
    chunk: Chunk
        Retrieved chunk.
    channels: tuple[str, ...]
        Channels that returned this chunk.

    """

    score: float
    chunk: Chunk
    channels: tuple[str, ...]


def hybrid_search(
    *,
    semantic: Sequence[RankedResult] = (),
    lexical: Sequence[RankedResult] = (),
    code: Sequence[RankedResult] = (),
    count: int = 5,
    rank_constant: int = 60,
) -> list[HybridResult]:
    """Fuse ranked retrieval lists without comparing unlike raw scores.

    Parameters
    ----------
    semantic: collections.abc.Sequence[RankedResult], optional
        Semantic results in best-first order.
    lexical: collections.abc.Sequence[RankedResult], optional
        Lexical results in best-first order.
    code: collections.abc.Sequence[RankedResult], optional
        Code-aware results in best-first order.
    count: int, optional
        Maximum number of combined results.
    rank_constant: int, optional
        Reciprocal Rank Fusion smoothing constant.

    Returns
    -------
    list[HybridResult]
        Highest combined rank first.

    Raises
    ------
    ValueError
        If count or rank_constant is invalid.

    """

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant must be a positive whole number.")

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    channels: dict[str, set[str]] = {}
    rankings = (
        ("semantic", semantic),
        ("lexical", lexical),
        ("code", code),
    )
    for channel, results in rankings:
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            identifier = result.chunk.identifier
            if identifier in seen:
                continue
            seen.add(identifier)
            chunks[identifier] = result.chunk
            channels.setdefault(identifier, set()).add(channel)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (
                rank_constant + rank
            )

    combined = [
        HybridResult(
            score=score,
            chunk=chunks[identifier],
            channels=tuple(sorted(channels[identifier])),
        )
        for identifier, score in scores.items()
    ]
    combined.sort(
        key=lambda result: (
            -result.score,
            result.chunk.identifier,
        )
    )
    return combined[: min(count, len(combined))]
