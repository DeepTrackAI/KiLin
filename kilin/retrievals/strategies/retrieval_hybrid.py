"""Combine retrieval rankings through reciprocal-rank fusion.

Each nonempty channel contributes reciprocal rank, not its raw score. Results
are merged by chunk identifier across semantic, lexical, and code rankings.
A duplicate within a channel contributes only once, and final ties are broken
by identifier. The operation does not query models, inspect source files, or
recompute the supplied rankings.

Key Features
------------

- **Channel-independent scores**

    Fuse rankings whose original score scales differ.

- **Deduplicated evidence**

    Merge channel contributions for each chunk identifier.

- **Inspectable provenance**

    Record which channels contributed to each fused result.

Module Structure
----------------

- ``HybridResult``

    One chunk ranked across retrieval channels.

- ``hybrid_search``

    Fuse ranked retrieval lists without comparing unlike raw scores.

Examples
--------
>>> import kilin
>>> results = kilin.hybrid_search(count=5)

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kilin.chunking import Chunk
from kilin.defaults import load_defaults_section, require_default_int


__all__ = [
    "HybridResult",
    "hybrid_search",
]

# RRF smoothing is independent of the project's number of returned results.
_RETRIEVAL_DEFAULTS = load_defaults_section("RETRIEVAL")
_DEFAULT_RRF_RANK_CONSTANT = require_default_int(
    _RETRIEVAL_DEFAULTS, "RRF_RANK_CONSTANT", "RETRIEVAL"
)


class RankedResult(Protocol):
    """Structural typing protocol for a ranked item carrying a source chunk.

    Attributes
    ----------
    chunk: Chunk
        Source chunk associated with this position in a ranking.

    """

    chunk: Chunk


@dataclass(frozen=True, slots=True)
class HybridResult:
    """One chunk ranked across retrieval channels.

    Parameters
    ----------
    score: float
        Reciprocal Rank Fusion score.
    chunk: Chunk
        Retrieved chunk.
    channels: tuple[str, ...]
        Channels that returned this chunk.

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
    rank_constant: int = _DEFAULT_RRF_RANK_CONSTANT,
) -> list[HybridResult]:
    """Fuse ranked retrieval lists without comparing unlike raw scores.

    Parameters
    ----------
    semantic: Sequence[RankedResult], optional
        Semantic results in best-first order. Defaults to ``()``.
    lexical: Sequence[RankedResult], optional
        Lexical results in best-first order. Defaults to ``()``.
    code: Sequence[RankedResult], optional
        Code-aware results in best-first order. Defaults to ``()``.
    count: int, optional
        Maximum number of combined results. Defaults to ``5``.
    rank_constant: int, optional
        Reciprocal Rank Fusion smoothing constant, from
        ``RETRIEVAL.RRF_RANK_CONSTANT`` (currently 60).

    Returns
    -------
    list[HybridResult]
        Highest combined rank first.

    Raises
    ------
    ValueError
        If count or rank_constant is invalid.

    Notes
    -----
    Each first occurrence of an identifier in a channel contributes 1 /
    (rank_constant + one-based rank). Duplicate positions still count toward
    subsequent ranks. All channels have equal weight; original scores are
    ignored. Final ties use the chunk identifier, and channels are stored in
    sorted order.

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
