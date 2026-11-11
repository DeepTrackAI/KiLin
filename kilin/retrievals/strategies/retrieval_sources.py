"""Rank sources across four channels, then select original passages.

Unlike passage-first retrieval, source-first fusion gives each source one
position in each available channel. A relevant source card can therefore
identify a document even when none of its chunks is among the global top
passages. Cards influence source discovery, never answer evidence.

Key Features
------------

- **Independent source rankings**

    Convert semantic, lexical, graph, and card rankings to distinct source
    orders before Reciprocal Rank Fusion.

- **Deterministic fusion**

    Give each available channel equal weight, with path-based tie-breaking.

- **Source-backed evidence**

    Reserve a first original passage for each selected source, then fill
    remaining places using the complete passage ranking.

Module Structure
----------------

- ``SourceResult``

    One source ranked across the contributing retrieval channels.

- ``source_first_search``

    Fuse the four supplied rankings at source level.

- ``select_source_passages``

    Select bounded original passages from previously ranked sources.

Examples
--------
>>> from kilin.retrievals.strategies.retrieval_sources import SourceResult
>>> from pathlib import Path
>>> SourceResult(Path("example.pdf"), 1.0, (("cards", 1),)).path.name
'example.pdf'

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kilin.defaults import load_defaults_section, require_default_int
from kilin.retrievals.strategies.retrieval_hybrid import (
    HybridResult,
    RankedResult,
)

__all__ = [
    "SourceResult",
    "select_source_passages",
    "source_first_search",
]

_RRF_RANK_CONSTANT = require_default_int(
    load_defaults_section("RETRIEVAL"), "RRF_RANK_CONSTANT", "RETRIEVAL"
)


class _CardRankedResult(Protocol):
    """Describe one ranked card's source path without a card import.

    Attributes
    ----------
    path: pathlib.Path
        Original source path associated with a ranked card.

    """

    path: Path


@dataclass(frozen=True, slots=True)
class SourceResult:
    """One source ranked by independent channel positions.

    Parameters
    ----------
    path: pathlib.Path
        Original source-relative path.
    score: float
        Sum of channel-specific reciprocal ranks.
    channel_ranks: tuple[tuple[str, int], ...]
        Channel name and one-based source rank, in channel-name order.

    """

    path: Path
    score: float
    channel_ranks: tuple[tuple[str, int], ...]

    @property
    def channels(self) -> tuple[str, ...]:
        """Return the channel names that found this source.

        Returns
        -------
        tuple[str, ...]
            Names in deterministic order.

        """

        return tuple(channel for channel, _ in self.channel_ranks)


def source_first_search(
    *,
    semantic: Sequence[RankedResult] = (),
    lexical: Sequence[RankedResult] = (),
    code: Sequence[RankedResult] = (),
    cards: Sequence[_CardRankedResult] = (),
    count: int = 5,
    rank_constant: int = _RRF_RANK_CONSTANT,
) -> list[SourceResult]:
    """Fuse one rank per distinct source in each available channel.

    Parameters
    ----------
    semantic: Sequence[RankedResult], optional
        Best-first semantic chunk ranking. Defaults to ``()``.
    lexical: Sequence[RankedResult], optional
        Best-first lexical chunk ranking. Defaults to ``()``.
    code: Sequence[RankedResult], optional
        Best-first graph chunk ranking. Defaults to ``()``.
    cards: Sequence[_CardRankedResult], optional
        Best-first source-card ranking. Multiple kinds of the same source
        contribute only once. Defaults to ``()``.
    count: int, optional
        Maximum number of sources returned. Defaults to ``5``.
    rank_constant: int, optional
        RRF smoothing constant from retrieval defaults.

    Returns
    -------
    list[SourceResult]
        Highest-scoring sources first, ties broken by relative path.

    Raises
    ------
    ValueError
        If ``count`` or ``rank_constant`` is not a positive integer.

    Notes
    -----
    Repeated chunks and cards do not change a source's channel position.
    Source ranks are consecutive after deduplication, rather than the
    positions of the original chunks. Raw similarity scores are never
    compared between channels. No model or index is queried here.

    """

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant must be a positive whole number.")

    positions: dict[Path, dict[str, int]] = {}
    rankings = (
        ("semantic", (item.chunk.path for item in semantic)),
        ("lexical", (item.chunk.path for item in lexical)),
        ("code", (item.chunk.path for item in code)),
        ("cards", (item.path for item in cards)),
    )
    for channel, paths in rankings:
        seen: set[Path] = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            positions.setdefault(path, {})[channel] = len(seen)

    results = []
    for path, ranks in positions.items():
        score = sum(
            1.0 / (rank_constant + rank) for rank in ranks.values()
        )
        results.append(SourceResult(
            path=path,
            score=score,
            channel_ranks=tuple(sorted(ranks.items())),
        ))
    results.sort(key=lambda item: (-item.score, item.path.as_posix()))
    return results[:count]


def select_source_passages(
    ranked: Sequence[HybridResult],
    sources: Sequence[Path],
    *,
    count: int = 5,
    per_source: int = 1,
) -> list[HybridResult]:
    """Choose original evidence from selected sources within a hard count.

    Parameters
    ----------
    ranked: Sequence[HybridResult]
        Complete, best-first fused ranking of original chunks.
    sources: Sequence[pathlib.Path]
        Chosen sources in source-relevance order.
    count: int, optional
        Maximum number of original chunks returned. Defaults to ``5``.
    per_source: int, optional
        Desired original passages per selected source, after each source
        receives one. A target subject to ``count``. Defaults to ``1``.

    Returns
    -------
    list[HybridResult]
        Each selected source's best passage, if present and space permits,
        followed by the best remaining passages from the selected sources.

    Raises
    ------
    ValueError
        If ``count`` is not a positive integer.

    Notes
    -----
    Selection does not expand passages or claim exhaustive coverage.
    The later context-expansion stage remains independently configurable.
    Repeated source paths or chunk identifiers are counted only once.

    """

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")
    if type(per_source) is not int or per_source < 1:
        raise ValueError("per_source must be a positive whole number.")

    chosen = tuple(dict.fromkeys(sources))[:count]
    allowed = set(chosen)
    first: dict[Path, HybridResult] = {}
    for result in ranked:
        if result.chunk.path in allowed:
            first.setdefault(result.chunk.path, result)

    selected: list[HybridResult] = []
    seen: set[str] = set()
    for path in chosen:
        result = first.get(path)
        if result is not None and result.chunk.identifier not in seen:
            selected.append(result)
            seen.add(result.chunk.identifier)
    # Fill additional per-source targets without taking the first passage
    # away from another selected source.
    for path in chosen:
        present = sum(result.chunk.path == path for result in selected)
        for result in ranked:
            if len(selected) >= count or present >= per_source:
                break
            if (
                result.chunk.path == path
                and result.chunk.identifier not in seen
            ):
                selected.append(result)
                seen.add(result.chunk.identifier)
                present += 1
    for result in ranked:
        if len(selected) >= count:
            break
        if (
            result.chunk.path in allowed
            and result.chunk.identifier not in seen
        ):
            selected.append(result)
            seen.add(result.chunk.identifier)
    return selected
