"""Combine independent retrieval rankings without comparing raw scores."""

from __future__ import annotations

from dataclasses import dataclass

from qachat.chunking import Chunk
from qachat.retrieval import RetrievalMethod, RetrievalResult

METHOD_REASONS: dict[RetrievalMethod, str] = {
    "semantic": "Meaning-based retrieval ranked this chunk.",
    "lexical": "Exact words or identifiers ranked this chunk.",
    "metadata": "Path, symbol, section, or kind ranked this chunk.",
}


@dataclass(frozen=True, slots=True)
class RankContribution:
    """Explain one retrieval method's contribution to a fused result.

    Attributes
    ----------
    method: {"semantic", "lexical", "metadata"}
        Retrieval method.
    rank: int
        One-based position in that method's result list.
    reason: str
        Plain-language reason the method can retrieve a chunk.

    """

    method: RetrievalMethod
    rank: int
    reason: str


@dataclass(frozen=True, slots=True)
class HybridResult:
    """One deduplicated chunk with a transparent rank-based score.

    Attributes
    ----------
    chunk: Chunk
        Matching source chunk.
    score: float
        Sum of reciprocal ranks, never a sum of method-specific scores.
    contributions: tuple[RankContribution, ...]
        Retrieval methods and ranks that contributed to the result.

    """

    chunk: Chunk
    score: float
    contributions: tuple[RankContribution, ...]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    *,
    count: int = 5,
    rank_constant: int = 60,
) -> list[HybridResult]:
    """Combine ranked result lists using reciprocal-rank fusion.

    Each method keeps its own score scale. Fusion uses only a result's rank
    within each input list, so semantic, lexical, and metadata scores are
    never added or compared.

    Parameters
    ----------
    result_lists: list[list[RetrievalResult]]
        Independent ranked result lists.
    count: int, optional
        Maximum number of fused results.
    rank_constant: int, optional
        Positive constant that controls how quickly rank influence decreases.

    Returns
    -------
    list[HybridResult]
        Deduplicated results in deterministic fused-rank order.

    Raises
    ------
    ValueError
        If settings or result values are invalid.

    """

    _check_settings(count, rank_constant)
    if not isinstance(result_lists, list):
        raise ValueError("result_lists must be a list of ranked lists.")

    chunks: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    contributions: dict[str, list[RankContribution]] = {}

    for ranked_results in result_lists:
        if not isinstance(ranked_results, list):
            raise ValueError("Each ranked result collection must be a list.")

        seen: set[str] = set()
        for rank, result in enumerate(ranked_results, start=1):
            if not isinstance(result, RetrievalResult):
                raise ValueError(
                    "Ranked lists must contain RetrievalResult values."
                )
            if result.method not in METHOD_REASONS:
                raise ValueError("A result has an unknown retrieval method.")

            identifier = result.chunk.identifier
            if identifier in seen:
                continue
            seen.add(identifier)

            chunks.setdefault(identifier, result.chunk)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (
                rank_constant + rank
            )
            contributions.setdefault(identifier, []).append(
                RankContribution(
                    method=result.method,
                    rank=rank,
                    reason=METHOD_REASONS[result.method],
                )
            )

    fused = [
        HybridResult(
            chunk=chunks[identifier],
            score=score,
            contributions=tuple(contributions[identifier]),
        )
        for identifier, score in scores.items()
    ]
    return sorted(
        fused,
        key=lambda result: (
            -result.score,
            min(contribution.rank for contribution in result.contributions),
            result.chunk.identifier,
        ),
    )[:count]


def _check_settings(count: int, rank_constant: int) -> None:
    """Check reciprocal-rank fusion settings.

    Parameters
    ----------
    count: int
        Maximum number of fused results.
    rank_constant: int
        Rank influence constant.

    Raises
    ------
    ValueError
        If either setting is not a positive whole number.

    """

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant must be a positive whole number.")
