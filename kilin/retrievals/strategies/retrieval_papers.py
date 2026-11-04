"""Choose papers first, then passages inside them.

Plain hybrid retrieval keeps the best few passages of the whole collection.
In a literature project those can all come from one paper, and a paper
that answers the question can be missing from them because none of its
passages ranks high on its own. Paper retrieval ranks the papers first, by
their cards (``paper_cards.py``) and by their best passages, then fills the
evidence from the chosen papers: the best passages overall, with the best
paper always represented, or a guaranteed number of passages from each
chosen paper when sources are compared.

Key Features
------------

- **Paper ranking**

    Fuse the similarity of the question to each paper's card with the rank
    of the paper's best passage, by Reciprocal Rank Fusion.

- **Passage selection**

    Fill the evidence from the chosen papers, with a minimum number of
    passages for each paper when requested.

Module Structure
----------------

- ``PaperResult``

    One paper ranked for a question.

- ``card_similarities``

    Similarity of a query vector to each paper's card.

- ``rank_papers``

    Rank papers by their cards and their best passages.

- ``select_paper_passages``

    Fill the evidence from the chosen papers.

Examples
--------
>>> from pathlib import Path
>>> from kilin.retrievals.strategies.retrieval_papers import rank_papers
>>> best = rank_papers([], {Path("a.pdf"): 0.9, Path("b.pdf"): 0.2})[0]
>>> best.path.as_posix()
'a.pdf'

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from kilin.defaults import load_defaults_section, require_default_int
from kilin.retrievals.strategies.retrieval_hybrid import HybridResult

__all__ = [
    "PaperResult",
    "card_similarities",
    "rank_papers",
    "select_paper_passages",
]

# The same smoothing as passage fusion.
_RRF_RANK_CONSTANT = require_default_int(
    load_defaults_section("RETRIEVAL"), "RRF_RANK_CONSTANT", "RETRIEVAL"
)


@dataclass(frozen=True, slots=True)
class PaperResult:
    """One paper ranked for a question.

    Parameters
    ----------
    path: pathlib.Path
        Paper path.
    score: float
        Fused score; higher is better.
    card_rank: int | None
        One-based rank of the paper's card, or ``None`` without a card.
    passage_rank: int | None
        One-based rank of the paper's best passage among papers, or
        ``None`` when none of its passages was ranked.

    Attributes
    ----------
    path: pathlib.Path
        Paper path.
    score: float
        Fused score; higher is better.
    card_rank: int | None
        One-based rank of the paper's card, or ``None`` without a card.
    passage_rank: int | None
        One-based rank of the paper's best passage among papers.

    """

    path: Path
    score: float
    card_rank: int | None
    passage_rank: int | None


def card_similarities(
    query_vector: Sequence[float],
    vectors: Mapping[Path, Sequence[Sequence[float]]],
) -> dict[Path, float]:
    """Return the similarity of a query vector to each paper's card.

    Parameters
    ----------
    query_vector: Sequence[float]
        Embedded question.
    vectors: Mapping[pathlib.Path, Sequence[Sequence[float]]]
        Search vectors of each card (``card_texts``).

    Returns
    -------
    dict[pathlib.Path, float]
        Highest cosine similarity of the query to any of a card's vectors;
        papers without vectors, or with vectors of another size, are left
        out.

    """

    query_norm = sqrt(sum(value * value for value in query_vector))
    if not query_norm:
        return {}
    scores: dict[Path, float] = {}
    for path, card_vectors in vectors.items():
        best = None
        for vector in card_vectors:
            if len(vector) != len(query_vector):
                continue
            norm = sqrt(sum(value * value for value in vector))
            if not norm:
                continue
            score = sum(
                a * b for a, b in zip(query_vector, vector)
            ) / (query_norm * norm)
            best = score if best is None else max(best, score)
        if best is not None:
            scores[path] = best
    return scores


def rank_papers(
    ranked: Sequence[HybridResult],
    card_scores: Mapping[Path, float],
    *,
    rank_constant: int = _RRF_RANK_CONSTANT,
) -> list[PaperResult]:
    """Rank papers by their cards and their best passages.

    Parameters
    ----------
    ranked: Sequence[HybridResult]
        Fused passage ranking, best first.
    card_scores: Mapping[pathlib.Path, float]
        Similarity of the question to each paper's card.
    rank_constant: int, optional
        Reciprocal Rank Fusion constant. Defaults to the retrieval setting
        ``RRF_RANK_CONSTANT``.

    Returns
    -------
    list[PaperResult]
        Every paper with a card or a ranked passage, best first. Each list
        contributes ``1 / (rank_constant + rank)``; ties keep the order of
        the passage ranking, then of the paths.

    """

    passage_order: list[Path] = []
    for result in ranked:
        if result.chunk.path not in passage_order:
            passage_order.append(result.chunk.path)
    card_order = sorted(
        card_scores, key=lambda path: (-card_scores[path], path.as_posix())
    )
    passage_rank = {path: rank for rank, path in
                    enumerate(passage_order, start=1)}
    card_rank = {path: rank for rank, path in enumerate(card_order, start=1)}
    papers = []
    for path in dict.fromkeys([*passage_order, *card_order]):
        score = sum(
            1.0 / (rank_constant + ranks[path])
            for ranks in (passage_rank, card_rank)
            if path in ranks
        )
        papers.append(PaperResult(
            path, score, card_rank.get(path), passage_rank.get(path)
        ))
    tie = {path: index for index, path in enumerate(
        dict.fromkeys([*passage_order, *sorted(card_order)])
    )}
    papers.sort(key=lambda paper: (-paper.score, tie[paper.path]))
    return papers


def select_paper_passages(
    ranked: Sequence[HybridResult],
    papers: Sequence[Path],
    *,
    count: int,
    per_paper: int = 0,
) -> list[HybridResult]:
    """Fill the evidence from the chosen papers.

    Parameters
    ----------
    ranked: Sequence[HybridResult]
        Fused passage ranking, best first.
    papers: Sequence[pathlib.Path]
        Chosen papers, best first.
    count: int
        Passages to keep at least.
    per_paper: int, optional
        Passages guaranteed to each chosen paper; zero guarantees one to
        the best paper only. Defaults to ``0``.

    Returns
    -------
    list[HybridResult]
        With ``per_paper`` zero, the ``count`` best passages of the chosen
        papers, the best paper's best passage replacing the last one when
        the best paper has none among them. Otherwise each chosen paper's
        ``per_paper`` best passages, then the best remaining passages of the
        chosen papers, up to ``max(count, per_paper * len(papers))``
        passages. Passages keep their ranking order in the result.

    """

    chosen = set(papers)
    pool = [result for result in ranked if result.chunk.path in chosen]
    if not pool:
        return []
    if per_paper <= 0:
        kept = pool[:count]
        best_paper = papers[0] if papers else None
        if best_paper is not None and all(
            result.chunk.path != best_paper for result in kept
        ):
            best = next(
                (r for r in pool if r.chunk.path == best_paper), None
            )
            if best is not None:
                kept = [*kept[:max(count - 1, 0)], best]
        return [result for result in pool if result in kept]
    total = max(count, per_paper * len(papers))
    picked: list[HybridResult] = []
    for paper in papers:
        picked.extend(
            [result for result in pool if result.chunk.path == paper][
                :per_paper
            ]
        )
    for result in pool:
        if len(picked) >= total:
            break
        if result not in picked:
            picked.append(result)
    selected = {result.chunk.identifier for result in picked}
    return [result for result in pool if result.chunk.identifier in selected]
