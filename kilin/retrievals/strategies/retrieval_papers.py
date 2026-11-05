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

- **Relevant papers only**

    Keep the papers with a passage among the best passages of the whole
    collection, unless the question asks about all papers.

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

- ``asks_for_all_papers``

    Whether a question asks about every paper.

- ``relevant_papers``

    Keep the ranked papers relevant to a question.

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
import re
from typing import Final

from kilin.defaults import load_defaults_section, require_default_int
from kilin.retrievals.strategies.retrieval_hybrid import HybridResult

__all__ = [
    "PaperResult",
    "asks_for_all_papers",
    "card_similarities",
    "rank_papers",
    "relevant_papers",
    "select_paper_passages",
]

# "all five papers", "each study", "across the articles": every paper.
_DOCUMENTS: Final[str] = (
    r"(?:papers?|articles?|stud(?:y|ies)|documents?|sources?|pdfs?|works?)"
)
_PLURAL_DOCUMENTS: Final[str] = (
    r"(?:papers|articles|studies|documents|sources|pdfs|works)"
)
_ALL_PAPERS: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:all|each|every)\b(?:\W+\w+){{0,3}}?\W+{_DOCUMENTS}\b"
    r"|\b(?:all|each|every one) of (?:them|these|those)\b"
    rf"|\bacross (?:the |these |those )?{_DOCUMENTS}\b"
    r"|\b(?:two|three|four|five|six|seven|eight|nine|ten|[1-9]\d?)\W+"
    rf"(?:\w+\W+)?{_PLURAL_DOCUMENTS}\b",
    re.IGNORECASE,
)
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


def asks_for_all_papers(question: str) -> bool:
    """Return whether a question asks about every paper.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    bool
        Whether it speaks of all, each, or a number of papers ("all five
        project papers", "each study", "across the articles", "the five
        articles").

    """

    return bool(_ALL_PAPERS.search(question))


def relevant_papers(
    papers: Sequence[PaperResult],
    ranked: Sequence[HybridResult],
    *,
    depth: int,
    minimum: int = 1,
    required: Sequence[Path] = (),
) -> list[PaperResult]:
    """Keep the ranked papers relevant to a question.

    Parameters
    ----------
    papers: Sequence[PaperResult]
        Ranked papers, best first (``rank_papers``).
    ranked: Sequence[HybridResult]
        Fused passage ranking, best first.
    depth: int
        Best passages of the whole collection a relevant paper has one of.
    minimum: int, optional
        Fewest papers kept, best first, even when fewer are relevant; a
        comparison needs two. Defaults to ``1``.
    required: Sequence[pathlib.Path], optional
        Papers always kept first, such as the paper the question means.
        Defaults to ``()``.

    Returns
    -------
    list[PaperResult]
        The required papers, then the other papers with a passage among
        the ``depth`` best passages, in ranking order; topped up with the
        next best papers to ``minimum``.

    Notes
    -----
    Card similarity alone never makes a paper relevant: every card is
    somewhat similar to any question, so a comparison of two papers would
    otherwise take in every paper of the collection.

    """

    near = {result.chunk.path for result in ranked[:depth]}
    keep = [paper for paper in papers if paper.path in required]
    keep += [
        paper for paper in papers
        if paper.path not in required and paper.path in near
    ]
    for paper in papers:
        if len(keep) >= minimum:
            break
        if paper not in keep:
            keep.append(paper)
    return keep


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
