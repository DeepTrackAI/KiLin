"""Rate citation suitability from checked evidence, not search rank.

A retrieval rank measures relative similarity among search hits, and is
not a calibrated probability that a paper supports a statement. Both
reference verification and citation suggestion use the same rubric. This
module gives eligible citations a deterministic ordinal star rating. The
existing support verdict sets the rating band; confirmed quotations
and overlap of distinctive statement words with the cited passage
provide evidence within that band.

Key Features
------------

- **Verdict-first rating**

    A partly supported statement cannot earn the same stars as a directly
    supported statement merely by sharing words with a passage.

- **No inference calls**

    The rating uses the already checked model verdict and passages, with
    no network access, embedding call, or additional language-model call.

Module Structure
----------------

- ``SuitabilityRating``

    A bounded, uncalibrated score and its star representation.

- ``citation_suitability``

    Calculate an evidence-aware rating for an eligible citation.

Examples
--------
>>> from kilin.tools._shared._rating import SuitabilityRating
>>> SuitabilityRating(78, 4).stars
4

"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from kilin.tools._shared.claims import (
    VERDICT_BACKGROUND,
    VERDICT_PARTLY,
    VERDICT_SUPPORTED,
    Support,
    Verdict,
)

__all__ = [
    "SuitabilityRating", "citation_suitability", "suitability_stars",
]

_WORD: Final[re.Pattern[str]] = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS: Final[frozenset[str]] = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "these", "this", "to", "was", "were",
    "which", "with",
))
_BASE: Final[dict[str, int]] = {
    VERDICT_SUPPORTED: 70,
    VERDICT_PARTLY: 40,
    VERDICT_BACKGROUND: 10,
}
# Score thresholds deliberately preserve the ordering of verdict bands.
_THRESHOLDS: Final[tuple[int, ...]] = (25, 55, 70, 85)


@dataclass(frozen=True, slots=True)
class SuitabilityRating:
    """A heuristic score and an ordinal number of stars.

    Attributes
    ----------
    score: int
        Uncalibrated 0-100 suitability indicator, not a probability.
    stars: int
        Ordinal rating from one to five.

    """

    score: int
    stars: int


def citation_suitability(
    support: Support | Verdict,
) -> SuitabilityRating | None:
    """Rate whether a paper is suitable as a citation for a statement.

    Parameters
    ----------
    support: Support | Verdict
        Checked claim or statement, candidate paper, support verdict,
        confirmed quotation when present, and retrieved passages.

    Returns
    -------
    SuitabilityRating | None
        Verdict-first heuristic rating for a citation; None for
        unsupported, contradicted, failed or indeterminate assessments.

    Notes
    -----
    The model's qualitative verdict is not converted into a statistical
    confidence. Textual overlap only refines ratings *within* that
    verdict's band; it never overrides the verdict. A model-claimed
    quotation is counted only when Python verified it in the passages.

    """

    base = _BASE.get(support.verdict)
    if base is None:
        return None

    confirmed = bool(support.quote and support.quote_chunk is not None)
    evidence = (
        support.quote if confirmed else _best_passage(support)
    )
    overlap = _word_coverage(_statement(support), evidence)
    score = base + (10 if confirmed else 0) + round(20 * overlap)
    # High overlap without a checked quote does not raise the star band.
    if not confirmed:
        limits = {
            VERDICT_SUPPORTED: 84,
            VERDICT_PARTLY: 54,
            VERDICT_BACKGROUND: 24,
        }
        score = min(score, limits[support.verdict])
    stars = 1 + sum(score >= threshold for threshold in _THRESHOLDS)
    return SuitabilityRating(score=score, stars=stars)


def _best_passage(support: Support | Verdict) -> str:
    """Select the passage covering most distinctive statement words.

    Parameters
    ----------
    support: Support | Verdict
        Candidate and passages already checked.

    Returns
    -------
    str
        Highest-coverage passage text, or an empty string.

    """

    return max(
        (passage.text for passage in support.passages),
        key=lambda text: _word_coverage(_statement(support), text),
        default="",
    )


def suitability_stars(rating: SuitabilityRating | None) -> str:
    """Format a rating, keeping inconclusive checks explicitly unrated.

    Parameters
    ----------
    rating: SuitabilityRating | None
        The heuristic rating of a supported, partly supported, or
        background citation; None when no rating is justified.

    Returns
    -------
    str
        Stars and an ordinal label, or unfilled stars marked unrated.

    """

    if rating is None:
        return "☆☆☆☆☆ (unrated)"
    stars = "★" * rating.stars + "☆" * (5 - rating.stars)
    return f"{stars} ({rating.stars}/5 suitability)"


def _statement(support: Support | Verdict) -> str:
    """Select claim context or a suggestion's supplied statement.

    Parameters
    ----------
    support: Support | Verdict
        Evaluated citation or proposed reference.

    Returns
    -------
    str
        Words relevant to the cited evidence, including an antecedent
        when a citation says only "this study" or similar.

    """

    if isinstance(support, Support):
        return support.statement
    claim = support.claim
    if claim.before:
        return f"{claim.before} {claim.sentence.text}"
    return claim.sentence.text


def _word_coverage(statement: str, evidence: str) -> float:
    """Measure coverage of the statement's distinctive word types.

    Parameters
    ----------
    statement: str
        Statement requiring a citation.
    evidence: str
        Confirmed quote or retrieved passage.

    Returns
    -------
    float
        Share of statement words present in the evidence, from 0 to 1.

    """

    wanted = {
        word.casefold() for word in _WORD.findall(statement)
        if word.casefold() not in _STOP_WORDS
    }
    if not wanted:
        return 0.0
    found = {word.casefold() for word in _WORD.findall(evidence)}
    return len(wanted & found) / len(wanted)
