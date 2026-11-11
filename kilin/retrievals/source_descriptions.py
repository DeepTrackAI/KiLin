"""Resolve papers a question describes instead of naming.

"The 2017 paper", "the BRAPH 2 paper" or "the earlier BRAPH framework"
mean one document without naming its file. Source cards record each
source's title and year, so such a description can be matched to one
source, which then limits the search as a named file does. Nothing is
guessed: a description resolves only when exactly one source fits it.

Key Features
------------

- **Year and title words**

    "The 2017 paper" matches the card year; "the BRAPH 2 paper" matches
    every word against the card title.

- **Earlier and later**

    "The earlier BRAPH framework" picks the oldest source whose title has
    the words; "the newer one" picks the most recent.

- **No guessing**

    Ambiguous or unmatched descriptions resolve to nothing.

Module Structure
----------------

- ``described_sources``

    Return the sources a question describes.

Examples
--------
>>> from pathlib import Path
>>> cards = {
...     Path("a.pdf"): {"title": "BRAPH: graph software", "year": "2017"},
...     Path("b.pdf"): {"title": "BRAPH 2: a framework", "year": "2026"},
... }
>>> described_sources("How does it compare with the 2017 paper?", cards)
[PosixPath('a.pdf')]

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Final

__all__ = ["described_sources"]

# Nouns that end a description of one document.
_NOUN: Final[str] = (
    r"(?:paper|article|study|work|publication|preprint|manuscript|version"
    r"|framework|software|toolbox|toolkit)"
)
# "the 2017 paper", "the earlier BRAPH framework", "this deep learning
# article": a determiner, up to four words, and a document noun.
_DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:the|that|this)\s+(?P<words>(?:[\w-]+\s+){{1,4}}?){_NOUN}\b",
    re.IGNORECASE,
)
# "the paper from 2017", "that study published in 2019"; not "which
# paper from 2024", a search among the papers.
_YEAR_AFTER: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:the|that|this)\s+{_NOUN}\s+(?:from|of|in|published\s+in)\s+"
    r"(?P<year>(?:19|20)\d\d)\b",
    re.IGNORECASE,
)
_YEAR: Final[re.Pattern[str]] = re.compile(r"^(?:19|20)\d\d$")
# An author-year file name: "smith2017", "patil25deep".
_STEM_YEAR: Final[re.Pattern[str]] = re.compile(
    r"^[a-z]+(?P<year>\d{4}|\d{2})(?![0-9])"
)
# Words that order sources by year instead of describing one.
_OLDEST: Final[frozenset[str]] = frozenset(
    {"earlier", "original", "first", "previous", "older", "old"}
)
_NEWEST: Final[frozenset[str]] = frozenset(
    {"later", "newer", "latest", "new", "recent", "updated"}
)
# Words that describe no particular source.
_IGNORED: Final[frozenset[str]] = frozenset(
    {"a", "an", "the", "of", "on", "in", "about", "for", "and", "same",
     "other", "whole", "main", "entire", "one", "two", "both", "each"}
)


def described_sources(
    question: str, cards: Mapping[Path, Mapping[str, str]]
) -> list[Path]:
    """Return the sources a question describes by year or title.

    Parameters
    ----------
    question: str
        The user's question.
    cards: Mapping[pathlib.Path, Mapping[str, str]]
        For each source with a card, its metadata; ``title`` and ``year``
        are used.

    Returns
    -------
    list[pathlib.Path]
        In the order the question describes them, each source that exactly
        one description fits: a year equal to the source's year (from its
        card, else from an author-year file name; a source with no known
        year fits any year), title words all found in the card's title
        (ignoring case and hyphens), or both;
        with "earlier", "original" (or "later", "newer") the oldest (or
        newest) of the sources that fit. A description with neither a year
        nor a title word ("the earlier paper") resolves to nothing.

    """

    found: list[Path] = []
    for match in _DESCRIPTION.finditer(question):
        words = _tokens(match.group("words"))
        path = _resolve(words, cards)
        if path is not None and path not in found:
            found.append(path)
    for match in _YEAR_AFTER.finditer(question):
        path = _resolve([match.group("year")], cards)
        if path is not None and path not in found:
            found.append(path)
    return found


def _resolve(
    words: list[str], cards: Mapping[Path, Mapping[str, str]]
) -> Path | None:
    """Return the one source a description fits, if any.

    Parameters
    ----------
    words: list[str]
        The description's words, lowercase.
    cards: Mapping[pathlib.Path, Mapping[str, str]]
        Source metadata.

    Returns
    -------
    pathlib.Path | None
        The single fitting source, or None.

    """

    order = (
        "oldest" if _OLDEST & set(words)
        else "newest" if _NEWEST & set(words) else None
    )
    years = [word for word in words if _YEAR.match(word)]
    title_words = [
        word for word in words
        if word not in _OLDEST | _NEWEST | _IGNORED and word not in years
    ]
    if not years and not title_words:
        return None

    known = {path: _year(path, card) for path, card in cards.items()}
    # A source without a known year may be the one meant, so it stays a
    # candidate for any year and keeps the description ambiguous.
    fitting = [
        path for path, card in cards.items()
        if all(known[path] in (None, int(year)) for year in years)
        and set(title_words) <= set(_tokens(card.get("title", "")))
    ]
    if len(fitting) > 1 and order is not None:
        dated = [
            (known[path], path) for path in fitting
            if known[path] is not None
        ]
        if len(dated) == len(fitting):
            dated.sort(key=lambda item: item[0])
            pick = dated[0] if order == "oldest" else dated[-1]
            same = [item for item in dated if item[0] == pick[0]]
            fitting = [pick[1]] if len(same) == 1 else fitting
    return fitting[0] if len(fitting) == 1 else None


def _year(path: Path, card: Mapping[str, str]) -> int | None:
    """Return a source's year: from its card, else from its file name.

    Parameters
    ----------
    path: pathlib.Path
        The source.
    card: Mapping[str, str]
        Its card metadata.

    Returns
    -------
    int | None
        The card's year; otherwise the year an author-year file name
        starts with ("smith2017.pdf", "patil25deep.pdf" is 2025, a
        two-digit year below 50 being in the 2000s); otherwise None.

    """

    year = card.get("year", "").strip()
    if _YEAR.match(year):
        return int(year)
    stem = _STEM_YEAR.match(path.stem.casefold())
    if stem is None:
        return None
    digits = stem.group("year")
    if len(digits) == 4:
        return int(digits) if _YEAR.match(digits) else None
    return 2000 + int(digits) if int(digits) < 50 else 1900 + int(digits)


def _tokens(text: str) -> list[str]:
    """Return lowercase words, splitting at hyphens and punctuation.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    list[str]
        Its words and numbers.

    """

    return re.findall(r"[^\W_]+", text.casefold())
