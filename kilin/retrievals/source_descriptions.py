"""Resolve papers a question describes instead of naming.

"The 2017 paper", "the BRAPH 2 paper" or "the earlier BRAPH framework"
can describe indexed documents without naming their files. Source cards
record each source's title and year; ambiguous descriptions do not match.
Explicit versioned names ("BRAPH 2", "BRAPH2") also resolve when a
unique source-card title starts with that name.

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

    Return unambiguous described or version-named sources.

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
# Recognize an explicitly mentioned versioned title, with or without
# a space before the version ("BRAPH 2", "BRAPH2"). Match the start
# of a title, not arbitrary topical words within it.
_VERSION_NAME: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)(?P<name>[a-z][a-z_-]{2,})[\s-]*"
    r"(?P<version>[1-9]\d?)(?!\w)", re.IGNORECASE,
)
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
        In question order, each uniquely matched source. Descriptions
        match card years (or author-year stems) and title words, with
        "earlier" or "newer" selecting a unique oldest or newest paper.
        Versioned names can match a unique card-title prefix without a
        document noun; joined and spaced version numbers are equivalent.
        Uninformative descriptions ("the earlier paper") match nothing.

    """

    matches: list[tuple[int, Path]] = []
    for match in _DESCRIPTION.finditer(question):
        path = _resolve(_tokens(match.group("words")), cards)
        if path is not None:
            matches.append((match.start(), path))
    for match in _YEAR_AFTER.finditer(question):
        path = _resolve([match.group("year")], cards)
        if path is not None:
            matches.append((match.start(), path))
    for match in _VERSION_NAME.finditer(question):
        name = match.group("name").strip("-_").casefold()
        version = match.group("version")
        candidates = [
            path for path, card in cards.items()
            if _version_prefix(card.get("title", ""), name, version)
        ]
        if len(candidates) == 1:
            matches.append((match.start(), candidates[0]))
    found: list[Path] = []
    for _, path in sorted(matches, key=lambda item: item[0]):
        if path not in found:
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
        and set(title_words) <= _title_terms(card.get("title", ""))
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


def _version_prefix(title: str, name: str, version: str) -> bool:
    """Match an exact versioned name at the start of a document title.

    Parameters
    ----------
    title: str
        Source-card title.
    name: str
        Mentioned name, case-folded.
    version: str
        Mentioned version number.

    Returns
    -------
    bool
        Whether the title starts with that versioned name.

    """

    tokens = _tokens(title)
    return bool(
        tokens[:2] == [name, version]
        or (tokens and tokens[0] == name + version)
    )


def _title_terms(title: str) -> set[str]:
    """Return title words with conservative version-number aliases.

    Parameters
    ----------
    title: str
        The source-card title.

    Returns
    -------
    set[str]
        Exact words plus joined/spaced variants for numbered names.

    """

    tokens = _tokens(title)
    terms = set(tokens)
    for word, number in zip(tokens, tokens[1:]):
        if word.isalpha() and number.isdigit() and len(number) <= 2:
            terms.add(word + number)
    for token in tokens:
        match = re.fullmatch(r"([a-z]+)([1-9]\d?)", token)
        if match:
            terms.update(match.groups())
    return terms


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
