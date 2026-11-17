"""Find author-year citations and resolve them to reference-list entries.

Papers in author-year styles cite "(Smith 2020)", "(Smith et al., 2020;
Lee and Wu 2019a)" or "Smith et al. (2020)" instead of numbers, with an
unnumbered, alphabetical reference list. This module, shared by
``reference_check`` (which cites by entry number) and ``writing_check``
(which must not let a suggestion change a citation), reads both sides:

1. ``find_markers`` finds parenthetical and narrative markers in a text,
   each with the works it cites (authors as cited, year, suffix).
2. ``entry_keys`` reads each entry's first and second author, number of
   authors and year, and gives entries of the same first author and year
   the suffixes "a", "b", ... in list order when none is printed.
3. ``resolve`` returns the entries a cited work can be: one when it is
   resolved, several when ambiguous, none when not in the list.

Names are compared by their last word, without accents and case, so
"(van den Heuvel 2020)", "(Van den Heuvel, 2020)" and "van den Heuvel MP"
agree. Anything that only looks like a citation ("(Phase 2020)") is found
too: callers count only markers that resolve.

Key Features
------------

- **Parenthetical and narrative markers**

    Several works per marker (";" or ","), several years per author
    ("Smith 2020a, b, 2021"), prefixes ("e.g.", "see"), locators
    ("p. 5"), "et al.", "and"/"&", "n.d." and "in press".

- **Entry keys**

    First and second surname, author count and year from the common
    entry styles (APA, Harvard, Nature, Vancouver, Chicago).

Module Structure
----------------

- ``CitedWork`` and ``AuthorYearMarker``

    One work a marker cites, and one marker with its works.

- ``EntryKey``, ``entry_key`` and ``entry_keys``

    What an entry is cited by.

- ``find_markers`` and ``resolve``

    Markers in a text, and the entries a work can be.

- ``normalized_name``

    A surname's last word as compared.

Examples
--------
>>> from kilin.tools._shared.author_year import (
...     entry_keys, find_markers, resolve)
>>> keys = entry_keys([(1, "Lee, B. (2019). T. J 1, 2."),
...                    (2, "Smith, J., Wu, C., & Ode, D. (2020). T.")])
>>> [m.text for m in find_markers("As shown (Smith et al., 2020).")]
['(Smith et al., 2020)']
>>> work = find_markers("As shown (Smith et al., 2020).")[0].works[0]
>>> resolve(work, keys)
(2,)

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
import re
from typing import Final
import unicodedata

__all__ = [
    "AuthorYearMarker",
    "CitedWork",
    "EntryKey",
    "entry_key",
    "entry_keys",
    "find_markers",
    "normalized_name",
    "resolve",
]

# A capital letter, including accented and other Latin ones: any letter
# that is not a lowercase ASCII or Latin-1 letter.
_UPPER: Final[str] = r"[^\W\d_a-zß-ÿ]"
_WORD: Final[str] = rf"{_UPPER}[\w'’\-]*"
_PARTICLE: Final[str] = (
    r"(?:(?i:van|von|der|den|de|del|della|des|di|da|dos|du|la|le|ten|ter|"
    r"al|el|bin|ibn|mac|st\.?)\s+)"
)
# One surname, possibly with particles; in parentheses also of two words
# ("García López"), which in running text would take the word before.
_NAME_ONE: Final[str] = rf"{_PARTICLE}*{_WORD}"
_NAME_TWO: Final[str] = rf"{_PARTICLE}*{_WORD}(?:\s+{_WORD})?"
_ET_AL: Final[str] = r"\s+et\s+al\.?"


def _authors(name: str) -> str:
    """Return the pattern of an author group built from one name pattern.

    Parameters
    ----------
    name: str
        The pattern of one surname.

    Returns
    -------
    str
        "Smith", "Smith et al.", "Smith and Lee", "Smith & Lee",
        "Smith, Lee and Wu", each name a ``name`` match.

    """

    return (
        rf"{name}(?:\s*,\s*{name})*(?:\s*,?\s+(?:and|&)\s+{name})?"
        rf"(?:{_ET_AL})?"
    )


_YEAR: Final[str] = (
    r"(?:(?:1[89]\d{2}|20\d{2})[a-z]?|n\.\s?d\.|in\s+press|forthcoming)"
)
_YEARS: Final[str] = rf"{_YEAR}(?:\s*,\s*(?:{_YEAR}|[a-z]\b(?!\.)))*"
_LOCATOR: Final[str] = (
    r"(?:\s*[,:]\s*(?i:pp?\.|chap\.|ch\.|sec\.|§|fig\.|figs\.|table)\s*"
    r"[\w–\-]+(?:\s*[–\-]\s*\w+)?)"
)
_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"(?i:see\s+also|see|e\.\s?g\.|i\.\s?e\.|cf\.|for\s+example|"
    r"for\s+a?\s*reviews?|reviewed\s+in|but\s+see|as\s+in|after)\s*,?\s+"
)
_WORK: Final[re.Pattern[str]] = re.compile(
    rf"(?P<authors>{_authors(_NAME_TWO)})\s*,?\s+(?P<years>{_YEARS})"
    rf"(?:{_LOCATOR})?"
)
_PARENTHESES: Final[re.Pattern[str]] = re.compile(
    r"\((?P<inner>[^()]{4,400})\)"
)
_NARRATIVE: Final[re.Pattern[str]] = re.compile(
    rf"(?<![\w(])(?P<authors>{_authors(_NAME_ONE)})(?:['’]s)?\s+"
    rf"\((?P<years>{_YEARS})(?:{_LOCATOR})?\)"
)
_COMMA: Final[re.Pattern[str]] = re.compile(r"\s*,\s*")
_NAME_SPLIT: Final[re.Pattern[str]] = re.compile(
    r"\s*,\s*(?:(?:and|&)\s+)?|\s*,?\s+(?:and|&)\s+"
)
_YEAR_ITEM: Final[re.Pattern[str]] = re.compile(
    r"(?P<year>1[89]\d{2}|20\d{2}|n\.\s?d\.|in\s+press|forthcoming)?"
    r"(?P<suffix>[a-z])?$"
)
# Words that start a sentence or a parenthesis without being a surname.
_NOT_NAMES: Final[frozenset[str]] = frozenset({
    "and", "or", "in", "on", "the", "a", "an", "since", "from", "until",
    "by", "for", "after", "before", "during", "between", "phase", "figure",
    "fig", "table", "section", "chapter", "step", "version", "eq", "ref",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "spring",
    "summer", "autumn", "fall", "winter", "covid", "sars", "who",
    "however", "moreover", "furthermore", "thus", "therefore", "hence",
    "indeed", "here", "there", "then", "also", "yet", "still", "later",
    "first", "second", "third", "finally", "instead", "conversely",
    "similarly", "likewise", "notably", "previously", "recently", "this",
    "these", "that", "those", "as", "but", "while", "whereas", "although",
    "when", "where", "if", "unlike", "like", "following", "according",
})

# Entries: the year (preferably in parentheses), and the author styles.
_ENTRY_YEAR: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\((?P<year>1[89]\d{2}|20\d{2})(?P<suffix>[a-z]?)[),;]"),
    re.compile(
        r"(?:^|[\s.,;])(?P<year>1[89]\d{2}|20\d{2})(?P<suffix>[a-z]?)"
        r"(?=[.;:,)]|\s|$)"
    ),
    re.compile(r"\((?P<year>n\.\s?d\.|in\s+press|forthcoming)\)", re.I),
)
_INITIALS: Final[str] = r"(?:[A-Z][a-z]?\.(?:\s*-?\s*[A-Z][a-z]?\.)*)"
# "Smith, J. A., Lee, B., & Wu, C." (APA, Harvard, Nature).
_COMMA_AUTHOR: Final[re.Pattern[str]] = re.compile(
    rf"\s*(?:,\s*)?(?:(?:and|&)\s+)?(?P<name>{_NAME_TWO}),\s+{_INITIALS}"
)
# "Smith JA, Lee B, Wu C." (Vancouver).
_SPACE_AUTHOR: Final[re.Pattern[str]] = re.compile(
    rf"\s*(?:,\s*)?(?:(?:and|&)\s+)?(?P<name>{_NAME_TWO})\s+"
    r"[A-Z]{1,3}(?=[,.\s])"
)
_ENTRY_ET_AL: Final[re.Pattern[str]] = re.compile(r"\bet\s+al\b", re.I)
_AUTHORS_MANY: Final[int] = 3


@dataclass(frozen=True, slots=True)
class CitedWork:
    """One work an author-year marker cites.

    Attributes
    ----------
    first: str
        The first author's surname as compared (``normalized_name``).
    second: str
        The second author's, when two are cited ("Smith and Lee"); else
        ``""``.
    authors: int
        Authors named: 1, 2, or 3 for three or more ("et al.", or three
        names).
    year: str
        "2020", "n.d.", "in press" or "forthcoming".
    suffix: str
        The letter after the year ("a"), or ``""``.
    cited_as: str
        The work as printed ("Smith et al., 2020a").

    """

    first: str
    second: str
    authors: int
    year: str
    suffix: str
    cited_as: str


@dataclass(frozen=True, slots=True)
class AuthorYearMarker:
    """One author-year citation in a text.

    Attributes
    ----------
    start: int
        Where it starts: at "(" for a parenthetical marker, at the first
        name for a narrative one.
    end: int
        Where it ends, after ")".
    text: str
        The marker as printed.
    works: tuple[CitedWork, ...]
        The works it cites, in order.
    narrative: bool
        True for "Smith et al. (2020)", False for "(Smith et al., 2020)".

    """

    start: int
    end: int
    text: str
    works: tuple[CitedWork, ...]
    narrative: bool = False


@dataclass(frozen=True, slots=True)
class EntryKey:
    """What a reference-list entry is cited by.

    Attributes
    ----------
    number: int
        The entry's number (printed or inferred).
    first: str
        The first author's surname as compared.
    second: str
        The second author's, or ``""`` (one author, or not read).
    authors: int | None
        1, 2, or 3 for three or more; None when not read.
    year: str
        "2020", "n.d.", "in press" or "forthcoming".
    suffix: str
        The letter after the year, printed or given in list order
        (``entry_keys``), or ``""``.

    """

    number: int
    first: str
    second: str
    authors: int | None
    year: str
    suffix: str


def normalized_name(name: str) -> str:
    """Return a surname's last word, as compared between text and list.

    Parameters
    ----------
    name: str
        A surname as printed ("van den Heuvel", "García-López", "O’Brien").

    Returns
    -------
    str
        Its last word, without accents, apostrophes or case ("heuvel",
        "garcia-lopez", "obrien"); ``""`` for an empty name.

    """

    words = name.replace("’s", "").replace("'s", "").split()
    if not words:
        return ""
    plain = unicodedata.normalize("NFKD", words[-1])
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return re.sub(r"['’.]", "", plain).casefold()


def find_markers(text: str) -> list[AuthorYearMarker]:
    """Return the author-year markers in a text.

    Parameters
    ----------
    text: str
        A sentence or passage.

    Returns
    -------
    list[AuthorYearMarker]
        Parenthetical markers ("(Smith 2020; Lee et al., 2019a, b)", with
        an optional prefix such as "e.g." and locators such as "p. 5", and
        words after the last work ignored: "(Smith 2020 for a review)")
        and narrative ones ("Smith et al. (2020)", "Smith and Lee
        (2020a, 2021)"), in order. A name that is a common word ("Phase",
        "Figure", a month) is not taken for an author.

    """

    found: list[AuthorYearMarker] = []
    for match in _NARRATIVE.finditer(text):
        authors = match["authors"]
        # "However, Smith and Lee (2020)": the opener is not an author.
        while (opener := re.match(r"(\S+?)\s*,\s+", authors)) and (
            opener.group(1).casefold() in _NOT_NAMES
            or opener.group(1).casefold().endswith("ly")
        ):
            authors = authors[opener.end():]
        works = _works(authors, match["years"])
        if works:
            start = match.start("authors") + len(match["authors"]) - len(
                authors
            )
            found.append(AuthorYearMarker(
                start, match.end(), text[start:match.end()], works, True
            ))
    taken = [(marker.start, marker.end) for marker in found]
    for match in _PARENTHESES.finditer(text):
        if any(start < match.end() and match.start() < end
               for start, end in taken):
            continue
        works = _parenthetical(match["inner"])
        if works:
            found.append(AuthorYearMarker(
                match.start(), match.end(), match.group(), works
            ))
    return sorted(found, key=lambda marker: marker.start)


def _parenthetical(inner: str) -> tuple[CitedWork, ...]:
    """Return the works cited inside one pair of parentheses.

    Parameters
    ----------
    inner: str
        The text between the parentheses.

    Returns
    -------
    tuple[CitedWork, ...]
        The works of each ";"-separated group, read from its start (after
        an optional prefix) while works follow, separated by commas; empty
        when the first group does not start with a work.

    """

    works: list[CitedWork] = []
    for index, group in enumerate(inner.split(";")):
        position = len(group) - len(group.lstrip())
        prefix = _PREFIX.match(group, position)
        if prefix:
            position = prefix.end()
        while True:
            match = _WORK.match(group, position)
            if match is None:
                break
            works.extend(_works(match["authors"], match["years"]))
            position = match.end()
            separator = _COMMA.match(group, position)
            if separator is None:
                break
            position = separator.end()
        if index == 0 and not works:
            return ()
    return tuple(works)


def _works(authors: str, years: str) -> tuple[CitedWork, ...]:
    """Return the works an author group and its years cite.

    Parameters
    ----------
    authors: str
        The names as printed ("Smith et al.", "Smith and Lee").
    years: str
        The years as printed ("2020a, b, 2021").

    Returns
    -------
    tuple[CitedWork, ...]
        One work per year (a lone letter takes the year before it); empty
        when the first name is a common word.

    """

    et_al = bool(re.search(_ET_AL + "$", authors))
    names = [
        name for name in _NAME_SPLIT.split(re.sub(_ET_AL + "$", "", authors))
        if name.strip()
    ]
    if not names or names[0].split()[0].casefold() in _NOT_NAMES or any(
        name.split()[-1].casefold() in _NOT_NAMES for name in names
    ):
        return ()
    count = _AUTHORS_MANY if et_al else min(len(names), _AUTHORS_MANY)
    first = normalized_name(names[0])
    second = normalized_name(names[1]) if count == 2 else ""
    works: list[CitedWork] = []
    year = ""
    for item in re.split(r"\s*,\s*", years.strip()):
        parts = _YEAR_ITEM.match(" ".join(item.split()).casefold())
        if parts is None:
            continue
        if parts["year"]:
            year = {"n. d.": "n.d."}.get(parts["year"], parts["year"])
        if not year:
            continue
        suffix = parts["suffix"] or ""
        cited_as = f"{authors.strip()}, {year}{suffix}"
        works.append(CitedWork(first, second, count, year, suffix, cited_as))
    return tuple(works)


def entry_key(number: int, entry: str) -> EntryKey | None:
    """Return what one reference-list entry is cited by.

    Parameters
    ----------
    number: int
        The entry's number.
    entry: str
        The entry as printed.

    Returns
    -------
    EntryKey | None
        Its first and second surnames and number of authors (from
        "Smith, J. A., Lee, B., & Wu, C.", "Smith JA, Lee B, Wu C." or, for
        other styles, the name before the first comma), year and printed
        suffix; None without a year or a first author.

    """

    year = suffix = ""
    for pattern in _ENTRY_YEAR:
        match = pattern.search(entry)
        if match:
            year = " ".join(match["year"].split()).casefold()
            year = {"n. d.": "n.d."}.get(year, year)
            suffix = match.groupdict().get("suffix") or ""
            break
    if not year:
        return None
    names: list[str] = []
    for pattern in (_COMMA_AUTHOR, _SPACE_AUTHOR):
        position = 0
        while match := pattern.match(entry, position):
            names.append(match["name"])
            position = match.end()
        if names:
            break
    authors: int | None
    if names:
        many = len(names) >= _AUTHORS_MANY or bool(
            _ENTRY_ET_AL.search(entry[:position + 12])
        )
        authors = _AUTHORS_MANY if many else len(names)
    else:
        head = entry.split(",", 1)[0].split("(", 1)[0].strip()
        if not head or len(head.split()) > 4:
            return None
        names, authors = [head], None
    first = normalized_name(names[0])
    if not first:
        return None
    second = normalized_name(names[1]) if authors == 2 else ""
    return EntryKey(number, first, second, authors, year, suffix)


def entry_keys(entries: Sequence[tuple[int, str]]) -> list[EntryKey]:
    """Return the keys of a reference list, with suffixes given in order.

    Parameters
    ----------
    entries: Sequence[tuple[int, str]]
        Each entry's number and text, in list order.

    Returns
    -------
    list[EntryKey]
        One per entry with a year and a first author. Entries of the same
        first author, author count and year, none with a printed suffix,
        are given "a", "b", ... in list order, as author-year styles print
        them in the text.

    """

    keys = [
        key for number, text in entries
        if (key := entry_key(number, text)) is not None
    ]
    groups: dict[tuple[str, int | None, str], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[(key.first, key.authors, key.year)].append(index)
    for indexes in groups.values():
        if len(indexes) > 1 and not any(keys[i].suffix for i in indexes):
            for letter, index in zip("abcdefghijklmnopqrstuvwxyz", indexes):
                keys[index] = replace(keys[index], suffix=letter)
    return keys


def resolve(work: CitedWork, keys: Sequence[EntryKey]) -> tuple[int, ...]:
    """Return the entries a cited work can be.

    Parameters
    ----------
    work: CitedWork
        One work a marker cites.
    keys: Sequence[EntryKey]
        The reference list's keys (``entry_keys``).

    Returns
    -------
    tuple[int, ...]
        The numbers of the entries with the work's first author and year
        (and suffix, when the work has one), narrowed when that leaves
        any to those with the same second author and number of authors:
        one when resolved, several when ambiguous, none when the list has
        no such entry.

    """

    found = [
        key for key in keys
        if key.first == work.first and key.year == work.year
        and (not work.suffix or key.suffix == work.suffix)
    ]
    if work.second:
        found = [
            key for key in found if key.second == work.second
        ] or found
    known = [key for key in found if key.authors is not None]
    if known:
        same = [key for key in known if key.authors == work.authors]
        found = same or found
    return tuple(key.number for key in found)
