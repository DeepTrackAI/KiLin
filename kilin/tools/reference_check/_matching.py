"""Match a paper's reference entries to the files of a project.

Private to ``reference_check``. The checker must know which cited works
are in the project, and which file each one is. Reference entries rarely
carry a DOI (Nature and Vancouver styles print none), so entries are
matched mainly by words:

1. **DOI**: the file's DOI, printed on its first page, appears in the
   entry. This settles the match.
2. **Title**: when the file's PDF metadata title is trustworthy (enough
   distinctive words, most of them printed on the first page, so not
   "Microsoft Word - draft.docx"), the share of its words that the entry
   contains.
3. **First page**: otherwise, the share of the entry's words found on the
   file's first page, where title, authors and journal are printed.

The entry's first-author surname must also be printed among the file's
metadata authors or on its first page. A title match whose first author
differs is only *probable*: usually another version of the same work,
such as the preprint of a published paper. Names are compared without
spaces, accents or punctuation, because PDF text often letter-spaces them
("L u d v i gS t o r m").

Everything here is deterministic and model-free; nothing is invented, and
every decision records its reason. The thresholds come from the tool's
validated ``defaults.json`` as ``MatchRules``.

Key Features
------------

- **No title parsing**

    The entry is compared with the file's title and first page as a whole,
    so citation styles do not need to be recognised.

- **Honest uncertainty**

    Each entry is ``found``, ``probable`` (with the reason), ``missing``,
    or ``unreadable`` when it is too short to identify (usually a
    fragment of a badly split reference list).

- **Several candidates reported**

    Other files that also match (duplicates, versions) are listed.

Module Structure
----------------

- ``MatchRules``

    The caller's thresholds and vocabulary.

- ``PaperIdentity`` and ``paper_identity``

    What identifies a project file: title, authors, DOI, first page.

- ``first_author``

    The first author's surname as an entry prints it.

- ``ReferenceMatch``, ``match_reference`` and ``match_references``

    Match entries to files, with a status and a reason.

Examples
--------
>>> from pathlib import Path
>>> from kilin.tools.reference_check._matching import (
...     MatchRules,
...     PaperIdentity,
...     match_reference,
... )
>>> rules = MatchRules(
...     first_characters=1500, title_words_min=3, title_on_page=0.6,
...     title_found=0.8, title_probable=0.5, page_found=0.85,
...     page_probable=0.65, entry_words_min=4,
...     common_words=frozenset({"journal"}),
... )
>>> paper = PaperIdentity(
...     path=Path("smith.pdf"),
...     title="Counting cells in growing tissue",
...     authors="Jane Smith", doi="",
...     first_page="Counting cells in growing tissue Jane Smith",
... )
>>> entry = "Smith, J. Counting cells in growing tissue. Cell 1, 2 (2020)."
>>> match_reference(entry, 1, [paper], rules).status
'found'

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final
import unicodedata

from kilin.chunking import Chunk

__all__ = [
    "MATCH_FOUND",
    "MATCH_MISSING",
    "MATCH_PROBABLE",
    "MATCH_UNREADABLE",
    "MatchRules",
    "PaperIdentity",
    "ReferenceMatch",
    "first_author",
    "match_reference",
    "match_references",
    "paper_identity",
]

MATCH_FOUND: Final[str] = "found"
MATCH_PROBABLE: Final[str] = "probable"
MATCH_MISSING: Final[str] = "missing"
MATCH_UNREADABLE: Final[str] = "unreadable"

# A DOI as printed on a first page: "https://doi.org/10.1038/s41...",
# "doi: 10.1371/journal.pone.0178798", or bare.
_DOI: Final[re.Pattern[str]] = re.compile(
    r"\b(10\.\d{4,9}/[^\s\"<>]+)", flags=re.I
)
# Fewest characters of a DOI (without punctuation) to compare it safely.
_DOI_MIN_CHARACTERS: Final[int] = 10
# Distinctive words: four letters or more, after accents are removed.
_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z]{4,}")
# Where the first author's name ends: a comma, a sentence end ("Smith J.
# Title"), "&" or "and" before a second author, or an APA year "(2020)".
_AUTHOR_END: Final[re.Pattern[str]] = re.compile(
    r",|\.\s|\s&\s|\sand\s|\s\("
)
# Initials after a surname ("M", "MP", "Y.-W.", "J.A.").
_INITIALS: Final[re.Pattern[str]] = re.compile(r"[A-Z][A-Z.\-]{0,5}")
# Most words of a surname ("van den Heuvel"); more is not a person.
_SURNAME_WORDS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class MatchRules:
    """The caller's thresholds and vocabulary for matching.

    The tool builds one from its validated ``defaults.json``; this
    module has no defaults of its own.

    Attributes
    ----------
    first_characters: int
        Characters at the start of a file read as its first page.
    title_words_min: int
        Fewest distinctive words of a metadata title worth trusting.
    title_on_page: float
        Share of the metadata title's words that must be printed on the
        first page for the title to be trusted (0-1).
    title_found: float
        Share of a trusted title's words an entry must contain to be
        ``found`` (0-1).
    title_probable: float
        Lower share that makes the file ``probable`` (0-1).
    page_found: float
        Without a trusted title: share of the entry's words that must be
        on the first page to be ``found`` (0-1).
    page_probable: float
        Lower share that makes the file ``probable`` (0-1).
    entry_words_min: int
        Fewest distinctive words of an entry; shorter ones are
        ``unreadable``.
    common_words: frozenset[str]
        Lowercase words that do not tell works apart ("journal").

    """

    first_characters: int
    title_words_min: int
    title_on_page: float
    title_found: float
    title_probable: float
    page_found: float
    page_probable: float
    entry_words_min: int
    common_words: frozenset[str]


@dataclass(frozen=True, slots=True)
class PaperIdentity:
    """What identifies a project file.

    Attributes
    ----------
    path: pathlib.Path
        The file, as indexed.
    title: str
        The trusted metadata title, or empty.
    authors: str
        The metadata authors, as one string, or empty.
    doi: str
        The DOI printed on the first page, or empty.
    first_page: str
        The start of the file's text.

    """

    path: Path
    title: str
    authors: str
    doi: str
    first_page: str


@dataclass(frozen=True, slots=True)
class ReferenceMatch:
    """How one reference entry matches the project's files.

    Attributes
    ----------
    position: int
        The entry's position in the reference list, from one. It equals
        the printed number when the list was split correctly.
    entry: str
        The entry, as printed.
    status: str
        ``found``, ``probable``, ``missing``, or ``unreadable``.
    path: pathlib.Path | None
        The best-matching file; None when missing or unreadable.
    score: float
        The share behind the decision (1.0 for a DOI), 0 when missing.
    reason: str
        Why, in a few words ("DOI", "title and first author", "title;
        first author differs (another version?)").
    alternatives: tuple[pathlib.Path, ...]
        Other files that also match at least as ``probable``.

    """

    position: int
    entry: str
    status: str
    path: Path | None
    score: float
    reason: str
    alternatives: tuple[Path, ...] = ()


def paper_identity(
    path: Path,
    passages: Sequence[Chunk],
    file: Path | None,
    rules: MatchRules,
) -> PaperIdentity:
    """Return what identifies a project file.

    Parameters
    ----------
    path: pathlib.Path
        The file, as indexed (``Chunk.path``).
    passages: Sequence[Chunk]
        The file's passages, in order.
    file: pathlib.Path | None
        The file on disk, for its PDF metadata; None when unknown or not a
        PDF.
    rules: MatchRules
        The caller's thresholds.

    Returns
    -------
    PaperIdentity
        The metadata title only when it has at least
        ``rules.title_words_min`` distinctive words and at least
        ``rules.title_on_page`` of them are printed on the first page;
        otherwise an empty title. Unreadable metadata gives an empty title
        and authors.

    """

    first_page = "\n".join(chunk.text for chunk in passages)
    first_page = first_page[: rules.first_characters]
    title, authors = _metadata(file)

    title_words = set(_words(title, rules))
    if len(title_words) < rules.title_words_min or (
        len(title_words & set(_words(first_page, rules)))
        < rules.title_on_page * len(title_words)
    ):
        title = ""

    doi = _DOI.search(first_page)
    return PaperIdentity(
        path=path,
        title=title,
        authors=authors,
        doi=doi.group(1).rstrip(".,;)") if doi else "",
        first_page=first_page,
    )


def first_author(entry: str) -> str | None:
    """Return the first author's surname as an entry prints it.

    Parameters
    ----------
    entry: str
        A reference entry, as printed.

    Returns
    -------
    str | None
        The last word of the first name before a comma, sentence end, or
        second author, without initials: "Mijalkov" for "Mijalkov, M. et
        al." and "Mijalkov M, Kakaei E", "Heuvel" for "van den Heuvel MP".
        None when that name has more than three words (an organization,
        or a fragment of a badly split entry).

    """

    head = _AUTHOR_END.split(entry.strip(), maxsplit=1)[0]
    words = [
        word for word in head.split() if not _INITIALS.fullmatch(word)
    ]
    if not 0 < len(words) <= _SURNAME_WORDS:
        return None

    return words[-1]


def match_reference(
    entry: str,
    position: int,
    papers: Sequence[PaperIdentity],
    rules: MatchRules,
) -> ReferenceMatch:
    """Match one reference entry to the project's files.

    Parameters
    ----------
    entry: str
        The entry, as printed.
    position: int
        Its position in the reference list, from one.
    papers: Sequence[PaperIdentity]
        The candidate files (the citing paper itself excluded).
    rules: MatchRules
        The caller's thresholds.

    Returns
    -------
    ReferenceMatch
        The best candidate: a ``found`` match before a ``probable`` one,
        then the higher score; the others that match are listed as
        alternatives. ``unreadable`` when the entry has fewer than
        ``rules.entry_words_min`` distinctive words and no DOI matches.

    """

    words = set(_words(entry, rules))
    candidates = [
        candidate
        for paper in papers
        if (candidate := _candidate(entry, words, paper, rules))
    ]
    if not candidates and len(words) < rules.entry_words_min:
        return ReferenceMatch(
            position, entry, MATCH_UNREADABLE, None, 0.0,
            "too short to identify",
        )

    if not candidates:
        return ReferenceMatch(
            position, entry, MATCH_MISSING, None, 0.0, "no matching file"
        )

    candidates.sort(
        key=lambda c: (c[0] == MATCH_FOUND, c[1], str(c[3])), reverse=True
    )
    status, score, reason, path = candidates[0]
    return ReferenceMatch(
        position, entry, status, path, round(score, 3), reason,
        tuple(c[3] for c in candidates[1:]),
    )


def match_references(
    entries: Sequence[str],
    papers: Sequence[PaperIdentity],
    rules: MatchRules,
) -> list[ReferenceMatch]:
    """Match every reference entry to the project's files.

    Parameters
    ----------
    entries: Sequence[str]
        The paper's reference entries, in order (``reference_entries``).
    papers: Sequence[PaperIdentity]
        The candidate files (the citing paper itself excluded).
    rules: MatchRules
        The caller's thresholds.

    Returns
    -------
    list[ReferenceMatch]
        One match per entry, in order, positions from one.

    """

    return [
        match_reference(entry, position, papers, rules)
        for position, entry in enumerate(entries, start=1)
    ]


def _candidate(
    entry: str,
    words: set[str],
    paper: PaperIdentity,
    rules: MatchRules,
) -> tuple[str, float, str, Path] | None:
    """Return how well one file matches an entry, or None.

    Parameters
    ----------
    entry: str
        The entry, as printed.
    words: set[str]
        The entry's distinctive words.
    paper: PaperIdentity
        The candidate file.
    rules: MatchRules
        The caller's thresholds.

    Returns
    -------
    tuple[str, float, str, pathlib.Path] | None
        Status, score, reason and file; None below ``probable``.

    """

    doi = _flat(paper.doi)
    if len(doi) >= _DOI_MIN_CHARACTERS and doi in _flat(entry):
        return MATCH_FOUND, 1.0, "DOI", paper.path

    surname = first_author(entry)
    author: bool | None = None
    if surname:
        author = _flat(surname) in _flat(paper.authors + paper.first_page)

    if paper.title:
        title = set(_words(paper.title, rules))
        score = len(title & words) / len(title)
        found, probable, what = (
            rules.title_found, rules.title_probable, "title"
        )
    else:
        if not words:
            return None
        score = len(words & set(_words(paper.first_page, rules))) / len(
            words
        )
        found, probable, what = (
            rules.page_found, rules.page_probable, "first page"
        )

    if score >= found and author is not False:
        reason = (
            f"{what} and first author" if author
            else f"{what} (first author not identified)"
        )
        return MATCH_FOUND, score, reason, paper.path

    if score >= found:
        return (
            MATCH_PROBABLE, score,
            f"{what}; first author differs (another version?)", paper.path,
        )

    if score >= probable and author:
        return (
            MATCH_PROBABLE, score, f"part of the {what} and first author",
            paper.path,
        )

    return None


def _metadata(file: Path | None) -> tuple[str, str]:
    """Return a PDF's metadata title and authors, or empty strings.

    Parameters
    ----------
    file: pathlib.Path | None
        The file on disk.

    Returns
    -------
    tuple[str, str]
        Title and authors with whitespace joined; empty for other file
        types and unreadable metadata.

    """

    if file is None or file.suffix.lower() != ".pdf" or not file.is_file():
        return "", ""

    try:
        from pypdf import PdfReader

        metadata = PdfReader(file).metadata or {}
    except Exception:  # Unreadable metadata: identify by text only.
        return "", ""

    return (
        " ".join(str(metadata.get("/Title") or "").split()),
        " ".join(str(metadata.get("/Author") or "").split()),
    )


def _plain(text: str) -> str:
    """Return lowercase text without accents and ligatures.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        NFKD-decomposed ("ﬂ" becomes "fl"), combining marks and spacing
        diacritics ("Lundstr¨om") removed, case-folded.

    """

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        ch for ch in decomposed
        if not unicodedata.combining(ch) and ch not in "¨´`^~¸˚"
    ).casefold()


def _flat(text: str) -> str:
    """Return letters and digits only, for names and DOIs.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        ``_plain`` text without spaces or punctuation, so letter-spaced
        names ("S t o r m") and broken DOIs ("doi. org/10.") compare.

    """

    return re.sub(r"[^a-z0-9]", "", _plain(text))


def _words(text: str, rules: MatchRules) -> list[str]:
    """Return the distinctive words of a text.

    Parameters
    ----------
    text: str
        Any text.
    rules: MatchRules
        For the caller's common words.

    Returns
    -------
    list[str]
        Words of four or more letters, in order, without common words.

    """

    return [
        word for word in _WORD.findall(_plain(text))
        if word not in rules.common_words
    ]
