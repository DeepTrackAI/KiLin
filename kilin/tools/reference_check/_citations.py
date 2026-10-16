"""Find a paper's numbered in-text citations and their citing sentences.

Private to ``reference_check``. The checker needs, for each cited
reference, the sentence that cites it. This module finds numbered
citation markers in the paper's body (the passages before its reference
list) and maps each to reference numbers. Two styles are supported:

1. **Brackets**: ``[3]``, ``[3,4]``, ``[1–4]``, ``[2, 5–7]``. Brackets are
   unambiguous, so only numbers outside 1–``max_range`` are rejected.
2. **Superscripts** (Nature, Science, many medical journals): extracted
   text glues them to the word (``polygenic15``) or to a number
   (``BRAPH 2.087`` for 2.0 and 87, ``Braph288`` for 2 and 88), or leaves
   a space (``practices 1,8–14.``), where they look like ordinary numbers
   (``in 3 groups``). A text candidate is therefore kept only when the PDF
   prints those digits smaller than the rest of their line
   (``superscript_runs``), right after the same characters. Author and
   affiliation markers (after a capitalized name) and unit exponents
   (``cm⁻¹``) are rejected.

Author-year citations ("Smith et al., 2020") are not recognised; a paper
using them has style ``none``. Numbering is not assumed to follow first
appearance: journals print figure legends that cite Methods references
early.

Each reference number is the entry's position in the reference list, so
the caller must check that the list was split correctly: numbers cited
beyond the list's length are reported, and usually mean entries were
merged.

Everything here is deterministic and model-free. The thresholds come from
the tool's validated ``defaults.json`` as ``CitationRules``.

Key Features
------------

- **Font-confirmed superscripts**

    Text patterns propose, the PDF's glyph sizes confirm.

- **Citing sentences**

    Each citation keeps the sentence it appears in, with its source
    position, for evidence and checking.

- **Diagnostics**

    Reference numbers never cited, and numbers cited beyond the list.

Module Structure
----------------

- ``CitationRules``

    The caller's thresholds and vocabulary.

- ``Superscripts`` and ``superscript_runs``

    The digits a PDF prints as superscripts, with the text before them.

- ``cited_numbers``

    The reference numbers of one marker ("2, 5–7" is 2, 5, 6, 7).

- ``Citation``, ``CitationScan`` and ``find_citations``

    Find every citation in a paper's body.

Examples
--------
>>> from kilin.tools.reference_check._citations import cited_numbers
>>> cited_numbers("2, 5–7", max_range=50)
(2, 5, 6, 7)

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
from typing import Final

from kilin.chunking import Chunk
from kilin.tools._shared.sentences import Sentence, split_sentences

__all__ = [
    "STYLE_BRACKETS",
    "STYLE_NONE",
    "STYLE_SUPERSCRIPT",
    "Citation",
    "CitationRules",
    "CitationScan",
    "Superscripts",
    "cited_numbers",
    "find_citations",
    "superscript_runs",
]

STYLE_BRACKETS: Final[str] = "brackets"
STYLE_SUPERSCRIPT: Final[str] = "superscript"
STYLE_NONE: Final[str] = "none"

# A list of numbers and ranges: "3", "3,4", "1–4", "2, 5–7".
_NUMBERS: Final[str] = r"\d{1,3}(?:\s*[,–-]\s*\d{1,3})*"
# A bracketed citation: "[3]", "[1–4]", "[2, 5–7]".
_BRACKET: Final[re.Pattern[str]] = re.compile(rf"\[(?P<n>{_NUMBERS})\]")
# A superscript glued to the word or bracket before it, optionally after
# punctuation: "polygenic15", "MDD93", "regions)70", "stress.12". The PDF's
# glyph sizes, not this pattern, rule out "COVID19".
_GLUED: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z)\]][.,;:]?(?P<n>\d{1,3}(?:[,–-]\d{1,3})*)(?=[\s.,;:)]|$)"
)
# A superscript after a space: "practices 1,8–14.", "petioles 5,14,15 by".
_SPACED: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z)][.,;:]? (?P<n>\d{1,3}(?:,\d{1,3}|[–-]\d{1,3})*)"
    r"(?=[.,;:]?(?:\s|$))"
)
# A superscript glued to a number: "BRAPH 2.0⁸⁷" is extracted as "2.087"
# and "BRAPH 2⁸⁸" as "288". Every split after a digit is a candidate
# (overlapping, so "288" gives "88" and "8"); the PDF decides, because
# only digits it prints small after the same characters count, and "6.87"
# or "288" printed in full size never are.
_AFTER_DIGIT: Final[re.Pattern[str]] = re.compile(
    r"(?<=\d)(?=(?P<n>\d{1,3}(?:[,–-]\d{1,3})*)(?:[\s.,;:)]|$))"
)
# "10" ending a number ("1×10", "of 10"): digits glued to it are a power.
_POWER_OF_TEN: Final[re.Pattern[str]] = re.compile(r"(?<![\d.])10$")
# Fewest letters and digits of context that must agree between the PDF
# and the text; a superscript at the start of a PDF line has none.
_CONTEXT_MIN: Final[int] = 3
# A capitalized word ("Patil", "Nam-Hai", "Gothenburg"), for bylines.
_CAPITALIZED: Final[re.Pattern[str]] = re.compile(r"[A-Z][^\sA-Z]*[a-z]")
# Digits the PDF prints small that are part of a superscript run.
_RUN_CHARACTER: Final[re.Pattern[str]] = re.compile(r"[\d,–-]")

#: Superscript digits (without separators) mapped to the letters and digits
#: printed just before them on the same line, lowercase.
Superscripts = Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class CitationRules:
    """The caller's thresholds and vocabulary for finding citations.

    The tool builds one from its validated ``defaults.json``; this
    module has no defaults of its own.

    Attributes
    ----------
    bracket_markers_min: int
        Fewest bracketed markers for a PDF with superscripts to use
        brackets; without superscripts, one is enough.
    superscript_ratio: float
        A glyph smaller than this share of its line's median letter size
        is a superscript (0-1).
    signature_characters: int
        Letters and digits before a superscript compared between the PDF
        and the extracted text.
    max_range: int
        Longest range a marker may span ("1–4" spans 3); longer ones are
        not citations.
    unit_words: frozenset[str]
        Lowercase words after which digits are exponents, not citations
        ("cm", "log").
    byline_share: float
        Share of capitalized words above which a sentence is an author
        list or affiliations, whose superscripts are not citations (0-1).

    """

    bracket_markers_min: int
    superscript_ratio: float
    signature_characters: int
    max_range: int
    unit_words: frozenset[str]
    byline_share: float


@dataclass(frozen=True, slots=True)
class Citation:
    """One citation marker in the paper's body.

    Attributes
    ----------
    numbers: tuple[int, ...]
        The cited reference numbers, in the marker's order.
    marker: str
        The marker as printed ("2, 5–7").
    sentence: Sentence
        The sentence it appears in.

    """

    numbers: tuple[int, ...]
    marker: str
    sentence: Sentence


@dataclass(frozen=True, slots=True)
class CitationScan:
    """Every citation found in a paper's body.

    Attributes
    ----------
    style: str
        ``brackets``, ``superscript``, or ``none``.
    citations: tuple[Citation, ...]
        In reading order.
    uncited: tuple[int, ...]
        Reference numbers (1 to the entry count) no marker cites.
    beyond_list: tuple[int, ...]
        Cited numbers larger than the entry count: usually entries merged
        when the list was split, so later numbers may point to the wrong
        entry.

    """

    style: str
    citations: tuple[Citation, ...]
    uncited: tuple[int, ...]
    beyond_list: tuple[int, ...]


def cited_numbers(marker: str, max_range: int) -> tuple[int, ...] | None:
    """Return the reference numbers of one marker.

    Parameters
    ----------
    marker: str
        Numbers and ranges separated by commas ("2, 5–7").
    max_range: int
        Longest allowed range.

    Returns
    -------
    tuple[int, ...] | None
        The numbers in order, ranges expanded; None when a number is zero,
        a range runs backwards or spans more than ``max_range``, or a part
        has more than two ends.

    """

    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", marker.strip()):
        ends = [int(end) for end in re.split(r"\s*[–-]\s*", part)]
        if min(ends) < 1 or len(ends) > 2:
            return None

        if len(ends) == 2:
            if not ends[0] < ends[1] <= ends[0] + max_range:
                return None
            numbers.extend(range(ends[0], ends[1] + 1))
        else:
            numbers.append(ends[0])

    return tuple(numbers)


def superscript_runs(file: Path, rules: CitationRules) -> Superscripts:
    """Return the digits a PDF prints as superscripts.

    Parameters
    ----------
    file: pathlib.Path
        The PDF on disk.
    rules: CitationRules
        For ``superscript_ratio`` and ``signature_characters``.

    Returns
    -------
    Superscripts
        For each run of small digits, commas and dashes (separators
        dropped), the letters and digits printed before it on its line
        (at most ``signature_characters``, lowercase, earlier superscript
        digits included). Lines without
        letters are skipped. Empty when ``pdfminer.six`` cannot read the
        file.

    """

    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import (
            LAParams,
            LTChar,
            LTTextContainer,
            LTTextLine,
        )

        pages = list(extract_pages(str(file), laparams=LAParams()))
    except Exception:  # Unreadable PDF: no superscripts are confirmed.
        return {}

    runs: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        for box in page:
            if not isinstance(box, LTTextContainer):
                continue
            for line in box:
                if not isinstance(line, LTTextLine):
                    continue
                letters = [
                    glyph.size for glyph in line
                    if isinstance(glyph, LTChar) and glyph.get_text().isalpha()
                ]
                if not letters:
                    continue

                small = rules.superscript_ratio * statistics.median(letters)
                before, run = "", ""
                for glyph in [*line, None]:
                    text = glyph.get_text() if glyph is not None else ""
                    if (
                        isinstance(glyph, LTChar)
                        and glyph.size < small
                        and _RUN_CHARACTER.fullmatch(text)
                    ):
                        run += text
                        continue

                    digits = re.sub(r"\D", "", run)
                    if digits:
                        runs[digits].append(
                            _flat(before)[-rules.signature_characters:]
                        )
                    # Extracted text keeps superscript digits, so the
                    # context of the next run does too.
                    before += run + text
                    run = ""

    return {digits: tuple(found) for digits, found in runs.items()}


def find_citations(
    passages: Sequence[Chunk],
    entry_count: int,
    superscripts: Superscripts | None,
    rules: CitationRules,
) -> CitationScan:
    """Find every numbered citation in a paper's body.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's body, in order (``reference_body``).
    entry_count: int
        Number of entries in the paper's reference list.
    superscripts: Superscripts | None
        The PDF's superscripts (``superscript_runs``); None when the paper
        is not a PDF, so superscripts cannot be confirmed.
    rules: CitationRules
        The caller's thresholds.

    Returns
    -------
    CitationScan
        Bracketed markers when there are at least
        ``rules.bracket_markers_min`` of them, or any when there are no
        superscripts to look for; otherwise font-confirmed superscripts;
        otherwise style ``none`` with no citations.

    """

    sentences = split_sentences(passages, min_words=1)
    brackets = [
        (sentence, match)
        for sentence in sentences
        for match in _BRACKET.finditer(sentence.text)
    ]
    if len(brackets) >= rules.bracket_markers_min or (
        brackets and not superscripts
    ):
        style, found = STYLE_BRACKETS, brackets
    elif superscripts:
        style, found = STYLE_SUPERSCRIPT, []
        previous = ""
        for sentence in sentences:
            if _byline(sentence.text, rules):
                previous = sentence.text
                continue
            # Markers of every pattern, in their order in the sentence.
            matches = sorted(
                (
                    match
                    for pattern in (_GLUED, _SPACED, _AFTER_DIGIT)
                    for match in pattern.finditer(sentence.text)
                    if _confirmed(
                        sentence.text, match, superscripts, rules, previous
                    )
                ),
                key=lambda match: match.start("n"),
            )
            found.extend((sentence, match) for match in matches)
            previous = sentence.text
    else:
        return CitationScan(
            STYLE_NONE, (), tuple(range(1, entry_count + 1)), ()
        )

    citations: list[Citation] = []
    for sentence, match in found:
        numbers = cited_numbers(match.group("n"), rules.max_range)
        if not numbers or (
            # Far beyond the list, a small number is an exponent ("2^256").
            style == STYLE_SUPERSCRIPT
            and max(numbers) > entry_count + rules.max_range
        ):
            continue
        citations.append(Citation(numbers, match.group("n"), sentence))

    cited = {number for citation in citations for number in citation.numbers}
    return CitationScan(
        style=style,
        citations=tuple(citations),
        uncited=tuple(
            number for number in range(1, entry_count + 1)
            if number not in cited
        ),
        beyond_list=tuple(sorted(n for n in cited if n > entry_count)),
    )


def _confirmed(
    text: str,
    match: re.Match[str],
    superscripts: Superscripts,
    rules: CitationRules,
    previous: str = "",
) -> bool:
    """Return whether a text candidate is a superscript citation.

    Parameters
    ----------
    text: str
        The sentence.
    match: re.Match[str]
        A ``_GLUED`` or ``_SPACED`` candidate in it.
    superscripts: Superscripts
        The PDF's superscripts.
    rules: CitationRules
        For ``unit_words`` and ``signature_characters``.
    previous: str, optional
        The sentence before, whose end completes a short context (a
        sentence split at "Fig." leaves "5A)51,52."). Defaults to ``""``.

    Returns
    -------
    bool
        False after a single letter ("v 2"), a unit word, or the number
        10 (a power: "1×10⁸", "10⁵"); otherwise
        whether the PDF prints these digits as a superscript after the
        same characters. A PDF line can start just before a superscript,
        so the shorter of the two contexts must end the longer one, and
        both must have at least ``_CONTEXT_MIN`` characters.

    """

    before = text[: match.start("n")]
    if _POWER_OF_TEN.search(before):
        return False

    words = before.split()
    word = words[-1].rstrip(".,;:") if words else ""
    if (
        len(word) < 2
        or word.casefold() in rules.unit_words
    ):
        return False

    context = _flat(f"{previous} {before}")[-rules.signature_characters:]
    digits = re.sub(r"\D", "", match.group("n"))
    return len(context) >= _CONTEXT_MIN and any(
        len(printed) >= _CONTEXT_MIN
        and (context.endswith(printed) or printed.endswith(context))
        for printed in superscripts.get(digits, ())
    )


def _byline(text: str, rules: CitationRules) -> bool:
    """Return whether a sentence is an author list or affiliations.

    Parameters
    ----------
    text: str
        The sentence.
    rules: CitationRules
        For ``byline_share``.

    Returns
    -------
    bool
        True when more than ``rules.byline_share`` of its words of two or
        more letters are capitalized ("Mite Mijalkov1, Ludvig Storm2",
        "Department of Physics, University of Gothenburg, Sweden").

    """

    words = [word for word in re.findall(r"[^\W\d_][\w'’-]*", text)
             if len(word) > 1]
    if not words:
        return False

    capitalized = sum(bool(_CAPITALIZED.match(word)) for word in words)
    return capitalized > rules.byline_share * len(words)


def _flat(text: str) -> str:
    """Return lowercase letters and digits only.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        Without spaces, punctuation or symbols, so the PDF's glyph text
        and the extracted text compare.

    """

    return re.sub(r"[^0-9a-z]", "", text.casefold())
