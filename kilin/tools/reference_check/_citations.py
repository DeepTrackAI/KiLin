"""Find a paper's numbered in-text citations and their citing sentences.

Private to ``reference_check``. The checker needs, for each cited
reference, the sentence that cites it. This module finds numbered
citation markers in the paper's body (the passages before its reference
list) and maps each to reference numbers. Two styles are supported:

1. **Brackets**: ``[3]``, ``[3,4]``, ``[1–4]``, ``[2, 5–7]``. Brackets are
   unambiguous, so only numbers outside 1–``max_range`` are rejected.
   Parentheses (``(3)``, ``(15,30–32)``, Cell Press and PNAS) are used
   the same way when a paper has more of them than brackets; a number
   shortly after an equation sign is an equation number, not a citation.
2. **Superscripts** (Nature, Science, many medical journals): extracted
   text glues them to the word (``polygenic15``) or to a number
   (``BRAPH 2.087`` for 2.0 and 87, ``Braph288`` for 2 and 88), or leaves
   a space (``practices 1,8–14.``), where they look like ordinary numbers
   (``in 3 groups``). A text candidate is therefore kept only when the PDF
   prints those digits smaller than the rest of their line
   (``superscript_runs``), right after the same characters. Markers in
   author lists and affiliations, unit exponents (``cm⁻¹``) and powers of
   ten are rejected.
3. **Written out** in any style: ``Ref. 50``, ``refs. 12,13``. These need
   no confirmation.

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
    "STYLE_PARENTHESES",
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
STYLE_PARENTHESES: Final[str] = "parentheses"
STYLE_SUPERSCRIPT: Final[str] = "superscript"
STYLE_NONE: Final[str] = "none"

# A list of numbers and ranges: "3", "3,4", "1–4", "2, 5–7".
_NUMBERS: Final[str] = r"\d{1,3}(?:\s*[,–-]\s*\d{1,3})*"
# A bracketed citation: "[3]", "[1–4]", "[2, 5–7]".
_BRACKET: Final[re.Pattern[str]] = re.compile(rf"\[(?P<n>{_NUMBERS})\]")
# A parenthesized citation (Cell Press, PNAS): "(3)", "(6,7)", "(15,30–32)".
_PARENTHESIS: Final[re.Pattern[str]] = re.compile(rf"\((?P<n>{_NUMBERS})\)")
# Characters shortly before a parenthesized number that make it an
# equation number ("= 2Dt (1)"), not a citation.
_EQUATION: Final[re.Pattern[str]] = re.compile(r"[=<>≈∝∑∫±×]")
# How far back an equation sign makes a parenthesized number an equation
# number, in characters.
_EQUATION_REACH: Final[int] = 40
# A superscript glued to the word, bracket or symbol before it, optionally
# after punctuation: "polygenic15", "MDD93", "regions)70", "stress.12",
# "UNet3+86". The PDF's glyph sizes, not this pattern, rule out "COVID19".
_GLUED: Final[re.Pattern[str]] = re.compile(
    r"(?:[^\W\d_]|[)\]+*′″’”%])[.,;:]?(?P<n>\d{1,3}(?:[,–-]\d{1,3})*)"
    r"(?=[\s.,;:)]|$)"
)
# A superscript after a space: "practices 1,8–14.", "petioles 5,14,15 by",
# and after a symbol or number ("α/2 12,67" for "α/2¹²,⁶⁷").
_SPACED: Final[re.Pattern[str]] = re.compile(
    r"\S[.,;:]? (?P<n>\d{1,3}(?:,\d{1,3}|[–-]\d{1,3})*)"
    r"(?=[.,;:]?(?:\s|$))"
)
# A citation written out: "with Ref. 50", "see refs. 12,13" or "Refs. 4–6".
# Unambiguous in any style, so it needs no confirmation from the PDF.
_REF_WORD: Final[re.Pattern[str]] = re.compile(
    rf"\b[Rr]efs?\.\s?(?!0)(?P<n>{_NUMBERS})\b(?![-–/]?[A-Za-z0-9])"
)
# Glyphs a PDF cannot map to text ("(cid:12)"), which extracted text drops.
_UNMAPPED: Final[re.Pattern[str]] = re.compile(r"\(cid:\d+\)")
# A single Latin letter: a variable ("t2", "H1"), or a stamp ("v 2").
_LATIN_LETTER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]")
# The word before a superscript inside an open parenthesis ("log(σ2)"), or
# ending with a short parenthesized argument ("f(0)2", "ð0Þ2" in PDFs that
# encode parentheses as ð and Þ): the digits are an exponent. "(VAE)30" is
# a citation.
_MATH_WORD: Final[re.Pattern[str]] = re.compile(
    r"[(ð][^)Þ]*$|[(ð][^)Þ\s]{1,2}[)Þ]$"
)
# A Greek letter variable after a math sign ("N(µ, σ2)", "− µ2"): the
# digits are an exponent; "exponent α64" is a citation.
_GREEK_VARIABLE: Final[re.Pattern[str]] = re.compile(
    r"[−+=(,×·/ð]\s*[\u0370-\u03ff\u00b5]$"
)
# Words before a parenthesized number that make it an equation or a step,
# not a citation: "Equation (1)", "Eqs. (1) and (2)", "step (3)".
_NOT_CITED: Final[re.Pattern[str]] = re.compile(
    r"\b(?:equations?|eqs?\.|steps?)\s*(?:\(\d+\)\s*(?:,|and|or)\s*)*$",
    flags=re.I,
)
# Glyphs of equations, including those some PDFs use for "(", ")" and "="
# (ð, Þ, ¼).
_MATH_GLYPH: Final[re.Pattern[str]] = re.compile(r"[=¼ðÞ∑∫δΨ|∣±×∝≈<>]")
# Share of a sentence's tokens that are single characters or hold an
# equation glyph above which the "sentence" is an equation ("ðsÞ = 1 2",
# "j2H2 t2 j2H1"), whose superscripts are exponents. Prose, even with a
# formula, and tables of methods stay below it.
_MATH_SHARE: Final[float] = 0.3
# A superscript's glyphs sit at least this share of the line's letter size
# above the line's baseline; subscripts ("t₂") sit below it.
_RAISE: Final[float] = 0.15
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
        The marker's numbers as printed ("2, 5–7").
    sentence: Sentence
        The sentence it appears in.
    start: int
        Where the marker starts in the sentence's text: at "[" for a
        bracketed marker, at "R" for "Ref. 50", else at its first digit.
    end: int
        Where the marker ends in the sentence's text.

    """

    numbers: tuple[int, ...]
    marker: str
    sentence: Sentence
    start: int = 0
    end: int = 0


@dataclass(frozen=True, slots=True)
class CitationScan:
    """Every citation found in a paper's body.

    Attributes
    ----------
    style: str
        ``brackets``, ``parentheses``, ``superscript``, or ``none``.
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
        For each run of small, raised digits, commas and dashes
        (separators dropped; lowered subscripts are not runs), the text
        printed before it on its line without spaces (at most
        ``signature_characters``, lowercase, earlier superscript digits
        included). Lines without
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
                    glyph for glyph in line
                    if isinstance(glyph, LTChar) and glyph.get_text().isalpha()
                ]
                if not letters:
                    continue

                size = statistics.median(glyph.size for glyph in letters)
                small = rules.superscript_ratio * size
                base = statistics.median(
                    glyph.y0 for glyph in letters if glyph.size >= small
                ) if any(glyph.size >= small for glyph in letters) else None
                raised = None if base is None else base + _RAISE * size
                before, run = "", ""
                for glyph in [*line, None]:
                    text = glyph.get_text() if glyph is not None else ""
                    if (
                        isinstance(glyph, LTChar)
                        and glyph.size < small
                        and (raised is None or glyph.y0 >= raised)
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
        Bracketed (or, when they are more, parenthesized) markers when
        there are at least ``rules.bracket_markers_min`` of them, or any
        when there are no superscripts to look for; otherwise
        font-confirmed superscripts; otherwise style ``none`` with no
        citations. Written-out citations ("Ref. 50") count in every style.

    """

    sentences = split_sentences(passages, min_words=1)
    brackets = sum(len(_BRACKET.findall(s.text)) for s in sentences)
    parentheses = sum(
        _parenthesized(s.text, match)
        for s in sentences
        for match in _PARENTHESIS.finditer(s.text)
    )
    superscript_found = (
        _superscripts(sentences, superscripts, rules) if superscripts else []
    )
    enclosed = max(brackets, parentheses)
    if brackets < parentheses and superscript_found:
        # Superscript papers also number equations and steps in
        # parentheses: the style citing more references wins.
        cited = {
            number
            for sentence in sentences
            for match in _PARENTHESIS.finditer(sentence.text)
            if _parenthesized(sentence.text, match)
            for number in cited_numbers(match.group("n"), rules.max_range)
            or ()
            if number <= entry_count
        }
        raised = {
            number
            for _, match in superscript_found
            for number in cited_numbers(match.group("n"), rules.max_range)
            or ()
        }
        if len(raised) >= len(cited):
            enclosed = 0
    if enclosed >= rules.bracket_markers_min or (
        enclosed and not superscripts
    ):
        style, pattern = (
            (STYLE_BRACKETS, _BRACKET) if brackets >= parentheses
            else (STYLE_PARENTHESES, _PARENTHESIS)
        )
        found = []
        for sentence in sentences:
            found.extend(
                (sentence, match)
                for match in sorted(
                    (
                        *(
                            match
                            for match in pattern.finditer(sentence.text)
                            if pattern is _BRACKET
                            or _parenthesized(sentence.text, match)
                        ),
                        *_REF_WORD.finditer(sentence.text),
                    ),
                    key=lambda match: match.start(),
                )
            )
    elif superscripts:
        style, found = STYLE_SUPERSCRIPT, superscript_found
    else:
        return CitationScan(
            STYLE_NONE, (), tuple(range(1, entry_count + 1)), ()
        )

    citations: list[Citation] = []
    for sentence, match in found:
        numbers = cited_numbers(match.group("n"), rules.max_range)
        if not numbers or (
            # Far beyond the list, a number is an exponent ("2^256") or a
            # count ("(405)"), not a citation.
            style != STYLE_BRACKETS
            and max(numbers) > entry_count + rules.max_range
        ):
            continue
        # Brackets and "Ref." are shown whole; a superscript is its digits.
        start = (
            match.start() if match.re in (_BRACKET, _PARENTHESIS, _REF_WORD)
            else match.start("n")
        )
        end = (
            match.end() if match.re in (_BRACKET, _PARENTHESIS)
            else match.end("n")
        )
        citations.append(
            Citation(numbers, match.group("n"), sentence, start, end)
        )

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


def _superscripts(
    sentences: Sequence[Sentence],
    superscripts: Superscripts,
    rules: CitationRules,
) -> list[tuple[Sentence, re.Match[str]]]:
    """Return the font-confirmed superscript and written-out citations.

    Parameters
    ----------
    sentences: Sequence[Sentence]
        The paper's body, in sentences.
    superscripts: Superscripts
        The PDF's superscripts.
    rules: CitationRules
        The caller's thresholds.

    Returns
    -------
    list[tuple[Sentence, re.Match[str]]]
        In reading order. Superscripts in author lists, affiliations and
        equations are skipped; a written-out "Ref. 50" needs no
        confirmation, and its number is not also taken as a superscript.

    """

    found: list[tuple[Sentence, re.Match[str]]] = []
    previous = ""
    for sentence in sentences:
        written = list(_REF_WORD.finditer(sentence.text))
        if _byline(sentence.text, rules) or _equation(sentence.text):
            found.extend((sentence, match) for match in written)
            previous = sentence.text
            continue
        taken = {match.start("n") for match in written}
        matches = sorted(
            (
                *written,
                *(
                    match
                    for pattern in (_GLUED, _SPACED, _AFTER_DIGIT)
                    for match in pattern.finditer(sentence.text)
                    if match.start("n") not in taken
                    and _confirmed(
                        sentence.text, match, superscripts, rules, previous
                    )
                ),
            ),
            key=lambda match: match.start("n"),
        )
        found.extend((sentence, match) for match in matches)
        previous = sentence.text
    return found


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
        False after a single Latin letter ("v 2", "t2"), a unit word, the
        number 10 (a power: "1×10⁸", "10⁵"), inside parentheses opened in
        the same word or after a short argument ("log(σ²)", "f(0)²"), or
        after a Greek variable following a math sign ("N(µ, σ²)");
        otherwise
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
        not word
        or _LATIN_LETTER.fullmatch(word)
        or word.casefold() in rules.unit_words
        or _MATH_WORD.search(word)
        or _GREEK_VARIABLE.search(before.rstrip())
    ):
        return False

    context = _flat(f"{previous} {before}")[-rules.signature_characters:]
    digits = re.sub(r"\D", "", match.group("n"))
    return len(context) >= _CONTEXT_MIN and any(
        len(printed) >= _CONTEXT_MIN
        and (context.endswith(printed) or printed.endswith(context))
        for printed in superscripts.get(digits, ())
    )


def _parenthesized(text: str, match: re.Match[str]) -> bool:
    """Return whether a parenthesized number is a citation.

    Parameters
    ----------
    text: str
        The sentence.
    match: re.Match[str]
        A ``_PARENTHESIS`` match in it.

    Returns
    -------
    bool
        False for an equation or step number: an equation sign ("=",
        "≈", "∑", ...) within ``_EQUATION_REACH`` characters before it, or
        "Equation", "Eq.", "step" (or "and" in "Eqs. (1) and (2)") right
        before it, or a numbered list item: "(1)" starting the sentence or
        following a colon or semicolon, and followed by a word ("(1) video
        frames"; "(8), and" is a citation).

    """

    before = text[max(0, match.start() - _EQUATION_REACH): match.start()]
    stripped = before.rstrip()
    listed = (not stripped or stripped[-1] in ":;") and bool(
        re.match(r"\s+[^\W\d_]", text[match.end():])
    )
    return not (
        _EQUATION.search(before) or _NOT_CITED.search(before) or listed
    )


def _equation(text: str) -> bool:
    """Return whether a "sentence" is mostly an equation.

    Parameters
    ----------
    text: str
        The sentence.

    Returns
    -------
    bool
        True when at least ``_MATH_SHARE`` of its space-separated tokens
        are single characters or hold an equation glyph ("ðsÞ = 1 2").

    """

    tokens = text.split()
    math = sum(
        len(token) == 1 or bool(_MATH_GLYPH.search(token))
        for token in tokens
    )
    return bool(tokens) and math >= _MATH_SHARE * len(tokens)


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
    """Return text without spaces, for comparing contexts.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        Lowercase, without whitespace and unmapped glyphs, so the PDF's
        glyph text and the extracted text compare; symbols are kept
        ("H = α/2" gives "h=α/2"), so a short line start still has context.

    """

    return re.sub(r"\s+", "", _UNMAPPED.sub("", text.casefold()))
