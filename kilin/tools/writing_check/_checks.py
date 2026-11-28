"""Check a document's consistency and style in code, and build its glossary.

Private to ``writing_check``. Some problems are only visible across the
whole document, where a model reading one paragraph cannot see them, and
are found reliably by code:

- **Acronyms**: used before they are defined, never defined, or defined
  twice. The abstract is read on its own, as journals expect it to
  define its acronyms again.
- **Terms written several ways**: "data set", "data-set" and "dataset";
  the less frequent forms are reported, with the most frequent as the
  suggestion.
- **US and UK spelling**: the variant used less is reported.
- **Promotional words** ("novel", "unprecedented"), **repeated words**
  ("the the") and **long sentences**.

The same pass builds the glossary given to every model call, so that the
model keeps the authors' acronyms, terms and spelling.

Key Features
------------

- **No model call**

    Instant, and the same on every run.

- **Findings shaped like the model's**

    Each finding has a paragraph, a line, a level, the original words, a
    suggestion and a reason.

Module Structure
----------------

- ``Finding`` and ``Glossary``

    One problem found, and what the model is told about the document.

- ``check_document``

    Every check of this module, and the glossary.

- ``make_finding``

    A finding placed at the source line of its words, for any check.

- ``sentences``

    A paragraph's sentences, with where each starts.

Examples
--------
>>> from pathlib import Path
>>> from kilin.tools.writing_check._document import read_document
>>> from kilin.tools.writing_check._checks import check_document
>>> document = read_document(Path("a.txt"), "We used the the tool.")
>>> [finding.check for finding in check_document(document)[0]]
['repeated word']

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
import re
from typing import Final

from . import _configuration as _config
from ._document import Document, Paragraph

__all__ = [
    "LEVEL_CONSISTENCY",
    "LEVEL_GRAMMAR",
    "LEVEL_STYLE",
    "Finding",
    "Glossary",
    "check_document",
    "make_finding",
    "sentences",
]

LEVEL_GRAMMAR: Final[str] = "grammar"
LEVEL_STYLE: Final[str] = "style"
LEVEL_CONSISTENCY: Final[str] = "consistency"

# An acronym: two or more capitals, possibly after one lowercase letter
# ("fMRI") and with digits ("H2O" is not one, "COVID19" is), and a plural
# "s" ("GPUs").
_ACRONYM: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w-])(?P<short>[a-z]?[A-Z][A-Z0-9]*[A-Z][A-Z0-9]*)s?(?![\w-])"
)
_PIECES: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][a-z]*|[A-Z]+")
# Small words a definition's initials may skip ("rate of change (ROC)").
_SKIPPED_WORDS: Final[frozenset[str]] = frozenset({
    "of", "and", "the", "for", "in", "to", "a", "an", "with", "on", "by",
})
_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_REPEATED: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<word>[A-Za-z]+)\s+(?P=word)\b", re.IGNORECASE
)
# Repeated words that can be correct ("that that", "had had").
_REPEAT_ALLOWED: Final[frozenset[str]] = frozenset({"that", "had", "is"})
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bFig\.)"
    r"(?<!\bEq\.)(?<!\bRef\.)(?<!\bvs\.)\s+(?=[A-Z0-9\[(])"
)
# Words shown at the start of a long sentence in its finding.
_SENTENCE_START_WORDS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class Finding:
    """One problem found in the document.

    Attributes
    ----------
    paragraph: int
        Index of its paragraph in ``Document.paragraphs``.
    line: int
        Its source line, or the paragraph's first line when the words
        cannot be found in the source as written.
    level: str
        ``grammar``, ``style`` or ``consistency``.
    check: str
        Which check found it ("acronym", "term", "spelling", ...).
    original: str
        The words as they appear in the paragraph's readable text.
    suggestion: str
        What to write instead, or ``""`` when the finding only flags.
    reason: str
        One line explaining it.
    offset: int
        Where the words start in the paragraph's readable text, to keep
        findings on one line in reading order.

    """

    paragraph: int
    line: int
    level: str
    check: str
    original: str
    suggestion: str
    reason: str
    offset: int = 0


@dataclass(frozen=True, slots=True)
class Glossary:
    """What every model call is told about the document.

    Attributes
    ----------
    acronyms: dict[str, str]
        Each defined acronym and its long form.
    terms: dict[str, str]
        Each term written several ways, by its preferred form, mapped to
        that form ("data set" to "dataset").
    spelling: str
        ``US``, ``UK``, or ``""`` when the document shows neither.

    """

    acronyms: dict[str, str] = field(default_factory=dict)
    terms: dict[str, str] = field(default_factory=dict)
    spelling: str = ""

    def describe(self) -> str:
        """Return the glossary as lines for a prompt.

        Returns
        -------
        str
            One line per kind of entry, or ``""`` when it is empty.

        """

        lines = []
        if self.acronyms:
            listed = "; ".join(
                f"{short} = {long}" for short, long in self.acronyms.items()
            )
            lines.append(f"Acronyms defined in the document: {listed}.")
        preferred = sorted(set(self.terms.values()))
        if preferred:
            lines.append(
                "Terms the document writes this way: "
                + ", ".join(f"“{term}”" for term in preferred) + "."
            )
        if self.spelling:
            lines.append(f"Spelling: {self.spelling} English.")
        return "\n".join(lines)


def check_document(document: Document) -> tuple[list[Finding], Glossary]:
    """Run every code check on a document.

    Parameters
    ----------
    document: Document
        The document, read by ``read_document``.

    Returns
    -------
    tuple[list[Finding], Glossary]
        The findings in document order (by paragraph, then line), and the
        glossary for the model calls.

    """

    paragraphs = document.paragraphs
    abstract = _abstract_paragraphs(document)
    acronyms, acronym_findings = _acronyms(paragraphs, abstract)
    terms, term_findings = _terms(paragraphs)
    spelling, spelling_findings = _spelling(paragraphs)
    findings = [
        *acronym_findings, *term_findings, *spelling_findings,
        *_promotional(paragraphs), *_repeated(paragraphs),
        *_long_sentences(paragraphs),
    ]
    findings.sort(
        key=lambda found: (found.paragraph, found.line, found.offset)
    )
    return findings, Glossary(acronyms, terms, spelling)


def _abstract_paragraphs(document: Document) -> frozenset[int]:
    """Return the paragraphs of the abstract.

    Parameters
    ----------
    document: Document
        The document.

    Returns
    -------
    frozenset[int]
        Paragraph indices in a section titled "Abstract".

    """

    return frozenset(
        index for index, paragraph in enumerate(document.paragraphs)
        if paragraph.section is not None
        and document.sections[paragraph.section].title.casefold()
        == "abstract"
    )


def _acronyms(
    paragraphs: Sequence[Paragraph], abstract: frozenset[int]
) -> tuple[dict[str, str], list[Finding]]:
    """Find acronyms used before definition, never defined, or redefined.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.
    abstract: frozenset[int]
        Paragraphs of the abstract, checked on their own.

    Returns
    -------
    tuple[dict[str, str], list[Finding]]
        Each acronym's long form (from its first definition), and the
        findings: the first use before a definition, the first use of
        an acronym never defined (unless commonly known), and each
        definition after the first.

    """

    long_forms: dict[str, str] = {}
    findings = []
    for part in (
        [i for i in range(len(paragraphs)) if i in abstract],
        [i for i in range(len(paragraphs)) if i not in abstract],
    ):
        # Paragraph index and offset of each acronym's first definition
        # and first use, so a use earlier in the same paragraph counts.
        defined: dict[str, tuple[int, int]] = {}
        used: dict[str, tuple[int, int]] = {}
        for index in part:
            text = paragraphs[index].text
            for match in _ACRONYM.finditer(text):
                short = match["short"]
                if short.casefold() in _config.KNOWN_ACRONYMS:
                    continue
                definition = _definition(text, match)
                if definition:
                    long_forms.setdefault(short, definition)
                    if short in defined:
                        findings.append(make_finding(
                            paragraphs, index, LEVEL_CONSISTENCY, "acronym",
                            f"{definition} ({short})", short,
                            f"{short} is already defined (line "
                            f"{_line_of(paragraphs, *defined[short], short)}"
                            "); use the acronym alone.",
                            max(0, match.start() - len(definition) - 2),
                        ))
                    else:
                        defined[short] = (index, match.start())
                    continue
                used.setdefault(short, (index, match.start()))
        for short, (index, offset) in used.items():
            if short not in defined:
                findings.append(make_finding(
                    paragraphs, index, LEVEL_CONSISTENCY, "acronym", short,
                    "", f"{short} is never defined; spell it out at first "
                    "use.", offset,
                ))
            elif (index, offset) < defined[short]:
                findings.append(make_finding(
                    paragraphs, index, LEVEL_CONSISTENCY, "acronym", short,
                    "", f"{short} is used before its definition (line "
                    f"{_line_of(paragraphs, *defined[short], short)}); "
                    "define it here.", offset,
                ))
    return long_forms, findings


def _definition(text: str, match: re.Match[str]) -> str:
    """Return the long form an acronym is defined with, if this defines it.

    Parameters
    ----------
    text: str
        The paragraph.
    match: re.Match[str]
        One acronym in it.

    Returns
    -------
    str
        The words before "(ACR)" whose initials spell it ("single-particle
        tracking (SPT)"), or the words in brackets after it ("SPT
        (single-particle tracking)"); ``""`` otherwise.

    """

    short = match["short"]
    before = text[:match.start()]
    after = text[match.end():]
    if before.rstrip().endswith("(") and after.lstrip().startswith(")"):
        words = before.rstrip()[:-1].split()
        for count in range(1, len(short) + 4):
            candidate = words[-count:] if count <= len(words) else []
            if candidate and _initials_match(" ".join(candidate), short):
                return " ".join(candidate)
        return ""
    bracket = re.match(r"\s*\(([^()]{3,80})\)", after)
    if bracket and _initials_match(bracket.group(1), short):
        return bracket.group(1).strip()
    return ""


def _initials_match(words: str, short: str) -> bool:
    """Whether words spell an acronym by their initials.

    Parameters
    ----------
    words: str
        Candidate long form.
    short: str
        The acronym.

    Returns
    -------
    bool
        True when the first letters of the words (hyphenated parts and
        capitalized parts counted as words, small words optional) are the
        acronym's letters, case ignored.

    """

    letters = [c.casefold() for c in short if c.isalpha()]
    pieces = [
        piece for piece in _PIECES.findall(words.replace("-", " "))
    ]
    initials = [
        (piece[0].casefold(), piece.casefold() in _SKIPPED_WORDS)
        for piece in pieces
    ]

    return bool(letters) and _spells(letters, initials, 0, 0)


def _spells(
    letters: list[str], initials: list[tuple[str, bool]], at: int,
    start: int,
) -> bool:
    """Whether initials spell an acronym's letters from given positions.

    Parameters
    ----------
    letters: list[str]
        The acronym's letters, case-folded.
    initials: list[tuple[str, bool]]
        Each word's initial, and whether the word may be skipped.
    at: int
        Position in the letters.
    start: int
        Position in the initials.

    Returns
    -------
    bool
        True when the initials from ``start`` spell the letters from
        ``at`` exactly, skipping only small words.

    """

    if at == len(letters):
        return all(optional for _, optional in initials[start:])
    if start == len(initials):
        return False
    initial, optional = initials[start]
    if initial == letters[at] and _spells(letters, initials, at + 1,
                                          start + 1):
        return True
    return optional and _spells(letters, initials, at, start + 1)


def _terms(
    paragraphs: Sequence[Paragraph],
) -> tuple[dict[str, str], list[Finding]]:
    """Find terms written as one word, two words, and hyphenated.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.

    Returns
    -------
    tuple[dict[str, str], list[Finding]]
        Each less frequent form mapped to the most frequent one, and a
        finding for each use of a less frequent form. Ties are not
        reported: there is no preferred form to suggest.

    """

    counts: Counter[str] = Counter()
    for paragraph in paragraphs:
        words = [word.casefold() for word in _WORD.findall(paragraph.text)]
        counts.update(words)
        counts.update(
            f"{first} {second}" for first, second in zip(words, words[1:])
        )
    forms: dict[str, str] = {}
    for word in list(counts):
        if "-" in word or " " in word or len(word) < 6:
            continue
        for cut in range(3, len(word) - 2):
            first, second = word[:cut], word[cut:]
            variants = {
                word: counts[word],
                f"{first} {second}": counts[f"{first} {second}"],
                f"{first}-{second}": counts[f"{first}-{second}"],
            }
            used = {form: count for form, count in variants.items() if count}
            if len(used) < 2:
                continue
            ranked = sorted(used.items(), key=lambda item: -item[1])
            if ranked[0][1] == ranked[1][1]:
                continue
            for form, _ in ranked[1:]:
                forms[form] = ranked[0][0]
    findings = []
    for index, paragraph in enumerate(paragraphs):
        for form, preferred in forms.items():
            pattern = rf"(?<![\w-]){re.escape(form)}(?![\w-])"
            for match in re.finditer(pattern, paragraph.text, re.IGNORECASE):
                findings.append(make_finding(
                    paragraphs, index, LEVEL_CONSISTENCY, "term",
                    match.group(0), preferred,
                    f"the document mostly writes “{preferred}”.",
                    match.start(),
                ))
    return forms, findings


def _spelling(
    paragraphs: Sequence[Paragraph],
) -> tuple[str, list[Finding]]:
    """Find US and UK spellings mixed in one document.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.

    Returns
    -------
    tuple[str, list[Finding]]
        ``US`` or ``UK`` when one variant is used more, else ``""``; and a
        finding for each use of the other variant.

    """

    pairs = _spelling_forms()
    us_words = {us: uk for us, uk in pairs}
    uk_words = {uk: us for us, uk in pairs}
    hits = []
    for index, paragraph in enumerate(paragraphs):
        for match in _WORD.finditer(paragraph.text):
            word = match.group(0).casefold()
            if word in us_words:
                hits.append(("US", index, match, us_words[word]))
            elif word in uk_words:
                hits.append(("UK", index, match, uk_words[word]))
    tally = Counter(variant for variant, _, _, _ in hits)
    if tally["US"] == tally["UK"]:
        return "", []
    spelling = "US" if tally["US"] > tally["UK"] else "UK"
    findings = [
        make_finding(
            paragraphs, index, LEVEL_CONSISTENCY, "spelling",
            found.group(0), _same_case(found.group(0), other),
            f"the document uses {spelling} spelling.", found.start(),
        )
        for variant, index, found, other in hits if variant != spelling
    ]
    return spelling, findings


def _spelling_forms() -> list[tuple[str, str]]:
    """Return the US and UK spellings with their common inflections.

    Returns
    -------
    list[tuple[str, str]]
        Each configured pair, and for "-ize"/"-ise" and "-yze"/"-yse"
        verbs their "-d", "-s", "-ing" and "-ation" forms; for "-or"/"-our"
        and "-er"/"-re" nouns their plural.

    """

    forms = []
    for us, uk in _config.SPELLING_PAIRS:
        forms.append((us, uk))
        if us.endswith(("ize", "yze")) and uk.endswith(("ise", "yse")):
            forms += [(us + "d", uk + "d"), (us + "s", uk + "s"),
                      (us[:-1] + "ing", uk[:-1] + "ing"),
                      (us[:-1] + "ation", uk[:-1] + "ation"),
                      (us[:-1] + "ations", uk[:-1] + "ations"),
                      (us + "r", uk + "r"), (us + "rs", uk + "rs")]
        elif us.endswith(("or", "er")) and uk.endswith(("our", "re")):
            forms.append((us + "s", uk + "s"))
    return forms


def _same_case(original: str, replacement: str) -> str:
    """Return a replacement with the original's capital first letter.

    Parameters
    ----------
    original: str
        The word as written.
    replacement: str
        Its replacement, in lower case.

    Returns
    -------
    str
        The replacement, capitalized when the original is.

    """

    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _promotional(paragraphs: Sequence[Paragraph]) -> list[Finding]:
    """Find promotional words.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.

    Returns
    -------
    list[Finding]
        One style finding per use of a configured promotional word.

    """

    words = "|".join(
        re.escape(word) for word in sorted(
            _config.PROMOTIONAL_WORDS, key=len, reverse=True
        )
    )
    pattern = re.compile(rf"(?<![\w-])(?:{words})(?![\w-])", re.IGNORECASE)
    return [
        make_finding(
            paragraphs, index, LEVEL_STYLE, "promotional word",
            match.group(0), "",
            "promotional; let the results show it, or remove it.",
            match.start(),
        )
        for index, paragraph in enumerate(paragraphs)
        for match in pattern.finditer(paragraph.text)
    ]


def _repeated(paragraphs: Sequence[Paragraph]) -> list[Finding]:
    """Find a word written twice in a row.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.

    Returns
    -------
    list[Finding]
        One grammar finding per repeat ("the the"), except words that can
        correctly repeat ("that that", "had had").

    """

    return [
        make_finding(
            paragraphs, index, LEVEL_GRAMMAR, "repeated word",
            match.group(0), match["word"], "the word is written twice.",
            match.start(),
        )
        for index, paragraph in enumerate(paragraphs)
        for match in _REPEATED.finditer(paragraph.text)
        if match["word"].casefold() not in _REPEAT_ALLOWED
    ]


def sentences(text: str) -> list[tuple[int, str]]:
    """Return a paragraph's sentences, with where each starts.

    Parameters
    ----------
    text: str
        A paragraph's readable text.

    Returns
    -------
    list[tuple[int, str]]
        Each sentence's position in the text and its words, split after
        ".", "!" or "?" before a capital, digit or bracket ("et al.",
        "e.g.", "Fig." and similar do not end a sentence).

    """

    found = []
    offset = 0
    for sentence in _SENTENCE_END.split(text):
        start = text.find(sentence, offset)
        if start < 0:
            start = offset
        found.append((start, sentence))
        offset = start + len(sentence)
    return found


def _long_sentences(paragraphs: Sequence[Paragraph]) -> list[Finding]:
    """Find sentences longer than the configured number of words.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.

    Returns
    -------
    list[Finding]
        One style finding per long sentence, quoting its first words.

    """

    findings = []
    for index, paragraph in enumerate(paragraphs):
        for start, sentence in sentences(paragraph.text):
            words = sentence.split()
            if len(words) > _config.LONG_SENTENCE_WORDS:
                findings.append(make_finding(
                    paragraphs, index, LEVEL_STYLE, "long sentence",
                    " ".join(words[:_SENTENCE_START_WORDS]), "",
                    f"{len(words)} words; consider splitting it.",
                    start,
                ))
    return findings


def make_finding(
    paragraphs: Sequence[Paragraph],
    index: int,
    level: str,
    check: str,
    original: str,
    suggestion: str,
    reason: str,
    offset: int = 0,
) -> Finding:
    """Return a finding with the source line of its words.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.
    index: int
        The finding's paragraph.
    level: str
        Its level.
    check: str
        The check that found it.
    original: str
        Its words.
    suggestion: str
        The suggested words, or ``""``.
    reason: str
        Its reason.
    offset: int, optional
        Where the words are in the paragraph's readable text. Defaults to
        ``0``.

    Returns
    -------
    Finding
        The finding, at the source line of its words (``_line_of``).

    """

    line = _line_of(paragraphs, index, offset, original)
    return Finding(
        index, line, level, check, original, suggestion, reason, offset
    )


def _line_of(
    paragraphs: Sequence[Paragraph], index: int, offset: int, words: str
) -> int:
    """Return the source line of words in a paragraph.

    Parameters
    ----------
    paragraphs: Sequence[Paragraph]
        The document's paragraphs.
    index: int
        The paragraph.
    offset: int
        Where the words are in the paragraph's readable text.
    words: str
        The words.

    Returns
    -------
    int
        The source line holding the same occurrence of the words' first
        three words (else of the first word): the second "SPT" of the
        readable text is on the line of the second "SPT" of the source.
        The paragraph's first line when neither is found, as LaTeX
        commands may split the words.

    """

    paragraph = paragraphs[index]
    text = " ".join(paragraph.text.casefold().split())
    before = " ".join(paragraph.text[:offset].casefold().split())
    lines = [" ".join(line.casefold().split())
             for line in paragraph.source.split("\n")]
    split = words.casefold().split()
    for probe in (" ".join(split[:3]), split[0] if split else ""):
        if not probe or probe not in text:
            continue
        wanted = before.count(probe) + 1
        seen = 0
        for number, line in enumerate(lines):
            seen += line.count(probe)
            if seen >= wanted:
                return paragraph.start_line + number
    return paragraph.start_line
