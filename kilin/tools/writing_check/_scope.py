"""Choose the part of a document a question asks to check.

Private to ``writing_check``. "Check draft.tex" means the whole document,
but a question can name a part: "check the methods", "check section 2.3",
"the second paragraph of the introduction", "lines 120-180", or text
quoted or written after a colon. The part is read from the question in
code, without a model call, in this order:

1. **Quoted text** (at least four words, in quotes or after a colon): the
   paragraphs that contain it, or the text itself when it is not in the
   document.
2. **Lines**: the paragraphs touching the line range.
3. **A paragraph by position** ("the second paragraph of the methods",
   "the last paragraph"), within a section or the whole document.
4. **A section by number** ("section 2.3", "§2.3"), subsections included.
5. **A section by name**: "the methods section", "section on data", or a
   heading's title used in the question ("check the results").

A part the question names but the document does not have (a section
number or name with no heading) is not replaced by the whole document:
the result is a problem that lists the headings, so the user can choose.

Key Features
------------

- **No model call**

    The same question always gives the same part.

- **Names that match loosely**

    "the methods" finds "Materials and methods"; case, numbering and a
    plural "s" are ignored.

Module Structure
----------------

- ``Scope`` and ``ScopeProblem``

    The part to check, or why it could not be found.

- ``choose_scope``

    Read the part from a question.

- ``section_scope``

    One section's paragraphs, for a section the user picks.

Examples
--------
>>> from pathlib import Path
>>> from kilin.tools.writing_check._document import read_document
>>> from kilin.tools.writing_check._scope import choose_scope
>>> document = read_document(Path("a.md"), "# Results\\n\\nIt works.\\n")
>>> choose_scope("check the results", document).label
'section "Results"'

"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from ._document import Document, Paragraph, Section

__all__ = [
    "Scope",
    "ScopeProblem",
    "choose_scope",
    "section_scope",
]

# Fewest words of quoted text or text after a colon taken as the text to
# check ("check: the results is clear" is too short to be a passage).
_QUOTED_WORDS_MIN: Final[int] = 4
_QUOTED: Final[re.Pattern[str]] = re.compile(r"[\"“«]([^\"”»]+)[\"”»]")
_LINES: Final[re.Pattern[str]] = re.compile(
    r"\blines?\s+(\d+)(?:\s*(?:-|–|—|to|through)\s*(\d+))?", re.IGNORECASE
)
_ORDINALS: Final[dict[str, int]] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "last": -1, "final": -1,
}
_POSITION: Final[re.Pattern[str]] = re.compile(
    r"\b(?:(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth"
    r"|ninth|tenth|last|final)|(?P<count>\d+)(?:st|nd|rd|th))\s+paragraph"
    r"|\bparagraph\s+(?P<number>\d+)\b",
    re.IGNORECASE,
)
# What follows "of" or "in" after a paragraph position: a section.
_POSITION_SECTION: Final[re.Pattern[str]] = re.compile(
    r"^\s+(?:of|in)\s+(?:the\s+)?(?P<name>.+?)(?:\s+section)?\s*"
    r"(?:[?.!,;]|$|\s+(?:of|in)\s+\S+\.\w+)",
    re.IGNORECASE,
)
_SECTION_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:\bsection|\bsec\.|§|\bappendix)\s*"
    r"(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\b", re.IGNORECASE
)
_NAMED_SECTION: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"\bsection\s+(?:on|about|called|named|titled)\s+[\"“']?"
        r"(?P<name>[^\"”'?.!,;]+?)[\"”']?\s*(?:[?.!,;]|$|\s+(?:of|in)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the|this|that|my|our)\s+(?P<name>[A-Za-z][\w\s-]{0,60}?)"
        r"\s+section\b",
        re.IGNORECASE,
    ),
)
# Words after "the" that name what to check, not a section ("check the
# grammar of the draft").
_NOT_SECTIONS: Final[frozenset[str]] = frozenset({
    "grammar", "style", "spelling", "text", "document", "paper", "article",
    "manuscript", "draft", "chapter", "thesis", "report", "proposal",
    "file", "writing", "language", "english", "whole", "entire", "full",
    "wording", "punctuation", "tense", "consistency",
})
# A file name in the question ("draft.tex"), removed before names are
# matched so "methods.tex" does not name a section.
_FILE_NAME: Final[re.Pattern[str]] = re.compile(r"\S+\.[A-Za-z]{1,5}\b")


@dataclass(frozen=True, slots=True)
class Scope:
    """The part of a document to check.

    Attributes
    ----------
    paragraphs: tuple[int, ...]
        Indices of the paragraphs to check, in order.
    label: str
        How to name the part in the answer ("section 2.1 Data").
    text: str
        Text from the question to check on its own, when it is not in the
        document; ``""`` otherwise.

    """

    paragraphs: tuple[int, ...]
    label: str
    text: str = ""


@dataclass(frozen=True, slots=True)
class ScopeProblem:
    """Why the part a question names was not found.

    Attributes
    ----------
    message: str
        One sentence for the user.
    headings: tuple[str, ...]
        The document's headings as the user would name them ("2.1 Data"),
        to choose from.

    """

    message: str
    headings: tuple[str, ...]


def choose_scope(question: str, document: Document) -> Scope | ScopeProblem:
    """Return the part of a document a question asks to check.

    Parameters
    ----------
    question: str
        The question as typed.
    document: Document
        The document, read by ``read_document``.

    Returns
    -------
    Scope | ScopeProblem
        The paragraphs to check (every paragraph when the question names
        no part), or a problem when the part it names is not there.

    """

    quoted = _quoted(question)
    if quoted:
        return _quoted_scope(quoted, document)
    lines = _LINES.search(question)
    if lines:
        return _line_scope(int(lines.group(1)),
                           int(lines.group(2) or lines.group(1)), document)
    asked = _FILE_NAME.sub(" ", question)
    position = _POSITION.search(asked)
    if position:
        return _position_scope(position, asked, document)
    number = _SECTION_NUMBER.search(asked)
    if number:
        return _numbered_scope(number.group(1), document)
    for pattern in _NAMED_SECTION:
        named = pattern.search(asked)
        if named:
            return _named_scope(named["name"], document)
    for index in _titles_used(asked, document):
        return section_scope(index, document)
    return Scope(
        tuple(range(len(document.paragraphs))), "the whole document"
    )


def _quoted(question: str) -> str:
    """Return the text a question quotes or writes after a colon.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    str
        The longest quoted text, or the text after the first colon, when
        it has at least ``_QUOTED_WORDS_MIN`` words; otherwise ``""``.

    """

    quoted = max(
        (match.group(1).strip() for match in _QUOTED.finditer(question)),
        key=len, default="",
    )
    if len(quoted.split()) >= _QUOTED_WORDS_MIN:
        return quoted
    _, colon, after = question.partition(":")
    if colon and len(after.split()) >= _QUOTED_WORDS_MIN:
        return after.strip()
    return ""


def _quoted_scope(text: str, document: Document) -> Scope:
    """Return the paragraphs containing quoted text, or the text alone.

    Parameters
    ----------
    text: str
        Text from the question.
    document: Document
        The document.

    Returns
    -------
    Scope
        The paragraphs whose readable text contains it (case and spacing
        ignored); when none does, the text itself, checked on its own.

    """

    wanted = _squeezed(text)
    found = tuple(
        index for index, paragraph in enumerate(document.paragraphs)
        if wanted in _squeezed(paragraph.text)
        or wanted in _squeezed(paragraph.source)
    )
    if found:
        first = document.paragraphs[found[0]]
        return Scope(found, f"the quoted text (line {first.start_line})")
    return Scope((), "the text in the question", text)


def _line_scope(start: int, end: int, document: Document) -> Scope:
    """Return the paragraphs touching a line range.

    Parameters
    ----------
    start: int
        First line.
    end: int
        Last line; swapped with ``start`` when smaller.
    document: Document
        The document.

    Returns
    -------
    Scope
        The paragraphs with at least one line in the range (possibly
        none, when the lines hold no prose).

    """

    start, end = min(start, end), max(start, end)
    found = tuple(
        index for index, paragraph in enumerate(document.paragraphs)
        if paragraph.start_line <= end and paragraph.end_line >= start
    )
    label = f"line {start}" if start == end else f"lines {start}–{end}"
    return Scope(found, label)


def _position_scope(
    match: re.Match[str], question: str, document: Document
) -> Scope | ScopeProblem:
    """Return one paragraph by its position, in a section or overall.

    Parameters
    ----------
    match: re.Match[str]
        The position in the question (``_POSITION``).
    question: str
        The question, file names removed.
    document: Document
        The document.

    Returns
    -------
    Scope | ScopeProblem
        The paragraph at that position (counting the section's
        subsections too), or a problem when the section is not found or
        has fewer paragraphs.

    """

    if match["word"]:
        position = _ORDINALS[match["word"].lower()]
    else:
        position = int(match["count"] or match["number"])
    candidates = tuple(range(len(document.paragraphs)))
    where = "the document"
    following = _POSITION_SECTION.match(question[match.end():])
    if following:
        number = _SECTION_NUMBER.match(following["name"])
        part = (
            _numbered_scope(number.group(1), document) if number
            else _named_scope(following["name"], document)
        )
        if isinstance(part, ScopeProblem):
            return part
        candidates, where = part.paragraphs, part.label
    if not candidates or not -len(candidates) <= position <= len(
        candidates
    ) or position == 0:
        return ScopeProblem(
            f"{where[:1].upper()}{where[1:]} has {len(candidates)} "
            "paragraphs.",
            _headings(document),
        )
    index = candidates[position - 1 if position > 0 else position]
    word = match.group(0).strip()
    name = word if match["number"] else f"the {word}"
    return Scope((index,), f"{name} of {where}")


def _numbered_scope(number: str, document: Document) -> Scope | ScopeProblem:
    """Return a section by its printed number.

    Parameters
    ----------
    number: str
        The number ("2.3").
    document: Document
        The document.

    Returns
    -------
    Scope | ScopeProblem
        The section's paragraphs, subsections included, or a problem
        listing the headings when no section has that number.

    """

    for index, section in enumerate(document.sections):
        if section.number == number:
            return section_scope(index, document)
    return ScopeProblem(
        f"There is no section {number}.", _headings(document)
    )


def _named_scope(name: str, document: Document) -> Scope | ScopeProblem:
    """Return a section by its name.

    Parameters
    ----------
    name: str
        The name in the question ("methods", "the data analysis").
    document: Document
        The document.

    Returns
    -------
    Scope | ScopeProblem
        The section whose title is the name, else the only section whose
        title contains all its words (case, numbers and a final "s"
        ignored); a problem when none or several match, since the
        question clearly names a section.

    """

    wanted = _words(re.sub(r"^\s*(?:the|this|that|my|our)\s+", "", name))
    if not wanted:
        return ScopeProblem("Which section should I check?",
                            _headings(document))
    exact = [
        index for index, section in enumerate(document.sections)
        if _words(section.title) == wanted
    ]
    within = [
        index for index, section in enumerate(document.sections)
        if set(wanted) <= set(_words(section.title))
    ]
    found = exact or within
    if len(found) == 1:
        return section_scope(found[0], document)
    if len(found) > 1:
        return ScopeProblem(
            f"Several sections match “{name.strip()}”.",
            tuple(_heading(document.sections[index]) for index in found),
        )
    return ScopeProblem(
        f"There is no section called “{name.strip()}”.", _headings(document)
    )


def _titles_used(question: str, document: Document) -> list[int]:
    """Return the sections whose title the question uses after "the".

    Parameters
    ----------
    question: str
        The question, file names removed.
    document: Document
        The document.

    Returns
    -------
    list[int]
        Section indices, longest title first: a whole title after "the"
        ("check the results", "proofread the materials and methods"),
        else the only section whose title has every word of up to four
        words after "the" ("check the methods" for "Materials and
        methods"), words naming what to check ("the grammar") excepted.

    """

    asked = " ".join(_words(question))
    found = [
        index for index, section in enumerate(document.sections)
        if (title := " ".join(_words(section.title)))
        and re.search(rf"\bthe {re.escape(title)}\b", asked)
    ]
    if found:
        return sorted(
            found, key=lambda index: -len(document.sections[index].title)
        )
    titles = [set(_words(section.title)) for section in document.sections]
    # Phrases may overlap: "the style of the data" holds "the data".
    for phrase in re.finditer(r"\bthe (?=((?:[a-z]+ ){0,3}[a-z]+))", asked):
        words = phrase.group(1).split()
        while words and words[0] not in _NOT_SECTIONS:
            matches = [
                index for index, title in enumerate(titles)
                if set(words) <= title
            ]
            if len(matches) == 1:
                return matches
            words = words[:-1]
    return []


def section_scope(index: int, document: Document) -> Scope:
    """Return a section's paragraphs, subsections included.

    Parameters
    ----------
    index: int
        The section's index.
    document: Document
        The document.

    Returns
    -------
    Scope
        Paragraphs within the section's lines, labelled "section 2.1
        Data", or by title alone when it has no number.

    """

    section = document.sections[index]
    found = tuple(
        position for position, paragraph in enumerate(document.paragraphs)
        if _inside(paragraph, section)
    )
    if section.number:
        label = f"section {section.number} {section.title}"
    else:
        label = f'section "{section.title}"'
    return Scope(found, label)


def _inside(paragraph: Paragraph, section: Section) -> bool:
    """Whether a paragraph lies within a section's lines.

    Parameters
    ----------
    paragraph: Paragraph
        A paragraph.
    section: Section
        A section.

    Returns
    -------
    bool
        True when the paragraph starts within the section.

    """

    return section.start_line <= paragraph.start_line <= section.end_line


def _headings(document: Document) -> tuple[str, ...]:
    """Return the document's headings as a user would name them.

    Parameters
    ----------
    document: Document
        The document.

    Returns
    -------
    tuple[str, ...]
        "2.1 Data" for a numbered heading, its title otherwise.

    """

    return tuple(_heading(section) for section in document.sections)


def _heading(section: Section) -> str:
    """Return one heading as a user would name it.

    Parameters
    ----------
    section: Section
        A section.

    Returns
    -------
    str
        Number and title, or the title alone.

    """

    return f"{section.number} {section.title}".strip()


def _words(text: str) -> list[str]:
    """Return a name's words, case-folded, without numbers or a final "s".

    Parameters
    ----------
    text: str
        A title or a name from the question.

    Returns
    -------
    list[str]
        Words of letters, each without a final "s" ("methods" and
        "method" match).

    """

    return [
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in re.findall(r"[^\W\d_]+", text.casefold())
    ]


def _squeezed(text: str) -> str:
    """Return text in lower case with single spaces, for containment.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        Its words joined by single spaces, case-folded.

    """

    return " ".join(text.casefold().split())
