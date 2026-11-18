"""Take one document's notes for a project report, and check them.

Private to ``project_summary``. The map step of a project report reads
each document once (an excerpt, ``_documents``) and asks the model, in one
call, for:

1. **Notes** for each section the report writes from all documents
   (``notes`` sections): a few facts, or nothing.
2. **Checks** for each per-document section: whether the document meets
   its ``When:`` condition (yes, no or unclear), the words of the excerpt
   that decide it, and, when it does, the entry its ``Do:`` asks for (for a
   table, one value per column).

Code then checks the reply: a "yes" whose quoted words are not in the
excerpt, or that has no entry, becomes "unclear", so no document is listed
without evidence; entries are cut to their word limit and table rows to
their columns.

Key Features
------------

- **One call per document**

    Notes and checks for every section together.

- **Evidence for every entry**

    A document is listed only with words found in what was read.

Module Structure
----------------

- ``CHECK_YES``, ``CHECK_NO`` and ``CHECK_UNCLEAR``

    A check's answers.

- ``Check`` and ``DocumentNotes``

    One per-document check, and one document's notes and checks.

- ``topics``, ``per_document_sections`` and ``template_texts``

    The template's sections by kind, and its words for choosing passages.

- ``map_prompt`` and ``parse_notes``

    The model call for one document, and its checked result.

Examples
--------
>>> from kilin.tools.project_summary._mapping import quote_found
>>> quote_found("“trained a U-Net”", "We trained a\\nU-Net on 40 images.")
True

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re
from typing import Final

from kilin._model_completion import strip_thinking
from kilin.tools._shared.report_templates import (
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_PER_DOCUMENT,
    ReportTemplate,
    TemplateSection,
)
from kilin.tools.project_summary._documents import ProjectDocument

__all__ = [
    "CHECK_NO",
    "CHECK_UNCLEAR",
    "CHECK_YES",
    "Check",
    "DocumentNotes",
    "map_prompt",
    "parse_notes",
    "per_document_sections",
    "quote_found",
    "template_texts",
    "topics",
]

CHECK_YES: Final[str] = "yes"
CHECK_NO: Final[str] = "no"
CHECK_UNCLEAR: Final[str] = "unclear"
_LINE = re.compile(
    r"^\s*\**(NOTE|CHECK|QUOTE|ENTRY|VALUE)\s+([0-9]+|[A-Z][0-9]{0,2})"
    r"(?:\s+[^:\n]{1,60}?)?\**\s*:\s*(.*?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_MISSING: Final[str] = "not reported"
_NONE = re.compile(r"^(?:none|n/?a|-|—|not (?:reported|stated|mentioned))\.?$",
                   re.IGNORECASE)
_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# A quote's words are found when at least this many of them are in a row
# in the excerpt (shorter quotes must be found whole).
_QUOTE_WORDS_MIN: Final[int] = 4


@dataclass(frozen=True, slots=True)
class Check:
    """Whether one document meets a per-document section's condition.

    Attributes
    ----------
    answer: str
        ``CHECK_YES``, ``CHECK_NO`` or ``CHECK_UNCLEAR``.
    quote: str
        The excerpt's words that decide it, as the model quoted them.
    entry: str
        For a yes, what the section writes for the document; else ``""``.
    cells: tuple[str, ...]
        For a yes in a table, one value per column; else empty.
    why: str
        For an unclear answer that code decided, why ("the quoted words
        are not in the document"); else ``""``.

    """

    answer: str
    quote: str = ""
    entry: str = ""
    cells: tuple[str, ...] = ()
    why: str = ""


@dataclass(frozen=True, slots=True)
class DocumentNotes:
    """One document's notes and checks for a project report.

    Attributes
    ----------
    number: int
        The document's number in the report.
    notes: dict[str, str]
        For each ``notes`` section, by name, what the document says about
        it; sections it says nothing about are left out.
    checks: dict[str, Check]
        For each per-document section, by name, its check.
    failure: str
        Why the document could not be read, or ``""``.

    """

    number: int
    notes: dict[str, str] = field(default_factory=dict)
    checks: dict[str, Check] = field(default_factory=dict)
    failure: str = ""


def topics(template: ReportTemplate) -> list[TemplateSection]:
    """Return the sections written from all documents' notes.

    Parameters
    ----------
    template: ReportTemplate
        A project template.

    Returns
    -------
    list[TemplateSection]
        Its ``notes`` sections and labelled lines, in order.

    """

    return [s for s in template.sections if s.source == SECTION_SOURCE_NOTES]


def per_document_sections(template: ReportTemplate) -> list[TemplateSection]:
    """Return the sections with one entry per matching document.

    Parameters
    ----------
    template: ReportTemplate
        A project template.

    Returns
    -------
    list[TemplateSection]
        Its per-document sections, in order (at most 26).

    """

    sections = [
        s for s in template.sections
        if s.source == SECTION_SOURCE_PER_DOCUMENT
    ]
    if len(sections) > len(_LETTERS):
        raise ValueError("At most 26 per-document sections are supported.")
    return sections


def template_texts(template: ReportTemplate) -> list[str]:
    """Return what the template asks about, for choosing passages.

    Parameters
    ----------
    template: ReportTemplate
        A project template.

    Returns
    -------
    list[str]
        The names and instructions of its notes sections, and the names,
        conditions and actions of its per-document sections.

    """

    texts = [f"{s.name} {s.guidance}" for s in topics(template)]
    texts += [
        f"{s.name} {s.when} {s.do} {' '.join(s.columns)}"
        for s in per_document_sections(template)
    ]
    return texts


def map_prompt(
    template: ReportTemplate, document: ProjectDocument, excerpt: str
) -> tuple[str, str]:
    """Return the prompts taking one document's notes and checks.

    Parameters
    ----------
    template: ReportTemplate
        A project template.
    document: ProjectDocument
        The document.
    excerpt: str
        What is read from it (``document_excerpt``).

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for ``NOTE n:`` lines, one per notes
        section, and ``CHECK``, ``QUOTE`` and ``ENTRY`` lines per
        per-document section, from the excerpt only.

    """

    system = (
        "You take notes on one document for a report about a collection "
        "of documents. Use only the excerpt given; never add facts from "
        "elsewhere. Quote the excerpt's own words."
    )
    parts = [
        f"Report: {template.title}"
        + (f" — {template.description}" if template.description else ""),
        f'Document [{document.number}]: {document.title}\n\nExcerpt:\n"""\n'
        f'{excerpt}\n"""',
    ]
    format_lines: list[str] = []
    notes = topics(template)
    if notes:
        parts.append(
            "Topics: for each, write what this document contributes to it, "
            "in one or two short sentences of its facts; for a topic about "
            "the collection as a whole, what this document is about. Write "
            "NONE only when the excerpt says nothing about it.\n"
            + "\n".join(
                f"{index}. {s.name}: {s.guidance}".rstrip(": ")
                for index, s in enumerate(notes, start=1)
            )
        )
        format_lines += [
            f"NOTE {index}: <facts, or NONE>"
            for index in range(1, len(notes) + 1)
        ]
    checks = per_document_sections(template)
    if checks:
        questions = []
        for letter, s in zip(_LETTERS, checks):
            if s.columns:
                action = (
                    "the document's " + ", ".join(s.columns)
                    + ", one per VALUE line, writing \"not reported\" when "
                    "the excerpt does not say"
                )
            else:
                action = s.do + (
                    f" (at most {s.words[1]} words)" if s.words else ""
                )
            questions.append(
                f"{letter}. Does the document meet this: {s.when} "
                f"If yes, write: {action}."
                + (f" {s.guidance}" if s.guidance else "")
            )
        parts.append("Questions:\n" + "\n".join(questions))
        for letter, s in zip(_LETTERS, checks):
            format_lines += [
                f"CHECK {letter}: YES, NO or UNCLEAR",
                f"QUOTE {letter}: <the excerpt's exact words that decide "
                "it>",
            ]
            if s.columns:
                format_lines += [
                    f"VALUE {letter}{index} {column}: <for YES only: the "
                    f"document's {column.lower()}>"
                    for index, column in enumerate(s.columns, start=1)
                ]
            else:
                format_lines.append(f"ENTRY {letter}: <for YES only>")
    parts.append("Answer exactly in this format, one line each:\n"
                 + "\n".join(format_lines))
    return system, "\n\n".join(parts)


def _plain(text: str) -> str:
    """Return text for comparing quotes: words only, lower case.

    Parameters
    ----------
    text: str
        A quote or an excerpt.

    Returns
    -------
    str
        Its words in lower case, single-spaced, with words broken at a
        hyphen over lines joined.

    """

    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return " ".join(re.findall(r"\w+", text.casefold()))


def quote_found(quote: str, excerpt: str) -> bool:
    """Tell whether quoted words are in the excerpt.

    Parameters
    ----------
    quote: str
        The model's quote, possibly in quotation marks or with "...".
    excerpt: str
        What was read.

    Returns
    -------
    bool
        True when every part of the quote (split at an ellipsis) with at
        least one word is in the excerpt, ignoring case, punctuation and
        line breaks; parts of ``_QUOTE_WORDS_MIN`` or more words may also
        be found by their first ``_QUOTE_WORDS_MIN`` words.

    """

    text = _plain(excerpt)
    parts = [_plain(part) for part in re.split(r"\.\.\.|…", quote)]
    parts = [part for part in parts if part]
    if not parts:
        return False
    for part in parts:
        words = part.split()
        if f" {part} " in f" {text} ":
            continue
        head = " ".join(words[:_QUOTE_WORDS_MIN])
        if len(words) >= _QUOTE_WORDS_MIN and f" {head} " in f" {text} ":
            continue
        return False
    return True


def _cells(
    section: TemplateSection,
    letter: str,
    lines: dict[tuple[str, str], str],
    entry: str,
) -> tuple[str, ...]:
    """Return a table row's values, one per column, cleaned.

    Parameters
    ----------
    section: TemplateSection
        A per-document table section.
    letter: str
        Its letter in the reply.
    lines: dict[tuple[str, str], str]
        The reply's lines by kind and key.
    entry: str
        Its ``ENTRY`` line, read as values separated by ``|`` when there
        are no ``VALUE`` lines.

    Returns
    -------
    tuple[str, ...]
        One value per column: from ``VALUE A1``, ``VALUE A2``, ... lines,
        else the entry's ``|``-separated values (extra ones joined into the
        last). A column name written before a value ("Aim: ...") is
        removed; a value that is empty, "none", or only the column's name
        is "not reported".

    """

    size = len(section.columns)
    values = [lines.get(("VALUE", f"{letter}{index}"), "")
              for index in range(1, size + 1)]
    if not any(values) and entry:
        values = [cell.strip() for cell in entry.strip("| ").split("|")]
        if len(values) > size:
            values = values[:size - 1] + [" ".join(values[size - 1:])]
        values += [""] * (size - len(values))
    cleaned: list[str] = []
    for column, value in zip(section.columns, values):
        value = re.sub(
            rf"^\**\s*{re.escape(column)}\s*(?::\s*\**|\**\s*:)\s*", "",
            value.strip(), flags=re.IGNORECASE,
        ).strip()
        if not value or _NONE.match(value) or (
            value.casefold().strip(". ") == column.casefold()
        ):
            value = _MISSING
        cleaned.append(value)
    return tuple(cleaned)


def parse_notes(
    reply: str,
    template: ReportTemplate,
    number: int,
    excerpt: str,
) -> DocumentNotes:
    """Return one document's checked notes from the model's reply.

    Parameters
    ----------
    reply: str
        The model's reply to ``map_prompt``.
    template: ReportTemplate
        The project template.
    number: int
        The document's number.
    excerpt: str
        What was read, to check quotes.

    Returns
    -------
    DocumentNotes
        Notes by section (``NONE`` and empty notes left out) and a check
        per per-document section: missing or unreadable answers are
        unclear; a yes is unclear when its quote is not in the excerpt, it
        has no entry, or a table row has no value (``_cells``: values
        that only repeat their column's name count as not reported);
        entries are cut to the section's word limit.

    """

    lines: dict[tuple[str, str], str] = {}
    for kind, key, text in _LINE.findall(strip_thinking(reply)):
        lines.setdefault((kind.upper(), key.upper()), text.strip())
    notes: dict[str, str] = {}
    for index, section in enumerate(topics(template), start=1):
        note = lines.get(("NOTE", str(index)), "").strip()
        if note and not _NONE.match(note):
            notes[section.name] = note
    checks: dict[str, Check] = {}
    for letter, section in zip(_LETTERS, per_document_sections(template)):
        answer = lines.get(("CHECK", letter), "").split()
        answer = answer[0].strip(".:*").casefold() if answer else ""
        quote = lines.get(("QUOTE", letter), "").strip()
        entry = lines.get(("ENTRY", letter), "").strip()
        if answer == CHECK_NO:
            checks[section.name] = Check(CHECK_NO, quote)
        elif answer != CHECK_YES:
            checks[section.name] = Check(
                CHECK_UNCLEAR, quote,
                why="" if answer == CHECK_UNCLEAR else "no answer",
            )
        elif not quote or not quote_found(quote, excerpt):
            checks[section.name] = Check(
                CHECK_UNCLEAR, quote,
                why="the quoted words are not in the document",
            )
        elif section.columns:
            cells = _cells(section, letter, lines, entry)
            if all(cell == _MISSING for cell in cells):
                checks[section.name] = Check(
                    CHECK_UNCLEAR, quote, why="no values were written"
                )
            else:
                checks[section.name] = Check(
                    CHECK_YES, quote, " | ".join(cells), cells
                )
        elif not entry or _NONE.match(entry):
            checks[section.name] = Check(CHECK_UNCLEAR, quote,
                                         why="no entry was written")
        else:
            words = entry.split()
            if section.words and len(words) > section.words[1]:
                entry = " ".join(words[:section.words[1]]).rstrip(",;:") + "…"
            checks[section.name] = Check(CHECK_YES, quote, entry)
    return DocumentNotes(number, notes, checks)
