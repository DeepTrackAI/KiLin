"""Write a project report from every document's notes and checks.

Private to ``project_summary``. After the map step (``_mapping``) has read
each document, this module writes the report:

1. **Sections from all documents** (``notes``): one model call per
   section, from every document's note for it, citing documents as
   ``[n]``. When the notes do not fit in one call, they are first
   condensed in groups, keeping their citations.
2. **Chosen documents** (``{key_documents}``): the model picks document
   numbers; code writes their titles.
3. **Per-document sections**: assembled by code from the checks, one entry
   or table row per matching document, with how many documents match and
   which are unclear, so no model can drop or invent an entry.
4. **The documents covered** (``{documents}``) and the list closing the
   report, written by code.

Key Features
------------

- **Synthesis, not lists**

    Sections from all documents compare and group them, citing each.

- **Entries by code**

    Per-document sections and counts are never written by the model.

Module Structure
----------------

- ``note_groups``

    Split notes into groups that fit one call.

- ``condense_prompt``, ``section_prompt``, ``fix_prompt``,
  ``headline_prompt`` and ``key_documents_prompt``

    The model calls that write the report.

- ``chosen_numbers``

    The document numbers a reply chooses.

- ``per_document_markdown``, ``documents_line`` and ``documents_list``

    The parts written by code.

Examples
--------
>>> from kilin.tools.project_summary._reduce import chosen_numbers
>>> chosen_numbers("NUMBERS: 3, 7, 3, 99", 10, 2)
[3, 7]

"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Final

from kilin._model_completion import strip_thinking
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._shared.report_templates import (
    ReportTemplate,
    TemplateSection,
)
from kilin.tools.project_summary._documents import ProjectDocument
from kilin.tools.project_summary._mapping import (
    CHECK_UNCLEAR,
    CHECK_YES,
    DocumentNotes,
)

__all__ = [
    "chosen_numbers",
    "condense_prompt",
    "documents_line",
    "documents_list",
    "fix_prompt",
    "headline_prompt",
    "key_documents_prompt",
    "note_groups",
    "per_document_markdown",
    "section_prompt",
]

_SYSTEM: Final[str] = (
    "You write one section of a report on a collection of scientific "
    "documents, from notes on each document. Use only the notes. Cite the "
    "documents a statement rests on as [n] right after it, with the "
    "numbers the notes give. Compare and group the documents; do not "
    "describe them one by one."
)
_NUMBERS = re.compile(r"\d+")
# Unclear documents named in a per-document section's count line.
_UNCLEAR_NAMED: Final[int] = 10


def _length(section: TemplateSection) -> str:
    """Return how long a section should be, for a prompt.

    Parameters
    ----------
    section: TemplateSection
        The section.

    Returns
    -------
    str
        "about 50 words (40-60)", "one line only", or "a few concise
        sentences".

    """

    if section.inline:
        return "one line only"
    if section.words:
        low, high = section.words
        return f"about {(low + high) // 2} words ({low}-{high})"
    if section.items:
        low, high = section.items
        return f"{low}-{high} items, one per line starting with '- '"
    return "a few concise sentences"


def note_groups(
    notes: Sequence[tuple[int, str]], tokens: int
) -> list[list[tuple[int, str]]]:
    """Split notes into groups that each fit one model call.

    Parameters
    ----------
    notes: Sequence[tuple[int, str]]
        Each document's number and note for one section.
    tokens: int
        Most tokens of notes in one group.

    Returns
    -------
    list[list[tuple[int, str]]]
        The notes in order, in groups of at most ``tokens`` (a note larger
        than that is a group of its own); one group when all fit.

    """

    groups: list[list[tuple[int, str]]] = [[]]
    used = 0
    for number, note in notes:
        cost = estimate_tokens(note) + 4
        if groups[-1] and used + cost > tokens:
            groups.append([])
            used = 0
        groups[-1].append((number, note))
        used += cost
    return [group for group in groups if group]


def _notes_text(notes: Sequence[tuple[int, str]]) -> str:
    """Return notes as prompt lines, each with its document's number.

    Parameters
    ----------
    notes: Sequence[tuple[int, str]]
        Each document's number and note.

    Returns
    -------
    str
        "[3] note" lines.

    """

    return "\n".join(f"[{number}] {note}" for number, note in notes)


def condense_prompt(
    section: TemplateSection, notes: Sequence[tuple[int, str]]
) -> tuple[str, str]:
    """Return the prompts condensing one group of notes for a section.

    Parameters
    ----------
    section: TemplateSection
        The section the notes are for.
    notes: Sequence[tuple[int, str]]
        One group's notes.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the group's facts as short
        lines, each keeping the numbers of the documents it comes from.

    """

    user = (
        f"Notes on several documents for the report section "
        f"“{section.name}” ({section.guidance}):\n\n{_notes_text(notes)}"
        "\n\nCondense them into short lines of facts, grouping documents "
        "that say the same and keeping each line's document numbers, such "
        "as “U-Net models segment nuclei [3, 7].” Keep differences between "
        "documents. Add nothing."
    )
    return _SYSTEM, user


def section_prompt(
    template: ReportTemplate,
    section: TemplateSection,
    notes: str,
    documents: int,
) -> tuple[str, str]:
    """Return the prompts writing one section from all documents' notes.

    Parameters
    ----------
    template: ReportTemplate
        The project template.
    section: TemplateSection
        The section to write.
    notes: str
        The notes for it: "[n] note" lines, or condensed lines citing
        documents.
    documents: int
        How many documents the report covers.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the section's content only, of
        the section's length, citing documents as ``[n]``.

    """

    user = (
        f"Report: {template.title} on {documents} documents"
        + (f" — {template.description}" if template.description else "")
        + f"\n\nSection: {section.name}\nWhat it should say: "
        f"{section.guidance or section.name}\nLength: "
        f"{_length(section)}\n\nNotes:\n{notes or '(no document says '
        'anything about this)'}\n\nWrite only the section's content, "
        "without its heading. When the notes say nothing about it, say so "
        "in one sentence."
    )
    return _SYSTEM, user


def fix_prompt(
    section: TemplateSection, content: str, problem: str
) -> tuple[str, str]:
    """Return the prompts fixing one section's length or form.

    Parameters
    ----------
    section: TemplateSection
        The section.
    content: str
        Its content as written.
    problem: str
        What is wrong (``section_problem``).

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the content rewritten to fit,
        keeping its facts and citations.

    """

    user = (
        f"This report section has a problem: {problem}\nLength wanted: "
        f"{_length(section)}\n\nSection “{section.name}”:\n{content}\n\n"
        "Rewrite it to fit, keeping its facts and its [n] citations and "
        "adding nothing. Write only the content."
    )
    return _SYSTEM, user


def headline_prompt(
    template: ReportTemplate, sections: Sequence[tuple[str, str]]
) -> tuple[str, str]:
    """Return the prompts writing a project report's headline.

    Parameters
    ----------
    template: ReportTemplate
        The template, with its headline limit and instruction.
    sections: Sequence[tuple[str, str]]
        The written sections' names and content.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for one line of at most
        ``template.headline_characters`` characters.

    """

    written = "\n\n".join(
        f"{name}:\n{content[:600]}" for name, content in sections
    )
    user = (
        f"Write the headline of this report ({template.title}): "
        f"{template.headline_guidance or 'what the documents show'}. At "
        f"most {template.headline_characters} characters, one line, no "
        f"citations, no full stop.\n\n{written}"
    )
    return _SYSTEM, user


def key_documents_prompt(
    section: TemplateSection,
    documents: Sequence[ProjectDocument],
    notes: dict[int, str],
) -> tuple[str, str]:
    """Return the prompts choosing documents for a section.

    Parameters
    ----------
    section: TemplateSection
        A ``{key_documents}`` section, with its item limit.
    documents: Sequence[ProjectDocument]
        The documents.
    notes: dict[int, str]
        A line on each document (its notes), by number.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for ``NUMBERS:`` and the chosen
        documents' numbers, without writing their titles.

    """

    low, high = section.items or (1, 3)
    listed = "\n".join(
        f"[{d.number}] {d.title}: {notes.get(d.number, '')[:300]}"
        for d in documents
    )
    user = (
        f"Choose {low}-{high} documents for the section “{section.name}”: "
        f"{section.guidance}\n\nDocuments:\n{listed}\n\nAnswer with one "
        "line: NUMBERS: <the chosen numbers, separated by commas>"
    )
    return _SYSTEM, user


def chosen_numbers(reply: str, documents: int, most: int) -> list[int]:
    """Return the document numbers a reply chooses.

    Parameters
    ----------
    reply: str
        The model's reply to ``key_documents_prompt``.
    documents: int
        How many documents there are.
    most: int
        Most numbers kept.

    Returns
    -------
    list[int]
        The numbers on the ``NUMBERS:`` line (or the whole reply), in
        order, each once, between 1 and ``documents``.

    """

    text = strip_thinking(reply)
    line = next(
        (line for line in text.splitlines() if "NUMBERS" in line.upper()),
        text,
    )
    chosen: list[int] = []
    for found in _NUMBERS.findall(line):
        number = int(found)
        if 1 <= number <= documents and number not in chosen:
            chosen.append(number)
    return chosen[:most]


def per_document_markdown(
    section: TemplateSection,
    documents: Sequence[ProjectDocument],
    notes: dict[int, DocumentNotes],
) -> str:
    """Return a per-document section, assembled from the checks.

    Parameters
    ----------
    section: TemplateSection
        A per-document section.
    documents: Sequence[ProjectDocument]
        The documents, in order.
    notes: dict[int, DocumentNotes]
        Each read document's notes and checks, by number.

    Returns
    -------
    str
        A count line ("3 of 12 documents."), with the unclear ones and why
        ("Unclear: [4] (the quoted words are not in the document).") and
        those not read; then one "- **Title**: entry [n]" line per matching
        document, or for a table a Markdown table with a Document column.
        "None of the documents" when none match.

    """

    matching: list[tuple[ProjectDocument, object]] = []
    unclear: list[str] = []
    unread: list[int] = []
    for document in documents:
        found = notes.get(document.number)
        if found is None or found.failure:
            unread.append(document.number)
            continue
        check = found.checks.get(section.name)
        if check is None:
            continue
        if check.answer == CHECK_YES:
            matching.append((document, check))
        elif check.answer == CHECK_UNCLEAR:
            unclear.append(
                f"[{document.number}]"
                + (f" ({check.why})" if check.why else "")
            )
    count = (
        f"{len(matching)} of {len(documents)} documents."
        if matching else
        f"None of the {len(documents)} documents."
    )
    if unclear:
        named = unclear[:_UNCLEAR_NAMED]
        more = len(unclear) - len(named)
        count += " Unclear: " + ", ".join(named) + (
            f" and {more} more" if more else ""
        ) + "."
    if unread:
        count += " Not read: " + ", ".join(f"[{n}]" for n in unread) + "."
    lines = [count]
    if not matching:
        return "\n".join(lines)
    lines.append("")
    if section.columns:
        def cell(text: str) -> str:
            """Return a table cell's text.

            Parameters
            ----------
            text: str
                A value.

            Returns
            -------
            str
                On one line, with ``|`` escaped.

            """

            return " ".join(text.split()).replace("|", "\\|")

        lines.append("| Document | " + " | ".join(section.columns) + " |")
        lines.append("| --- |" + " --- |" * len(section.columns))
        for document, check in matching:
            lines.append(
                f"| {cell(document.title)} [{document.number}] | "
                + " | ".join(cell(value) for value in check.cells) + " |"
            )
    else:
        for document, check in matching:
            lines.append(
                f"- **{document.title}**: {check.entry} [{document.number}]"
            )
    return "\n".join(lines)


def documents_line(documents: Sequence[ProjectDocument]) -> str:
    """Return the content of a ``{documents}`` line.

    Parameters
    ----------
    documents: Sequence[ProjectDocument]
        The documents covered.

    Returns
    -------
    str
        "12 documents, listed at the end of the report [1–12]."

    """

    count = len(documents)
    return (
        f"{count} document{'s' if count != 1 else ''}, listed at the end of "
        f"the report [1{f'–{count}' if count > 1 else ''}]."
    )


def documents_list(documents: Sequence[ProjectDocument]) -> str:
    """Return the numbered list of documents closing a project report.

    Parameters
    ----------
    documents: Sequence[ProjectDocument]
        The documents covered.

    Returns
    -------
    str
        "**Documents**" and one "[n] Title (`path`)" line per document, so
        each ``[n]`` of the report can be traced.

    """

    lines = ["**Documents**", ""]
    for document in documents:
        path = document.path.as_posix()
        lines.append(
            f"[{document.number}] {document.title}"
            + (f" (`{path}`)" if document.title != document.path.name
               else "")
            + "  "
        )
    return "\n".join(lines).rstrip()
