"""Read a document into sections and paragraphs of prose.

Private to ``writing_check``. The checker works one paragraph at a time
and reports each finding with its section and line, so a document is read
from its source file into:

1. **Sections**: each heading with its level, its number ("2.3", from
   LaTeX's counters or a numbered Markdown title) and its line range,
   subsections included. A LaTeX ``abstract`` environment is a section.
2. **Paragraphs**: runs of prose lines, with their source text, their
   readable text (``_shared.latex_text`` for LaTeX; Markdown links and
   emphasis reduced to words), their line range, and their position in
   their section.

Lines that are not prose are left out: LaTeX setup, math and float
environments, verbatim and code, Markdown code fences and tables. A
paragraph ends at a blank line, a heading, a list item, or a line that
prints nothing. Plain text has paragraphs and no sections.

Key Features
------------

- **Source lines kept**

    Every paragraph keeps the lines it came from, so findings point at
    the file.

- **Section numbers as printed**

    "Section 2.3" means what the reader sees in the compiled document.

Module Structure
----------------

- ``Section``, ``Paragraph`` and ``Document``

    The parts of a document.

- ``read_document``

    Read a ``.tex``, ``.md`` or ``.txt`` source into sections and
    paragraphs.

Examples
--------
>>> from pathlib import Path
>>> from kilin.tools.writing_check._document import read_document
>>> document = read_document(Path("notes.md"), "# Results\\n\\nIt works.\\n")
>>> [(p.section, p.text) for p in document.paragraphs]
[(0, 'It works.')]

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from kilin.sources.extensions.sources_md import markdown_headings
from kilin.sources.extensions.sources_tex import latex_headings
from kilin.tools._shared.latex_text import (
    LATEX_COMMENT,
    NOT_CITING,
    readable_latex,
)

__all__ = [
    "Document",
    "Paragraph",
    "Section",
    "read_document",
]

# File types read as LaTeX, Markdown, or plain text.
LATEX_SUFFIXES: Final[frozenset[str]] = frozenset({".tex"})
MARKDOWN_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".markdown"})
TEXT_SUFFIXES: Final[frozenset[str]] = frozenset({".txt"})

# LaTeX environments whose content is not prose to check.
_SKIPPED: Final[frozenset[str]] = frozenset({
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "displaymath",
    "math", "figure", "figure*", "table", "table*", "tabular", "tabular*",
    "tabularx", "longtable", "verbatim", "verbatim*", "Verbatim",
    "lstlisting", "minted", "comment", "thebibliography", "tikzpicture",
    "algorithm", "algorithmic", "wrapfigure", "subfigure",
})
_BEGIN: Final[re.Pattern[str]] = re.compile(r"\\begin\s*\{([^{}]+)\}")
_END: Final[re.Pattern[str]] = re.compile(r"\\end\s*\{([^{}]+)\}")
_DISPLAY_OPEN: Final[re.Pattern[str]] = re.compile(r"^\s*(\\\[|\$\$)")
_DISPLAY_CLOSE: Final[re.Pattern[str]] = re.compile(r"(\\\]|\$\$)\s*$")
_ITEM: Final[re.Pattern[str]] = re.compile(r"^\s*\\item\b")
_SECTIONING: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\(?:part|chapter|section|subsection|subsubsection)(?![A-Za-z])"
)
_ABSTRACT_BEGIN: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\begin\s*\{abstract\}"
)
_ABSTRACT_END: Final[re.Pattern[str]] = re.compile(
    r"\\end\s*\{abstract\}"
)
# Markdown lines that are not prose, and list items.
_FENCE: Final[re.Pattern[str]] = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_TABLE_ROW: Final[re.Pattern[str]] = re.compile(r"^\s*\|")
_MARKDOWN_ITEM: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+"
)
_LINK: Final[re.Pattern[str]] = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPHASIS: Final[re.Pattern[str]] = re.compile(r"(\*{1,3}|_{1,3}|`)")
# A heading title that starts with its number ("2.3 Data").
_NUMBERED_TITLE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+"
)


@dataclass(frozen=True, slots=True)
class Section:
    """One heading of a document and the lines under it.

    Attributes
    ----------
    title: str
        The heading as plain words.
    number: str
        The number the reader sees ("2.3"), or ``""``.
    level: int
        Nesting level; smaller is higher (a LaTeX ``\\section`` is 1, a
        Markdown ``##`` is 2).
    start_line: int
        The heading's line, one-based.
    end_line: int
        The last line before the next heading of the same or a higher
        level: subsections are inside their section.

    """

    title: str
    number: str
    level: int
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class Paragraph:
    """One paragraph of prose.

    Attributes
    ----------
    section: int | None
        Index of the innermost section holding it, or None before the
        first heading.
    position: int
        Its position among the paragraphs of that section (subsections
        not counted), from one.
    start_line: int
        First source line, one-based.
    end_line: int
        Last source line.
    source: str
        The source lines as written.
    text: str
        Its readable text on one line: LaTeX reduced to what it prints,
        citations as "[cite]", Markdown links and emphasis as words.

    """

    section: int | None
    position: int
    start_line: int
    end_line: int
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class Document:
    """A document read into sections and paragraphs.

    Attributes
    ----------
    path: pathlib.Path
        The document's path, as indexed.
    sections: tuple[Section, ...]
        Its headings, in order.
    paragraphs: tuple[Paragraph, ...]
        Its paragraphs of prose, in order.

    """

    path: Path
    sections: tuple[Section, ...]
    paragraphs: tuple[Paragraph, ...]


def read_document(path: Path, text: str) -> Document:
    """Read a source file into sections and paragraphs.

    Parameters
    ----------
    path: pathlib.Path
        The document's path; its suffix chooses how it is read.
    text: str
        Its content.

    Returns
    -------
    Document
        Sections and paragraphs. A suffix that is not LaTeX or Markdown
        is read as plain text: paragraphs separated by blank lines, no
        sections.

    """

    lines = text.splitlines()
    suffix = path.suffix.lower()
    if suffix in LATEX_SUFFIXES:
        sections, prose = _latex(text, lines)
    elif suffix in MARKDOWN_SUFFIXES:
        sections, prose = _markdown(lines)
    else:
        sections, prose = [], _plain_lines(lines)
    return Document(path, tuple(sections), _paragraphs(prose, sections, lines))


def _latex(
    text: str, lines: list[str]
) -> tuple[list[Section], dict[int, str]]:
    """Return a LaTeX document's sections and readable prose lines.

    Parameters
    ----------
    text: str
        LaTeX source.
    lines: list[str]
        Its lines.

    Returns
    -------
    tuple[list[Section], dict[int, str]]
        Sections (the abstract environment included), and the readable
        text of each prose line by line number. Lines that are not prose
        are absent; a line that breaks a paragraph without being prose
        (a heading, ``\\item``, a command printing nothing) maps to ``""``.

    """

    first, last, headings = latex_headings(text)
    marks = [
        (heading.line, heading.level, heading.title, heading.number)
        for heading in headings
    ]
    prose: dict[int, str] = {}
    skipped: str | None = None
    display = False
    for number in range(first, last + 1):
        line = lines[number - 1]
        code = LATEX_COMMENT.sub("", line)
        if skipped is not None:
            if any(name == skipped for name in _END.findall(code)):
                skipped = None
            prose[number] = ""
            continue
        if display:
            if _DISPLAY_CLOSE.search(code):
                display = False
            prose[number] = ""
            continue
        opened = [name for name in _BEGIN.findall(code) if name in _SKIPPED]
        if opened:
            if opened[-1] not in _END.findall(code):
                skipped = opened[-1]
            prose[number] = ""
            continue
        if _DISPLAY_OPEN.match(code) and not _DISPLAY_CLOSE.search(
            code[_DISPLAY_OPEN.match(code).end():]
        ):
            display = True
            prose[number] = ""
            continue
        if _ABSTRACT_BEGIN.match(code):
            marks.append((number, 1, "Abstract", ""))
        if not code.strip():
            if line.strip():
                continue  # A comment line does not break a paragraph.
            prose[number] = ""
            continue
        if _SECTIONING.match(code) or _ITEM.match(code):
            readable = "" if _SECTIONING.match(code) else _latex_line(code)
            prose[number] = ""
            if readable.strip():
                prose[number] = "\0" + readable.strip()
            continue
        prose[number] = _latex_line(code)
    marks.sort()
    return _sections(marks, last), prose


def _latex_line(line: str) -> str:
    """Return one LaTeX line as readable text, citations as "[cite]".

    Parameters
    ----------
    line: str
        One line.

    Returns
    -------
    str
        Its readable text (``readable_latex``); ``\\citeauthor`` and
        similar commands as "[author]" or "[year]", ``\\nocite`` removed.

    """

    def cite(match: re.Match[str]) -> str:
        """Return the placeholder of one citation command.

        Parameters
        ----------
        match: re.Match[str]
            The citation command.

        Returns
        -------
        str
            The text the paragraph shows in its place.

        """

        name = match["name"].lower()
        if name == "nocite":
            return ""
        if name in NOT_CITING:
            return "[author]" if "author" in name else "[year]"
        return "[cite]"

    return readable_latex(line, cite)


def _markdown(lines: list[str]) -> tuple[list[Section], dict[int, str]]:
    """Return Markdown sections and readable prose lines.

    Parameters
    ----------
    lines: list[str]
        Markdown lines.

    Returns
    -------
    tuple[list[Section], dict[int, str]]
        Sections from the ATX headings (numbered when the title starts
        with a number), and the readable text of each prose line; code
        fences and table rows are not prose.

    """

    headings = markdown_headings(lines)
    marks = []
    for line, level, title in headings:
        numbered = _NUMBERED_TITLE.match(title)
        marks.append((
            line, level,
            title[numbered.end():] if numbered else title,
            numbered.group(1) if numbered else "",
        ))
    heading_lines = {line for line, _, _ in headings}
    prose: dict[int, str] = {}
    fence: str | None = None
    for number, line in enumerate(lines, start=1):
        opening = _FENCE.match(line)
        if fence is not None:
            if opening and opening.group(1)[0] == fence[0]:
                fence = None
            prose[number] = ""
            continue
        if opening:
            fence = opening.group(1)
            prose[number] = ""
            continue
        if number in heading_lines or _TABLE_ROW.match(line):
            prose[number] = ""
            continue
        text = line.lstrip("> ").rstrip()
        item = _MARKDOWN_ITEM.match(text)
        if item:
            text = "\0" + text[item.end():]
        text = _LINK.sub(r"\1", text)
        prose[number] = _EMPHASIS.sub("", text)
    return _sections(marks, len(lines)), prose


def _plain_lines(lines: list[str]) -> dict[int, str]:
    """Return plain-text lines as prose.

    Parameters
    ----------
    lines: list[str]
        Text lines.

    Returns
    -------
    dict[int, str]
        Each line by number, stripped of trailing space.

    """

    return {
        number: line.rstrip() for number, line in enumerate(lines, start=1)
    }


def _sections(
    marks: list[tuple[int, int, str, str]], last: int
) -> list[Section]:
    """Return sections from headings, each up to the next one at its level.

    Parameters
    ----------
    marks: list[tuple[int, int, str, str]]
        Line, level, title and number of each heading, in line order.
    last: int
        The document's last line.

    Returns
    -------
    list[Section]
        One section per heading; each ends before the next heading of the
        same or a higher level, or at ``last``.

    """

    sections = []
    for index, (line, level, title, number) in enumerate(marks):
        end = last
        for later, later_level, _, _ in marks[index + 1:]:
            if later_level <= level:
                end = later - 1
                break
        sections.append(Section(title, number, level, line, end))
    return sections


def _paragraphs(
    prose: dict[int, str], sections: list[Section], lines: list[str]
) -> tuple[Paragraph, ...]:
    """Group prose lines into paragraphs.

    Parameters
    ----------
    prose: dict[int, str]
        Readable text by line number: ``""`` breaks a paragraph, a text
        starting with ``"\\0"`` starts a new one (a list item), and an
        absent line is ignored.
    sections: list[Section]
        The document's sections.
    lines: list[str]
        Source lines.

    Returns
    -------
    tuple[Paragraph, ...]
        Paragraphs in order, each with its innermost section and its
        position among that section's paragraphs.

    """

    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for number in sorted(prose):
        text = prose[number]
        if text.startswith("\0"):
            if current:
                runs.append(current)
            current = [(number, text[1:])]
        elif text.strip():
            current.append((number, text))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    paragraphs = []
    counts: dict[int | None, int] = {}
    for run in runs:
        start, end = run[0][0], run[-1][0]
        section = _innermost(sections, start)
        counts[section] = counts.get(section, 0) + 1
        paragraphs.append(Paragraph(
            section,
            counts[section],
            start,
            end,
            "\n".join(lines[start - 1:end]),
            " ".join(" ".join(text.split()) for _, text in run),
        ))
    return tuple(paragraphs)


def _innermost(sections: list[Section], line: int) -> int | None:
    """Return the index of the innermost section holding a line.

    Parameters
    ----------
    sections: list[Section]
        The document's sections.
    line: int
        A line number.

    Returns
    -------
    int | None
        The section that starts last before the line and still holds it;
        None before the first heading.

    """

    found = None
    for index, section in enumerate(sections):
        if section.start_line <= line <= section.end_line:
            found = index
    return found
