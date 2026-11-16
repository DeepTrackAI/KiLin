"""Partition LaTeX sources at sectioning commands.

A ``.tex`` file is split into the front matter (the document's text before
its first section: title, authors, abstract) and one unit per ``\\part``,
``\\chapter``, ``\\section``, ``\\subsection`` or ``\\subsubsection``. The
setup before ``\\begin{document}`` (``\\documentclass``, packages, macros)
and anything after ``\\end{document}`` are left out; a file without
``\\begin{document}``, such as a chapter included by ``\\input``, is all
document text. Unit text is the source as written, commands included.

Key Features
------------

- **Section boundaries**

    Sectioning commands start sections, including starred forms and
    optional short titles (``\\section*{...}``, ``\\section[short]{...}``),
    but not in comment lines or verbatim environments.

- **Readable section names**

    A section's name is its title with simple formatting commands
    removed (``\\emph{BRAPH}`` becomes ``BRAPH``).

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one LaTeX source file.

- ``latex_title``

    The document title from ``\\title{...}``, when it has one.

- ``LatexHeading`` and ``latex_headings``

    The document's text lines and its sectioning commands, with their
    level and number.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".tex"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "LatexHeading",
    "extract_source_units",
    "latex_headings",
    "latex_title",
]

# A sectioning command at the start of a line, up to its title's brace.
_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\(?:part|chapter|section|subsection|subsubsection)\*?"
    r"\s*(?:\[[^\]]*\])?\s*\{"
)
# The title command, up to its brace.
_TITLE: Final[re.Pattern[str]] = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{")
_BEGIN_DOCUMENT: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\begin\s*\{document\}"
)
_END_DOCUMENT: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\end\s*\{document\}"
)
# Environments whose text is shown as written, so commands in them are not
# headings.
_VERBATIM: Final[re.Pattern[str]] = re.compile(
    r"\\begin\s*\{(verbatim\*?|lstlisting|minted|comment|Verbatim)\}"
)
# A formatting command around a word ("\emph{word}"): the word is kept.
_WRAPPED: Final[re.Pattern[str]] = re.compile(
    r"\\[A-Za-z]+\*?\s*(?:\[[^\]]*\])?\{([^{}]*)\}"
)
# Commands left after unwrapping ("\LaTeX", "\label{...}" removed above).
_COMMAND: Final[re.Pattern[str]] = re.compile(r"\\[A-Za-z]+\*?")
# Commands whose argument is not part of a title.
_DROPPED: Final[re.Pattern[str]] = re.compile(
    r"\\(?:label|footnote|thanks|cite[a-z]*|ref|index)\*?\s*"
    r"(?:\[[^\]]*\])?\{[^{}]*\}"
)


def _braced(text: str, start: int) -> str | None:
    """Return the text inside balanced braces.

    Parameters
    ----------
    text: str
        Text that continues after an opening brace.
    start: int
        Position just after the opening brace.

    Returns
    -------
    str | None
        The text up to the matching closing brace, or ``None`` when the
        braces do not close in ``text``.

    """

    depth = 1
    for position in range(start, len(text)):
        character = text[position]
        if character == "\\":
            continue
        if character == "{" and text[position - 1] != "\\":
            depth += 1
        elif character == "}" and text[position - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start:position]
    return None


def _plain(title: str) -> str:
    """Return a LaTeX title as plain words.

    Parameters
    ----------
    title: str
        Title source, possibly with commands ("The \\emph{BRAPH} tool").

    Returns
    -------
    str
        The title without labels, citations or formatting commands, with
        ``~`` as a space and whitespace collapsed ("The BRAPH tool").

    """

    text = _DROPPED.sub("", title)
    previous = None
    while previous != text:
        previous, text = text, _WRAPPED.sub(r"\1", text)
    text = _COMMAND.sub("", text).replace("~", " ")
    text = text.replace("{", "").replace("}", "").replace("\\\\", " ")
    return " ".join(text.split())


def _document_lines(lines: list[str]) -> tuple[int, int]:
    """Return the first and last line of the document text.

    Parameters
    ----------
    lines: list[str]
        Source lines.

    Returns
    -------
    tuple[int, int]
        One-based inclusive line range: after ``\\begin{document}`` and
        before ``\\end{document}`` when present, otherwise the whole file.

    """

    first, last = 1, len(lines)
    for number, line in enumerate(lines, start=1):
        if _is_comment(line):
            continue
        if first == 1 and _BEGIN_DOCUMENT.match(line):
            first = number + 1
        elif _END_DOCUMENT.match(line):
            last = number - 1
            break
    return first, last


def _is_comment(line: str) -> bool:
    """Whether a line is a LaTeX comment.

    Parameters
    ----------
    line: str
        One source line.

    Returns
    -------
    bool
        True when its first visible character is ``%``.

    """

    return line.lstrip().startswith("%")


def _headings(
    lines: list[str], first: int, last: int
) -> list[tuple[int, str]]:
    """Find sectioning commands in the document text.

    Parameters
    ----------
    lines: list[str]
        Source lines.
    first: int
        First line of the document text, one-based.
    last: int
        Last line of the document text, one-based.

    Returns
    -------
    list[tuple[int, str]]
        One-based line numbers and plain titles, outside comment lines and
        verbatim environments. A title whose braces do not close on its
        line or the next two is read from those lines as far as it goes.

    """

    headings: list[tuple[int, str]] = []
    verbatim: str | None = None
    for number in range(first, last + 1):
        line = lines[number - 1]
        if verbatim is not None:
            if re.search(rf"\\end\s*\{{{re.escape(verbatim)}\}}", line):
                verbatim = None
            continue
        if _is_comment(line):
            continue
        opening = _VERBATIM.search(line)
        if opening is not None:
            rest = line[opening.end():]
            name = opening.group(1)
            if not re.search(rf"\\end\s*\{{{re.escape(name)}\}}", rest):
                verbatim = name
            continue
        match = _HEADING.match(line)
        if match is None:
            continue
        text = "".join(lines[number - 1:min(number + 2, last)])
        start = match.end()
        title = _braced(text, start)
        if title is None:
            title = text[start:].split("\n", 1)[0]
        headings.append((number, _plain(title)))
    return headings


# Sectioning commands by level; a part sits above chapters.
_LEVELS: Final[dict[str, int]] = {
    "part": -1, "chapter": 0, "section": 1, "subsection": 2,
    "subsubsection": 3,
}
_COMMAND_NAME: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\(part|chapter|section|subsection|subsubsection)(\*?)"
)
_APPENDIX: Final[re.Pattern[str]] = re.compile(r"^\s*\\appendix\b")
_SET_COUNTER: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\setcounter\s*\{"
    r"(chapter|section|subsection|subsubsection)\}\s*\{(-?\d+)\}"
)


def _appendix_number(number: int) -> str:
    """Return a standard appendix letter, including AA after Z.

    Parameters
    ----------
    number: int
        One-based appendix counter.

    Returns
    -------
    str
        Uppercase appendix letters.

    """

    parts = []
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        parts.append(chr(ord("A") + remainder))
    return "".join(reversed(parts))




@dataclass(frozen=True, slots=True)
class LatexHeading:
    """One sectioning command of a LaTeX document.

    Attributes
    ----------
    line: int
        Its one-based line.
    level: int
        -1 for ``\\part``, 0 ``\\chapter``, 1 ``\\section``, 2
        ``\\subsection``, 3 ``\\subsubsection``.
    title: str
        The title as plain words.
    number: str
        Number inferred from standard heading counters ("2.3"). Common
        ``\\appendix`` and ``\\setcounter`` commands are recognized;
        custom counter macros are not. Empty for starred headings or parts.

    """

    line: int
    level: int
    title: str
    number: str


def latex_headings(text: str) -> tuple[int, int, list[LatexHeading]]:
    """Return a LaTeX document's text lines and sectioning commands.

    Parameters
    ----------
    text: str
        LaTeX source.

    Returns
    -------
    tuple[int, int, list[LatexHeading]]
        The first and last line of the document text (one-based,
        inclusive: after ``\\begin{document}`` and before
        ``\\end{document}`` when present), and its headings in order,
        outside comments and verbatim environments. Numbers start at the
        highest level used (chapters when there are any, else sections),
        and a heading resets the counters below it. Appendix lettering
        and direct ``\\setcounter`` commands are recognized, but arbitrary
        macro redefinitions cannot be evaluated.

    """

    lines = text.splitlines(keepends=True)
    if not lines:
        return 1, 0, []
    first, last = _document_lines(lines)
    found = []
    for number, title in _headings(lines, first, last):
        match = _COMMAND_NAME.match(lines[number - 1])
        if match is not None:
            found.append(
                (number, _LEVELS[match.group(1)], title, not match.group(2))
            )
    top = min(
        (level for _, level, _, _ in found if level >= 0), default=1
    )
    counters: dict[int, int] = {}
    headings = []
    by_line = {line: (level, title, numbered)
               for line, level, title, numbered in found}
    appendix = False
    verbatim: str | None = None
    for line_number in range(first, last + 1):
        source = lines[line_number - 1]
        if verbatim is not None:
            if re.search(rf"\\end\s*\{{{re.escape(verbatim)}\}}", source):
                verbatim = None
            continue
        if _is_comment(source):
            continue
        opening = _VERBATIM.search(source)
        if opening is not None:
            name = opening.group(1)
            if not re.search(
                rf"\\end\s*\{{{re.escape(name)}\}}",
                source[opening.end():],
            ):
                verbatim = name
            continue
        if _APPENDIX.match(source):
            appendix = True
            counters.clear()
        reset = _SET_COUNTER.match(source)
        if reset is not None:
            level = _LEVELS[reset.group(1)]
            counters[level] = int(reset.group(2))
            for deeper in [key for key in counters if key > level]:
                del counters[deeper]
        heading = by_line.get(line_number)
        if heading is None:
            continue
        level, title, numbered = heading
        label = ""
        if numbered and level >= 0:
            counters[level] = counters.get(level, 0) + 1
            for deeper in [key for key in counters if key > level]:
                del counters[deeper]
            parts = [
                _appendix_number(counters.get(step, 0))
                if appendix and step == top
                else str(counters.get(step, 0))
                for step in range(top, level + 1)
            ]
            label = ".".join(parts)
        headings.append(LatexHeading(line_number, level, title, label))
    return first, last, headings


def latex_title(text: str) -> str | None:
    """Return the title of a LaTeX document.

    Parameters
    ----------
    text: str
        LaTeX source.

    Returns
    -------
    str | None
        The plain words of the first ``\\title{...}`` outside comments, or
        ``None`` when there is none or it is empty.

    """

    uncommented = "".join(
        re.sub(r"(?<!\\)%.*$", "", line)
        for line in text.splitlines(keepends=True)
    )
    match = _TITLE.search(uncommented)
    if match is None:
        return None
    title = _braced(uncommented, match.end())
    if title is None:
        return None
    return _plain(title) or None


def _latex_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify the front matter and the sections of a LaTeX document.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        ``latex_front`` (text before the first section) and
        ``latex_section`` units, in source order.

    """

    first, last = _document_lines(lines)
    headings = _headings(lines, first, last)

    units: list[SourceUnit] = []
    first_heading = headings[0][0] if headings else last + 1
    front = _source_unit(
        path, lines, first, first_heading - 1, kind="latex_front"
    )
    if front is not None:
        units.append(front)

    for index, (start_line, title) in enumerate(headings):
        end_line = (
            headings[index + 1][0] - 1
            if index + 1 < len(headings)
            else last
        )
        section = _source_unit(
            path,
            lines,
            start_line,
            end_line,
            kind="latex_section",
            section=title or None,
        )
        if section is not None:
            units.append(section)

    return units


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one LaTeX source file.

    Parameters
    ----------
    source_file: SourceFile
        Source file converted to text.

    Returns
    -------
    list[SourceUnit]
        Natural units in source order: the front matter, then one unit per
        sectioning command. The setup before ``\\begin{document}`` is not
        included.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []

    return _latex_units(source_file.path, lines)
