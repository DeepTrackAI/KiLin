"""Find a paper's reference list and split it into entries, as printed.

Several registered tools need a paper's own reference entries: the report
tool copies some into a briefing, and a citation checker matches them
against other files. This private implementation keeps the layout rules
in one place, so tools do not import one another to reuse them.

The rules are deterministic and model-free. They read the text after the
paper's last "References" heading and split it at entry numbers, or (for
two-column PDFs, which print the numbers apart from the entries) at the
ends of entries; numbers printed alone are given back to their entries.
Indexed passages overlap, so their shared lines are read once. Extracted
PDF text can interleave the two columns of a reference list, so when the
caller passes the PDF, its pages are read again in column order. Every
entry is the paper's own text; an unusual layout can still split or join
entries wrongly.

The numeric thresholds are not owned here: each caller passes a
``ReferenceLayout`` built from its own validated ``defaults.json``.

Key Features
------------

- **Nothing invented**

    Entries are copied from the paper, with lines joined and words
    hyphenated across lines rejoined.

- **Layout-aware splitting**

    Numbered lists split at their numbers; others split after a line that
    ends an entry (a year, volume and pages, a PMID, or a DOI).

- **Page furniture removed**

    Page numbers, lone entry numbers, and running heads and footers are
    dropped.

Module Structure
----------------

- ``ReferenceLayout``

    The caller's thresholds for finding and splitting entries.

- ``ReferenceEntry``

    One entry, with its printed (or inferred) reference number.

- ``reference_body``

    Return the paper's passages before its reference list.

- ``ReferenceList`` and ``reference_lists``

    Every reference list of a paper (a main and a supplementary one),
    with its heading and entries.

- ``reference_list``

    Return one of the paper's reference lists, found by its heading.

- ``reference_entries`` and ``numbered_reference_entries``

    Split the paper's reference list into entries, as printed, with or
    without their reference numbers.

- ``joined_text``

    Join passages without repeating the lines they overlap on.

- ``pdf_text``

    Read a PDF's pages in column order.

Examples
--------
>>> from kilin.tools._shared.references import (
...     ReferenceLayout,
...     reference_entries,
... )
>>> layout = ReferenceLayout(
...     max_characters=40000,
...     numbered_entries_min=5,
...     running_line_repeats=3,
...     running_line_words=12,
...     entry_min_words=5,
... )
>>> reference_entries((), layout)
[]

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import re
from typing import Final

from kilin.chunking import Chunk

__all__ = [
    "ReferenceEntry",
    "ReferenceLayout",
    "ReferenceList",
    "joined_text",
    "numbered_reference_entries",
    "pdf_text",
    "reference_body",
    "reference_entries",
    "reference_list",
    "reference_lists",
]

# The heading that starts a reference list, also for a supplementary or
# methods list ("Supplementary References", "Methods references",
# "References and Notes"). A paper can have several; the caller picks one.
_REFERENCES_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^\s*#*\s*(?:(?:supplementary|supplemental|additional|online|si|"
    r"methods?)\s+)?(references(?:\s+and\s+notes)?|notes\s+and\s+references"
    r"|bibliography|literature cited|works cited)\s*:?\s*$",
    flags=re.I | re.M,
)
# A numbered reference entry: "12. Smith, J. ...", "[12] Smith ...".
_ENTRY_START: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+(\S.*)$"
)
# The end of an entry: "(2019).", "2019.", "2023;24(9):557–74.", a year
# followed by pages ("1995; 289–300.", "2016, 160422."), a PMID, or a DOI.
# Two-column PDFs print the numbers apart from the entries, so ends
# separate entries more reliably than numbers.
_ENTRY_END: Final[re.Pattern[str]] = re.compile(
    r"(?:\((?:19|20)\d\d[a-z]?\)\.?"
    r"|\b(?:19|20)\d\d[a-z]?\."
    r"|\d+(?:\(\d+\))?:\s*[\w–-]+\."
    r"|\b(?:19|20)\d\d[a-z]?[;,]\s*[\w–-]+\."
    r"|PMID:\s*\d+"
    r"|doi\.org/\S+[0-9A-Za-z])\s*$"
)
# A number printed alone ("2.", "[2]") in a column of its own.
_LONE_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s*$"
)
# Share of a page's text lines starting in its right half that makes it a
# two-column page.
_RIGHT_COLUMN_SHARE: Final[float] = 0.25
# Running heads and footers: "PLOS ONE | https://...", "12 / 25".
_FURNITURE: Final[re.Pattern[str]] = re.compile(
    r"\|\s*https?://|\b\d+\s*/\s*\d+\s*$"
)
# Headings after the reference list, where it ends: alone on the line, or
# followed by text ("Acknowledgements This work ..."). Not "Methods", which
# starts journal names ("Nat\nMethods 18, ...").
_AFTER_REFERENCES: Final[re.Pattern[str]] = re.compile(
    r"^\s*#*\s*(acknowledge?ments?|author contributions|funding|competing "
    r"interests|additional information|supplementary information|data "
    r"availability|code availability|appendix|"
    r"(?:supplementary|supplemental)\s+"
    r"(?:methods?|results?|discussion))\s*(?::|$|\s+[A-Z][a-z])",
    flags=re.I,
)
# A table or figure caption after the list ("Table 1. ...", "Fig. 2 |",
# "Supplementary Figure S3:"), where the list ends: its rows and legend
# are not references.
_CAPTION: Final[re.Pattern[str]] = re.compile(
    r"^\s*#*\s*(?:supplementary\s+|extended\s+data\s+)?"
    r"(?:table|fig\.?|figure|box)\s*S?\d+[a-z]?\s*(?:[.:|]|$|\s+[A-Z])",
    flags=re.I,
)
# Most a printed entry number may skip ahead of the previous one (a
# missing or merged entry); a larger jump, or a number going back, is a
# number inside an entry ("…919–\n927. https://doi…"), not a new entry.
_NUMBER_STEP_MAX: Final[int] = 3
# A line ending inside a URL or DOI ("https://doi.org/10.1016/j.x.2013."),
# which continues on the next line even when it looks like an entry end.
_BROKEN_URL: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://|doi\.\s*org/|\b10\.\d{4,9}/)\S*[./_-]\s*$"
)
# The start of an author-year entry: "Smith, J. A., ... (2020)", "Smith,
# J. and Lee, B. (2020a)"; in an unnumbered list it starts a new entry once
# the entry before has its year.
_AUTHOR_YEAR_HEAD: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?i:van|von|der|den|de|del|di|da|du|la|le)\s+)*"
    r"[^\W\d_a-zß-ÿ][\w'’\-]*(?:\s+[^\W\d_a-zß-ÿ][\w'’\-]*)?,\s+"
    r"[A-Z][a-z]?\.[^()]{0,200}\((?:19|20)\d\d[a-z]?\)"
)
_YEAR_IN_PARENTHESES: Final[re.Pattern[str]] = re.compile(
    r"\((?:19|20)\d\d[a-z]?\)"
)
# A line that always continues the entry before it (its DOI or PMID).
_CONTINUATION: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:https?://|doi\b|pmid\b|pmcid\b)", re.I
)
# Page numbers between entries ("12", "Page 12") are skipped.
_PAGE_LINE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}\s*$", flags=re.I
)


@dataclass(frozen=True, slots=True)
class ReferenceLayout:
    """The caller's thresholds for finding and splitting entries.

    Each tool builds one from its own validated ``defaults.json``; this
    module has no defaults of its own.

    Attributes
    ----------
    max_characters: int
        Most characters read after the references heading; longer lists
        are cut.
    numbered_entries_min: int
        Fewest numbered lines ("12. ...", "[12] ...") for entries to be
        split at their numbers.
    running_line_repeats: int
        Repetitions in the paper that make a short line a running head or
        footer ("J. Smith et al.", "Journal Name 35 (2026) 102791").
    running_line_words: int
        Most words of a line that can be a running head or footer.
    entry_min_words: int
        Fewest words of an entry; shorter ones are page furniture.

    """

    max_characters: int
    numbered_entries_min: int
    running_line_repeats: int
    running_line_words: int
    entry_min_words: int


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    """One entry of a paper's reference list.

    Attributes
    ----------
    number: int
        The entry's reference number: as printed, or (when ``printed`` is
        False) the number after the previous entry's.
    text: str
        The entry as printed, without its number.
    printed: bool
        Whether the paper prints the number with this entry.

    """

    number: int
    text: str
    printed: bool


@dataclass(frozen=True, slots=True)
class ReferenceList:
    """One reference list of a paper, as found by its heading.

    Attributes
    ----------
    index: int
        Its position among the paper's lists, from zero.
    heading: str
        The heading as printed ("References", "Supplementary References").
    entries: tuple[ReferenceEntry, ...]
        Its entries (``numbered_reference_entries``).

    """

    index: int
    heading: str
    entries: tuple[ReferenceEntry, ...]

    @property
    def label(self) -> str:
        """Return a short description for asking the user.

        Returns
        -------
        str
            The heading, the entry count and the number range, such as
            "Supplementary References (12 entries, 1–12)".

        """

        if not self.entries:
            return f"{self.heading} (no entries)"
        first, last = self.entries[0].number, self.entries[-1].number
        count = len(self.entries)
        return (
            f"{self.heading} ({count} "
            f"{'entry' if count == 1 else 'entries'}, {first}–{last})"
        )


def reference_body(
    passages: Sequence[Chunk], which: int = -1
) -> tuple[Chunk, ...]:
    """Return the paper's passages that cite one of its reference lists.

    In-text citations are looked for here, so the reference list's own
    numbers are not mistaken for citations.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    which: int, optional
        The reference list, as an index into the paper's lists (``-1``,
        the default, is the last).

    Returns
    -------
    tuple[Chunk, ...]
        The passages before that list's heading ("References",
        "Supplementary References", ...); the passage holding the heading
        is cut just before it. For a later list, only the text after the
        end of the list before it (a supplement citing its own list).
        Every passage when there is no heading. Text after the list
        (figure legends printed at the end, for example) is not included.

    """

    headings = [
        (index, match)
        for index, passage in enumerate(passages)
        for match in _REFERENCES_HEADING.finditer(passage.text)
    ]
    if not headings:
        return tuple(passages)
    try:
        index, match = headings[which]
    except IndexError:
        index, match = headings[-1]
    position = headings.index((index, match))

    body = _cut_before(passages, index, match.start())
    if position == 0:
        return body

    # A later list's citing text starts where the list before it ends.
    previous_index, previous = headings[position - 1]
    stop = _list_end(passages, previous_index, previous.end())
    if stop is None:
        return body[previous_index:] if previous_index < len(body) else ()
    stop_index, offset = stop
    kept = list(body[stop_index:])
    if kept:
        first = kept[0]
        head = first.text[:offset]
        kept[0] = replace(
            first,
            text=first.text[offset:],
            start_line=first.start_line + head.count("\n"),
        )
    return tuple(passage for passage in kept if passage.text.strip())


def _cut_before(
    passages: Sequence[Chunk], index: int, offset: int
) -> tuple[Chunk, ...]:
    """Return the passages up to a position, the last one cut there.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    index: int
        The passage holding the position.
    offset: int
        Where to cut in its text.

    Returns
    -------
    tuple[Chunk, ...]
        The passages before ``index``, and the start of passage ``index``
        when it has text before ``offset``.

    """

    kept = passages[index].text[:offset]
    body = tuple(passages[:index])
    if kept.strip():
        body += (
            replace(
                passages[index],
                text=kept,
                end_line=passages[index].start_line + kept.count("\n"),
            ),
        )
    return body


def _list_end(
    passages: Sequence[Chunk], index: int, offset: int
) -> tuple[int, int] | None:
    """Find where a reference list that starts at a position ends.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    index: int
        The passage holding the list's heading.
    offset: int
        The end of the heading in its text.

    Returns
    -------
    tuple[int, int] | None
        The passage and offset of the first line that ends the list
        (``_ends_list``); None when no line does.

    """

    for number in range(index, len(passages)):
        text = passages[number].text
        start = offset if number == index else 0
        position = start
        for line in text[start:].splitlines(keepends=True):
            if _ends_list(line):
                return number, position
            position += len(line)
    return None


def _ends_list(line: str) -> bool:
    """Return whether a line ends a reference list.

    Parameters
    ----------
    line: str
        A line after the list's heading.

    Returns
    -------
    bool
        True for a following heading ("Acknowledgements", "Supplementary
        information"), a table or figure caption, or another reference
        list's heading.

    """

    return bool(
        _AFTER_REFERENCES.match(line)
        or _CAPTION.match(line)
        or _REFERENCES_HEADING.match(line)
    )


def reference_list(
    passages: Sequence[Chunk],
    layout: ReferenceLayout,
    file: Path | None = None,
    which: int = -1,
) -> str:
    """Return one of the paper's reference lists, found by its heading.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    layout: ReferenceLayout
        The caller's thresholds; ``max_characters`` limits the list.
    file: pathlib.Path | None, optional
        The paper on disk. For a PDF, the list is read again from the
        page layout in column order (``pdf_text``), because extracted text
        can interleave the columns of a two-column list. Defaults to
        ``None``: the passages' text.
    which: int, optional
        The list, as an index into the paper's lists (``-1``, the
        default, is the last).

    Returns
    -------
    str
        Text after that list's heading ("References", "Supplementary
        References", "Bibliography", "Literature cited", "Works cited"), up
        to the next list's heading and ``layout.max_characters`` at most;
        empty when there is none. Lines that overlapping passages repeat
        are included once. The passages are used when the PDF cannot be
        read or has no such heading.

    """

    text = _source_text(passages, file)
    headings = list(_REFERENCES_HEADING.finditer(text))
    if not headings:
        return ""
    try:
        heading = headings[which]
    except IndexError:
        heading = headings[-1]
    position = headings.index(heading)

    start = heading.end()
    stop = (
        headings[position + 1].start()
        if position + 1 < len(headings) else len(text)
    )
    stop = min(stop, start + layout.max_characters)
    return text[start:stop].strip()


def reference_lists(
    passages: Sequence[Chunk],
    layout: ReferenceLayout,
    file: Path | None = None,
    *,
    accept_short: Callable[[str], bool] | None = None,
) -> list[ReferenceList]:
    """Return every reference list of the paper that has entries.

    A manuscript can print a main and a supplementary list, each numbered
    from 1; a citation number then means a different entry in each, so
    the caller checks one list and, when there are several, asks which.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    layout: ReferenceLayout
        The caller's thresholds.
    file: pathlib.Path | None, optional
        The paper on disk, for a PDF's column order. Defaults to ``None``.
    accept_short: Callable[[str], bool] | None, optional
        As for ``numbered_reference_entries``.

    Returns
    -------
    list[ReferenceList]
        In the paper's order; lists without entries (a heading in a table
        of contents, for example) are left out, but ``index`` still counts
        them, so it can be passed as ``which``.

    """

    text = _source_text(passages, file)
    found: list[ReferenceList] = []
    for index, heading in enumerate(_REFERENCES_HEADING.finditer(text)):
        entries = numbered_reference_entries(
            passages, layout, file, accept_short=accept_short, which=index
        )
        if entries:
            found.append(
                ReferenceList(
                    index, " ".join(heading.group(0).split()).strip("#: "),
                    tuple(entries),
                )
            )
    return found


def reference_entries(
    passages: Sequence[Chunk],
    layout: ReferenceLayout,
    file: Path | None = None,
) -> list[str]:
    """Split the paper's reference list into entries, as printed.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    layout: ReferenceLayout
        The caller's thresholds for splitting and filtering entries.
    file: pathlib.Path | None, optional
        The paper on disk, for a PDF's column order (``reference_list``).
        Defaults to ``None``.

    Returns
    -------
    list[str]
        The texts of ``numbered_reference_entries``, in order.

    """

    return [
        entry.text
        for entry in numbered_reference_entries(passages, layout, file)
    ]


def numbered_reference_entries(
    passages: Sequence[Chunk],
    layout: ReferenceLayout,
    file: Path | None = None,
    *,
    accept_short: Callable[[str], bool] | None = None,
    which: int = -1,
) -> list[ReferenceEntry]:
    """Split one of the paper's reference lists into numbered entries.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    layout: ReferenceLayout
        The caller's thresholds for splitting and filtering entries.
    file: pathlib.Path | None, optional
        The paper on disk, for a PDF's column order (``reference_list``).
        Defaults to ``None``.
    accept_short: Callable[[str], bool] | None, optional
        Optional rule retaining an explicitly numbered short citation
        that carries enough bibliographic information for the caller.
        The default preserves the original minimum-word filtering.
    which: int, optional
        The list, as an index into the paper's lists (``-1``, the
        default, is the last).

    Returns
    -------
    list[ReferenceEntry]
        One entry per reference, without the paper's own number, with
        lines joined (and words hyphenated across lines rejoined). When at
        least ``layout.numbered_entries_min`` lines start with a number
        ("12. ...", "[12] ...") and no number stands alone, a new entry
        starts at each number; otherwise (as two-column PDFs print them)
        also after a line that ends an entry (a year, volume and pages, a
        PMID, or a DOI), unless the next line is a DOI or PMID, and at a
        line that starts like an author-year entry ("Smith, J. A., ...
        (2020)") once the entry before has its year in parentheses. A number
        that does not follow the previous entry's (more than
        ``_NUMBER_STEP_MAX`` ahead, or back) starts no entry: it is part
        of the entry ("…919–\\n927. https://doi…"); a list that starts
        again at 1 ends there. Numbers printed alone ("6.", "7.") are
        given, in order, to the entries that start without one. Page
        numbers, running heads and footers (short lines repeated
        ``layout.running_line_repeats`` or more times in the paper), and
        entries of fewer than ``layout.entry_min_words`` words are dropped
        unless an explicitly numbered short entry passes
        ``accept_short``. The list stops at a following heading such as
        "Acknowledgements", a table or figure caption, or another list's
        heading. An entry without a printed number takes the number after
        the previous entry's (``printed`` is then False). Empty when the
        paper has no reference list.

    """

    lines = reference_list(passages, layout, file, which).splitlines()

    # Running heads and footers repeat across pages; drop them first. They
    # are counted in the text the list was read from.
    repeats = Counter(
        " ".join(line.split())
        for line in _source_text(passages, file).splitlines()
        if 0 < len(line.split()) <= layout.running_line_words
    )
    lines = [
        line for line in lines
        if repeats[" ".join(line.split())] < layout.running_line_repeats
    ]

    numbered = sum(bool(_ENTRY_START.match(line)) for line in lines)
    by_number = numbered >= layout.numbered_entries_min and not any(
        _LONE_NUMBER.match(line) for line in lines
    )

    # The list in order: entries (printed number or None, lines) and the
    # numbers printed alone, which belong to entries without a number.
    sequence: list[tuple[int | None, list[str]] | int] = []
    complete = True
    last_printed: int | None = None
    for line in lines:
        if _ends_list(line):
            break

        if lone_number := _LONE_NUMBER.match(line):
            number = int(lone_number.group(1) or lone_number.group(2))
            sequence.append(number)
            if last_printed is None or (
                last_printed < number <= last_printed + _NUMBER_STEP_MAX
            ):
                last_printed = number
            continue

        if (
            not line.strip()
            or _PAGE_LINE.match(line)
            or _FURNITURE.search(line)
        ):
            continue

        match = _ENTRY_START.match(line)
        if match and last_printed is not None:
            number = int(match.group(1) or match.group(2))
            if number == 1 and last_printed >= layout.numbered_entries_min:
                break  # Numbering starts again: another list.
            if not last_printed < number <= last_printed + _NUMBER_STEP_MAX:
                match = None  # A number inside the entry.
        text = (match.group(3) if match else line).strip()
        entries = [item for item in sequence if isinstance(item, tuple)]
        if not entries or match or (
            not by_number and complete and not _CONTINUATION.match(text)
        ) or (
            not by_number and _AUTHOR_YEAR_HEAD.match(text)
            and _YEAR_IN_PARENTHESES.search(" ".join(entries[-1][1]))
        ):
            number = int(match.group(1) or match.group(2)) if match else None
            if number is not None:
                last_printed = number
            sequence.append((number, [text]))
        else:
            entries[-1][1].append(text)
        complete = bool(_ENTRY_END.search(text)) and not _BROKEN_URL.search(
            text
        )

    # Short fragments are dropped before lone numbers are paired, so they
    # cannot take a number.
    kept: list[tuple[int | None, str] | int] = []
    for item in sequence:
        if isinstance(item, int):
            kept.append(item)
            continue
        joined = re.sub(r"-\s+(?=[a-z])", "", " ".join(item[1]))
        joined = " ".join(joined.split())
        if (
            len(joined.split()) >= layout.entry_min_words
            or (
                item[0] is not None
                and accept_short is not None and accept_short(joined)
            )
        ):
            kept.append((item[0], joined))

    numbers: dict[int, int] = {}
    unnumbered: list[int] = []  # Entries waiting for a lone number.
    lone: list[int] = []  # Lone numbers waiting for an entry.
    for index, item in enumerate(kept):
        if isinstance(item, int):
            lone.append(item)
        elif item[0] is None:
            unnumbered.append(index)
        while unnumbered and lone:
            numbers[unnumbered.pop(0)] = lone.pop(0)

    cleaned: list[ReferenceEntry] = []
    previous = 0
    for index, item in enumerate(kept):
        if isinstance(item, int):
            continue
        number = item[0] if item[0] is not None else numbers.get(index)
        previous = number if number is not None else previous + 1
        cleaned.append(ReferenceEntry(previous, item[1], number is not None))

    return cleaned


def _source_text(passages: Sequence[Chunk], file: Path | None) -> str:
    """Return the text a reference list is read from.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    file: pathlib.Path | None
        The paper on disk, or None.

    Returns
    -------
    str
        The PDF's column-ordered text when it has a references heading;
        otherwise the passages joined without repeated lines.

    """

    if file is not None:
        text = pdf_text(file)
        if _REFERENCES_HEADING.search(text):
            return text

    return joined_text(passages)


def joined_text(passages: Sequence[Chunk]) -> str:
    """Return the passages' text with overlapping lines included once.

    Indexed passages of the same page overlap by whole lines; joining them
    as they are repeats those lines.

    Parameters
    ----------
    passages: Sequence[Chunk]
        Passages in file and line order.

    Returns
    -------
    str
        The passages joined by newlines. When a passage starts on a line
        the previous passage of the same file and section already covers,
        and its first lines repeat the end of that passage exactly, those
        lines are left out.

    """

    parts: list[str] = []
    previous: Chunk | None = None
    for chunk in passages:
        text = chunk.text
        if (
            previous is not None
            and chunk.path == previous.path
            and chunk.section == previous.section
            and chunk.start_line <= previous.end_line
        ):
            shared = previous.end_line - chunk.start_line + 1
            lines = text.splitlines(keepends=True)
            if previous.text.splitlines()[-shared:] == [
                line.rstrip("\r\n") for line in lines[:shared]
            ]:
                text = "".join(lines[shared:])
        parts.append(text)
        previous = chunk

    return "\n".join(parts)


def pdf_text(file: Path) -> str:
    """Return a PDF's text with each page read in column order.

    ``pdfminer.six`` orders text boxes in a way that can interleave the
    two columns of a page (reference 1-4, then 23-28, then 5 ...). Here
    each page's text lines are read top to bottom: on a two-column page
    (at least ``_RIGHT_COLUMN_SHARE`` of its lines start in the right
    half), first the left column (lines starting in the left half,
    including full-width ones), then the right column; other pages keep
    ``pdfminer.six``'s order.

    Parameters
    ----------
    file: pathlib.Path
        A file on disk.

    Returns
    -------
    str
        Pages separated by form feeds, lines by newlines; empty for files
        that are not PDFs and PDFs ``pdfminer.six`` cannot read.

    """

    if file.suffix.lower() != ".pdf" or not file.is_file():
        return ""

    status = file.stat()
    return _pdf_text(str(file), status.st_mtime_ns, status.st_size)


@lru_cache(maxsize=8)
def _pdf_text(file: str, modified: int, size: int) -> str:
    """Return a PDF's column-ordered text, cached per file version.

    Parameters
    ----------
    file: str
        The PDF on disk.
    modified: int
        Its modification time in nanoseconds, so an edited file is read
        again.
    size: int
        Its size in bytes, for the same reason.

    Returns
    -------
    str
        As ``pdf_text``.

    """

    del modified, size  # Part of the cache key only.
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams, LTTextContainer, LTTextLine

        pages = list(extract_pages(file, laparams=LAParams()))
    except Exception:  # Unreadable PDF: the caller uses the passages.
        return ""

    texts: list[str] = []
    for page in pages:
        lines = [
            line
            for box in page
            if isinstance(box, LTTextContainer)
            for line in box
            if isinstance(line, LTTextLine)
        ]
        middle = page.width / 2
        right = sum(line.x0 >= middle for line in lines)
        if lines and right >= _RIGHT_COLUMN_SHARE * len(lines):
            lines.sort(key=lambda line: (line.x0 >= middle, -line.y1))
        texts.append(
            "\n".join(line.get_text().rstrip("\r\n") for line in lines)
        )

    return "\f".join(texts)
