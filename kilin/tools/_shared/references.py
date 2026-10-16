"""Find a paper's reference list and split it into entries, as printed.

Several registered tools need a paper's own reference entries: the report
tool copies some into a briefing, and a citation checker matches them
against other files. This private implementation keeps the layout rules
in one place, so tools do not import one another to reuse them.

The rules are deterministic and model-free. They read the text after the
paper's last "References" heading and split it at entry numbers, or (for
two-column PDFs, which print the numbers apart from the entries) at the
ends of entries. Every entry is the paper's own text; an unusual layout
can still split or join entries wrongly.

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

- ``reference_body``

    Return the paper's passages before its reference list.

- ``reference_list``

    Return the paper's reference list, found by its heading.

- ``reference_entries``

    Split the paper's reference list into entries, as printed.

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
from collections.abc import Sequence
from dataclasses import dataclass, replace
import re
from typing import Final

from kilin.chunking import Chunk

__all__ = [
    "ReferenceLayout",
    "reference_body",
    "reference_entries",
    "reference_list",
]

# The heading that starts a reference list; the last one in the paper wins.
_REFERENCES_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^\s*#*\s*(references|bibliography|literature cited|works cited)"
    r"\s*:?\s*$",
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
    r"^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s*$"
)
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
    r"availability|code availability|appendix)\s*(?::|$|\s+[A-Z][a-z])",
    flags=re.I,
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


def reference_body(passages: Sequence[Chunk]) -> tuple[Chunk, ...]:
    """Return the paper's passages before its reference list.

    In-text citations are looked for here, so the reference list's own
    numbers are not mistaken for citations.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.

    Returns
    -------
    tuple[Chunk, ...]
        The passages before the last "References" (or "Bibliography",
        "Literature cited", "Works cited") heading; the passage holding the
        heading is cut just before it. Every passage when there is no
        heading. Text after the reference list (figure legends printed at
        the end, for example) is not included.

    """

    for index in range(len(passages) - 1, -1, -1):
        headings = list(_REFERENCES_HEADING.finditer(passages[index].text))
        if not headings:
            continue

        start = headings[-1].start()
        kept = passages[index].text[:start]
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

    return tuple(passages)


def reference_list(
    passages: Sequence[Chunk], layout: ReferenceLayout
) -> str:
    """Return the paper's reference list, found by its heading.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    layout: ReferenceLayout
        The caller's thresholds; ``max_characters`` limits the list.

    Returns
    -------
    str
        Text after the last "References" (or "Bibliography", "Literature
        cited", "Works cited") heading, ``layout.max_characters`` at most;
        empty when there is none.

    """

    text = "\n".join(chunk.text for chunk in passages)
    headings = list(_REFERENCES_HEADING.finditer(text))
    if not headings:
        return ""

    start = headings[-1].end()
    return text[start : start + layout.max_characters].strip()


def reference_entries(
    passages: Sequence[Chunk], layout: ReferenceLayout
) -> list[str]:
    """Split the paper's reference list into entries, as printed.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's passages, in order.
    layout: ReferenceLayout
        The caller's thresholds for splitting and filtering entries.

    Returns
    -------
    list[str]
        One string per entry, without the paper's own number, with lines
        joined (and words hyphenated across lines rejoined). When at least
        ``layout.numbered_entries_min`` lines start with a number
        ("12. ...", "[12] ...") and no number stands alone, a new entry
        starts at each number; otherwise (as two-column PDFs print them)
        also after a line that ends an entry (a year, volume and pages, a
        PMID, or a DOI), unless the next line is a DOI or PMID. Lone
        numbers, page numbers, running heads and footers (short lines
        repeated ``layout.running_line_repeats`` or more times in the
        paper), and entries of fewer than ``layout.entry_min_words`` words
        are dropped; the list stops at a following heading such as
        "Acknowledgements". Empty when the paper has no reference list.

    """

    lines = reference_list(passages, layout).splitlines()

    # Running heads and footers repeat across pages; drop them first.
    repeats = Counter(
        " ".join(line.split())
        for chunk in passages
        for line in chunk.text.splitlines()
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

    entries: list[list[str]] = []
    complete = True
    for line in lines:
        if _AFTER_REFERENCES.match(line):
            break

        if (
            not line.strip()
            or _PAGE_LINE.match(line)
            or _LONE_NUMBER.match(line)
            or _FURNITURE.search(line)
        ):
            continue

        match = _ENTRY_START.match(line)
        text = (match.group(3) if match else line).strip()
        if not entries or match or (
            not by_number and complete and not _CONTINUATION.match(text)
        ):
            entries.append([text])
        else:
            entries[-1].append(text)
        complete = bool(_ENTRY_END.search(text))

    cleaned: list[str] = []
    for entry_lines in entries:
        joined = re.sub(r"-\s+(?=[a-z])", "", " ".join(entry_lines))
        joined = " ".join(joined.split())
        if len(joined.split()) >= layout.entry_min_words:
            cleaned.append(joined)

    return cleaned
