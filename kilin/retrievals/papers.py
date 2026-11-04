"""Describe the documents of a literature project for searching.

A passage of a paper rarely says which paper it belongs to or which part of
it it comes from: a methods paragraph does not repeat the paper's title,
and a results paragraph does not name the section it sits in. Search then
finds passages by their own words only, and a question about "the BRAPH 2
architecture" can miss the paragraph that describes it without the name.

This module gives each document passage a short header for searching: the
document's title, its first author and file name, the section the passage
belongs to, and its page. Only searching sees the header; evidence,
citations, and the answering model keep the passage text itself.

Titles come from the PDF metadata when it holds a plausible title, from the
first ``#`` heading of a Markdown file, and otherwise from the file name.
Section headings are recognized by name ("Methods", "Results", "Data
availability") or by numbering ("3.1. Software architecture").

Key Features
------------

- **Document identity**

    Find a title and first author for each document without a model.

- **Section headings**

    Follow the section each passage belongs to through a document.

- **Search headers**

    Build the header placed in front of each document passage for search.

Module Structure
----------------

- ``DOCUMENT_HEADER_FORMAT``

    Version of the header layout, recorded with saved indexes.

- ``DOCUMENT_KINDS``

    Passage kinds that receive a header.

- ``DocumentInfo``

    Title and first author of one document.

- ``document_info``

    Find the title and first author of one document.

- ``section_headings``

    Map each passage to the section heading it belongs to.

- ``document_headers``

    Build the search header of every document passage.

Examples
--------
>>> from pathlib import Path
>>> from kilin.chunking import Chunk
>>> from kilin.retrievals.papers import DocumentInfo, document_headers
>>> chunk = Chunk("c1", Path("smith20.pdf"), "1. Introduction\\nText.",
...               1, 2, "pdf_page", None, "Page 1")
>>> info = {Path("smith20.pdf"): DocumentInfo("A study", "Smith")}
>>> print(document_headers([chunk], info)["c1"])
A study
Smith et al. · smith20.pdf
1. Introduction · page 1

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from kilin.chunking import Chunk

__all__ = [
    "DOCUMENT_HEADER_FORMAT",
    "DOCUMENT_KINDS",
    "DocumentInfo",
    "document_headers",
    "document_info",
    "section_headings",
]

# Recorded with saved indexes; a change rebuilds them.
DOCUMENT_HEADER_FORMAT: Final[str] = "document-title-section-page-v1"
# Prose passages of documents; code passages have their own headers.
DOCUMENT_KINDS: Final[frozenset[str]] = frozenset(
    {"pdf_page", "markdown_preamble", "markdown_section", "text_paragraph"}
)
# Longest title kept; longer metadata is not a title.
_TITLE_MAX_CHARACTERS: Final[int] = 300
# Lines at the start of a passage where a heading still names that passage.
_HEADING_LEAD_LINES: Final[int] = 3
# Section names that stand alone on their line, optionally numbered.
_NAMED_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:[1-9]\d?\.)+\s*|[1-9]\d?\s+)?"
    r"(?:abstract|introduction|background|results?(?: and discussion)?"
    r"|discussion|conclusions?|methods?|materials and methods|methodology"
    r"|experimental(?: section| procedures)?|related work|references"
    r"|bibliography|acknowledg(?:e)?ments?"
    r"|supplementary (?:information|materials?|figures|tables)"
    r"|data availability|code availability|limitations"
    r"|future (?:work|developments?)|motivation and significance|impact"
    r"|summary|keywords)\.?$",
    re.IGNORECASE,
)
# Numbered headings such as "3.1. Software architecture"; numbers start at
# one, so values like "0.91." or "07.003" are not taken for headings.
_NUMBERED_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^(?:[1-9]\d?\.)+\s+[A-Z][^.!?]{1,70}$"
)
# Most words in a numbered heading; longer lines are sentences.
_HEADING_MAX_WORDS: Final[int] = 10
_PAGE_SECTION: Final[re.Pattern[str]] = re.compile(r"^Page (\d+)$")
_MARKDOWN_TITLE: Final[re.Pattern[str]] = re.compile(
    r"^#[ \t]+(.+?)[ \t#]*$", re.MULTILINE
)
# Metadata titles that name a tool or a file rather than the document.
_PLACEHOLDER_TITLE: Final[re.Pattern[str]] = re.compile(
    r"^(?:untitled|title|microsoft word\b.*|.*\.(?:pdf|docx?|tex|dvi|ps))$",
    re.IGNORECASE,
)
_AUTHOR_SEPARATORS: Final[re.Pattern[str]] = re.compile(
    r"\s*(?:;|,|\band\b|&)\s*"
)


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """Title and first author of one document.

    Parameters
    ----------
    title: str
        Document title, or the file name without suffix when none is known.
    first_author: str | None
        Surname of the first author, when known.

    Attributes
    ----------
    title: str
        Document title, or the file name without suffix when none is known.
    first_author: str | None
        Surname of the first author, when known.

    """

    title: str
    first_author: str | None


def document_info(
    source: Path, path: Path, chunks: Sequence[Chunk]
) -> DocumentInfo:
    """Find the title and first author of one document.

    Parameters
    ----------
    source: pathlib.Path
        Folder that holds the project's source files.
    path: pathlib.Path
        The document's path relative to ``source``.
    chunks: Sequence[Chunk]
        The document's passages, used for a Markdown title.

    Returns
    -------
    DocumentInfo
        The PDF metadata title and first author when plausible; the first
        ``#`` heading of a Markdown file; otherwise the file name without
        its suffix and no author.

    Notes
    -----
    Unreadable PDF metadata is ignored: the file name is used instead, so
    indexing never fails because of a title.

    """

    title: str | None = None
    author: str | None = None
    if path.suffix.lower() == ".pdf":
        title, author = _pdf_metadata(source / path)
    elif path.suffix.lower() == ".md":
        for chunk in chunks:
            match = _MARKDOWN_TITLE.search(chunk.text)
            if match is not None:
                title = _clean_title(match.group(1))
                break
    return DocumentInfo(title or path.stem, author)


def section_headings(chunks: Sequence[Chunk]) -> dict[str, str]:
    """Map each passage to the section heading it belongs to.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Indexed passages of any number of documents.

    Returns
    -------
    dict[str, str]
        Heading of each document passage that follows one, by passage
        identifier. Markdown passages keep their own section name.

    Notes
    -----
    Passages are read in page and line order within each document. A
    heading near the start of a passage names that passage; otherwise the
    passage belongs to the last heading before it, and a heading inside it
    starts the section of the passages that follow.

    """

    by_path: dict[Path, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.kind in DOCUMENT_KINDS:
            by_path.setdefault(chunk.path, []).append(chunk)
    headings: dict[str, str] = {}
    for passages in by_path.values():
        current: str | None = None
        for chunk in sorted(passages, key=_reading_position):
            if chunk.kind.startswith("markdown_"):
                if chunk.section:
                    headings[chunk.identifier] = chunk.section
                continue
            found = [
                (number, line)
                for number, line in enumerate(_lines(chunk.text))
                if _is_heading(line)
            ]
            if found and found[0][0] < _HEADING_LEAD_LINES:
                headings[chunk.identifier] = found[0][1]
            elif current is not None:
                headings[chunk.identifier] = current
            if found:
                current = found[-1][1]
    return headings


def document_headers(
    chunks: Sequence[Chunk], infos: Mapping[Path, DocumentInfo]
) -> dict[str, str]:
    """Build the search header of every document passage.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Indexed passages; only document passages (``DOCUMENT_KINDS``) of
        documents in ``infos`` receive a header.
    infos: Mapping[pathlib.Path, DocumentInfo]
        Title and first author of each document.

    Returns
    -------
    dict[str, str]
        Header lines by passage identifier: the title; the first author
        and file name; and the section heading and page, when known.

    """

    headings = section_headings(chunks)
    headers: dict[str, str] = {}
    for chunk in chunks:
        info = infos.get(chunk.path)
        if chunk.kind not in DOCUMENT_KINDS or info is None:
            continue
        origin = (
            f"{info.first_author} et al. · {chunk.path.name}"
            if info.first_author
            else chunk.path.name
        )
        place = [headings[chunk.identifier]] if (
            chunk.identifier in headings
        ) else []
        page = _page(chunk)
        if page is not None:
            place.append(f"page {page}")
        lines = [info.title, origin]
        if place:
            lines.append(" · ".join(place))
        headers[chunk.identifier] = "\n".join(lines)
    return headers


def _pdf_metadata(file: Path) -> tuple[str | None, str | None]:
    """Read a plausible title and first author from PDF metadata.

    Parameters
    ----------
    file: pathlib.Path
        PDF file.

    Returns
    -------
    tuple[str | None, str | None]
        Title and first author's surname; ``None`` for each one that is
        missing, implausible, or unreadable.

    """

    try:
        from pypdf import PdfReader

        metadata = PdfReader(file).metadata or {}
        title = metadata.get("/Title")
        author = metadata.get("/Author")
    except Exception:  # pylint: disable=broad-exception-caught
        # Any reader failure falls back to the file name.
        return None, None
    return (
        _clean_title(str(title)) if title else None,
        _first_surname(str(author)) if author else None,
    )


def _clean_title(value: str) -> str | None:
    """Return a title with normalized spaces, or None when implausible.

    Parameters
    ----------
    value: str
        Candidate title.

    Returns
    -------
    str | None
        The title on one line, or ``None`` when it is empty, too short or
        long, or names a file or tool instead of a document.

    """

    title = " ".join(value.split())
    if (
        len(title) < 4
        or len(title) > _TITLE_MAX_CHARACTERS
        or _PLACEHOLDER_TITLE.match(title)
    ):
        return None
    return title


def _first_surname(value: str) -> str | None:
    """Return the surname of the first author in an author list.

    Parameters
    ----------
    value: str
        Author metadata, such as ``"Mite Mijalkov, Ehsan Kakaei"``.

    Returns
    -------
    str | None
        Last word of the first listed name, or ``None`` when there is none.

    """

    names = [
        name for name in _AUTHOR_SEPARATORS.split(value.strip()) if name
    ]
    if not names:
        return None
    words = re.findall(r"[^\W\d_][\w'’-]*", names[0])
    return words[-1] if words else None


def _reading_position(chunk: Chunk) -> tuple[int, int]:
    """Return a passage's page and first line, for reading order.

    Parameters
    ----------
    chunk: Chunk
        Document passage.

    Returns
    -------
    tuple[int, int]
        Page number (zero when unknown) and first line.

    """

    return (_page(chunk) or 0, chunk.start_line)


def _page(chunk: Chunk) -> int | None:
    """Return the page a PDF passage comes from.

    Parameters
    ----------
    chunk: Chunk
        Document passage.

    Returns
    -------
    int | None
        One-based page number, or ``None`` for passages without pages.

    """

    match = _PAGE_SECTION.match(chunk.section or "")
    return int(match.group(1)) if match else None


def _lines(text: str) -> list[str]:
    """Return the nonblank lines of a passage, stripped.

    Parameters
    ----------
    text: str
        Passage text.

    Returns
    -------
    list[str]
        Nonblank lines in order.

    """

    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_heading(line: str) -> bool:
    """Return whether a line is a section heading.

    Parameters
    ----------
    line: str
        One stripped line.

    Returns
    -------
    bool
        Whether the line is a known section name, optionally numbered, or a
        short numbered heading such as ``3.1. Software architecture``.

    """

    if _NAMED_HEADING.match(line):
        return True
    return bool(
        _NUMBERED_HEADING.match(line)
        and len(line.split()) <= _HEADING_MAX_WORDS
    )
