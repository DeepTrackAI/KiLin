"""Split source units into token-budgeted retrieval chunks.

The chunker preserves substantive source text and provenance with a supplied
token-count function. Units already within the maximum size stay whole. An
oversized unit is split through its children only when they cover all of its
non-whitespace text. Otherwise, it is split along line boundaries and, if
necessary, within an individual long line.

Target size guides splitting; maximum size is the acceptance bound. Overlap
is retained as source lines or character suffixes. No embedding model is
loaded by this module.

Token-count assumptions
-----------------------

Chunk boundaries are selected using binary search over candidate fragments.
This assumes that token counts are approximately monotonic as text grows or
shrinks. Subword tokenizers do not strictly guarantee this: adding characters
can sometimes reduce the count because previously separate tokens merge.

Consequently, the chunker may produce smaller fragments than necessary or,
in unusual cases, fail to identify a fitting fragment. This is an intentional
performance trade-off: splitting efficiently takes precedence over finding
globally optimal boundaries. Every emitted chunk is checked using the supplied
token counter and must fit within ``max_tokens``. The caller should use the
token counter corresponding to the intended embedding model.

Key Features
------------

- **Provenance retention**

    Carry paths, inclusive line ranges, structural labels, notebook cells, and
    sections into each chunk.

- **Hierarchical splitting**

    Prefer complete units and use children only when no content is lost.

- **Stable identifiers**

    Hash an unambiguous encoding of text and provenance without including an
    absolute repository root.

- **Progress reporting**

    Report completed source units and cache text-token counts within one call.

Module Structure
----------------

- ``Chunk``

    A searchable source fragment with its original location.

- ``chunk_source_units``

    Create token-bounded chunks while preserving natural units.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
Use a simple character counter to illustrate the API without loading a model:

>>> import qachat
>>> from pathlib import Path
>>> unit = qachat.SourceUnit(
...     path=Path("notes.txt"), text="Brownian motion.",
...     start_line=1, end_line=1, kind="text",
... )
>>> chunks = qachat.chunk_source_units(
...     [unit], count_tokens=len,
...     target_tokens=20, max_tokens=30, overlap_tokens=5,
... )

For model use, pass that model's token counter instead of ``len``.

"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from qachat.sources.sources import SourceUnit

__all__ = [
    "Chunk",
    "chunk_source_units",
]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A searchable source fragment with its original location.

    Parameters
    ----------
    identifier: str
        Stable content-based identifier.
    path: pathlib.Path
        Relative source path.
    text: str
        Exact source fragment.
    start_line: int
        First one-based source line.
    end_line: int
        Last one-based source line.
    kind: str
        Original source-unit type.
    symbol: str | None
        Original symbol name.
    section: str | None
        Original document section.
    cell: int | None, optional
        One-based notebook cell number when available. Defaults to ``None``.

    Attributes
    ----------
    identifier: str
        Stable content-based identifier.
    path: pathlib.Path
        Relative source path.
    text: str
        Exact source fragment.
    start_line: int
        First one-based source line.
    end_line: int
        Last one-based source line.
    kind: str
        Original source-unit type.
    symbol: str | None
        Original symbol name.
    section: str | None
        Original document section.
    cell: int | None
        One-based notebook cell number when available.

    Notes
    -----
    Line ranges are inclusive and refer to the source-unit coordinate system.
    Notebook lines are cell-local; PDF lines are page-local with section
    provenance. Frozen dataclass fields prevent reassignment, not mutation of
    objects stored within them.

    """

    identifier: str
    path: Path
    text: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    section: str | None
    cell: int | None = None


def chunk_source_units(
    units: list[SourceUnit],
    *,
    count_tokens: Callable[[str], int],
    target_tokens: int = 500,
    max_tokens: int = 700,
    overlap_tokens: int = 80,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Chunk]:
    """Create token-bounded chunks while preserving natural units.

    Parameters
    ----------
    units: list[SourceUnit]
        Natural source units.
    count_tokens: Callable[[str], int]
        Token counter from the selected embedding model. Boundary selection
        assumes approximately monotonic token counts; see module notes.
    target_tokens: int, optional
        Preferred size when a unit must be split. Defaults to ``500``.
    max_tokens: int, optional
        Largest allowed chunk. Defaults to ``700``.
    overlap_tokens: int, optional
        Maximum repeated text between adjacent split chunks. Defaults to
        ``80``.
    progress_callback: Callable[[int, int], None] | None, optional
        Called after each top-level source unit with ``current`` and ``total``.
        Defaults to ``None``.

    Returns
    -------
    list[Chunk]
        Searchable chunks in source order.

    Raises
    ------
    ValueError
        If a size is invalid or the token counter returns an invalid value.

    Notes
    -----
    Identifiers now use an unambiguous JSON-array encoding instead of the
    previous colon-separated encoding. Existing indexes must be rebuilt.

    """

    _check_sizes(target_tokens, max_tokens, overlap_tokens)
    counts: dict[str, int] = {}

    def count(text: str) -> int:
        """Memorize validated token counts during one chunking call.

        Parameters
        ----------
        text: str
            Exact source substring whose count is needed. Empty text has count
            zero without invoking the tokenizer.

        Returns
        -------
        int
            Nonnegative token count returned by the caller, reused for
            identical text within this invocation.

        Raises
        ------
        ValueError
            If the callback returns a value other than a nonnegative exact
            integer.

        """

        if text not in counts:
            value = count_tokens(text) if text else 0
            if type(value) is not int or value < 0:
                raise ValueError(
                    "count_tokens must return a non-negative whole number."
                )
            counts[text] = value
        return counts[text]

    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None.")

    chunks: list[Chunk] = []
    total = len(units)
    for current, unit in enumerate(units, start=1):
        chunks.extend(
            _chunk_source_unit(
                unit,
                count=count,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
        if progress_callback is not None:
            progress_callback(current, total)

    return chunks


def _check_sizes(
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    """Check chunk size settings.

    Parameters
    ----------
    target_tokens: int
        Preferred split size.
    max_tokens: int
        Largest allowed chunk.
    overlap_tokens: int
        Maximum repeated text.

    Raises
    ------
    ValueError
        If the settings cannot produce bounded forward progress.

    """

    values = (target_tokens, max_tokens, overlap_tokens)
    if any(type(value) is not int for value in values):
        raise ValueError("Chunk sizes must be whole numbers.")

    if target_tokens < 1:
        raise ValueError("target_tokens must be at least 1.")

    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be at least target_tokens.")

    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError(
            "overlap_tokens must be non-negative and below target_tokens."
        )


def _chunk_source_unit(
    unit: SourceUnit,
    *,
    count: Callable[[str], int],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Keep one unit whole or split its smaller natural children.

    Parameters
    ----------
    unit: SourceUnit
        Unit to process.
    count: Callable[[str], int]
        Checked and cached token counter.
    target_tokens: int
        Preferred split size.
    max_tokens: int
        Largest allowed chunk.
    overlap_tokens: int
        Maximum repeated text.

    Returns
    -------
    list[Chunk]
        Zero or more bounded chunks.

    """

    if not unit.text.strip():
        return []

    if count(unit.text) <= max_tokens:
        return [
            _make_chunk(
                unit,
                unit.text,
                unit.start_line,
                unit.end_line,
            )
        ]

    if unit.children and _children_cover_text(unit):
        chunks: list[Chunk] = []
        for child in unit.children:
            chunks.extend(
                _chunk_source_unit(
                    child,
                    count=count,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        return chunks

    return _split_unit(
        unit,
        count=count,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def _children_cover_text(unit: SourceUnit) -> bool:
    """Check whether ordered children cover all substantive parent text.

    Parameters
    ----------
    unit: SourceUnit
        Parent with candidate natural child units.

    Returns
    -------
    bool
        ``True`` if the children match nonoverlapping substrings in source
        order, with only whitespace between or around those substrings.

    Notes
    -----
    Unmatched non-whitespace content requires splitting the parent instead.
    Exact substring matching also avoids using child text that was normalized
    differently from the original source or belongs to another file.

    """

    offset = 0
    for child in unit.children:
        if child.path != unit.path or not child.text:
            return False

        start = unit.text.find(child.text, offset)
        if start < 0 or unit.text[offset:start].strip():
            return False
        offset = start + len(child.text)

    return not unit.text[offset:].strip()


def _make_chunk(
    unit: SourceUnit,
    text: str,
    start_line: int,
    end_line: int,
    *,
    part: int = 0,
) -> Chunk:
    """Create one stable chunk.

    Parameters
    ----------
    unit: SourceUnit
        Source provenance.
    text: str
        Chunk text.
    start_line: int
        First source line.
    end_line: int
        Last source line.
    part: int, optional
        Character offset used to distinguish pieces of one long line. Defaults
        to ``0``.

    Returns
    -------
    Chunk
        New chunk with a content-based identifier.

    Notes
    -----
    A JSON array preserves field boundaries and distinguishes missing metadata
    from empty values before hashing. The first 16 hash digits are retained.

    """

    identity = json.dumps(
        [
            unit.path.as_posix(),
            start_line,
            end_line,
            unit.kind,
            unit.symbol,
            unit.section,
            unit.cell,
            part,
            text,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    identifier = sha256(identity.encode("utf-8")).hexdigest()[:16]
    return Chunk(
        identifier=identifier,
        path=unit.path,
        text=text,
        start_line=start_line,
        end_line=end_line,
        kind=unit.kind,
        symbol=unit.symbol,
        section=unit.section,
        cell=unit.cell,
    )


def _largest_prefix(
    text: str,
    limit: int,
    count: Callable[[str], int],
) -> int:
    """Find the largest character prefix within a token limit.

    Parameters
    ----------
    text: str
        Text to search.
    limit: int
        Token limit.
    count: Callable[[str], int]
        Token counter.

    Returns
    -------
    int
        Number of fitting characters.

    Notes
    -----
    The search assumes candidate token counts are sufficiently monotonic for
    binary search. The tokenizer callback is not checked for that property.

    """

    low = 1
    high = len(text)
    fitting = 0
    while low <= high:
        middle = (low + high) // 2
        if count(text[:middle]) <= limit:
            fitting = middle
            low = middle + 1
        else:
            high = middle - 1

    return fitting


def _largest_suffix(
    text: str,
    limit: int,
    count: Callable[[str], int],
) -> str:
    """Find the largest text suffix within a token limit.

    Parameters
    ----------
    text: str
        Text to search.
    limit: int
        Token limit.
    count: Callable[[str], int]
        Token counter.

    Returns
    -------
    str
        Fitting suffix.

    Notes
    -----
    The search assumes candidate token counts are sufficiently monotonic for
    binary search. The tokenizer callback is not checked for that property.

    """

    if limit == 0 or not text:
        return ""

    low = 0
    high = len(text)
    fitting_start = len(text)
    while low <= high:
        middle = (low + high) // 2
        if count(text[middle:]) <= limit:
            fitting_start = middle
            high = middle - 1
        else:
            low = middle + 1

    return text[fitting_start:]


def _largest_line_end(
    lines: list[tuple[str, int]],
    start: int,
    limit: int,
    count: Callable[[str], int],
) -> int:
    """Find the farthest whole-line boundary within a token limit.

    Parameters
    ----------
    lines: list[tuple[str, int]]
        Text and one-based line number pairs.
    start: int
        Zero-based starting item.
    limit: int
        Token limit.
    count: Callable[[str], int]
        Token counter.

    Returns
    -------
    int
        Exclusive ending item.

    Notes
    -----
    The search assumes candidate token counts are sufficiently monotonic for
    binary search. The tokenizer callback is not checked for that property.

    """

    low = start + 1
    high = len(lines)
    fitting = start
    while low <= high:
        middle = (low + high) // 2
        text = "".join(line for line, _ in lines[start:middle])
        if count(text) <= limit:
            fitting = middle
            low = middle + 1
        else:
            high = middle - 1

    return fitting


def _overlap_start(
    lines: list[tuple[str, int]],
    start: int,
    end: int,
    limit: int,
    count: Callable[[str], int],
) -> int:
    """Find the earliest whole-line suffix within an overlap limit.

    Parameters
    ----------
    lines: list[tuple[str, int]]
        Text and line number pairs.
    start: int
        Current zero-based starting item.
    end: int
        Current exclusive ending item.
    limit: int
        Token limit.
    count: Callable[[str], int]
        Token counter.

    Returns
    -------
    int
        Start of the overlapping suffix.

    Notes
    -----
    The search assumes candidate token counts are sufficiently monotonic for
    binary search. The tokenizer callback is not checked for that property.

    """

    if limit == 0:
        return end

    low = start + 1
    high = end
    fitting = end
    while low <= high:
        middle = (low + high) // 2
        text = "".join(line for line, _ in lines[middle:end])
        if count(text) <= limit:
            fitting = middle
            high = middle - 1
        else:
            low = middle + 1

    return fitting


def _split_long_line(
    unit: SourceUnit,
    *,
    count: Callable[[str], int],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split one oversized source line by characters.

    Parameters
    ----------
    unit: SourceUnit
        Single-line source unit.
    count: Callable[[str], int]
        Token counter.
    target_tokens: int
        Preferred split size.
    max_tokens: int
        Largest allowed chunk.
    overlap_tokens: int
        Maximum repeated text.

    Returns
    -------
    list[Chunk]
        Bounded chunks covering the source line.

    Raises
    ------
    ValueError
        If even one character exceeds the target budget or overlap prevents
        forward progress.

    """

    chunks: list[Chunk] = []
    remaining = unit.text
    start_character = 0
    while remaining:
        if count(remaining) <= max_tokens:
            chunks.append(
                _make_chunk(
                    unit,
                    remaining,
                    unit.start_line,
                    unit.end_line,
                    part=start_character,
                )
            )
            break

        fitting = _largest_prefix(remaining, target_tokens, count)
        if fitting == 0:
            raise ValueError(
                "The tokenizer reports that one character exceeds the "
                "target token budget."
            )

        piece = remaining[:fitting]
        chunks.append(
            _make_chunk(
                unit,
                piece,
                unit.start_line,
                unit.end_line,
                part=start_character,
            )
        )
        overlap = _largest_suffix(piece, overlap_tokens, count)
        consumed = fitting - len(overlap)
        if consumed <= 0:
            raise ValueError("Token overlap prevented chunking progress.")

        start_character += consumed
        remaining = overlap + remaining[fitting:]

    return chunks


def _split_unit(
    unit: SourceUnit,
    *,
    count: Callable[[str], int],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split an oversized unit at line boundaries when possible.

    Parameters
    ----------
    unit: SourceUnit
        Source unit to split.
    count: Callable[[str], int]
        Token counter.
    target_tokens: int
        Preferred split size.
    max_tokens: int
        Largest allowed chunk.
    overlap_tokens: int
        Maximum repeated text.

    Returns
    -------
    list[Chunk]
        Bounded chunks in source order.

    """

    source_lines = unit.text.splitlines(keepends=True)
    if len(source_lines) == 1:
        return _split_long_line(
            unit,
            count=count,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

    lines = [
        (line, unit.start_line + offset)
        for offset, line in enumerate(source_lines)
    ]
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        remaining = "".join(line for line, _ in lines[start:])
        if count(remaining) <= max_tokens:
            chunks.append(
                _make_chunk(
                    unit,
                    remaining,
                    lines[start][1],
                    lines[-1][1],
                )
            )
            break

        end = _largest_line_end(lines, start, target_tokens, count)
        if end == start:
            line, line_number = lines[start]
            line_unit = SourceUnit(
                path=unit.path,
                text=line,
                start_line=line_number,
                end_line=line_number,
                kind=unit.kind,
                symbol=unit.symbol,
                section=unit.section,
                cell=unit.cell,
            )
            chunks.extend(
                _split_long_line(
                    line_unit,
                    count=count,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            start += 1
            continue

        text = "".join(line for line, _ in lines[start:end])
        chunks.append(
            _make_chunk(
                unit,
                text,
                lines[start][1],
                lines[end - 1][1],
            )
        )
        if end == len(lines):
            break

        start = _overlap_start(
            lines,
            start,
            end,
            overlap_tokens,
            count,
        )

    return [chunk for chunk in chunks if chunk.text.strip()]
