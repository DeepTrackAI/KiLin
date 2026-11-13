"""Split indexed text into distinct sentences with source positions.

Algorithms shared by registered tools belong here rather than in
another executable tool. The module has no GUI dependency.

Key Features
------------

- **Source-aware sentence splitting**

    Deterministic operations reusable by multiple tool packages.

Module Structure
----------------

- ``Sentence and split_sentences``

    Public operations shared within the tools package.

Examples
--------
>>> from kilin.tools._shared.sentences import split_sentences
>>> split_sentences([])
[]

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from kilin.chunking import Chunk

# Abbreviations that end with a period but not a sentence ("see Ref. 50",
# "in Fig. 3", "Eq. 2"): no split after them.
_ABBREVIATIONS: Final[tuple[str, ...]] = (
    "Ref", "ref", "Refs", "refs", "Fig", "fig", "Figs", "figs", "Eq", "eq",
    "Eqs", "eqs", "Sec", "Secs", "No", "Nos", "Tab", "Vol", "ca", "cf",
    "vs", "approx",
)
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])"
    + "".join(rf"(?<!\b{word}\.)" for word in _ABBREVIATIONS)
    + r"\s+(?=[A-Z0-9\"'(\[])|\n\s*\n"
)
_FRAGMENT: Final[re.Pattern[str]] = re.compile(r"^\s*[a-z]")
# Text that goes on from an unfinished sentence: a lowercase word, or a
# citation marker printed first on the next line or page and then one
# ("45,46 in cells", "61 in plants").
_CONTINUES: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\s+)?[a-z]"
)
# A page number printed alone between the halves of a sentence that runs
# onto the next page ("13", "Page 13").
_PAGE_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}\s*$", re.I
)
# Recognizable footer lines, not arbitrary URLs, DOI links or table
# cells. Running headers and repeated footers are handled separately.
_FURNITURE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:\d+\s*/\s*\d+|[^|\n]+\|\s*(?:https?://\S*)?)\s*$",
    re.I,
)
# A lone link following an independently confirmed repeated running head
# can be a printed DOI footer, but other standalone links are content.
_STANDALONE_LINK: Final[re.Pattern[str]] = re.compile(
    r"https?://\S+", re.I
)
# Short lines repeated this often in a file are running heads or footers
# ("Article", "(2025) 16:2748", the paper's title).
_RUNNING_REPEATS: Final[int] = 3
# Most words of a running head or footer.
_RUNNING_WORDS: Final[int] = 15
# A passage that ends a sentence: its last piece ends with ".", "!" or "?",
# possibly followed by closing quotes or brackets.
_ENDED: Final[re.Pattern[str]] = re.compile(r"[.!?][\"'”’)\]]*\s*$")

__all__ = [
    "Sentence",
    "split_sentences",
]


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence of a passage.

    Attributes
    ----------
    path: pathlib.Path
        The file.
    line: int
        Its first line in the file.
    text: str
        The sentence, with whitespace joined.
    chunk: Chunk
        The passage it comes from, for citations.

    """

    path: Path
    line: int
    text: str
    chunk: Chunk


def split_sentences(
    chunks: Sequence[Chunk], *, min_words: int = 4
) -> list[Sentence]:
    """Split passages into sentences, without repeats.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Passages in file and line order.

    min_words: int, optional
        Minimum number of words. Defaults to ``4``.

    Returns
    -------
    list[Sentence]
        Sentences of at least ``min_words`` words; abbreviations such
        as "Ref.", "Fig." or "Eq." are not sentence boundaries. A sentence
        shared by overlapping passages is kept once, from the first passage.
        A passage that begins inside a sentence completes the previous
        passage's unfinished last sentence (their overlap counted once);
        its broken start is otherwise left out. A sentence broken by a
        blank line or a page number ("of the\\n\\n13\\n\\nprotein",
        "accumulation\\n13\\n45,46 in") is joined, without the number.

    """

    counts = Counter(
        (chunk.path, _line_key(line))
        for chunk in chunks
        for line in chunk.text.splitlines()
        if 0 < len(line.split()) <= _RUNNING_WORDS
    )
    running = {
        key for key, count in counts.items() if count >= _RUNNING_REPEATS
    }
    seen: set[tuple[Path, str]] = set()
    sentences: list[Sentence] = []
    previous: Path | None = None
    # The previous passage's unfinished last sentence: its text, and its
    # index in ``sentences`` (None when it was too short to keep), with
    # the line and passage it starts in.
    unfinished: tuple[str, int | None, int, Chunk] | None = None
    last_chunk: Chunk | None = None
    for chunk in chunks:
        lines = {line for path, line in running if path == chunk.path}
        pieces = _pieces(chunk.text, lines)
        if chunk.path == previous and unfinished is not None:
            # Page furniture at the top of the passage, before the rest of
            # a sentence the passage before left unfinished.
            while pieces and _furniture(pieces[0][1], lines, blank=True):
                pieces = pieces[1:]
            if pieces:
                pieces[0] = (
                    pieces[0][0], _without_furniture(pieces[0][1], lines)
                )
        starts_inside = bool(pieces) and (
            _FRAGMENT.match(pieces[0][1])
            or (unfinished is not None and _CONTINUES.match(pieces[0][1]))
        )
        # A passage whose first piece the passage before already holds
        # whole starts in their overlap, before the unfinished sentence: it
        # repeats that sentence instead of going on from it.
        earlier = (
            unfinished is not None
            and last_chunk is not None
            and bool(pieces)
            and _line_key(pieces[0][1]) in _line_key(last_chunk.text)
        )
        if chunk.path == previous and starts_inside:
            # A passage that starts inside a sentence ("tion for deep
            # learning ..."): the sentence started in the passage before.
            # Its end is here when that passage stopped before it ended.
            if unfinished is not None and not earlier:
                unfinished = _complete(
                    sentences, seen, unfinished, pieces[0][1], min_words
                )
            # A passage inside one long sentence leaves it unfinished.
            carried = (
                unfinished if len(pieces) == 1 and not earlier else None
            )
            pieces = pieces[1:]
        else:
            carried = None
        # The unfinished sentence, when this passage repeats it whole.
        repeated = unfinished if chunk.path == previous else None
        previous = chunk.path
        last_chunk = chunk
        unfinished = carried

        for index, (offset, piece) in enumerate(pieces):
            text = " ".join(piece.split())
            last = index == len(pieces) - 1
            line = chunk.start_line + chunk.text[: max(0, offset)].count("\n")
            open_end = last and bool(text) and not _ENDED.search(piece)
            if len(text.split()) < min_words:
                if open_end:
                    unfinished = (text, None, line, chunk)
                continue

            key = (chunk.path, text.casefold())
            if key in seen:
                continue

            seen.add(key)
            if (
                repeated is not None
                and repeated[1] is not None
                and text.startswith(repeated[0])
            ):
                # The whole of a sentence the passage before cut short.
                seen.discard((chunk.path, repeated[0].casefold()))
                sentences[repeated[1]] = Sentence(
                    chunk.path, repeated[2], text, repeated[3]
                )
                if open_end:
                    unfinished = (text, repeated[1], repeated[2], repeated[3])
                repeated = None
                continue
            if open_end:
                unfinished = (text, len(sentences), line, chunk)
            sentences.append(Sentence(chunk.path, line, text, chunk))

    return sentences


def _pieces(text: str, running: set[str]) -> list[tuple[int, str]]:
    """Split a passage at sentence ends, joining sentences broken apart.

    Parameters
    ----------
    text: str
        The passage's text.
    running: set[str]
        The file's running heads and footers (``_line_key``).

    Returns
    -------
    list[tuple[int, str]]
        Each piece's offset in ``text`` and its text. A piece that does
        not end a sentence is joined with the next one that goes on from
        it (a lowercase word, or a citation marker and then one), across
        blank lines and page furniture between them (a page number, a
        footer, a running head), which is left out.

    """

    found: list[tuple[int, str]] = []
    position = 0
    for piece in _SENTENCE_END.split(text):
        offset = text.find(piece, position)
        position = max(position, offset + len(piece))
        found.append((offset, piece))

    joined: list[tuple[int, str]] = []
    index = 0
    while index < len(found):
        if (
            index == 0 and len(found) > 2
            and _line_key(found[0][1]) in running
            and _STANDALONE_LINK.fullmatch(found[1][1].strip())
            and _CONTINUES.match(found[2][1])
        ):
            # Printed footer after a known running header, directly
            # before the continuation of a sentence from last page.
            # A lone link without this evidence remains source text.
            index = 2
        offset, piece = found[index]
        unfinished = bool(joined) and bool(joined[-1][1].strip()) and not (
            _ENDED.search(joined[-1][1])
            or _furniture(joined[-1][1], running)
        )
        if unfinished:
            # Skip furniture when the sentence goes on after it.
            ahead = index
            while ahead < len(found) and _furniture(
                found[ahead][1], running, blank=True
            ):
                ahead += 1
            if ahead < len(found):
                rest = _without_furniture(found[ahead][1], running)
                if _CONTINUES.match(rest):
                    joined[-1] = (joined[-1][0], f"{joined[-1][1]} {rest}")
                    index = ahead + 1
                    continue
        joined.append((offset, piece))
        index += 1

    # Furniture ending the passage after an unfinished sentence: the
    # sentence goes on in the next passage.
    end = len(joined)
    while end > 1 and _furniture(joined[end - 1][1], running, blank=True):
        end -= 1
    if end < len(joined) and not _ENDED.search(joined[end - 1][1]):
        joined = joined[:end]
    return joined


def _furniture(text: str, running: set[str], *, blank: bool = False) -> bool:
    """Return whether a piece is only page furniture.

    Parameters
    ----------
    text: str
        A piece of a passage.
    running: set[str]
        The file's running heads and footers (``_line_key``).
    blank: bool, optional
        Whether an empty piece counts. Defaults to ``False``.

    Returns
    -------
    bool
        True when every line is a page number, a recognizable footer,
        or a repeated running head or footer; for an empty piece,
        ``blank``. A lone URL or DOI is not page furniture.

    """

    lines = [line for line in text.splitlines() if _line_key(line)]
    if not lines:
        return blank
    return all(
        _PAGE_NUMBER.fullmatch(line)
        or _FURNITURE.search(line)
        or _line_key(line) in running
        for line in lines
    )


def _line_key(line: str) -> str:
    """Return a line for comparing repeats: one space, no form feed.

    Parameters
    ----------
    line: str
        A line of a passage.

    Returns
    -------
    str
        The line's words joined by single spaces.

    """

    return " ".join(line.replace("\f", " ").split())


def _without_furniture(text: str, running: set[str]) -> str:
    """Drop page furniture printed before the rest of a sentence.

    Parameters
    ----------
    text: str
        A piece of a passage that may go on from an unfinished sentence.
    running: set[str]
        The file's running heads and footers (``_line_key``).

    Returns
    -------
    str
        ``text`` without its first lines while they are page furniture
        and the rest goes on from a sentence ("13\\naccumulation of");
        otherwise ``text``.

    """

    lines = text.lstrip().split("\n")
    for start in range(1, len(lines)):
        if not _furniture(lines[start - 1], running) and _line_key(
            lines[start - 1]
        ):
            break
        rest = "\n".join(lines[start:])
        if _CONTINUES.match(rest):
            return rest
    return text


def _complete(
    sentences: list[Sentence],
    seen: set[tuple[Path, str]],
    unfinished: tuple[str, int | None, int, Chunk],
    fragment: str,
    min_words: int,
) -> tuple[str, int | None, int, Chunk]:
    """Join a sentence split by a passage boundary, in place.

    Parameters
    ----------
    sentences: list[Sentence]
        The sentences so far; the unfinished one is replaced by the whole.
    seen: set[tuple[pathlib.Path, str]]
        Sentences already taken, by file and case-folded text.
    unfinished: tuple[str, int | None, int, Chunk]
        The previous passage's unfinished last sentence: its text, its
        index in ``sentences`` or None when too short to keep, its line,
        and its passage.
    fragment: str
        The start of the next passage, up to its first sentence end.
    min_words: int
        Minimum number of words.

    Returns
    -------
    tuple[str, int | None, int, Chunk]
        The joined sentence in the form of ``unfinished``, for a sentence
        that goes on into the passage after.

    Notes
    -----
    Passages overlap, so the fragment usually repeats the end of the
    unfinished part; the longest such repeat is dropped. Without one, the
    two are simply joined.

    """

    text, index, line, chunk = unfinished
    head, tail = text.split(), fragment.split()
    overlap = next(
        (
            size for size in range(min(len(head), len(tail)), 0, -1)
            if head[-size:] == tail[:size]
        ),
        0,
    )
    whole = " ".join([*head, *tail[overlap:]])
    if index is not None:
        seen.discard((chunk.path, text.casefold()))
    key = (chunk.path, whole.casefold())
    if len(whole.split()) < min_words or key in seen:
        if index is not None:
            del sentences[index]
        return whole, None, line, chunk
    seen.add(key)
    sentence = Sentence(chunk.path, line, whole, chunk)
    if index is None:
        index = len(sentences)
        sentences.append(sentence)
    else:
        sentences[index] = sentence
    return whole, index, line, chunk

