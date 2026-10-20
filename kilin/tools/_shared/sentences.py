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
        Sentences of at least four words, not split after abbreviations
        such as "Ref.", "Fig." or "Eq.". A sentence that overlapping
        passages share is kept once, from its first passage. A passage
        that begins inside a sentence completes the previous passage's
        unfinished last sentence (their overlap counted once); its broken
        start is otherwise left out.

    """

    seen: set[tuple[Path, str]] = set()
    sentences: list[Sentence] = []
    previous: Path | None = None
    # The previous passage's unfinished last sentence: its text, and its
    # index in ``sentences`` (None when it was too short to keep), with
    # the line and passage it starts in.
    unfinished: tuple[str, int | None, int, Chunk] | None = None
    for chunk in chunks:
        start = 0
        pieces = _SENTENCE_END.split(chunk.text)
        if chunk.path == previous and pieces and _FRAGMENT.match(pieces[0]):
            # A passage that starts inside a sentence ("tion for deep
            # learning ..."): the sentence started in the passage before.
            # Its end is here when that passage stopped before it ended.
            start = len(pieces[0])
            if unfinished is not None:
                unfinished = _complete(
                    sentences, seen, unfinished, pieces[0], min_words
                )
            # A passage inside one long sentence leaves it unfinished.
            carried = unfinished if len(pieces) == 1 else None
            pieces = pieces[1:]
        else:
            carried = None
        # The unfinished sentence, when this passage repeats it whole.
        repeated = unfinished if chunk.path == previous else None
        previous = chunk.path
        unfinished = carried

        for index, piece in enumerate(pieces):
            offset = chunk.text.find(piece, start)
            start = max(start, offset + len(piece))
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

