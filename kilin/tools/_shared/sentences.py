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

_SENTENCE_END: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\s*\n"
)
_FRAGMENT: Final[re.Pattern[str]] = re.compile(r"^\s*[a-z]")

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
        Sentences of at least four words. A sentence that overlapping
        passages share is kept once, from its first passage; the broken
        start of a passage that begins inside a sentence is left out.

    """

    seen: set[tuple[Path, str]] = set()
    sentences: list[Sentence] = []
    previous: Path | None = None
    for chunk in chunks:
        start = 0
        pieces = _SENTENCE_END.split(chunk.text)
        if chunk.path == previous and pieces and _FRAGMENT.match(pieces[0]):
            # A passage that starts inside a sentence ("tion for deep
            # learning ..."): the sentence is in the passage before.
            start = len(pieces[0])
            pieces = pieces[1:]
        previous = chunk.path

        for piece in pieces:
            offset = chunk.text.find(piece, start)
            start = max(start, offset + len(piece))
            text = " ".join(piece.split())
            if len(text.split()) < min_words:
                continue

            key = (chunk.path, text.casefold())
            if key in seen:
                continue

            seen.add(key)
            line = chunk.start_line + chunk.text[: max(0, offset)].count("\n")
            sentences.append(Sentence(chunk.path, line, text, chunk))

    return sentences

