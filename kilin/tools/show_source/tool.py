"""Show the exact source text behind an earlier answer, and explain it.

"Show me that sentence" or "what is that passage talking about?" asks
about the text an earlier answer cited, not about the documents again.
This tool takes the evidence of the turn the question points back to and
picks the passages and the sentence the question means, by the words they
share with the question and the earlier answer. Words that every candidate
contains count for little, so rare shared words decide. Each sentence is
shown word for word, with the sentences before and after it taken from the
neighbouring passages in the index. The model then explains what the shown
text is talking about, from that text only. Nothing is searched again.

A cited passage is found in the index by its file and first line, or by
its text. When the index no longer holds it, the cited text is shown on
its own. When no sentence shares a word with the question or the earlier
answer, the whole passage is shown without a marked sentence.

Key Features
------------

- **No new search**

    The passages come from the earlier turn's evidence.

- **The sentence in context**

    The cited sentence is marked and shown with ``context_sentences``
    sentences on each side, across passage boundaries.

- **Explanation from the shown text**

    The model explains only the text it is shown and cites it.

Module Structure
----------------

- ``SourceSettings``

    Settings of ``show_source`` in a project type.

- ``parse_settings``

    Validate ``show_source`` settings from a project type.

- ``explain_prompt``

    Prompts that explain the shown source text.

- ``run``

    Show the cited sentences with their surroundings and explain them.

- ``newest_listed``

    Return the earlier turn whose evidence a follow-up is about.

Private helpers score words, pick the passages, and cut the window of
sentences; they are not part of the public API.

Examples
--------
``context`` is the ``ToolContext`` that KíLín passes to a tool, and
``earlier`` holds the turns the follow-up points back to:

>>> from kilin.tools import ToolRequest
>>> from kilin.tools.show_source import SourceSettings, run
>>> question = "show me that sentence"
>>> request = ToolRequest(question, question, earlier=earlier)
>>> result = run(SourceSettings(), request, context)

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Final

from kilin.chunking import Chunk
from kilin.tools.show_source import _configuration as _config
from kilin.tools._shared.earlier_turns import newest_listed
from kilin.tools._contract import (
    EarlierTurn,
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.sentences import Sentence, split_sentences

__all__ = [
    "SourceSettings",
    "explain_prompt",
    "newest_listed",
    "parse_settings",
    "run",
]

# A word: three or more letters or digits. Shorter words, such as "of" or
# "is", say little about which sentence is meant.
_WORD: Final[re.Pattern[str]] = re.compile(r"[^\W_]{3,}")


@dataclass(frozen=True, slots=True)
class SourceSettings:
    """Settings of ``show_source`` in a project type.

    Attributes
    ----------
    context_sentences: int
        Sentences shown before and after the cited sentence (0-10).
    max_passages: int
        Most cited passages shown (1-10).

    """

    context_sentences: int = _config.DEFAULT_CONTEXT_SENTENCES
    max_passages: int = _config.DEFAULT_MAX_PASSAGES


def parse_settings(data: dict, where: str) -> SourceSettings:
    """Validate ``show_source`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    SourceSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(
        where, data, ("context_sentences", "max_passages")
    )

    defaults = SourceSettings()
    return SourceSettings(
        context_sentences=check_int(
            where,
            data,
            "context_sentences",
            defaults.context_sentences,
            0,
            10,
        ),
        max_passages=check_int(
            where, data, "max_passages", defaults.max_passages, 1, 10
        ),
    )


def explain_prompt(
    question: str, windows: Sequence[tuple[int, str, str]]
) -> tuple[str, str]:
    """Return the prompts that explain the shown source text.

    Parameters
    ----------
    question: str
        The follow-up, such as "what is that sentence talking about?".
    windows: Sequence[tuple[int, str, str]]
        Per passage: its citation number, its file name, and its text with
        the cited sentence between ``>>`` and ``<<``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    system = (
        "You explain source text for KíLín. Use only the text shown, cite "
        "it like [1], and say plainly when the text does not make "
        "something clear."
    )
    shown = "\n\n".join(
        f"[{number}] {name}:\n{text}" for number, name, text in windows
    )
    user = f"""Source text (the cited sentence is between >> and <<):

{shown}

Explain what the cited sentence is talking about, using the sentences
around it. Keep it short unless the question asks for a length or format;
then follow it exactly.

Question: {question}"""
    return system, user


def run(
    settings: SourceSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Show the cited sentences with their surroundings and explain them.

    Parameters
    ----------
    settings: SourceSettings
        The intent's settings.
    request: ToolRequest
        The follow-up, with the earlier turns it points back to.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The source text word for word with an explanation, the shown text
        as evidence, and a summary.

    """

    summary: dict[str, object] = {"name": "show_source"}
    turn = newest_listed(request.earlier)
    if turn is None:
        context.done("No earlier sources", evidence_count=0)
        return ToolResult(
            "I couldn't find an earlier answer with sources to show. Ask "
            "the question first, then ask to see its source.",
            (),
            summary,
        )

    # The words of the question and of the earlier answer point at the
    # passages and the sentence meant.
    clue = _words(f"{request.question} {turn.answer}")
    items = _pick(turn, clue, settings.max_passages)
    context.progress(
        f"Showing {len(items)} cited "
        f"{'passage' if len(items) == 1 else 'passages'} from turn "
        f"{turn.number}"
    )
    shown: list[tuple[Path, list[Sentence], int | None]] = []
    for index in items:
        sentences, focus = _window(
            context.chunks,
            turn.paths[index],
            turn.lines[index] if turn.lines else 0,
            turn.texts[index],
            clue,
            settings.context_sentences,
        )
        shown.append((turn.paths[index], sentences, focus))
    context.done(
        f"Read {len(shown)} cited "
        f"{'passage' if len(shown) == 1 else 'passages'}",
        evidence_count=len(shown),
    )

    windows = [
        (number, path.name, _join(sentences, focus, ">>", "<<"))
        for number, (path, sentences, focus) in enumerate(shown, start=1)
    ]
    system, prompt = explain_prompt(request.question, windows)
    explanation = context.ask_model(system, prompt, None)

    lines: list[str] = []
    evidence: list[dict[str, object]] = []
    for number, (path, sentences, focus) in enumerate(shown, start=1):
        first = sentences[0].line if sentences else 0
        lines.append(f"**{path.name}, line {first}** [{number}]")
        lines.append("> " + _join(sentences, focus, "**", "**"))
        lines.append("")
        evidence.append(
            {
                "citation": number,
                "path": path,
                "start_line": first,
                "end_line": sentences[-1].line if sentences else first,
                "symbol": None,
                "text": " ".join(sentence.text for sentence in sentences),
                "cell": None,
            }
        )
    summary.update({"turn": turn.number, "passages": len(shown)})
    answer = "\n".join(lines) + explanation.strip()
    return ToolResult(answer, tuple(evidence), summary)




def _words(text: str) -> set[str]:
    """Return the words of a text, lower-case, without a final "s".

    Dropping the "s" lets "epoch" match "epochs"; three-letter words keep it.

    Parameters
    ----------
    text: str
        Text to analyze or transform.

    Returns
    -------
    set[str]
        the words of a text, lower-case, without a final "s".

    """

    return {
        word[:-1] if word.endswith("s") and len(word) > 3 else word
        for word in (match.casefold() for match in _WORD.findall(text))
    }


def _score(texts: Sequence[str], clue: set[str]) -> list[float]:
    """Score texts by the clue words they contain, rare words weighing most.

    A word in every text says little about which one is meant, so each
    word counts ``log((len(texts) + 1) / k)``, where ``k`` is the number of
    texts that contain it.

    Parameters
    ----------
    texts: Sequence[str]
        Text passages or fields to inspect.
    clue: set[str]
        Words used to locate the requested passage.

    Returns
    -------
    list[float]
        Relevance scores corresponding to the input texts.

    """

    words = [_words(text) & clue for text in texts]
    spread = Counter(word for found in words for word in found)
    return [
        sum(math.log((len(texts) + 1) / spread[word]) for word in found)
        for found in words
    ]


def _pick(turn: EarlierTurn, clue: set[str], limit: int) -> list[int]:
    """Return the evidence items the follow-up means, in their order.

    All items are kept when there are at most ``limit``; otherwise the
    ``limit`` best-scored ones, ties going to the earlier item.

    Parameters
    ----------
    turn: EarlierTurn
        Conversation turn to inspect.
    clue: set[str]
        Words used to locate the requested passage.
    limit: int
        Maximum number of items or characters to include.

    Returns
    -------
    list[int]
        the evidence items the follow-up means, in their order.

    """

    if len(turn.texts) <= limit:
        return list(range(len(turn.texts)))

    scores = _score(turn.texts, clue)
    best = sorted(
        range(len(scores)), key=lambda index: (-scores[index], index)
    )
    return sorted(best[:limit])


def _window(
    chunks: Sequence[Chunk],
    path: Path,
    line: int,
    text: str,
    clue: set[str],
    around: int,
) -> tuple[list[Sentence], int | None]:
    """Return the cited passage's sentences in context, and the focus.

    The cited passage is found in the index by its file and first line (or
    its text); its neighbours in the same file give the sentences before
    and after. The focus is the cited sentence that best matches the
    clue; None when no sentence matches, and then the whole passage is
    shown.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Indexed source chunks available for selection.
    path: Path
        Path to the target file or directory.
    line: int
        Source line around which to locate a passage.
    text: str
        Text to analyze or transform.
    clue: set[str]
        Words used to locate the requested passage.
    around: int
        Number of neighboring sentences to include.

    Returns
    -------
    tuple[list[Sentence], int | None]
        the cited passage's sentences in context, and the focus.

    """

    paper = [chunk for chunk in chunks if chunk.path == path]
    found = next(
        (
            index
            for index, chunk in enumerate(paper)
            if chunk.start_line == line or chunk.text == text
        ),
        None,
    )
    if found is None:
        # No longer indexed as it was: show the cited text on its own.
        alone = Chunk(
            identifier="cited",
            path=path,
            text=text,
            start_line=line,
            end_line=line,
            kind="text",
            symbol=None,
            section=None,
        )
        sentences = split_sentences([alone], min_words=_config.MIN_WORDS) or [
            Sentence(path, line, " ".join(text.split()), alone)
        ]
        return sentences, None

    # The passages before and after give the surrounding sentences.
    nearby = paper[max(0, found - 1) : found + 2]
    sentences = split_sentences(nearby, min_words=_config.MIN_WORDS)
    cited = " ".join(paper[found].text.split()).casefold()
    inside = [
        index
        for index, sentence in enumerate(sentences)
        if sentence.text.casefold() in cited
    ]
    if not inside:
        return sentences, None

    scores = _score([sentences[index].text for index in inside], clue)
    if max(scores) <= 0:
        return [sentences[index] for index in inside], None

    focus = inside[scores.index(max(scores))]
    start = max(0, focus - around)
    return sentences[start : focus + around + 1], focus - start


def _join(
    sentences: Sequence[Sentence], focus: int | None, before: str, after: str
) -> str:
    """Join sentences, marking the focus sentence with the given marks.

    Parameters
    ----------
    sentences: Sequence[Sentence]
        Sentences to arrange or render.
    focus: int | None
        Index of the sentence to emphasize, if any.
    before: str
        Prefix used to mark the selected sentence.
    after: str
        Suffix used to mark the selected sentence.

    Returns
    -------
    str
        Joined sentence text with the selected sentence delimited.

    """

    return " ".join(
        f"{before}{sentence.text}{after}" if index == focus else sentence.text
        for index, sentence in enumerate(sentences)
    )
