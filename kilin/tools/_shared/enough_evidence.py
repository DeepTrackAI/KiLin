"""Ask whether the passages found for a question are enough to answer it.

A quick answer is written from the few best-ranked passages, and the one
that states the fact may rank just below them: "which software does the
paper use?" can rank the introduction above the methods. Before writing,
a tool can ask the answering model, the one the user chose, whether the
passages it found answer the question, and read the next-ranked passages
when they do not. The model is told the passages are a search selection,
not the whole documents. The core uses ``passages_answer`` in the same way
for ordinary search answers.

The check only says yes or no. It never writes a search query: the next
passages come from the same ranking of the user's own words, so a search
again reads further down the list, not something the model imagined. An
unclear reply counts as enough, so a failed check never costs more reading.

Key Features
------------

- **Yes or no**

    ``passages_answer`` asks a model one short question and reads the
    first word of its reply.

- **Search again**

    ``rank_until_answered`` widens a ranked list one page at a time until
    the check says yes, the rounds run out, or no passages are left.

Module Structure
----------------

- ``passages_answer``

    Tell whether passages answer a question, according to a model.

- ``rank_until_answered``

    Take more ranked passages until they answer the question.

Examples
--------
>>> from kilin.tools._shared.enough_evidence import rank_until_answered
>>> picked, searches = rank_until_answered(
...     question, ranked, texts, page=10, rounds=1, ask=context.ask_check
... )

``ranked`` lists passage indices best first and ``texts`` holds the passage
texts; with the first ten passages judged not enough, ``picked`` holds the
first twenty and ``searches`` is 1.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re
from typing import Final

__all__ = ["passages_answer", "rank_until_answered"]

# The model replies with one word; a few tokens leave room for "Yes.".
_CHECK_MAX_TOKENS: Final[int] = 8
_MAX_CHECK_CHARACTERS: Final[int] = 12000
_FIRST_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z]+")


def passages_answer(
    question: str,
    passages: Sequence[str],
    ask: Callable[[str, str, int], str],
    *,
    max_characters: int = _MAX_CHECK_CHARACTERS,
) -> bool | None:
    """Tell whether passages answer a question, according to a model.

    Parameters
    ----------
    question: str
        The user's question.
    passages: Sequence[str]
        The passage texts found for it.
    ask: Callable[[str, str, int], str]
        Asks a model ``(system, prompt, max_new_tokens)`` with thinking
        off, such as ``ToolContext.ask_check``.
    max_characters: int, optional
        Maximum characters in the complete checking prompt. Larger
        evidence selections skip the check, not silently omit passages.
        Defaults to ``12000``.

    Returns
    -------
    bool | None
        True for "yes", False for "no"; None when the call fails, the
        prompt is too long, or the reply starts with neither word.

    """

    if type(max_characters) is not int or max_characters < 1:
        raise ValueError("max_characters must be positive.")
    system = (
        "You check search results for KíLín. You never answer the "
        "question. You reply with one word: yes or no."
    )
    blocks = "\n\n".join(
        f"[{number}] {text.strip()}"
        for number, text in enumerate(passages, start=1)
    )
    prompt = f"""A search of the documents selected these passages; they are
not the whole documents.

Passages:

{blocks}

Question: {question}

Do these passages state what is needed to answer the question? Reply yes
or no."""
    if len(system) + len(prompt) > max_characters:
        # A partial selection could create a spurious "no" and widen
        # retrieval unnecessarily. Preserve the evidence and answer.
        return None
    try:
        reply = ask(system, prompt, _CHECK_MAX_TOKENS)
    except Exception:  # A failed check must not stop the answer.
        return None
    word = _FIRST_WORD.match(reply.strip().lower())
    if word is None or word.group(0) not in ("yes", "no"):
        return None
    return word.group(0) == "yes"


def rank_until_answered(
    question: str,
    ranked: Sequence[int],
    texts: Sequence[str],
    *,
    page: int,
    rounds: int,
    ask: Callable[[str, str, int], str],
    max_characters: int = _MAX_CHECK_CHARACTERS,
) -> tuple[list[int], int]:
    """Take more ranked passages until they answer the question.

    Parameters
    ----------
    question: str
        The user's question.
    ranked: Sequence[int]
        Indices into ``texts``, best-ranked first.
    texts: Sequence[str]
        Passage texts.
    page: int
        Passages taken first, and added by each search again.
    rounds: int
        Most searches again; zero takes the first page without a check.
    ask: Callable[[str, str, int], str]
        Asks a model with thinking off, as for ``passages_answer``.
    max_characters: int, optional
        Maximum characters of the complete check prompt. Defaults to
        ``12000``.

    Returns
    -------
    tuple[list[int], int]
        The picked indices, best-ranked first, and the number of searches
        again that were made.

    """

    if type(page) is not int or page < 1:
        raise ValueError("page must be positive.")
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be nonnegative.")
    picked = list(ranked[:page])
    searches = 0
    while searches < rounds and len(picked) < len(ranked):
        found = [texts[index] for index in picked]
        # Only a clear "no" reads further; an unclear reply is enough.
        if passages_answer(
            question, found, ask, max_characters=max_characters
        ) is not False:
            break
        searches += 1
        picked = list(ranked[: page * (searches + 1)])
    return picked, searches
