"""Judge the items of an earlier answer against a follow-up question.

"Among those 7, how many are about XYZ 2?" or "of these sentences, which
discuss limitations?" asks about the items an earlier answer listed, not
about the documents again. This tool takes the evidence of the turn the
question points back to (the listed mentions, sentences, or passages) and
has the model judge each item against the follow-up, in batches; the model
replies with item numbers only. The answer lists the items that match with
their files and lines. Nothing is searched again, and the items are copied,
not rewritten.

The earlier turn is the newest one the router related to the question,
otherwise the newest turn with evidence (see
``show_source.newest_listed``). Each item is cut to ``item_characters``
characters, both for the model and in the answer; the evidence keeps the
full text. Numbers in a reply that are not in the batch are ignored. A
failed model call stops the tool; it is not retried.

Key Features
------------

- **No new search**

    The items come from the earlier turn's evidence, in its order.

- **Judged by number**

    The model only gives item numbers, so the listed items are copied word
    for word.

Module Structure
----------------

- ``FilterSettings``

    Settings of ``filter_results`` in a project type.

- ``parse_settings``

    Validate ``filter_results`` settings from a project type.

- ``filter_prompt``

    Prompts that ask the model which items match a follow-up.

- ``run``

    Judge each item of the earlier answer and list those that match.

Examples
--------
``context`` is the ``ToolContext`` that QAChat passes to a tool, and
``earlier`` holds the turns the follow-up points back to:

>>> from qachat.tools import ToolRequest
>>> from qachat.tools.filter_results import FilterSettings, run
>>> question = "among those, which are about XYZ 2?"
>>> request = ToolRequest(question, question, earlier=earlier)
>>> result = run(FilterSettings(), request, context)

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Final

from qachat.retrievals.document_reading import estimate_tokens
from qachat.tools.contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)
from qachat.tools.show_source import newest_listed

__all__ = [
    "FilterSettings",
    "filter_prompt",
    "parse_settings",
    "run",
]

# Estimated tokens that an item's number and blank line add to a batch.
_ITEM_NUMBER_TOKENS: Final[int] = 4
# Longest reply of one batch: its matching item numbers.
_REPLY_TOKENS: Final[int] = 120


@dataclass(frozen=True, slots=True)
class FilterSettings:
    """Settings of ``filter_results`` in a project type.

    Attributes
    ----------
    batch_tokens: int
        Estimated tokens of items per model call (500-8000).
    item_characters: int
        Longest item text shown to the model and in the answer (100-2000).

    """

    batch_tokens: int = 2500
    item_characters: int = 400


def parse_settings(data: dict, where: str) -> FilterSettings:
    """Validate ``filter_results`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    FilterSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, ("batch_tokens", "item_characters"))

    defaults = FilterSettings()
    return FilterSettings(
        batch_tokens=check_int(
            where, data, "batch_tokens", defaults.batch_tokens, 500, 8000
        ),
        item_characters=check_int(
            where,
            data,
            "item_characters",
            defaults.item_characters,
            100,
            2000,
        ),
    )


def filter_prompt(
    question: str, items: Sequence[tuple[int, str]]
) -> tuple[str, str]:
    """Return the prompts that pick the items matching a follow-up.

    Parameters
    ----------
    question: str
        The follow-up, such as "among those, which are about XYZ 2?".
    items: Sequence[tuple[int, str]]
        Item numbers and texts of one batch.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is item numbers.

    """

    system = (
        "You judge which items of a list match a request. You reply with "
        "item numbers only, separated by commas, or NONE."
    )
    listed = "\n\n".join(f"{number}. {text}" for number, text in items)
    user = f"""Request: {question}

Which of these items match the request? Judge each item by what it says.
Reply with their numbers, for example: 2, 5. Reply NONE when no item
matches.

{listed}"""
    return system, user


def run(
    settings: FilterSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Judge each item of the earlier answer and list those that match.

    Parameters
    ----------
    settings: FilterSettings
        The intent's settings.
    request: ToolRequest
        The follow-up, with the earlier turns it points back to.
    context: ToolContext
        QAChat's services.

    Returns
    -------
    ToolResult
        The count and the matching items with their locations, the items
        as evidence, and a summary.

    """

    summary: dict[str, object] = {"name": "filter_results"}
    turn = newest_listed(request.earlier)
    if turn is None:
        context.done("No earlier list", evidence_count=0)
        return ToolResult(
            "I couldn't find an earlier answer with a list to narrow down. "
            "Ask the list question first, then ask about its items.",
            (),
            summary,
        )

    items = [
        (number, " ".join(text.split())[: settings.item_characters])
        for number, text in enumerate(turn.texts, start=1)
    ]
    batches = _batches(items, settings.batch_tokens)
    picked: list[int] = []
    for index, batch in enumerate(batches, start=1):
        context.progress(
            f"Judging the {len(items)} items of turn {turn.number} "
            f"(batch {index} of {len(batches)})"
        )
        system, prompt = filter_prompt(request.question, batch)
        reply = context.ask_model(system, prompt, _REPLY_TOKENS)
        numbers = {number for number, _ in batch}
        picked += sorted(
            {int(n) for n in re.findall(r"\d+", reply) if int(n) in numbers}
        )
    summary.update(
        {"turn": turn.number, "items": len(items), "matched": len(picked)}
    )
    context.done(
        f"{len(picked)} of {len(items)} items match",
        evidence_count=len(picked),
    )

    lines = [
        f"**{len(picked)} of {len(items)}** items from turn {turn.number} "
        "match."
    ]
    evidence = []
    for citation, number in enumerate(picked, start=1):
        path = turn.paths[number - 1]
        line = turn.lines[number - 1] if turn.lines else 0
        lines.append(
            f"{citation}. {path.name}, line {line}: “{items[number - 1][1]}” "
            f"[{citation}]"
        )
        evidence.append(
            {
                "citation": citation,
                "path": path,
                "start_line": line,
                "end_line": line,
                "symbol": None,
                "text": turn.texts[number - 1],
                "cell": None,
            }
        )
    if not picked:
        lines.append("None of the items matches.")
    return ToolResult("\n".join(lines), tuple(evidence), summary)


def _batches(
    items: Sequence[tuple[int, str]], tokens: int
) -> list[list[tuple[int, str]]]:
    """Pack consecutive items into batches of about ``tokens`` tokens.

    An item larger than ``tokens`` gets a batch of its own.

    """

    batches: list[list[tuple[int, str]]] = []
    used = 0
    for item in items:
        size = estimate_tokens(item[1]) + _ITEM_NUMBER_TOKENS
        if batches and used + size <= tokens:
            batches[-1].append(item)
            used += size
        else:
            batches.append([item])
            used = size
    return batches
